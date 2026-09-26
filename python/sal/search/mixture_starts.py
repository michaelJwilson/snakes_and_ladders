"""Starts of a count-pair emission mixture, each polished by EM and timed through ``track`` (issue #891).

The joint count-pair mixture of :mod:`sal.sim.emission_mixture`
alone --- no lattice, no chain --- started every way the package can start it,
then polished by :func:`sal.opt.emission_mixture.expectation_maximization`
at one budget. Every start ends at the same seam, the instance's
:data:`~sal.opt.emission_mixture.ComponentsAt`, so what
separates two starts is where they place the components and nothing after.

**Four kinds of start.** A prior draw reads no pair. Rules over the pairs
place a component on each of ``C`` chosen observations: the uniform draw,
``Emission_Mixture++``, k-means++ on the raw pair, and a short EM burn-in on a
subsample from the uniform draw. Starts that take an ``Objective`` run on a
surrogate, the Gaussian mixture over both channels, because the count-pair
mixture has no ``theta`` and is not an ``Objective``: the four of
:mod:`sal.opt.initialize` and the three of
:mod:`sal.sample.initialize`. The Gaussian-mixture EM start runs
on the first channel alone, since
:func:`sal.opt.mixture.expectation_maximization` fits one
channel. A surrogate start yields locations, and :func:`at_locations` realizes
them at the observations nearest them.

**A start that iterates records its iterations.** The Gaussian EM records its
surrogate log-likelihood per iteration and keeps the components it would hand
over every :data:`PATH_STRIDE` iterations; the burn-in keeps each of its
iterations and ``FromChain`` each draw it keeps after its warm-up. The
annealing and tempering runs return their best point and not the path to it,
so they carry no path. :class:`TimedStart` evaluates each path entry's
count-pair log-likelihood after the clock stops, so the evaluation costs the
start nothing.

**Cost is counted in passes and in seconds.** A pass is every observation
scored under every component, one per EM iteration and two per surrogate
gradient; the seconds are what ``track`` records.

**One budget in seconds covers a start and its polish** (issue #898). The
polish runs until the log-likelihood's relative change between two iterations
is at most :data:`POLISH_TOLERANCE` or until its next iteration would pass
the seconds the start left, and :class:`TimedStart` reports the cell's whole
seconds against the budget. A polish of a fixed number of passes remains, for
a comparison that holds the passes equal instead.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from sal.cost import Cost
from sal.emissions import EmissionFamily
from sal.opt.budget import Budget, Outcome
from sal.opt.em import EmConfig
from sal.opt.emission_mixture import (
    ComponentsAt,
    SeedMethod,
    expectation_maximization,
    seed,
)
from sal.opt.initialize import (
    FromObjective,
    Perturbed,
    RandomRestart,
    quantile_locations,
)
from sal.opt.mixture import (
    GaussianMixtureObjective,
    KMeansPlusPlus,
    mixture_log_likelihood,
    responsibilities_torch,
)
from sal.opt.mixture import (
    expectation_maximization as gaussian_expectation_maximization,
)
from sal.opt.starts import Curve, Polished, Trial
from sal.opt.termination import Stop, Termination
from sal.parallel import Pool, map_tasks
from sal.sample.chain import torch_stream
from sal.sample.initialize import FromAnnealing, FromChain, FromTempering
from sal.sample.mixture_anneal import anneal_assignments
from sal.sample.schedule import ExponentialTempSchedule, ladder
from sal.search.projection import (
    ANNEAL_STEPS,
    CHAIN_BURN_IN,
    CHAIN_DRAWS,
    CHAIN_STEP,
    CHAIN_TRAJECTORY,
    GAUSSIAN_EM_ITERATIONS,
    PASSES_PER_GRADIENT,
    RESTART_SCALE,
    RESTARTS,
    TEMPERATURES,
    TEMPERING_ROUNDS,
    Seeding,
    match_components,
)
from sal.sim.emission_mixture import SimulatedEmissionMixtureDataset
from sal.track import MemoryRun, current, track

#: Iterations of the Gaussian EM between two entries of its path: 20 entries
#: over its 500 iterations.
PATH_STRIDE = 25

#: The context the handover is recorded under, so its one sample is a series
#: of its own beside the polish's.
HANDOVER = {"phase": "handover"}


@dataclass(frozen=True)
class MixtureInstance:
    """One drawn mixture, its truth, and the seam every start ends at.

    Parameters
    ----------
    observations : np.ndarray
        The pairs, shape ``(n_samples, 2)``.
    labels : np.ndarray
        The generating component of each pair, shape ``(n_samples,)``.
    weights : np.ndarray
        The generating mixing weights.
    truth : EmissionFamily
        The generating components.
    at : ComponentsAt
        Places a component on each of a set of pairs.
    covariate : np.ndarray | None
        Per-pair covariate, shape ``(n_samples, 2)``: every fit and every
        likelihood conditions on it (issue #933).
    seeding_rows : np.ndarray | None
        Where the starts read the pairs from, when that is not the pairs
        themselves: under a covariate,
        :func:`~sal.sim.count_pairs.rate_space`'s rows, so a
        start places components at rates rather than at raw counts.
    """

    observations: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    truth: EmissionFamily
    at: ComponentsAt
    covariate: np.ndarray | None = None
    seeding_rows: np.ndarray | None = None

    @property
    def rows(self) -> np.ndarray:
        """The pairs a start seeds from: :attr:`seeding_rows`, or the observations."""
        return self.observations if self.seeding_rows is None else self.seeding_rows

    @property
    def conditioned(self) -> torch.Tensor | None:
        """:attr:`covariate` as the tensor the fits take."""
        if self.covariate is None:
            return None
        return torch.as_tensor(self.covariate, dtype=torch.float64)

    @property
    def n_components(self) -> int:
        """Components the truth carries, which every start seeds."""
        return self.truth.n_states

    @property
    def n_samples(self) -> int:
        """Pairs drawn."""
        return int(self.observations.shape[0])

    @property
    def reference(self) -> float:
        """The log-likelihood the generating parameters reach on this draw."""
        values = torch.as_tensor(self.observations, dtype=torch.float64)
        log_weight = torch.log(torch.as_tensor(self.weights, dtype=torch.float64))
        return float(
            mixture_log_likelihood(
                values, log_weight, self.truth, covariate=self.conditioned
            )
        )


def instance_from(
    dataset: SimulatedEmissionMixtureDataset, at: ComponentsAt
) -> MixtureInstance:
    """The instance a simulated dataset and a seam make.

    Returns
    -------
    MixtureInstance
    """
    return MixtureInstance(
        observations=np.asarray(dataset.observations),
        labels=np.asarray(dataset.labels),
        weights=np.asarray(dataset.weights, dtype=np.float64),
        truth=dataset.components,
        at=at,
    )


def surrogate(instance: MixtureInstance) -> GaussianMixtureObjective:
    """The Gaussian mixture over both channels: the ``Objective`` a surrogate start reads.

    Returns
    -------
    GaussianMixtureObjective
    """
    return GaussianMixtureObjective(
        np.asarray(instance.rows, dtype=np.float64), instance.n_components
    )


def at_locations(instance: MixtureInstance, locations: torch.Tensor) -> EmissionFamily:
    """The components seeded on the observation nearest each location.

    ``locations`` is ``(C, 2)`` for a location in both channels, where
    nearest is Euclidean over the pair, or ``(C,)`` for one in the first
    channel alone, where it is nearest in that channel.

    Returns
    -------
    EmissionFamily
    """
    rows = np.asarray(instance.rows, dtype=np.float64)
    located = np.asarray(locations.detach().numpy(), dtype=np.float64)
    if located.ndim == 1:
        distance = np.abs(rows[None, :, 0] - located[:, None])
    else:
        distance = ((rows[None, :, :] - located[:, None, :]) ** 2).sum(axis=-1)
    return instance.at(rows[distance.argmin(axis=1)])


def prior_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """Components drawn from a prior over the observed range, reading no pair.

    A total log-uniform on the observed range of totals and an allele fraction
    uniform on ``(0, 1)``, the family's own support, each handed to the seam
    as the pair ``(total, fraction * total)``: the control that says whether
    reading the data earns its cost.

    Returns
    -------
    Seeding
    """
    totals = np.asarray(instance.rows, dtype=np.float64)[:, 0]
    low, high = float(max(totals.min(), 1.0)), float(max(totals.max(), 2.0))
    means = np.exp(rng.uniform(np.log(low), np.log(high), size=instance.n_components))
    rates = rng.uniform(0.0, 1.0, size=instance.n_components)
    return Seeding(instance.at(np.stack([means, rates * means], axis=1)), 0.0)


def data_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``uniform_start``: components on pairs drawn uniformly without replacement.

    Returns
    -------
    Seeding
    """
    start = seed(
        instance.observations,
        instance.n_components,
        instance.at,
        method=SeedMethod.UNIFORM,
        rng=rng,
        rows=instance.rows,
    )
    return Seeding(start.components, 0.0)


def emission_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``plus_plus_start``: D-squared sampling under the family's Bregman divergence.

    Returns
    -------
    Seeding
    """
    start = seed(
        instance.observations,
        instance.n_components,
        instance.at,
        method=SeedMethod.PLUS_PLUS,
        rng=rng,
        rows=instance.rows,
    )
    return Seeding(start.components, 1.0)


def kmeans_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``kmeans_plus_plus`` on the raw pair, its centres handed to the seam.

    Returns
    -------
    Seeding
    """
    start = seed(
        instance.observations,
        instance.n_components,
        instance.at,
        method=SeedMethod.KMEANS,
        rng=rng,
        rows=instance.rows,
    )
    return Seeding(start.components, 1.0)


def gaussian_em_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """The Gaussian mixture fitted to the first channel from k-means++, its means the locations.

    One EM iteration per call, so each is recorded into the enclosing run,
    with the components its means would seed every :data:`PATH_STRIDE`
    iterations and at the last.

    Returns
    -------
    Seeding
    """
    channel = np.asarray(instance.rows, dtype=np.float64)[:, 0]
    objective = GaussianMixtureObjective(channel, instance.n_components)
    start = KMeansPlusPlus(1, rng).starts(objective)[0]
    weights = torch.exp(objective.constrain(start)["log_weight"]).detach()
    components = objective.components(start)
    tracked = current()
    path: list[tuple[int, EmissionFamily]] = []
    for iteration in range(GAUSSIAN_EM_ITERATIONS):
        fitted = gaussian_expectation_maximization(
            channel,
            weights,
            components,
            config=EmConfig(max_iterations=1, tolerance=0.0),
        )
        weights, components = fitted.weights, fitted.components
        tracked.record(iteration, surrogate_log_likelihood=fitted.log_likelihood)
        last = iteration == GAUSSIAN_EM_ITERATIONS - 1
        if iteration % PATH_STRIDE == 0 or last:
            path.append((iteration, at_locations(instance, components.mean)))
    return Seeding(
        path[-1][1],
        float(GAUSSIAN_EM_ITERATIONS),
        f"budget after {GAUSSIAN_EM_ITERATIONS}",
        tuple(path),
    )


#: Fraction of the pairs the burn-in fits on, and the iterations it runs.
BURN_IN_FRACTION = 0.2
BURN_IN_ITERATIONS = 3


def burn_in_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """A short EM fit on a subsample, started from the data draw.

    :data:`BURN_IN_ITERATIONS` iterations on a :data:`BURN_IN_FRACTION`
    subsample, each charged its fraction of a pass; each iteration's
    components are an entry of the path.

    Returns
    -------
    Seeding
    """
    components = data_seeding(instance, rng).components
    size = max(instance.n_components, int(BURN_IN_FRACTION * instance.n_samples))
    chosen = rng.choice(instance.n_samples, size=size, replace=False)
    subsample = instance.observations[chosen]
    covariate = None if instance.covariate is None else instance.covariate[chosen]
    weights = torch.full(
        (instance.n_components,), 1.0 / instance.n_components, dtype=torch.float64
    )
    tracked = current()
    path: list[tuple[int, EmissionFamily]] = []
    for iteration in range(BURN_IN_ITERATIONS):
        fitted = expectation_maximization(
            subsample,
            weights,
            components,
            covariate=covariate,
            config=EmConfig(max_iterations=1, tolerance=0.0),
        )
        weights, components = fitted.weights, fitted.components
        tracked.record(iteration, subsample_log_likelihood=fitted.log_likelihood)
        path.append((iteration, components))
    return Seeding(
        components,
        BURN_IN_ITERATIONS * size / instance.n_samples,
        f"{size} pairs, {BURN_IN_ITERATIONS} iterations",
        tuple(path),
    )


def perturbation() -> Perturbed:
    """The tilt :func:`perturbed_seeding` applies: ``Perturbed`` at its declared magnitude."""
    return Perturbed()


def restart_initializer(rng: np.random.Generator) -> RandomRestart:
    """The restart set :func:`restart_seeding` draws: :data:`RESTARTS` points at :data:`RESTART_SCALE`."""
    return RandomRestart(RESTARTS, RESTART_SCALE, rng)


def chain_initializer(rng: np.random.Generator) -> FromChain:
    """The chain :func:`chain_seeding` runs, its stream derived from ``rng``, at ``FromChain``'s default warm-up."""
    return FromChain(
        CHAIN_DRAWS,
        CHAIN_STEP,
        torch_stream(rng),
        n_steps=CHAIN_TRAJECTORY,
        burn_in=CHAIN_BURN_IN,
    )


def annealing_initializer(rng: np.random.Generator) -> FromAnnealing:
    """The annealing run :func:`annealed_seeding` makes: hottest rung to 1 over :data:`ANNEAL_STEPS`."""
    return FromAnnealing(
        ExponentialTempSchedule(float(TEMPERATURES[-1]), 1.0, ANNEAL_STEPS),
        CHAIN_STEP,
        torch_stream(rng),
        n_steps=CHAIN_TRAJECTORY,
    )


def tempering_initializer(rng: np.random.Generator) -> FromTempering:
    """The ladder :func:`tempered_seeding` runs: :data:`TEMPERATURES`, :data:`TEMPERING_ROUNDS` rounds."""
    return FromTempering(
        TEMPERATURES,
        TEMPERING_ROUNDS,
        CHAIN_STEP,
        torch_stream(rng),
        n_steps=CHAIN_TRAJECTORY,
    )


def objective_seeding(
    instance: MixtureInstance, _rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``FromObjective``: the surrogate's own nominated point.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    theta = FromObjective().starts(objective)[0]
    return Seeding(at_locations(instance, objective.components(theta).mean), 0.0)


def perturbed_seeding(
    instance: MixtureInstance, _rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``Perturbed``: that point, tilted off a symmetry it may be stationary at.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    theta = perturbation().starts(objective)[0]
    return Seeding(at_locations(instance, objective.components(theta).mean), 0.0)


def restart_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``RandomRestart``: the best by surrogate value of points drawn around it.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    thetas = restart_initializer(rng).starts(objective)
    best = min(thetas, key=lambda theta: float(objective(theta)))
    return Seeding(
        at_locations(instance, objective.components(best).mean), float(len(thetas))
    )


def quantile_seeding(
    instance: MixtureInstance, _rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``quantile_locations``: each channel's evenly spaced quantiles, paired in order.

    Returns
    -------
    Seeding
    """
    values = torch.as_tensor(
        np.asarray(instance.rows, dtype=np.float64), dtype=torch.float64
    )
    return Seeding(
        at_locations(
            instance, quantile_locations(values, instance.n_components, dim=0)
        ),
        0.0,
    )


def chain_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``FromChain``: a short Hamiltonian chain on the surrogate, warmed up; its last draw seeds.

    The warm-up is ``FromChain``'s default,
    :data:`~sal.sample.initialize.CHAIN_ADAPTATION` (issue
    #898), and its gradients are in the passes charged.

    Returns
    -------
    Seeding
        Its path is every kept draw, at the step the chain recorded it.
    """
    objective = surrogate(instance)
    chain = chain_initializer(rng).chain(objective)
    path = tuple(
        (draw, at_locations(instance, objective.components(theta).mean))
        for draw, theta in enumerate(chain.draws)
    )
    return Seeding(
        path[-1][1],
        PASSES_PER_GRADIENT * chain.spent,
        f"acceptance {chain.acceptance_rate:.2f}"
        + (
            f", adapted step {chain.adapted.step_size:.3g}"
            if chain.adapted is not None
            else ""
        ),
        path,
    )


def annealed_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``FromAnnealing``: the best point of a falling temperature.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    run = annealing_initializer(rng).run(objective)
    return Seeding(
        at_locations(instance, objective.components(run.best).mean),
        PASSES_PER_GRADIENT * run.spent,
        f"acceptance {run.acceptance_rate:.2f}",
    )


def tempered_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``FromTempering``: the best point at any temperature of a ladder.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    run = tempering_initializer(rng).run(objective)
    return Seeding(
        at_locations(instance, objective.components(run.best).mean),
        PASSES_PER_GRADIENT * run.spent,
        f"cold acceptance {float(run.acceptance_rate[0]):.2f}, lowest swap "
        f"{float(run.swap_acceptance.min()):.2f}",
    )


def gibbs_schedule() -> tuple[float, ...]:
    """The temperatures :func:`annealed_gibbs_seeding` sweeps at: ``anneal``'s schedule, hottest rung to 1 over :data:`ANNEAL_STEPS`."""
    return ladder(ExponentialTempSchedule(float(TEMPERATURES[-1]), 1.0, ANNEAL_STEPS))


def annealed_gibbs_seeding(
    instance: MixtureInstance, rng: np.random.Generator
) -> Seeding[EmissionFamily]:
    """``anneal_assignments``: heat-bath sweeps over the assignments at a falling temperature, on the count-pair likelihood itself.

    Starts at the data draw, sweeps once per temperature of
    :func:`gibbs_schedule` --- the schedule ``anneal`` runs on the surrogate
    --- re-estimating the components at each sweep's hard assignments, and
    hands over the best state visited (issue #901). Each sweep scores every
    pair once, and the start scores the data draw once, so it charges one
    pass more than it sweeps; each sweep's components are an entry of the
    path.

    Returns
    -------
    Seeding
    """
    components = data_seeding(instance, rng).components
    weights = torch.full(
        (instance.n_components,), 1.0 / instance.n_components, dtype=torch.float64
    )
    run = anneal_assignments(
        instance.observations,
        weights,
        components,
        gibbs_schedule(),
        rng,
        covariate=instance.covariate,
    )
    return Seeding(
        run.components,
        float(len(run.temperatures) + 1),
        f"best at sweep {run.best_step} of {len(run.temperatures)}",
        tuple(enumerate(run.path)),
    )


#: Every start, in the order the notebook takes them: the prior control, the
#: three rules over the pairs, the Gaussian EM, the four of `opt.initialize`,
#: the burn-in and the three of `sample.initialize` with the annealed Gibbs
#: start after `anneal`, the one schedule read two ways. Module-level, so a
#: process pool can run each.
STARTS: dict[
    str, Callable[[MixtureInstance, np.random.Generator], Seeding[EmissionFamily]]
] = {
    "prior": prior_seeding,
    "data": data_seeding,
    "kmeans++": kmeans_seeding,
    "emission++": emission_seeding,
    "gaussian-em": gaussian_em_seeding,
    "objective": objective_seeding,
    "perturbed": perturbed_seeding,
    "restart": restart_seeding,
    "quantile": quantile_seeding,
    "burn-in": burn_in_seeding,
    "hmc": chain_seeding,
    "anneal": annealed_seeding,
    "gibbs-anneal": annealed_gibbs_seeding,
    "tempering": tempered_seeding,
}

#: The starts that read no generator: one run of each is every run.
DETERMINISTIC = frozenset({"objective", "perturbed", "quantile"})


@dataclass(frozen=True)
class _Seeding:
    """One seeding of a :class:`BestOf`: the start, and its polish when asked.

    Module-level and frozen, so the process pool can pickle it and a thread
    pool can share it: every task reads it and none writes to it.
    """

    name: str
    instance: MixtureInstance
    passes: int | None = None
    seconds: float | None = None
    tolerance: float = 0.0


@dataclass(frozen=True)
class _Seeded:
    """What one seeding produced: the start, its score at equal weights, and its fit."""

    seeded: Seeding[EmissionFamily]
    score: float
    fit: MixturePolished | None


def _run_seeding(task: _Seeding, generator: np.random.Generator) -> _Seeded:
    """Run one seeding on its own generator, and polish it when ``task`` says so.

    Thread-safe: the start draws only from ``generator``, which no other task
    holds; it records into a ``track`` block of its own, whose run is a
    :class:`contextvars.ContextVar` and so private to the thread; and the
    instance is read, never written.
    """
    opened = time.perf_counter()
    values = torch.as_tensor(task.instance.observations, dtype=torch.float64)
    uniform = torch.full(
        (task.instance.n_components,),
        -math.log(task.instance.n_components),
        dtype=torch.float64,
    )
    with track(MemoryRun()):
        seeded = lookup(task.name)(task.instance, generator)
    score = float(mixture_log_likelihood(values, uniform, seeded.components))
    fit: MixturePolished | None = None
    if task.passes is not None:
        with track(MemoryRun()):
            fit = polish(task.instance, seeded.components, passes=task.passes)
    elif task.seconds is not None:
        left = task.seconds - (time.perf_counter() - opened)
        with track(MemoryRun()):
            fit = polish(
                task.instance,
                seeded.components,
                seconds=max(left, 0.0),
                tolerance=task.tolerance,
            )
    return _Seeded(seeded, score, fit)


class Selection(StrEnum):
    """What a :class:`BestOf` selects on: the seedings, or their EM fits.

    ``SEEDED`` scores each seeding at equal weights and polishes the best
    once. ``POLISHED`` polishes every seeding by EM and keeps the fit of
    highest log-likelihood: ``n`` polishes where ``SEEDED`` runs one, since
    the seeding that scores best is not always the one EM ends best from
    (#912: ``gibbs-anneal``'s best of five ended 16.9 nats below the
    reference where one seeding ended 0.7 above).
    """

    SEEDED = "seeded"
    POLISHED = "polished"


@dataclass(frozen=True)
class BestOf:
    """A stochastic start run ``n`` times, the seeding of highest log-likelihood handed over (issue #905).

    Seeding ``i`` draws from the ``i``-th generator spawned from the cell's
    (:meth:`numpy.random.Generator.spawn`), so the seedings are independent
    tasks and run through :func:`sal.parallel.map_tasks`: the
    result is the same, bitwise, at every worker count and on every pool, and
    one seed reproduces all ``n``. Each seeding runs in a ``track`` block of
    its own and is scored at equal weights, the value the polish would begin
    from; the scored seedings are the path, one entry per seeding in order,
    recorded once all have returned, so the curve shows the selection. The
    charge is the ``n`` seedings' passes and one scoring pass each; the
    seconds are the whole call's, since :class:`TimedStart` times it.
    :func:`~sal.opt.initialize.RandomRestart` does this on the
    surrogate; this is the same policy at the seam.

    With ``select=Selection.POLISHED`` every seeding is polished instead and
    the best fit kept (:meth:`polished`); :class:`TimedStart` runs it under
    the cell's one budget, shared evenly over the rounds of seedings the
    workers run.

    Parameters
    ----------
    name : str
        A key of :data:`STARTS` that reads its generator.
    n : int
        Seedings, at least 1.
    select : Selection
        What is selected on; the seedings by default.
    workers : int
        Seedings run at once; ``1``, the default, runs them in the calling
        thread. Measured on ``emission_mixture/stress`` at five seedings and
        one intra-op thread (#912): four processes take the polished
        selection from 6.35 to 4.56 s (``emission++``), 12.2 to 7.0 s
        (``gibbs-anneal``) and 42.8 to 19.6 s (``hmc``), and lose where the
        seedings are cheaper than the pool's 2.5 s (``emission++`` seeded,
        0.96 against 2.86 s). Inside :func:`~sal.opt.budget.compare`'s
        four workers the pools nest, and the polished group took 69.8 s
        against 54.8 s at one worker, which is why one is the default.
    pool : Pool
        ``"processes"``, the default, or ``"threads"``, when ``workers > 1``.
        Every task is thread-safe (:func:`_run_seeding`), and both pools are
        pinned bitwise to one worker; threads did not pay here, 1.62 against
        0.96 s and 6.97 against 6.35 s for ``emission++``, since the starts
        hold the GIL.

    Raises
    ------
    ValueError
        If ``name`` is deterministic or not a start, ``n`` or ``workers`` is
        below 1, or ``pool`` is not a pool.
    """

    name: str
    n: int
    select: Selection = Selection.SEEDED
    workers: int = 1
    pool: Pool = "processes"

    def __post_init__(self) -> None:
        if self.name not in STARTS or self.name in DETERMINISTIC:
            msg = f"best-of takes a stochastic start, got {self.name!r}"
            raise ValueError(msg)
        if self.n < 1:
            msg = f"best-of needs at least one seeding, got {self.n}"
            raise ValueError(msg)
        if self.workers < 1:
            msg = f"best-of runs at least one worker, got {self.workers}"
            raise ValueError(msg)
        if self.pool not in ("threads", "processes"):
            msg = f"best-of runs on threads or processes, got {self.pool!r}"
            raise ValueError(msg)

    @property
    def key(self) -> str:
        """The start's name: ``f"{name}x{n}"``, and ``+em`` after it when every seeding is polished."""
        suffix = "+em" if self.select is Selection.POLISHED else ""
        return f"{self.name}x{self.n}{suffix}"

    def _seedings(
        self, task: _Seeding, rng: np.random.Generator
    ) -> tuple[list[_Seeded], tuple[tuple[int, EmissionFamily], ...], float]:
        """Every seeding on its spawned generator, the path, and the passes charged."""
        results = map_tasks(
            _run_seeding,
            [task] * self.n,
            workers=self.workers,
            pool=self.pool if self.workers > 1 else "serial",
            intra_op_threads=1,
            generator=rng,
        )
        tracked = current()
        for index, result in enumerate(results):
            tracked.record(index, seeding_log_likelihood=result.score)
        path = tuple((index, r.seeded.components) for index, r in enumerate(results))
        passes = sum(
            r.seeded.passes + (1.0 if r.fit is None else float(r.fit.iterations))
            for r in results
        )
        return results, path, passes

    def __call__(
        self, instance: MixtureInstance, rng: np.random.Generator
    ) -> Seeding[EmissionFamily]:
        """The best of ``n`` seedings, each scored at equal weights.

        Returns
        -------
        Seeding

        Raises
        ------
        ValueError
            If every seeding is to be polished: that runs through
            :meth:`polished`, which is handed the seconds or passes.
        """
        if self.select is Selection.POLISHED:
            msg = f"{self.key} polishes every seeding: call polished() with its budget"
            raise ValueError(msg)
        results, path, passes = self._seedings(_Seeding(self.name, instance), rng)
        best = int(np.argmax([r.score for r in results]))
        return Seeding(
            results[best].seeded.components,
            passes,
            f"best of {self.n}: seeding {best}",
            path,
        )

    def polished(
        self,
        instance: MixtureInstance,
        rng: np.random.Generator,
        *,
        seconds: float | None = None,
        passes: int | None = None,
        tolerance: float | None = None,
    ) -> tuple[Seeding[EmissionFamily], MixturePolished]:
        """Every seeding polished by EM, and the fit of highest log-likelihood.

        Given ``seconds``, the seedings run in ``ceil(n / workers)`` rounds
        and each is handed the round's share, its start and its polish
        together; given ``passes``, each polish runs exactly that many
        iterations. ``tolerance`` is the polish's stop,
        :data:`POLISH_TOLERANCE` when omitted. Ties go to the earlier
        seeding. The charge is the seedings' passes and every polish's
        iterations, one pass each.

        Returns
        -------
        tuple[Seeding[EmissionFamily], MixturePolished]
            The chosen seeding, as :meth:`__call__` returns one, and its fit.

        Raises
        ------
        ValueError
            Unless exactly one of ``seconds`` and ``passes`` is given.
        """
        if (seconds is None) == (passes is None):
            msg = "a polished best-of stops at seconds or at passes, exactly one"
            raise ValueError(msg)
        rounds = math.ceil(self.n / min(self.workers, self.n))
        task = _Seeding(
            self.name,
            instance,
            passes=passes,
            seconds=None if seconds is None else seconds / rounds,
            tolerance=POLISH_TOLERANCE if tolerance is None else tolerance,
        )
        results, path, charged = self._seedings(task, rng)
        fits = [r.fit for r in results]
        finals = [float(f.log_likelihoods[-1]) for f in fits if f is not None]
        best = int(np.argmax(finals))
        chosen = fits[best]
        if chosen is None:  # pragma: no cover - every task above polishes
            msg = "a polished best-of polishes every seeding"
            raise TypeError(msg)
        return (
            Seeding(
                results[best].seeded.components,
                charged,
                f"best of {self.n} after EM: seeding {best}",
                path,
            ),
            chosen,
        )


def best_of(
    name: str,
    n: int,
    select: Selection = Selection.SEEDED,
    *,
    workers: int = 1,
    pool: Pool = "processes",
) -> BestOf:
    """The start that runs ``name`` ``n`` times and hands over the best seeding or fit.

    Returns
    -------
    BestOf
    """
    return BestOf(name, n, select, workers, pool)


#: Seedings per best-of start in the notebook's group (issue #905).
BEST_OF = 5

#: The best-of group: every stochastic start at :data:`BEST_OF` seedings, in
#: :data:`STARTS`' order, keyed ``f"{name}x{BEST_OF}"``.
BEST_OF_STARTS: dict[str, BestOf] = {
    start.key: start
    for start in (
        best_of(name, BEST_OF) for name in STARTS if name not in DETERMINISTIC
    )
}


#: The best-of group polished throughout: every stochastic start at
#: :data:`BEST_OF` seedings, each seeding polished by EM, keyed
#: ``f"{name}x{BEST_OF}+em"`` (#912).
BEST_OF_EM_STARTS: dict[str, BestOf] = {
    start.key: start
    for start in (
        best_of(name, BEST_OF, Selection.POLISHED)
        for name in STARTS
        if name not in DETERMINISTIC
    )
}


def lookup(
    name: str,
) -> Callable[[MixtureInstance, np.random.Generator], Seeding[EmissionFamily]]:
    """A start by name, from :data:`STARTS`, :data:`BEST_OF_STARTS` or :data:`BEST_OF_EM_STARTS`.

    Returns
    -------
    Callable[[MixtureInstance, np.random.Generator], Seeding[EmissionFamily]]
    """
    if name in STARTS:
        return STARTS[name]
    if name in BEST_OF_STARTS:
        return BEST_OF_STARTS[name]
    return BEST_OF_EM_STARTS[name]


#: Relative change in the log-likelihood between two EM iterations at which
#: the polish stops: ``|L_i - L_{i-1}| <= POLISH_TOLERANCE * |L_{i-1}|``. On
#: the stress draw, whose generating parameters reach -28,268.1 nats, 1e-6 is
#: a change of 0.028 nats an iteration, under the 0.1 nat the notebook prints
#: a log-likelihood to (issue #898). Measured at seed 0 on the 4-core host, one
#: process a core at a 1-minute load of 3 to 6: at 1e-6, 7 of 8 starts stop
#: inside 120 s, `restart` and `quantile` at iteration 133 within 0.2 nats of
#: where an unstopped run is at 120 s; at 1e-7, 4 of the 8 do. The rule reads a
#: slowdown, not a maximum: `prior` passes 1e-6 at iteration 136 at a gap of
#: -9.7 nats and reaches -23.0 by iteration 270.
POLISH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MixturePolished(Polished):
    """The EM fit a start hands over to: :class:`~sal.opt.starts.Polished` at the seam.

    ``value`` is the negative log-likelihood where the polish stopped and
    ``termination`` how: at its tolerance (:attr:`~sal.opt.termination.Stop.CONVERGED`),
    at an emptied component (:attr:`~sal.opt.termination.Stop.REFUSED`)
    or at its budget.

    Parameters
    ----------
    components : EmissionFamily
        The fitted components.
    weights : torch.Tensor
        The fitted mixing weights.
    log_likelihoods : np.ndarray
        The log-likelihood at the handover and after every iteration, shape
        ``(iterations + 1,)``.
    emptied : bool
        Whether the polish stopped because EM emptied a component and its M
        step refused, :func:`polish`'s third stop.
    """

    components: EmissionFamily
    weights: torch.Tensor
    log_likelihoods: np.ndarray
    emptied: bool = False

    @classmethod
    def ended(
        cls,
        components: EmissionFamily,
        weights: torch.Tensor,
        log_likelihoods: np.ndarray,
        *,
        converged: bool = False,
        emptied: bool = False,
    ) -> MixturePolished:
        """The fit at the trace's last value, its termination read off the two flags."""
        reason = (
            Stop.CONVERGED if converged else Stop.REFUSED if emptied else Stop.BUDGET
        )
        return cls(
            value=-float(log_likelihoods[-1]),
            termination=Termination(
                converged, int(log_likelihoods.shape[0]) - 1, reason
            ),
            components=components,
            weights=weights,
            log_likelihoods=log_likelihoods,
            emptied=emptied,
        )


def polish(
    instance: MixtureInstance,
    components: EmissionFamily,
    *,
    passes: int | None = None,
    seconds: float | None = None,
    tolerance: float = POLISH_TOLERANCE,
) -> MixturePolished:
    """EM iterations from ``components`` at equal weights, one pass each, to one of two stops.

    Given ``passes``, exactly that many iterations. Given ``seconds``, the
    iterations run until the log-likelihood's relative change between two of
    them is at most ``tolerance`` or until the next would pass ``seconds``:
    an iteration starts only while the seconds spent, plus twice the longest
    iteration so far --- the iteration and the closing E step, each bounded
    by it --- fit in ``seconds``. The first iteration's cost is unknown until
    it has run, so it runs whenever ``seconds`` is positive, and a caller
    holding a ceiling checks the whole spend.

    **A component EM has emptied stops the polish** under ``seconds``. EM
    can drive a weight to underflow while the log-likelihood still rises:
    from the ``prior`` start at seed 3 on the stress draw, one weight falls
    from 2.6e-7 at iteration 156 to 1.5e-231 at 179, the likelihood climbing
    a nat an iteration, and the next E step underflows its responsibilities
    to zero, where the family's M step has no data to solve on and refuses.
    The polish stops there, :attr:`~sal.search.mixture_starts.MixturePolished.emptied`, and hands over the fit
    of the iteration before: the refusal is caught only when the E step at
    that fit leaves some component no responsibility at all, and raised
    otherwise. A small weight is not by itself the stop: from the same start
    at seed 0 one dips under one pair's share by iteration 18 and recovers,
    and EM converges 9.7 nats past the generating parameters.

    Each iteration is one call of
    :func:`~sal.opt.emission_mixture.expectation_maximization`,
    recorded into the enclosing run as ``log_likelihood`` at step ``i``: the
    value at the components iteration ``i`` was handed, recorded once that
    iteration has produced the next. The value at the last components is the
    E step alone, recorded at the step after the last iteration.

    Returns
    -------
    MixturePolished

    Raises
    ------
    ValueError
        Unless exactly one of ``passes`` and ``seconds`` is given.
    """
    if (passes is None) == (seconds is None):
        msg = "a polish stops at passes or at seconds, exactly one of them"
        raise ValueError(msg)
    opened = time.perf_counter()
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    weights = torch.full(
        (instance.n_components,), 1.0 / instance.n_components, dtype=torch.float64
    )
    tracked = current()
    trace: list[float] = []
    longest = 0.0
    converged = False
    emptied = False
    iteration = 0
    while True:
        if passes is not None and iteration >= passes:
            break
        if seconds is not None and (
            time.perf_counter() - opened + 2.0 * longest > seconds
        ):
            break
        began = time.perf_counter()
        try:
            step = expectation_maximization(
                instance.observations,
                weights,
                components,
                covariate=instance.covariate,
                config=EmConfig(max_iterations=1, tolerance=0.0),
            )
        except ValueError:
            # The one refusal this stop reads: a component the E step leaves
            # no responsibility on, whose M step has nothing to solve on.
            owned = responsibilities_torch(
                values,
                torch.log(weights),
                components,
                covariate=instance.conditioned,
            ).sum(dim=0)
            if seconds is None or float(owned.min()) > 0.0:
                raise
            emptied = True
            break
        longest = max(longest, time.perf_counter() - began)
        trace.append(step.log_likelihood)
        tracked.record(iteration, log_likelihood=step.log_likelihood)
        weights, components = step.weights, step.components
        iteration += 1
        if (
            seconds is not None
            and len(trace) > 1
            and abs(trace[-1] - trace[-2]) <= tolerance * abs(trace[-2])
        ):
            converged = True
            break
    final = float(
        mixture_log_likelihood(
            values, torch.log(weights), components, covariate=instance.conditioned
        )
    )
    trace.append(final)
    tracked.record(iteration, log_likelihood=final)
    tensors = [weights, *components.named_parameters().values()]
    tracked.record_cost(
        iteration, sum(int(t.element_size() * t.nelement()) for t in tensors)
    )
    return MixturePolished.ended(
        components, weights, np.asarray(trace), converged=converged, emptied=emptied
    )


@dataclass(frozen=True)
class MixtureTrial(Trial):
    """One timed start and its polish, as :class:`TimedStart` reports it.

    The seconds, the start's and its polish's together, and the polished
    state's bytes are :class:`~sal.opt.starts.Trial`'s;
    ``curve[handover][0]`` is the start's seconds alone.

    Parameters
    ----------
    name : str
        The start.
    seeded : Seeding[EmissionFamily]
        What the start produced, without its path.
    polished : MixturePolished
        The fit it handed over to.
    curve : tuple[tuple[float, float], ...]
        ``(seconds, log_likelihood)`` for every entry of the start's path and
        every state of the polish, seconds from the start's first call, each
        read from a ``track`` sample.
    handover : int
        The index in ``curve`` of the seeded components.
    recovery : float
        Fraction of pairs the fit assigns to their generating component, up to
        the best renaming of the components.
    mean_error : float
        Largest relative error in a component's first-channel mean over that
        renaming.
    """

    name: str
    seeded: Seeding[EmissionFamily]
    polished: MixturePolished
    curve: tuple[tuple[float, float], ...]
    handover: int
    recovery: float
    mean_error: float


def _first_per_step(series: list[tuple[int, float]]) -> dict[int, float]:
    """The first value recorded at each step."""
    first: dict[int, float] = {}
    for step, value in series:
        first.setdefault(step, float(value))
    return first


@dataclass(frozen=True)
class TimedStart:
    """One start and its polish as an :mod:`sal.opt.budget` method whose spend is seconds.

    The start runs in a ``track`` block nested in the block the polish runs
    in, so its own samples carry its own clock, offset to the outer one; the
    handover is one sample of the outer run.

    **One budget covers the cell** (issue #898). The budget
    :func:`~sal.opt.budget.compare` hands in is in
    :attr:`~sal.cost.Cost.SECONDS` and covers the start and
    its polish: the polish is handed the seconds the start left and stops at
    ``tolerance`` or before it would pass them, and the spend reported is the
    cell's whole seconds rounded up, so ``compare`` refuses a cell over it.
    Given ``passes``, the polish runs exactly that many iterations instead,
    and the spend reported against the seconds budget is the start's alone,
    to the handover.

    Parameters
    ----------
    name : str
        A key of :data:`STARTS` or :data:`BEST_OF_STARTS`.
    passes : Budget | None
        A polish of fixed length in :attr:`~sal.cost.Cost.PASSES`;
        ``None`` for the one budget.
    tolerance : float
        The polish's stop under the one budget, :data:`POLISH_TOLERANCE`.
    """

    name: str
    passes: Budget | None = None
    tolerance: float = POLISH_TOLERANCE

    def __call__(
        self, instance: MixtureInstance, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        """The negative log-likelihood reached, the seconds spent, and the :class:`MixtureTrial`.

        Returns
        -------
        Outcome

        Raises
        ------
        ValueError
            If ``budget`` is not in seconds, or ``passes`` is not in passes.
        """
        if budget.unit is not Cost.SECONDS:
            msg = f"a timed start is budgeted in seconds, not {budget.unit}"
            raise ValueError(msg)
        if self.passes is not None and self.passes.unit is not Cost.PASSES:
            msg = f"a fixed polish is counted in passes, not {self.passes.unit}"
            raise ValueError(msg)
        start = lookup(self.name)
        every = isinstance(start, BestOf) and start.select is Selection.POLISHED
        chosen: MixturePolished | None = None
        with track(MemoryRun()) as outer:
            with track(MemoryRun()) as inner:
                if isinstance(start, BestOf) and every:
                    seeded, chosen = start.polished(
                        instance,
                        rng,
                        seconds=None
                        if self.passes is not None
                        else budget.size - (time.perf_counter() - outer.started),
                        passes=None if self.passes is None else self.passes.size,
                        tolerance=self.tolerance,
                    )
                else:
                    seeded = start(instance, rng)
            outer.record(0, context=HANDOVER, handed_over=1.0)
            if chosen is not None:
                # Every seeding was polished inside the start; the handover
                # is the best fit, and nothing is left to polish. One sample
                # and its cost close the cell's clock, as a polish's do.
                polished = chosen
                outer.record(0, log_likelihood=float(chosen.log_likelihoods[-1]))
                tensors = [
                    chosen.weights,
                    *chosen.components.named_parameters().values(),
                ]
                outer.record_cost(
                    0, sum(int(t.element_size() * t.nelement()) for t in tensors)
                )
            elif self.passes is None:
                polished = polish(
                    instance,
                    seeded.components,
                    seconds=budget.size - (time.perf_counter() - outer.started),
                    tolerance=self.tolerance,
                )
            else:
                polished = polish(instance, seeded.components, passes=self.passes.size)
        inner_run, outer_run = inner.run, outer.run
        if not isinstance(inner_run, MemoryRun) or not isinstance(
            outer_run, MemoryRun
        ):  # pragma: no cover - bound above
            msg = "a timed start records into the MemoryRuns it opened"
            raise TypeError(msg)

        # The path's values, computed now that the clock has stopped.
        values = torch.as_tensor(instance.observations, dtype=torch.float64)
        uniform = torch.full(
            (instance.n_components,),
            -math.log(instance.n_components),
            dtype=torch.float64,
        )
        offset = inner.started - outer.started
        start_seconds = (
            _first_per_step(inner_run.series("seconds")) if seeded.path else {}
        )
        curve = [
            (
                offset + start_seconds[step],
                float(
                    mixture_log_likelihood(
                        values, uniform, family, covariate=instance.conditioned
                    )
                ),
            )
            for step, family in seeded.path
        ]
        handover = len(curve)
        handed = float(outer_run.last("seconds", HANDOVER))
        polish_seconds = _first_per_step(outer_run.series("seconds"))
        if chosen is not None:
            # The polishes ran inside the start, each in a block of its own,
            # so the curve holds the chosen fit's value from the handover.
            curve.append((handed, float(polished.log_likelihoods[-1])))
        else:
            curve.append((handed, float(polished.log_likelihoods[0])))
            curve.extend(
                (polish_seconds[index - 1], float(value))
                for index, value in enumerate(polished.log_likelihoods)
                if index > 0
            )

        posterior = responsibilities_torch(
            values,
            torch.log(polished.weights),
            polished.components,
            covariate=instance.conditioned,
        )
        columns = match_components(polished.components, instance.truth)
        assigned = np.asarray(posterior.argmax(dim=1).numpy())
        fitted_mean = polished.components.alignment_key()[:, 0].numpy()
        true_mean = instance.truth.alignment_key()[:, 0].numpy()[columns]
        seconds = float(outer_run.last("seconds"))
        trial = MixtureTrial(
            name=self.name,
            seeded=Seeding(seeded.components, seeded.passes, seeded.diagnostics),
            polished=polished,
            curve=tuple(curve),
            handover=handover,
            seconds=seconds,
            state_bytes=int(outer_run.last("state_bytes")),
            recovery=float(np.mean(columns[assigned] == instance.labels)),
            mean_error=float(np.max(np.abs(fitted_mean - true_mean) / true_mean)),
        )
        spent = math.ceil(seconds if self.passes is None else handed)
        return Outcome(-float(polished.log_likelihoods[-1]), spent, trial)


@dataclass(frozen=True)
class StartRow:
    """One start's row: what its trials reached, what they cost, and what they recovered.

    Every tuple carries one entry per trial, in seed order.

    Parameters
    ----------
    start : str
        The start.
    deterministic : bool
        Whether the start reads no generator, so one trial is every trial.
    seeded : tuple[float, ...]
        The log-likelihood at the handover.
    seeded_gap : tuple[float, ...]
        The reference less ``seeded``, in nats.
    reached : tuple[float, ...]
        The log-likelihood where the polish stopped: at its tolerance, at its
        budget, or after its fixed passes.
    gap : tuple[float, ...]
        The reference less ``reached``, in nats.
    passes : float
        The start's own cost in passes, the mean over the trials.
    state_bytes : int
        What the polished state holds.
    seconds : tuple[float, ...]
        Wall clock of the start and its polish.
    seeding_seconds : tuple[float, ...]
        Wall clock of the start alone, to the handover.
    recovery : tuple[float, ...]
        Fraction of pairs assigned to their generating component.
    mean_error : tuple[float, ...]
        Largest relative error in a component's first-channel mean.
    converged : tuple[bool, ...]
        Whether the polish stopped at its tolerance.
    emptied : tuple[bool, ...]
        Whether the polish stopped at an emptied component.
    iterations : tuple[int, ...]
        EM iterations the polish ran.
    """

    start: str
    deterministic: bool
    seeded: tuple[float, ...]
    seeded_gap: tuple[float, ...]
    reached: tuple[float, ...]
    gap: tuple[float, ...]
    passes: float
    state_bytes: int
    seconds: tuple[float, ...]
    seeding_seconds: tuple[float, ...]
    recovery: tuple[float, ...]
    mean_error: tuple[float, ...]
    converged: tuple[bool, ...]
    emptied: tuple[bool, ...]
    iterations: tuple[int, ...]

    @property
    def trials(self) -> int:
        """Trials run."""
        return len(self.reached)

    @classmethod
    def from_trials(
        cls, start: str, trials: list[MixtureTrial], reference: float
    ) -> StartRow:
        """The row of one start's trials, read against the reference.

        Returns
        -------
        StartRow
        """
        seeded = tuple(float(t.polished.log_likelihoods[0]) for t in trials)
        reached = tuple(float(t.polished.log_likelihoods[-1]) for t in trials)
        return cls(
            start=start,
            deterministic=start in DETERMINISTIC,
            seeded=seeded,
            seeded_gap=tuple(reference - value for value in seeded),
            reached=reached,
            gap=tuple(reference - value for value in reached),
            passes=float(np.mean([t.seeded.passes for t in trials])),
            state_bytes=trials[0].state_bytes,
            seconds=tuple(t.seconds for t in trials),
            seeding_seconds=tuple(t.curve[t.handover][0] for t in trials),
            recovery=tuple(t.recovery for t in trials),
            mean_error=tuple(t.mean_error for t in trials),
            converged=tuple(t.polished.converged for t in trials),
            emptied=tuple(t.polished.emptied for t in trials),
            iterations=tuple(t.polished.iterations for t in trials),
        )


@dataclass(frozen=True)
class GapBand:
    """One start's gap over its trials on a common grid of seconds (issue #898).

    Parameters
    ----------
    seconds : np.ndarray
        The grid, shape ``(n,)``.
    mean : np.ndarray
        The mean gap over the trials at each grid time, ``nan`` before every
        trial has recorded a point.
    std : np.ndarray
        The sample standard deviation over the trials, ``nan`` where ``mean``
        is and everywhere at one trial.
    handover : tuple[float, float]
        The mean seconds and the mean gap at the handover.
    """

    seconds: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    handover: tuple[float, float]


def curve_band(curves: Sequence[Curve], seconds: np.ndarray) -> GapBand:
    """Each trial's gap held from each entry to the next, read on ``seconds``.

    A trial's curve is a sequence of states, so between two samples the gap
    is the earlier one's, and after the last it is the last: the fit the cell
    ended on. The mean and the sample standard deviation are taken over
    trials at the same wall clock, where every trial has an entry, which is
    what a reader of a runtime axis compares; the handover is the mean over
    trials of each one's. The one band reader both starts notebooks draw
    through (issue #926): :func:`gap_band` reads a mixture trial as a
    :class:`~sal.opt.starts.Curve` and calls this.

    Returns
    -------
    GapBand

    Raises
    ------
    ValueError
        If ``curves`` is empty.
    """
    if not curves:
        msg = "a band needs at least one trial"
        raise ValueError(msg)
    held = np.full((len(curves), seconds.shape[0]), np.nan)
    for row, curve in enumerate(curves):
        # The last sample at or before each grid time; -1 before the first.
        index = np.searchsorted(curve.seconds, seconds, side="right") - 1
        known = index >= 0
        held[row, known] = curve.gaps[index[known]]
    started = ~np.isnan(held).any(axis=0)
    mean = np.full(seconds.shape[0], np.nan)
    std = np.full(seconds.shape[0], np.nan)
    mean[started] = held[:, started].mean(axis=0)
    if len(curves) > 1:
        std[started] = held[:, started].std(axis=0, ddof=1)
    handover = (
        float(np.mean([c.seconds[c.handover] for c in curves])),
        float(np.mean([c.gaps[c.handover] for c in curves])),
    )
    return GapBand(seconds, mean, std, handover)


def trial_curve(trial: MixtureTrial, reference: float, index: int = 0) -> Curve:
    """A mixture trial as the :class:`~sal.opt.starts.Curve` :func:`curve_band` reads.

    Its samples' times, values, and gaps below ``reference``; its handover
    index as it is. ``index`` fills the curve's trial slot, which a band
    does not read.

    Returns
    -------
    Curve
    """
    values = np.asarray([point[1] for point in trial.curve])
    return Curve(
        0,
        index,
        np.asarray([point[0] for point in trial.curve]),
        values,
        reference - values,
        trial.handover,
    )


def gap_band(
    trials: list[MixtureTrial], reference: float, seconds: np.ndarray
) -> GapBand:
    """Each trial's gap below ``reference``, read on ``seconds`` by :func:`curve_band`.

    Returns
    -------
    GapBand

    Raises
    ------
    ValueError
        If ``trials`` is empty.
    """
    return curve_band(
        [trial_curve(trial, reference, index) for index, trial in enumerate(trials)],
        seconds,
    )
