"""Starts of a count-pair emission mixture, each polished by EM and timed through ``track`` (issue #891).

The joint count-pair mixture of :mod:`snakes_and_ladders.sim.emission_mixture`
alone --- no lattice, no chain --- started every way the package can start it,
then polished by :func:`snakes_and_ladders.opt.emission_mixture.expectation_maximization`
at one budget. Every start ends at the same seam, the instance's
:data:`~snakes_and_ladders.opt.emission_mixture.ComponentsAt`, so what
separates two starts is where they place the components and nothing after.

**Four kinds of start.** A prior draw reads no pair. Rules over the pairs
place a component on each of ``C`` chosen observations: the uniform draw,
``Emission_Mixture++``, k-means++ on the raw pair, and a short EM burn-in on a
subsample from the uniform draw. Starts that take an ``Objective`` run on a
surrogate, the Gaussian mixture over both channels, because the count-pair
mixture has no ``theta`` and is not an ``Objective``: the four of
:mod:`snakes_and_ladders.opt.initialize` and the three of
:mod:`snakes_and_ladders.sample.initialize`. The Gaussian-mixture EM start runs
on the first channel alone, since
:func:`snakes_and_ladders.opt.mixture.expectation_maximization` fits one
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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.cost import Cost
from snakes_and_ladders.emissions import EmissionFamily
from snakes_and_ladders.opt.budget import Budget, Outcome
from snakes_and_ladders.opt.emission_mixture import (
    ComponentsAt,
    expectation_maximization,
    plus_plus_start,
    uniform_start,
)
from snakes_and_ladders.opt.initialize import (
    FromObjective,
    Perturbed,
    RandomRestart,
    quantile_locations,
)
from snakes_and_ladders.opt.mixture import (
    GaussianMixtureObjective,
    KMeansPlusPlus,
    kmeans_plus_plus,
    mixture_log_likelihood,
    responsibilities,
)
from snakes_and_ladders.opt.mixture import (
    expectation_maximization as gaussian_expectation_maximization,
)
from snakes_and_ladders.sample.initialize import FromAnnealing, FromChain, FromTempering
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule
from snakes_and_ladders.search.projection import (
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
    match_components,
)
from snakes_and_ladders.sim.emission_mixture import SimulatedEmissionMixtureDataset
from snakes_and_ladders.track import MemoryRun, current, track

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
    """

    observations: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    truth: EmissionFamily
    at: ComponentsAt

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
        return float(mixture_log_likelihood(values, log_weight, self.truth))


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


@dataclass(frozen=True)
class Seeded:
    """What one start produced, what it charged, and the path it took there.

    Parameters
    ----------
    components : EmissionFamily
        The seeded components.
    passes : float
        The start's own cost, in passes over the data.
    diagnostics : str
        Empty for a rule; for a chain or a fit, what says how it ended.
    path : tuple[tuple[int, EmissionFamily], ...]
        For a start that iterates, the step of its run at which it held each
        of these components; empty for one that does not.
    """

    components: EmissionFamily
    passes: float
    diagnostics: str = ""
    path: tuple[tuple[int, EmissionFamily], ...] = ()


def surrogate(instance: MixtureInstance) -> GaussianMixtureObjective:
    """The Gaussian mixture over both channels: the ``Objective`` a surrogate start reads.

    Returns
    -------
    GaussianMixtureObjective
    """
    return GaussianMixtureObjective(
        np.asarray(instance.observations, dtype=np.float64), instance.n_components
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
    rows = np.asarray(instance.observations, dtype=np.float64)
    located = np.asarray(locations.detach().numpy(), dtype=np.float64)
    if located.ndim == 1:
        distance = np.abs(rows[None, :, 0] - located[:, None])
    else:
        distance = ((rows[None, :, :] - located[:, None, :]) ** 2).sum(axis=-1)
    return instance.at(rows[distance.argmin(axis=1)])


def _generator(rng: np.random.Generator) -> torch.Generator:
    """A torch stream derived from the caller's generator, so one seed runs the start."""
    return torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))


def prior_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """Components drawn from a prior over the observed range, reading no pair.

    A total log-uniform on the observed range of totals and an allele fraction
    uniform on ``(0, 1)``, the family's own support, each handed to the seam
    as the pair ``(total, fraction * total)``: the control that says whether
    reading the data earns its cost.

    Returns
    -------
    Seeded
    """
    totals = np.asarray(instance.observations, dtype=np.float64)[:, 0]
    low, high = float(max(totals.min(), 1.0)), float(max(totals.max(), 2.0))
    means = np.exp(rng.uniform(np.log(low), np.log(high), size=instance.n_components))
    rates = rng.uniform(0.0, 1.0, size=instance.n_components)
    return Seeded(instance.at(np.stack([means, rates * means], axis=1)), 0.0)


def data_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """``uniform_start``: components on pairs drawn uniformly without replacement.

    Returns
    -------
    Seeded
    """
    return Seeded(
        uniform_start(instance.observations, instance.n_components, instance.at, rng),
        0.0,
    )


def emission_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """``plus_plus_start``: D-squared sampling under the family's Bregman divergence.

    Returns
    -------
    Seeded
    """
    return Seeded(
        plus_plus_start(instance.observations, instance.n_components, instance.at, rng),
        1.0,
    )


def kmeans_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """``kmeans_plus_plus`` on the raw pair, its centres handed to the seam.

    Returns
    -------
    Seeded
    """
    centres = kmeans_plus_plus(
        np.asarray(instance.observations, dtype=np.float64), instance.n_components, rng
    )
    return Seeded(instance.at(centres), 1.0)


def gaussian_em_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """The Gaussian mixture fitted to the first channel from k-means++, its means the locations.

    One EM iteration per call, so each is recorded into the enclosing run,
    with the components its means would seed every :data:`PATH_STRIDE`
    iterations and at the last.

    Returns
    -------
    Seeded
    """
    channel = np.asarray(instance.observations, dtype=np.float64)[:, 0]
    objective = GaussianMixtureObjective(channel, instance.n_components)
    start = KMeansPlusPlus(1, rng).starts(objective)[0]
    weights = torch.exp(objective.constrain(start)["log_weight"]).detach()
    components = objective.components(start)
    tracked = current()
    path: list[tuple[int, EmissionFamily]] = []
    for iteration in range(GAUSSIAN_EM_ITERATIONS):
        fitted = gaussian_expectation_maximization(
            channel, weights, components, max_iterations=1, tolerance=0.0
        )
        weights, components = fitted.weights, fitted.components
        tracked.record(iteration, surrogate_log_likelihood=fitted.log_likelihood)
        last = iteration == GAUSSIAN_EM_ITERATIONS - 1
        if iteration % PATH_STRIDE == 0 or last:
            path.append((iteration, at_locations(instance, components.mean)))
    return Seeded(
        path[-1][1],
        float(GAUSSIAN_EM_ITERATIONS),
        f"budget after {GAUSSIAN_EM_ITERATIONS}",
        tuple(path),
    )


#: Fraction of the pairs the burn-in fits on, and the iterations it runs.
BURN_IN_FRACTION = 0.2
BURN_IN_ITERATIONS = 3


def burn_in_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """A short EM fit on a subsample, started from the data draw.

    :data:`BURN_IN_ITERATIONS` iterations on a :data:`BURN_IN_FRACTION`
    subsample, each charged its fraction of a pass; each iteration's
    components are an entry of the path.

    Returns
    -------
    Seeded
    """
    components = data_seeding(instance, rng).components
    size = max(instance.n_components, int(BURN_IN_FRACTION * instance.n_samples))
    subsample = instance.observations[
        rng.choice(instance.n_samples, size=size, replace=False)
    ]
    weights = torch.full(
        (instance.n_components,), 1.0 / instance.n_components, dtype=torch.float64
    )
    tracked = current()
    path: list[tuple[int, EmissionFamily]] = []
    for iteration in range(BURN_IN_ITERATIONS):
        fitted = expectation_maximization(
            subsample, weights, components, max_iterations=1, tolerance=0.0
        )
        weights, components = fitted.weights, fitted.components
        tracked.record(iteration, subsample_log_likelihood=fitted.log_likelihood)
        path.append((iteration, components))
    return Seeded(
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
        _generator(rng),
        n_steps=CHAIN_TRAJECTORY,
        burn_in=CHAIN_BURN_IN,
    )


def annealing_initializer(rng: np.random.Generator) -> FromAnnealing:
    """The annealing run :func:`annealed_seeding` makes: hottest rung to 1 over :data:`ANNEAL_STEPS`."""
    return FromAnnealing(
        ExponentialTempSchedule(float(TEMPERATURES[-1]), 1.0, ANNEAL_STEPS),
        CHAIN_STEP,
        _generator(rng),
        n_steps=CHAIN_TRAJECTORY,
    )


def tempering_initializer(rng: np.random.Generator) -> FromTempering:
    """The ladder :func:`tempered_seeding` runs: :data:`TEMPERATURES`, :data:`TEMPERING_ROUNDS` rounds."""
    return FromTempering(
        TEMPERATURES,
        TEMPERING_ROUNDS,
        CHAIN_STEP,
        _generator(rng),
        n_steps=CHAIN_TRAJECTORY,
    )


def objective_seeding(instance: MixtureInstance, _rng: np.random.Generator) -> Seeded:
    """``FromObjective``: the surrogate's own nominated point.

    Returns
    -------
    Seeded
    """
    objective = surrogate(instance)
    theta = FromObjective().starts(objective)[0]
    return Seeded(at_locations(instance, objective.components(theta).mean), 0.0)


def perturbed_seeding(instance: MixtureInstance, _rng: np.random.Generator) -> Seeded:
    """``Perturbed``: that point, tilted off a symmetry it may be stationary at.

    Returns
    -------
    Seeded
    """
    objective = surrogate(instance)
    theta = perturbation().starts(objective)[0]
    return Seeded(at_locations(instance, objective.components(theta).mean), 0.0)


def restart_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """``RandomRestart``: the best by surrogate value of points drawn around it.

    Returns
    -------
    Seeded
    """
    objective = surrogate(instance)
    thetas = restart_initializer(rng).starts(objective)
    best = min(thetas, key=lambda theta: float(objective(theta)))
    return Seeded(
        at_locations(instance, objective.components(best).mean), float(len(thetas))
    )


def quantile_seeding(instance: MixtureInstance, _rng: np.random.Generator) -> Seeded:
    """``quantile_locations``: each channel's evenly spaced quantiles, paired in order.

    Returns
    -------
    Seeded
    """
    values = torch.as_tensor(
        np.asarray(instance.observations, dtype=np.float64), dtype=torch.float64
    )
    return Seeded(
        at_locations(
            instance, quantile_locations(values, instance.n_components, dim=0)
        ),
        0.0,
    )


def chain_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """``FromChain``: a short Hamiltonian chain on the surrogate, warmed up; its last draw seeds.

    The warm-up is ``FromChain``'s default,
    :data:`~snakes_and_ladders.sample.initialize.CHAIN_ADAPTATION` (issue
    #898), and its gradients are in the passes charged.

    Returns
    -------
    Seeded
        Its path is every kept draw, at the step the chain recorded it.
    """
    objective = surrogate(instance)
    chain = chain_initializer(rng).chain(objective)
    path = tuple(
        (draw, at_locations(instance, objective.components(theta).mean))
        for draw, theta in enumerate(chain.theta)
    )
    return Seeded(
        path[-1][1],
        PASSES_PER_GRADIENT * chain.force_evaluations,
        f"acceptance {chain.acceptance_rate:.2f}"
        + (
            f", adapted step {chain.adapted.step_size:.3g}"
            if chain.adapted is not None
            else ""
        ),
        path,
    )


def annealed_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """``FromAnnealing``: the best point of a falling temperature.

    Returns
    -------
    Seeded
    """
    objective = surrogate(instance)
    run = annealing_initializer(rng).run(objective)
    return Seeded(
        at_locations(instance, objective.components(run.theta).mean),
        PASSES_PER_GRADIENT * run.force_evaluations,
        f"acceptance {run.acceptance_rate:.2f}",
    )


def tempered_seeding(instance: MixtureInstance, rng: np.random.Generator) -> Seeded:
    """``FromTempering``: the best point at any temperature of a ladder.

    Returns
    -------
    Seeded
    """
    objective = surrogate(instance)
    run = tempering_initializer(rng).run(objective)
    return Seeded(
        at_locations(instance, objective.components(run.theta).mean),
        PASSES_PER_GRADIENT * run.force_evaluations,
        f"cold acceptance {float(run.acceptance_rate[0]):.2f}, lowest swap "
        f"{float(run.swap_acceptance.min()):.2f}",
    )


#: Every start, in the order the notebook takes them: the prior control, the
#: three rules over the pairs, the Gaussian EM, the four of `opt.initialize`,
#: the burn-in and the three of `sample.initialize`. Module-level, so a
#: process pool can run each.
STARTS: dict[str, Callable[[MixtureInstance, np.random.Generator], Seeded]] = {
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
    "tempering": tempered_seeding,
}

#: The starts that read no generator: one run of each is every run.
DETERMINISTIC = frozenset({"objective", "perturbed", "quantile"})


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
class Polished:
    """The EM fit a start hands over to.

    Parameters
    ----------
    components : EmissionFamily
        The fitted components.
    weights : torch.Tensor
        The fitted mixing weights.
    log_likelihoods : np.ndarray
        The log-likelihood at the handover and after every iteration, shape
        ``(iterations + 1,)``.
    converged : bool
        Whether the polish stopped at its tolerance rather than at its budget.
    emptied : bool
        Whether the polish stopped because EM emptied a component and its M
        step refused, :func:`polish`'s third stop.
    """

    components: EmissionFamily
    weights: torch.Tensor
    log_likelihoods: np.ndarray
    converged: bool = False
    emptied: bool = False

    @property
    def iterations(self) -> int:
        """EM iterations run, one pass each."""
        return int(self.log_likelihoods.shape[0]) - 1


def polish(
    instance: MixtureInstance,
    components: EmissionFamily,
    *,
    passes: int | None = None,
    seconds: float | None = None,
    tolerance: float = POLISH_TOLERANCE,
) -> Polished:
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
    The polish stops there, :attr:`~snakes_and_ladders.search.mixture_starts.Polished.emptied`, and hands over the fit
    of the iteration before: the refusal is caught only when the E step at
    that fit leaves some component no responsibility at all, and raised
    otherwise. A small weight is not by itself the stop: from the same start
    at seed 0 one dips under one pair's share by iteration 18 and recovers,
    and EM converges 9.7 nats past the generating parameters.

    Each iteration is one call of
    :func:`~snakes_and_ladders.opt.emission_mixture.expectation_maximization`,
    recorded into the enclosing run as ``log_likelihood`` at step ``i``: the
    value at the components iteration ``i`` was handed, recorded once that
    iteration has produced the next. The value at the last components is the
    E step alone, recorded at the step after the last iteration.

    Returns
    -------
    Polished

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
                max_iterations=1,
                tolerance=0.0,
            )
        except ValueError:
            # The one refusal this stop reads: a component the E step leaves
            # no responsibility on, whose M step has nothing to solve on.
            owned = responsibilities(values, torch.log(weights), components).sum(dim=0)
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
    final = float(mixture_log_likelihood(values, torch.log(weights), components))
    trace.append(final)
    tracked.record(iteration, log_likelihood=final)
    tensors = [weights, *components.named_parameters().values()]
    tracked.record_cost(
        iteration, sum(int(t.element_size() * t.nelement()) for t in tensors)
    )
    return Polished(components, weights, np.asarray(trace), converged, emptied)


@dataclass(frozen=True)
class Trial:
    """One timed start and its polish, as :class:`TimedStart` reports it.

    Parameters
    ----------
    name : str
        The start.
    seeded : Seeded
        What the start produced, without its path.
    polished : Polished
        The fit it handed over to.
    curve : tuple[tuple[float, float], ...]
        ``(seconds, log_likelihood)`` for every entry of the start's path and
        every state of the polish, seconds from the start's first call, each
        read from a ``track`` sample.
    handover : int
        The index in ``curve`` of the seeded components.
    seconds : float
        Wall clock of the start and its polish; ``curve[handover][0]`` is the
        start's alone.
    state_bytes : int
        What the polished state holds.
    recovery : float
        Fraction of pairs the fit assigns to their generating component, up to
        the best renaming of the components.
    mean_error : float
        Largest relative error in a component's first-channel mean over that
        renaming.
    """

    name: str
    seeded: Seeded
    polished: Polished
    curve: tuple[tuple[float, float], ...]
    handover: int
    seconds: float
    state_bytes: int
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
    """One start and its polish as an :mod:`snakes_and_ladders.opt.budget` method whose spend is seconds.

    The start runs in a ``track`` block nested in the block the polish runs
    in, so its own samples carry its own clock, offset to the outer one; the
    handover is one sample of the outer run.

    **One budget covers the cell** (issue #898). The budget
    :func:`~snakes_and_ladders.opt.budget.compare` hands in is in
    :attr:`~snakes_and_ladders.cost.Cost.SECONDS` and covers the start and
    its polish: the polish is handed the seconds the start left and stops at
    ``tolerance`` or before it would pass them, and the spend reported is the
    cell's whole seconds rounded up, so ``compare`` refuses a cell over it.
    Given ``passes``, the polish runs exactly that many iterations instead,
    and the spend reported against the seconds budget is the start's alone,
    to the handover.

    Parameters
    ----------
    name : str
        A key of :data:`STARTS`.
    passes : Budget | None
        A polish of fixed length in :attr:`~snakes_and_ladders.cost.Cost.PASSES`;
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
        """The negative log-likelihood reached, the seconds spent, and the :class:`Trial`.

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
        with track(MemoryRun()) as outer:
            with track(MemoryRun()) as inner:
                seeded = STARTS[self.name](instance, rng)
            outer.record(0, context=HANDOVER, handed_over=1.0)
            if self.passes is None:
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
                float(mixture_log_likelihood(values, uniform, family)),
            )
            for step, family in seeded.path
        ]
        handover = len(curve)
        handed = float(outer_run.last("seconds", HANDOVER))
        polish_seconds = _first_per_step(outer_run.series("seconds"))
        curve.append((handed, float(polished.log_likelihoods[0])))
        curve.extend(
            (polish_seconds[index - 1], float(value))
            for index, value in enumerate(polished.log_likelihoods)
            if index > 0
        )

        posterior = responsibilities(
            values, torch.log(polished.weights), polished.components
        )
        columns = match_components(polished.components, instance.truth)
        assigned = np.asarray(posterior.argmax(dim=1).numpy())
        fitted_mean = polished.components.alignment_key()[:, 0].numpy()
        true_mean = instance.truth.alignment_key()[:, 0].numpy()[columns]
        seconds = float(outer_run.last("seconds"))
        trial = Trial(
            name=self.name,
            seeded=Seeded(seeded.components, seeded.passes, seeded.diagnostics),
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
    def from_trials(cls, start: str, trials: list[Trial], reference: float) -> StartRow:
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


def gap_band(trials: list[Trial], reference: float, seconds: np.ndarray) -> GapBand:
    """Each trial's gap below ``reference`` held from each point to the next, read on ``seconds``.

    A trial's curve is a sequence of states, so between two samples the gap
    is the earlier one's, and after the last it is the last: the fit the cell
    ended on. The mean and the band are taken over trials at the same wall
    clock, which is what a reader of a runtime axis compares.

    Returns
    -------
    GapBand

    Raises
    ------
    ValueError
        If ``trials`` is empty.
    """
    if not trials:
        msg = "a band needs at least one trial"
        raise ValueError(msg)
    held = np.full((len(trials), seconds.shape[0]), np.nan)
    for row, trial in enumerate(trials):
        times = np.asarray([point[0] for point in trial.curve])
        gaps = reference - np.asarray([point[1] for point in trial.curve])
        # The last sample at or before each grid time; -1 before the first.
        index = np.searchsorted(times, seconds, side="right") - 1
        known = index >= 0
        held[row, known] = gaps[index[known]]
    started = ~np.isnan(held).any(axis=0)
    mean = np.full(seconds.shape[0], np.nan)
    std = np.full(seconds.shape[0], np.nan)
    mean[started] = held[:, started].mean(axis=0)
    if len(trials) > 1:
        std[started] = held[:, started].std(axis=0, ddof=1)
    handover = (
        float(np.mean([t.curve[t.handover][0] for t in trials])),
        float(np.mean([reference - t.curve[t.handover][1] for t in trials])),
    )
    return GapBand(seconds, mean, std, handover)
