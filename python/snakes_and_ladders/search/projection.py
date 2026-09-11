"""The coupled model in projection, and the seedings its emission parameters start from (issue #541).

Drop the spatial coupling and the Markov chain from ``eq:joint`` and each
observation is an independent draw from a mixture over the ``M x K`` (class,
state) pairs. That projected mixture is what a seeding of the emission
parameters sees before the block-coordinate ascent of
:func:`~snakes_and_ladders.search.spatio_sequential.fit_spatio_sequential`
runs, so it is where the seedings are compared.

**The projection discards exactly the structure the full model exploits**, so
a seeding that wins here need not win once the spatial term is restored. Every
candidate is therefore scored twice, and
:mod:`tests.regression.search.test_projection_seeding` states both orderings.

**Every candidate ends at the same seam.** A seeding chooses ``M x K``
observations and ``ComponentsAt`` places a component on each
(:mod:`snakes_and_ladders.opt.emission_mixture`), so what separates the
candidates is the choice and nothing downstream of it. The three that sample
a surface --- :class:`~snakes_and_ladders.opt.initialize.FromChain`,
:class:`~snakes_and_ladders.opt.initialize.FromAnnealing` and
:class:`~snakes_and_ladders.opt.initialize.FromTempering` --- run on a
Gaussian mixture over the first channel, which is an ``Objective`` and needs
no new code; their component means are realized as the observations nearest
them, so they reach the same seam as the rest.

**Cost is counted in passes over the data**, one pass being every observation
scored under every component. An expectation-maximization iteration spends one
on its E step, and the M step is not counted because every candidate pays the
identical one whatever seeded it. A gradient is two, forward and backward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    NegativeBinomialEmission,
)
from snakes_and_ladders.opt.budget import Budget, Outcome
from snakes_and_ladders.opt.emission_mixture import (
    ComponentsAt,
    expectation_maximization,
    plus_plus_start,
    uniform_start,
)
from snakes_and_ladders.opt.initialize import (
    FromAnnealing,
    FromChain,
    FromTempering,
)
from snakes_and_ladders.opt.mixture import (
    GaussianMixtureObjective,
    emission_mixture_plus_plus,
)
from snakes_and_ladders.opt.schedule import Exponential
from snakes_and_ladders.sim.count_pairs import (
    TOTAL,
    IndependentCountPair,
    planted_labels,
)
from snakes_and_ladders.sim.spatio_sequential import SpatioSequentialParams

#: Passes charged to one gradient of the surrogate objective: the forward
#: evaluation and the backward sweep over the same ``N x C`` densities.
PASSES_PER_GRADIENT = 2.0

#: Leapfrog step for every chain-based candidate, the largest measured stable
#: on the key model's surrogate: 0.1 rejects every proposal and 1.0 diverges to
#: a non-positive scale inside the integrator. The acceptance rate each
#: candidate reports is what says whether the step suited the surface.
CHAIN_STEP = 2.0e-2

#: Leapfrog steps per proposal, below :data:`snakes_and_ladders.opt.hmc.DEFAULT_STEPS`
#: because a seeding is charged for every gradient it takes.
CHAIN_TRAJECTORY = 8

#: The ladder :class:`FromTempering` runs, coldest first.
TEMPERATURES = (1.0, 2.0, 4.0, 8.0)

#: Relative improvement below which the projected fit has converged; absolute
#: would not transfer across data sizes (``DEV.md``, issue #111).
TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class ProjectedCounts:
    """The coupled instance with its coupling dropped: one mixture, and its truth.

    Parameters
    ----------
    observations : np.ndarray
        The drawn pairs, shape ``(n_samples, 2)``.
    components : np.ndarray
        The generating component of each observation, ``class * K + state``,
        shape ``(n_samples,)``. The simulated truth a recovery is refereed
        against.
    truth : IndependentCountPair
        The ``M x K`` generating families, flattened in the same order.
    weights : np.ndarray
        The generating mixture weights, shape ``(M * K,)``.
    trials : float
        The second channel's trial count, one number shared by every
        component: a property of the assay, as
        :class:`snakes_and_ladders.opt.emission_mixture.CountPairSeeding`
        states.
    """

    observations: np.ndarray
    components: np.ndarray
    truth: IndependentCountPair
    weights: np.ndarray
    trials: float

    @property
    def n_components(self) -> int:
        """``M * K``, the components the projection mixes over."""
        return int(self.weights.shape[0])

    @property
    def n_samples(self) -> int:
        """Observations drawn."""
        return int(self.observations.shape[0])


def flatten(params: SpatioSequentialParams) -> IndependentCountPair:
    """The ``M`` per-class families as one ``M x K`` family, class-major.

    Parameters
    ----------
    params : SpatioSequentialParams
        Its emissions :class:`IndependentCountPair`.

    Returns
    -------
    IndependentCountPair
        One family over ``M * K`` components.

    Raises
    ------
    TypeError
        If a class's family is not the two-channel count emission.
    """
    families = []
    for family in params.emissions:
        if not isinstance(family, IndependentCountPair):
            msg = (
                f"the projection is defined for the two-channel count emission, "
                f"not for {type(family).__name__}"
            )
            raise TypeError(msg)
        families.append(family)
    return IndependentCountPair(
        NegativeBinomialEmission(
            torch.cat([f.total.dispersion for f in families]),
            torch.cat([f.total.mean for f in families]),
        ),
        BetaBinomialEmission(
            torch.cat([f.successes.trials for f in families]),
            torch.cat([f.successes.alpha for f in families]),
            torch.cat([f.successes.beta for f in families]),
        ),
    )


def project(
    params: SpatioSequentialParams, n_samples: int, rng: np.random.Generator
) -> ProjectedCounts:
    """Draw ``n_samples`` observations from the projection of a coupled model.

    A class is drawn from the share of vertices
    :func:`snakes_and_ladders.sim.count_pairs.planted_labels` gives it, a
    state uniformly --- the shared transition is circulant, so its stationary
    distribution is uniform --- and the pair from that class's family at that
    state.

    **Drawn rather than subsampled from the coupled instance's own counts**:
    the coupled draw fixes one state per class per position, so a subsample of
    it carries ``M`` states and not ``M x K`` unless it runs to the full
    length. The parameters are the instance's, and
    :func:`snakes_and_ladders.sim.count_pairs.binned_model` supplies them at a
    bin factor without the 33 s draw.

    Parameters
    ----------
    params : SpatioSequentialParams
        The coupled model, its emissions :class:`IndependentCountPair`.
    n_samples : int
        Observations to draw, at least one.
    rng : np.random.Generator
        Passed in rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    ProjectedCounts

    Raises
    ------
    ValueError
        If fewer than one observation is asked for.
    """
    if n_samples < 1:
        msg = f"n_samples must be at least 1, got {n_samples}"
        raise ValueError(msg)
    n_classes, n_states = params.n_classes, params.n_states
    share = np.bincount(planted_labels(params, n_classes), minlength=n_classes) / float(
        params.graph.n_nodes
    )
    weights = np.repeat(share / n_states, n_states)

    component = rng.choice(n_classes * n_states, size=n_samples, p=weights)
    observations = np.empty((n_samples, 2), dtype=np.int64)
    for class_index in range(n_classes):
        drawn = component // n_states == class_index
        if bool(drawn.any()):
            observations[drawn] = params.emissions[class_index].sample(
                component[drawn] % n_states, rng
            )
    truth = flatten(params)
    return ProjectedCounts(
        observations=observations,
        components=np.asarray(component, dtype=np.int64),
        truth=truth,
        weights=weights,
        trials=float(truth.successes.trials[0]),
    )


@dataclass(frozen=True)
class CountPairAt:
    """Places an :class:`IndependentCountPair` component on each observed pair.

    The independent form's counterpart of
    :class:`snakes_and_ladders.opt.emission_mixture.CountPairSeeding`: the
    total says the component's negative-binomial mean and the successes, with
    the Jeffreys prior's half-count so ``(n, 0)`` does not seed a rate of
    exactly zero, its beta-binomial rate. One observation carries no shape, so
    the dispersion and the concentration start at declared values shared by
    every component.

    Parameters
    ----------
    dispersion : float
        Starting negative-binomial ``r`` for every component.
    concentration : float
        Starting beta-binomial ``a + b`` for every component.
    trials : float
        The fixed trial count, the assay's rather than a component's.
    """

    dispersion: float
    concentration: float
    trials: float

    def __call__(self, rows: np.ndarray) -> IndependentCountPair:
        """The family seeded on these pairs, one component per row.

        Parameters
        ----------
        rows : np.ndarray
            Observed pairs, shape ``(n_components, 2)``.

        Returns
        -------
        IndependentCountPair
        """
        pairs = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
        rate = (pairs[:, 1] + 0.5) / (self.trials + 1.0)
        return IndependentCountPair(
            NegativeBinomialEmission(
                np.full(pairs.shape[0], self.dispersion),
                np.maximum(pairs[:, 0], 1.0),
            ),
            BetaBinomialEmission(
                np.full(pairs.shape[0], self.trials),
                rate * self.concentration,
                (1.0 - rate) * self.concentration,
            ),
        )


@dataclass(frozen=True)
class Seeding:
    """What one candidate produced, and what it charged for it.

    Parameters
    ----------
    components : IndependentCountPair
        The seeded components.
    passes : float
        The seeding's own cost, in passes over the data.
    diagnostics : str
        Empty for a heuristic; for a chain, the acceptance rate and whatever
        else says the draw is a draw and not a random restart.
    """

    components: IndependentCountPair
    passes: float
    diagnostics: str = ""


def _euclidean_score(rows: np.ndarray) -> Callable[..., np.ndarray]:
    """``(index, indices) -> squared Euclidean distance between those pairs``.

    k-means++ as it stands, transplanted onto a pair: the score
    :func:`snakes_and_ladders.opt.mixture.kmeans_plus_plus` uses, over two
    channels rather than one, so the draws are that function's draw for draw.
    """

    def score(seed: float, candidates: np.ndarray) -> np.ndarray:
        centre = rows[int(seed)]
        return np.asarray(
            ((rows[candidates.astype(np.int64)] - centre) ** 2).sum(axis=1),
            dtype=float,
        )

    return score


def _at_indices(
    observations: np.ndarray, chosen: np.ndarray, at: ComponentsAt
) -> IndependentCountPair:
    """The family seeded on the observations these indices name."""
    seeded = at(np.asarray(observations)[np.asarray(chosen, dtype=np.int64)])
    if not isinstance(seeded, IndependentCountPair):
        msg = f"the projection seeds a count pair, not {type(seeded).__name__}"
        raise TypeError(msg)
    return seeded


def euclidean_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """k-means++ as it stands: D-squared sampling under squared Euclidean distance.

    Returns
    -------
    Seeding
    """
    indices = np.arange(instance.n_samples, dtype=np.float64)
    chosen = emission_mixture_plus_plus(
        indices,
        instance.n_components,
        _euclidean_score(np.asarray(instance.observations, dtype=np.float64)),
        rng,
    )
    return Seeding(_at_indices(instance.observations, chosen, at), 1.0)


def emission_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """``Emission_Mixture++``: the same scheme under the family's own negative log-density.

    Returns
    -------
    Seeding
    """
    return Seeding(
        _count_pair(
            plus_plus_start(instance.observations, instance.n_components, at, rng)
        ),
        1.0,
    )


def data_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """Components placed on observations drawn uniformly, without replacement.

    Returns
    -------
    Seeding
    """
    return Seeding(
        _count_pair(
            uniform_start(instance.observations, instance.n_components, at, rng)
        ),
        0.0,
    )


def prior_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """Components drawn from a prior over the observed range, reading no observation.

    The control that says whether structure in a seeding earns its cost: a
    mean log-uniform on the observed total range and a rate uniform on
    ``(0, 1)``, which is the family's own support and not the data's.

    Returns
    -------
    Seeding
    """
    totals = np.asarray(instance.observations, dtype=np.float64)[:, TOTAL]
    low, high = float(max(totals.min(), 1.0)), float(max(totals.max(), 2.0))
    means = np.exp(rng.uniform(np.log(low), np.log(high), size=instance.n_components))
    rates = rng.uniform(0.0, 1.0, size=instance.n_components)
    rows = np.stack([means, rates * instance.trials], axis=1)
    seeded = at(rows)
    if not isinstance(seeded, IndependentCountPair):
        msg = f"the projection seeds a count pair, not {type(seeded).__name__}"
        raise TypeError(msg)
    return Seeding(seeded, 0.0)


#: Fraction of the observations the burn-in fits on, and the iterations it
#: runs there. The coupled model's own start (``Graph_BurnIn++``) fits the
#: chains on what the data alone supports before the prior tightens; in
#: projection there is no prior to temper, so what is left of it is a short
#: fit on a subsample, at ``BURN_IN_ITERATIONS * BURN_IN_FRACTION`` passes.
BURN_IN_FRACTION = 0.2
BURN_IN_ITERATIONS = 3


def burn_in_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """The coupled model's own start in projection: a short fit on a subsample.

    Returns
    -------
    Seeding
    """
    seeded = data_seeding(instance, at, rng).components
    size = max(instance.n_components, int(BURN_IN_FRACTION * instance.n_samples))
    subsample = rng.choice(instance.n_samples, size=size, replace=False)
    weights = torch.full(
        (instance.n_components,), 1.0 / instance.n_components, dtype=torch.float64
    )
    fit = expectation_maximization(
        instance.observations[subsample],
        weights,
        seeded,
        max_iterations=BURN_IN_ITERATIONS,
        tolerance=0.0,
    )
    return Seeding(
        _count_pair(fit.components),
        fit.iterations * size / instance.n_samples,
    )


def surrogate(instance: ProjectedCounts) -> GaussianMixtureObjective:
    """The ``Objective`` a chain-based candidate samples: a Gaussian mixture on the first channel.

    The projected count mixture has no ``theta``, and building one would put a
    401-parameter posterior in front of a chain that has to stay cheap. The
    total channel's Gaussian mixture is an ``Objective`` already, its
    component means are in observation units, and what a candidate takes from
    it is a set of locations --- which is all the seam downstream consumes.

    Parameters
    ----------
    instance : ProjectedCounts
        Its first channel is the surrogate's data.

    Returns
    -------
    GaussianMixtureObjective
    """
    return GaussianMixtureObjective(
        np.asarray(instance.observations, dtype=np.float64)[:, TOTAL],
        instance.n_components,
    )


def _nearest(instance: ProjectedCounts, means: torch.Tensor) -> np.ndarray:
    """The observation nearest each mean in the first channel, as indices.

    Canonicalization and the map to the seam in one step: the means are sorted
    first, so a chain that crossed modes and returned its components under
    other names lands on the same seeding.
    """
    totals = np.asarray(instance.observations, dtype=np.float64)[:, TOTAL]
    located = np.sort(means.detach().numpy())
    return np.asarray(np.abs(totals[None, :] - located[:, None]).argmin(axis=1))


def _from_theta(
    instance: ProjectedCounts, theta: torch.Tensor, at: ComponentsAt
) -> IndependentCountPair:
    """The family seeded where a surrogate's ``theta`` puts its component means."""
    means = surrogate(instance).components(theta).mean
    return _at_indices(instance.observations, _nearest(instance, means), at)


def _count_pair(family: object) -> IndependentCountPair:
    """``family`` as an :class:`IndependentCountPair`, refusing anything else."""
    if not isinstance(family, IndependentCountPair):
        msg = f"the projection seeds a count pair, not {type(family).__name__}"
        raise TypeError(msg)
    return family


#: Draws a chain keeps, and the proposals it discards first. The three
#: chain-based candidates are set to spend the same 72 gradients --- 8
#: proposals of :data:`CHAIN_TRAJECTORY` steps, and one more per proposal for
#: the value at it --- so what separates them is where the proposals go and
#: not how many there are.
CHAIN_DRAWS = 4
CHAIN_BURN_IN = 4

#: Proposals the annealing control runs, and rounds each tempering replica
#: runs; ``TEMPERING_ROUNDS * len(TEMPERATURES)`` is the annealing control's
#: proposal count.
ANNEAL_STEPS = 8
TEMPERING_ROUNDS = 2


def _generator(rng: np.random.Generator) -> torch.Generator:
    """A torch stream derived from the caller's generator, so one seed runs the cell."""
    return torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))


def chain_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """A short Hamiltonian chain on the surrogate; the last draw is the seeding.

    Returns
    -------
    Seeding
        Its diagnostics carry the acceptance rate and the mean energy error,
        because a seed drawn from a chain that has not mixed is a random
        restart with a longer bill.
    """
    initializer = FromChain(
        CHAIN_DRAWS,
        CHAIN_STEP,
        _generator(rng),
        n_steps=CHAIN_TRAJECTORY,
        burn_in=CHAIN_BURN_IN,
    )
    chain = initializer.chain(surrogate(instance))
    return Seeding(
        _from_theta(instance, chain.theta[-1], at),
        PASSES_PER_GRADIENT * chain.force_evaluations,
        f"acceptance {chain.acceptance_rate:.2f}, mean energy error "
        f"{float(chain.energy_error.mean()):.2e}",
    )


def annealed_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """The single-chain control for tempering: the best point of a falling temperature.

    Returns
    -------
    Seeding
    """
    run = FromAnnealing(
        Exponential(float(TEMPERATURES[-1]), 1.0, ANNEAL_STEPS),
        CHAIN_STEP,
        _generator(rng),
        n_steps=CHAIN_TRAJECTORY,
    ).run(surrogate(instance))
    return Seeding(
        _from_theta(instance, run.theta, at),
        PASSES_PER_GRADIENT * run.force_evaluations,
        f"acceptance {run.acceptance_rate:.2f}",
    )


def tempered_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding:
    """Parallel tempering on the surrogate; the best point at any temperature.

    Returns
    -------
    Seeding
        Its diagnostics carry the coldest replica's acceptance rate and the
        lowest swap acceptance on the ladder: a pair the ladder never crosses
        makes the replicas independent chains and the run a restart set.
    """
    run = FromTempering(
        TEMPERATURES,
        TEMPERING_ROUNDS,
        CHAIN_STEP,
        _generator(rng),
        n_steps=CHAIN_TRAJECTORY,
    ).run(surrogate(instance))
    return Seeding(
        _from_theta(instance, run.theta, at),
        PASSES_PER_GRADIENT * run.force_evaluations,
        f"cold acceptance {float(run.acceptance_rate[0]):.2f}, lowest swap "
        f"{float(run.swap_acceptance.min()):.2f}",
    )


#: Every candidate, in the order the ticket states them. ``prior`` and ``data``
#: are the two cheap controls; the rest read structure out of the data or out
#: of the surface.
SEEDINGS: dict[str, Callable[..., Seeding]] = {
    "prior": prior_seeding,
    "data": data_seeding,
    "kmeans++": euclidean_seeding,
    "emission++": emission_seeding,
    "burn-in": burn_in_seeding,
    "hmc": chain_seeding,
    "anneal": annealed_seeding,
    "tempering": tempered_seeding,
}


@dataclass(frozen=True)
class Fitted:
    """One seeding, and the fit it started.

    Parameters
    ----------
    name : str
        The candidate.
    seeding : Seeding
        What it produced and charged.
    log_likelihoods : np.ndarray
        The projected log-likelihood at the seeding and after every
        iteration, shape ``(iterations + 1,)``. Non-decreasing.
    iterations : int
        Expectation-maximization iterations run, at most the budget.
    recovery : float
        The fraction of observations assigned to their generating component,
        up to the best renaming of the components.
    mean_error : float
        The largest relative error in a component's negative-binomial mean,
        over the matching that renaming gives. The simulated truth the fit is
        refereed against.
    """

    name: str
    seeding: Seeding
    log_likelihoods: np.ndarray
    iterations: int
    recovery: float
    mean_error: float

    @property
    def log_likelihood(self) -> float:
        """The projected log-likelihood the fit reached."""
        return float(self.log_likelihoods[-1])


def _matching(fitted: IndependentCountPair, truth: IndependentCountPair) -> np.ndarray:
    """Which true component each fitted one stands for, by linear assignment.

    Both families' :meth:`alignment_key` are in observation units, so the cost
    is the summed absolute difference of the two channels' means; ``C!`` is
    past enumeration at ``C = 100`` and the assignment is cubic.
    """
    cost = (
        (fitted.alignment_key()[:, None, :] - truth.alignment_key()[None, :, :])
        .abs()
        .sum(dim=-1)
        .numpy()
    )
    _, columns = linear_sum_assignment(cost)
    return np.asarray(columns)


def fit_projection(
    instance: ProjectedCounts,
    name: str,
    at: ComponentsAt,
    budget: Budget,
    rng: np.random.Generator,
) -> Fitted:
    """Seed by one candidate, then fit the projected mixture within the budget.

    Parameters
    ----------
    instance : ProjectedCounts
        The projected data and its truth.
    name : str
        A key of :data:`SEEDINGS`.
    at : ComponentsAt
        The seam every candidate ends at.
    budget : Budget
        Held equal across candidates; its size is the iterations the fit may
        run, one pass each.
    rng : np.random.Generator
        Passed in.

    Returns
    -------
    Fitted

    Raises
    ------
    KeyError
        If ``name`` is not a candidate.
    """
    seeding = SEEDINGS[name](instance, at, rng)
    weights = torch.full(
        (instance.n_components,), 1.0 / instance.n_components, dtype=torch.float64
    )
    components = seeding.components

    # One iteration at a time, so the curve is the fit's own E step and not a
    # second pass over the data: `expectation_maximization` reports the
    # log-likelihood at the parameters it was given, before the step it takes.
    trace: list[float] = []
    iterations = 0
    for _ in range(budget.size):
        step = expectation_maximization(
            instance.observations, weights, components, max_iterations=1, tolerance=0.0
        )
        trace.append(step.log_likelihood)
        weights, components = step.weights, _count_pair(step.components)
        iterations += 1
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) <= TOLERANCE * abs(trace[-1]):
            break
    final = expectation_maximization(
        instance.observations, weights, components, max_iterations=1, tolerance=0.0
    )
    trace.append(final.log_likelihood)

    assigned = np.asarray(final.responsibilities.argmax(dim=1).numpy())
    columns = _matching(components, instance.truth)
    fitted_mean = components.total.mean.numpy()
    true_mean = instance.truth.total.mean.numpy()[columns]
    return Fitted(
        name=name,
        seeding=seeding,
        log_likelihoods=np.array(trace),
        iterations=iterations,
        recovery=float(np.mean(columns[assigned] == np.asarray(instance.components))),
        mean_error=float(np.max(np.abs(fitted_mean - true_mean) / true_mean)),
    )


@dataclass(frozen=True)
class SeededFit:
    """One candidate as a :mod:`snakes_and_ladders.opt.budget` method.

    A dataclass rather than a closure so that
    :func:`snakes_and_ladders.opt.budget.compare` can run its cells on a
    process pool.

    Parameters
    ----------
    name : str
        A key of :data:`SEEDINGS`.
    at : ComponentsAt
        The seam every candidate ends at.
    """

    name: str
    at: ComponentsAt

    def __call__(
        self, instance: ProjectedCounts, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        """The negative log-likelihood the seeded fit reached, and the passes it spent.

        Returns
        -------
        Outcome
        """
        fitted = fit_projection(instance, self.name, self.at, budget, rng)
        return Outcome(-fitted.log_likelihood, fitted.iterations)
