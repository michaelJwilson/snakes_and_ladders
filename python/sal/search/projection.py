"""The coupled model in projection, and the seedings its emission parameters start from (issue #541).

Drop the spatial coupling and the Markov chain from ``eq:joint`` and each
observation is an independent draw from a mixture over the ``M x K`` (class,
state) pairs. That projected mixture is what a seeding of the emission
parameters sees before the block-coordinate ascent of
:func:`~sal.search.spatio_sequential.fit_spatio_sequential`
runs, so it is where the seedings are compared.

**The projection discards exactly the structure the full model exploits**, so
a seeding that wins here need not win once the spatial term is restored. Every
candidate is therefore scored twice, and
:mod:`tests.regression.search.test_projection_seeding` states both orderings.

**Every candidate ends at the same seam.** A seeding chooses ``M x K``
observations and ``ComponentsAt`` places a component on each
(:mod:`sal.opt.emission_mixture`), so what separates the
candidates is the choice and nothing downstream of it. The three that sample
a surface --- :class:`~sal.sample.initialize.FromChain`,
:class:`~sal.sample.initialize.FromAnnealing` and
:class:`~sal.sample.initialize.FromTempering` --- run on a
Gaussian mixture over the first channel, which is an ``Objective`` and needs
no new code; their component means are realized as the observations nearest
them, so they reach the same seam as the rest.

**Cost is counted in passes over the data**, one pass being every observation
scored under every component. An expectation-maximization iteration spends one
on its E step, and the M step is not counted because every candidate pays the
identical one whatever seeded it. A gradient is two, forward and backward.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sal.emissions import (
    BetaBinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
)
from sal.opt.budget import Budget, Outcome
from sal.opt.constrain import (
    free_from_log_simplex,
    free_from_positive,
    log_simplex,
    positive,
)
from sal.opt.em import EM, EmConfig
from sal.opt.emission_mixture import (
    ComponentsAt,
    expectation_maximization,
    plus_plus_start,
    uniform_start,
)
from sal.opt.initialize import (
    FromObjective,
    Initializer,
    Perturbed,
    RandomRestart,
    quantile_locations,
)
from sal.opt.mixture import (
    GaussianMixtureObjective,
    KMeansPlusPlus,
    emission_mixture_plus_plus,
    mixture_log_likelihood,
    responsibilities,
)
from sal.opt.mixture import (
    expectation_maximization as gaussian_expectation_maximization,
)
from sal.opt.objective import Objective
from sal.opt.starts import PolishedPoint, Trial
from sal.opt.termination import Termination
from sal.sample.chain import torch_stream
from sal.sample.initialize import FromAnnealing, FromChain, FromTempering
from sal.sample.schedule import ExponentialTempSchedule
from sal.sim.count_pairs import (
    TOTAL,
    IndependentCountPair,
    planted_labels,
)
from sal.sim.spatio_sequential import SpatioSequentialParams
from sal.track import MemoryRun, current, track

#: Passes charged to one gradient of the surrogate objective: the forward
#: evaluation and the backward sweep over the same ``N x C`` densities.
PASSES_PER_GRADIENT = 2.0

#: Leapfrog step for every chain-based candidate, the largest measured stable
#: on the key model's surrogate: 0.1 rejects every proposal and 1.0 diverges to
#: a non-positive scale inside the integrator. The acceptance rate each
#: candidate reports is what says whether the step suited the surface.
CHAIN_STEP = 2.0e-2

#: Leapfrog steps per proposal, below :data:`sal.sample.hmc.DEFAULT_STEPS`
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
        :class:`sal.opt.emission_mixture.CountPairSeeding`
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
    :func:`sal.sim.count_pairs.planted_labels` gives it, a
    state uniformly --- the shared transition is circulant, so its stationary
    distribution is uniform --- and the pair from that class's family at that
    state.

    **Drawn rather than subsampled from the coupled instance's own counts**:
    the coupled draw fixes one state per class per position, so a subsample of
    it carries ``M`` states and not ``M x K`` unless it runs to the full
    length. The parameters are the instance's, and
    :func:`sal.sim.count_pairs.binned_model` supplies them at a
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
    :class:`sal.opt.emission_mixture.CountPairSeeding`: the
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
class Seeding[F: EmissionFamily]:
    """What one start produced, what it charged, and the path it took there.

    The one result every seeding rule returns, here and in
    :data:`sal.search.mixture_starts.STARTS`, whose ``Seeded`` carried the
    same fields and ``path`` (issue #1059).

    Parameters
    ----------
    components : F
        The seeded components, of the family the rule seeds.
    passes : float
        The seeding's own cost, in passes over the data.
    diagnostics : str
        Empty for a heuristic; for a chain or a fit, what says how it ended:
        the acceptance rate, and whatever else says the draw is a draw and
        not a random restart.
    path : tuple[tuple[int, F], ...]
        For a start that iterates, the step of its run at which it held each
        of these components; empty for one that does not.
    """

    components: F
    passes: float
    diagnostics: str = ""
    path: tuple[tuple[int, F], ...] = ()


def _euclidean_score(rows: np.ndarray) -> Callable[..., np.ndarray]:
    """``(index, indices) -> squared Euclidean distance between those pairs``.

    k-means++ as it stands, transplanted onto a pair: the score
    :func:`sal.opt.mixture.kmeans_plus_plus` uses, over two
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
    return _count_pair(seeded)


def euclidean_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
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
) -> Seeding[IndependentCountPair]:
    """``Emission_Mixture++``: the same scheme under the family's own Bregman divergence.

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
) -> Seeding[IndependentCountPair]:
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
) -> Seeding[IndependentCountPair]:
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
    seeded = _count_pair(seeded)
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
) -> Seeding[IndependentCountPair]:
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
        config=EmConfig(max_iterations=BURN_IN_ITERATIONS, tolerance=0.0),
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


def seed_at_means(
    instance: ProjectedCounts, means: torch.Tensor, at: ComponentsAt
) -> IndependentCountPair:
    """The family seeded on the observation nearest each mean in the first channel.

    The step between a candidate that returns *locations* --- a chain on the
    surrogate, an initializer of it, a Gaussian mixture fitted to the first
    channel --- and the seam every candidate ends at. The three chain-based
    candidates reach it through :func:`_from_theta`, which is this function
    with the ``theta`` of :func:`surrogate` read first; a caller holding means
    already calls this one (issue #887).

    Parameters
    ----------
    instance : ProjectedCounts
        The projected data the locations are realized on.
    means : torch.Tensor
        One location per component, in the first channel's units.
    at : ComponentsAt
        The seam.

    Returns
    -------
    IndependentCountPair
    """
    return _at_indices(instance.observations, _nearest(instance, means), at)


def _from_theta(
    instance: ProjectedCounts, theta: torch.Tensor, at: ComponentsAt
) -> IndependentCountPair:
    """The family seeded where a surrogate's ``theta`` puts its component means."""
    return seed_at_means(instance, surrogate(instance).components(theta).mean, at)


def state_bytes(weights: torch.Tensor, components: IndependentCountPair) -> int:
    """What the projected fit's state holds: its weights and its component parameters.

    The quantity :meth:`sal.track.TrackedOptimization.record_cost`
    takes, written here because the state is a family and a weight vector
    rather than one array with an ``nbytes``.

    Parameters
    ----------
    weights : torch.Tensor
        The mixing weights.
    components : IndependentCountPair
        The fitted family.

    Returns
    -------
    int
        Bytes.
    """
    tensors = (
        weights,
        components.total.dispersion,
        components.total.mean,
        components.successes.trials,
        components.successes.alpha,
        components.successes.beta,
    )
    return sum(int(tensor.element_size() * tensor.nelement()) for tensor in tensors)


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


def chain_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
    """A short Hamiltonian chain on the surrogate; the last draw is the seeding.

    Returns
    -------
    Seeding
        Its diagnostics carry the acceptance rate and the mean energy error,
        because a seed drawn from a chain that has not mixed is a random
        restart with a longer bill.
    """
    # No warm-up: the three chain candidates are matched at 72 gradients, and
    # FromChain's default 300-proposal warm-up would add 2,700 to this one
    # alone (issue #898).
    initializer = FromChain(
        CHAIN_DRAWS,
        CHAIN_STEP,
        torch_stream(rng),
        n_steps=CHAIN_TRAJECTORY,
        burn_in=CHAIN_BURN_IN,
        adaptation=None,
    )
    chain = initializer.chain(surrogate(instance))
    return Seeding(
        _from_theta(instance, chain.draws[-1], at),
        PASSES_PER_GRADIENT * chain.force_evaluations,
        f"acceptance {chain.acceptance_rate:.2f}, mean energy error "
        f"{float(chain.energy_error.mean()):.2e}",
    )


def annealed_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
    """The single-chain control for tempering: the best point of a falling temperature.

    Returns
    -------
    Seeding
    """
    run = FromAnnealing(
        ExponentialTempSchedule(float(TEMPERATURES[-1]), 1.0, ANNEAL_STEPS),
        CHAIN_STEP,
        torch_stream(rng),
        n_steps=CHAIN_TRAJECTORY,
    ).run(surrogate(instance))
    return Seeding(
        _from_theta(instance, run.best, at),
        PASSES_PER_GRADIENT * run.spent,
        f"acceptance {run.acceptance_rate:.2f}",
    )


def tempered_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
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
        torch_stream(rng),
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
SEEDINGS: dict[str, Callable[..., Seeding[IndependentCountPair]]] = {
    "prior": prior_seeding,
    "data": data_seeding,
    "kmeans++": euclidean_seeding,
    "emission++": emission_seeding,
    "burn-in": burn_in_seeding,
    "hmc": chain_seeding,
    "anneal": annealed_seeding,
    "tempering": tempered_seeding,
}


#: Iterations the Gaussian-mixture start runs on the first channel: the
#: ceiling `qa.mixture_seeding` fits experiment 010's control at.
GAUSSIAN_EM_ITERATIONS = 500

#: Restarts :func:`restart_seeding` draws, and the spread it draws them at in
#: unconstrained coordinates. A restart costs one surrogate evaluation, which
#: is charged as one pass.
RESTARTS = 4
RESTART_SCALE = 0.5


def gaussian_em_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
    """The Gaussian mixture fitted to the first channel, its means the locations.

    Experiment 010's control: expectation--maximization of
    :class:`~sal.opt.mixture.GaussianMixtureObjective` from one
    k-means++ start, for :data:`GAUSSIAN_EM_ITERATIONS` iterations, one pass
    each --- the calls :func:`sal.qa.mixture_seeding.fitted`
    makes, here so a process pool can run it (issues #887, #891).

    Returns
    -------
    Seeding
        Its diagnostics carry how the Gaussian fit ended.
    """
    channel = np.asarray(instance.observations, dtype=np.float64)[:, TOTAL]
    objective = GaussianMixtureObjective(channel, instance.n_components)
    start = KMeansPlusPlus(1, rng).starts(objective)[0]
    fitted = gaussian_expectation_maximization(
        channel,
        torch.exp(objective.constrain(start)["log_weight"]).detach(),
        objective.components(start),
        config=replace(EM, max_iterations=GAUSSIAN_EM_ITERATIONS),
    )
    ended = fitted.termination
    return Seeding(
        seed_at_means(instance, fitted.components.mean, at),
        float(fitted.iterations),
        "" if ended is None else f"{ended.reason.value} after {ended.iterations}",
    )


def objective_seeding(
    instance: ProjectedCounts, at: ComponentsAt, _rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
    """The surrogate's own nominated point, through :class:`FromObjective`.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    theta = FromObjective().starts(objective)[0]
    return Seeding(seed_at_means(instance, objective.components(theta).mean, at), 0.0)


def perturbed_seeding(
    instance: ProjectedCounts, at: ComponentsAt, _rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
    """That point, tilted off a symmetry it may be stationary at, through :class:`Perturbed`.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    theta = Perturbed().starts(objective)[0]
    return Seeding(seed_at_means(instance, objective.components(theta).mean, at), 0.0)


def restart_seeding(
    instance: ProjectedCounts, at: ComponentsAt, rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
    """The best by surrogate value of :data:`RESTARTS` points drawn by :class:`RandomRestart`.

    Returns
    -------
    Seeding
    """
    objective = surrogate(instance)
    thetas = RandomRestart(RESTARTS, RESTART_SCALE, rng).starts(objective)
    best = min(thetas, key=lambda theta: float(objective(theta)))
    return Seeding(
        seed_at_means(instance, objective.components(best).mean, at),
        float(len(thetas)),
    )


def quantile_seeding(
    instance: ProjectedCounts, at: ComponentsAt, _rng: np.random.Generator
) -> Seeding[IndependentCountPair]:
    """Evenly spaced quantiles of the first channel, through :func:`quantile_locations`.

    Returns
    -------
    Seeding
    """
    channel = torch.as_tensor(
        np.asarray(instance.observations, dtype=np.float64)[:, TOTAL]
    )
    return Seeding(
        seed_at_means(instance, quantile_locations(channel, instance.n_components), at),
        0.0,
    )


#: The starts that produce *locations* rather than observations: the
#: Gaussian-mixture control and the four of :mod:`sal.opt.initialize`,
#: each realized at the seam by :func:`seed_at_means`. Held apart from
#: :data:`SEEDINGS`, which is experiment 009's candidate set and what the
#: suite parametrizes over; module-level, so a process pool can run them
#: (issue #891).
LOCATION_SEEDINGS: dict[str, Callable[..., Seeding[IndependentCountPair]]] = {
    "gaussian-em": gaussian_em_seeding,
    "objective": objective_seeding,
    "perturbed": perturbed_seeding,
    "restart": restart_seeding,
    "quantile": quantile_seeding,
}

#: The starts that read no generator: one run of each is every run.
DETERMINISTIC = frozenset({"objective", "perturbed", "quantile"})


@dataclass(frozen=True)
class Fitted:
    """One seeding, and the fit it started.

    Parameters
    ----------
    name : str
        The candidate.
    seeding : Seeding[IndependentCountPair]
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
    termination : Termination | None
        Whether the fit met its relative tolerance or spent the budget, and
        after how many iterations (issue #860).
    components : IndependentCountPair | None
        The fitted family, so a caller draws where the fit left the
        components beside where the seeding put them (issue #891).
    """

    name: str
    seeding: Seeding[IndependentCountPair]
    log_likelihoods: np.ndarray
    iterations: int
    recovery: float
    mean_error: float
    termination: Termination = dataclass_field(kw_only=True)
    components: IndependentCountPair | None = None

    @property
    def log_likelihood(self) -> float:
        """The projected log-likelihood the fit reached."""
        return float(self.log_likelihoods[-1])


def match_components(fitted: EmissionFamily, truth: EmissionFamily) -> np.ndarray:
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
    *,
    seeding: Seeding[IndependentCountPair] | None = None,
) -> Fitted:
    """Seed by one candidate, then fit the projected mixture within the budget.

    Inside a :func:`sal.track.track` block it records
    ``log_likelihood`` once per iteration, and ``objective`` as its negative:
    :attr:`Fitted.log_likelihoods` entry by entry, so a caller comparing
    candidates reads one curve per fit from the run rather than from a second
    loop of its own (issue #887).

    Parameters
    ----------
    instance : ProjectedCounts
        The projected data and its truth.
    name : str
        A key of :data:`SEEDINGS`, or the label ``seeding`` is reported under.
    at : ComponentsAt
        The seam every candidate ends at.
    budget : Budget
        Held equal across candidates; its size is the iterations the fit may
        run, one pass each.
    rng : np.random.Generator
        Passed in.
    seeding : Seeding[IndependentCountPair] | None
        A seeding produced elsewhere, so a start that is not one of
        :data:`SEEDINGS` is fitted through this same loop at this same budget
        rather than through a second one beside it (issue #887). ``None`` runs
        the candidate ``name`` names.

    Returns
    -------
    Fitted

    Raises
    ------
    KeyError
        If ``name`` is not a candidate and no ``seeding`` is given.
    """
    seeding = SEEDINGS[name](instance, at, rng) if seeding is None else seeding
    weights = torch.full(
        (instance.n_components,), 1.0 / instance.n_components, dtype=torch.float64
    )
    projected = _projected_em(instance, weights, seeding.components, budget)
    recovery, mean_error = _scored(instance, projected.weights, projected.components)
    return Fitted(
        name=name,
        seeding=seeding,
        log_likelihoods=np.array(projected.trace),
        iterations=projected.termination.iterations,
        recovery=recovery,
        mean_error=mean_error,
        termination=projected.termination,
        components=projected.components,
    )


@dataclass(frozen=True)
class _ProjectedRun:
    """What the projected loop ends at: the curve, the parameters, and how it ended."""

    trace: list[float]
    weights: torch.Tensor
    components: IndependentCountPair
    termination: Termination


def _projected_em(
    instance: ProjectedCounts,
    weights: torch.Tensor,
    components: IndependentCountPair,
    budget: Budget,
) -> _ProjectedRun:
    """Expectation-maximization of the projected mixture within the budget, one iteration a pass.

    The loop :func:`fit_projection` and :func:`polish_projected` share, so a
    fit through :mod:`sal.opt.starts` is this arithmetic from
    the point it is handed (issue #894).
    """
    # One iteration at a time, so the curve is the fit's own E step and not a
    # second pass over the data: `expectation_maximization` reports the
    # log-likelihood at the parameters it was given, before the step it takes.
    trace: list[float] = []
    iterations = 0
    converged = False
    tracked = current()
    for _ in range(budget.size):
        step = expectation_maximization(
            instance.observations,
            weights,
            components,
            config=EmConfig(max_iterations=1, tolerance=0.0),
        )
        trace.append(step.log_likelihood)
        tracked.record(
            len(trace) - 1,
            objective=-step.log_likelihood,
            log_likelihood=step.log_likelihood,
        )
        weights, components = step.weights, _count_pair(step.components)
        iterations += 1
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) <= TOLERANCE * abs(trace[-1]):
            converged = True
            break
    # The value at the last parameters: the E step the loop above runs,
    # without the M step a further EM iteration would take and discard. The M
    # step was 4.6 s of a 5.0 s iteration at 100 components (issue #891); the
    # two numbers are the same calls on the same inputs.
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    final_log_likelihood = float(
        mixture_log_likelihood(values, torch.log(weights), components)
    )
    trace.append(final_log_likelihood)
    tracked.record(
        len(trace) - 1,
        objective=-final_log_likelihood,
        log_likelihood=final_log_likelihood,
    )
    tracked.record_cost(len(trace) - 1, state_bytes(weights, components))
    return _ProjectedRun(
        trace, weights, components, Termination.after(iterations, converged=converged)
    )


def _scored(
    instance: ProjectedCounts, weights: torch.Tensor, components: IndependentCountPair
) -> tuple[float, float]:
    """The recovery and the mean error at these parameters, against the draw's truth.

    Returns
    -------
    tuple[float, float]
        The fraction of observations assigned their generating component, up
        to the best renaming, and the largest relative error in a
        component's negative-binomial mean over that renaming.
    """
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    posterior = responsibilities(values, torch.log(weights), components)
    assigned = np.asarray(posterior.argmax(dim=1).numpy())
    columns = match_components(components, instance.truth)
    fitted_mean = components.total.mean.numpy()
    true_mean = instance.truth.total.mean.numpy()[columns]
    return (
        float(np.mean(columns[assigned] == np.asarray(instance.components))),
        float(np.max(np.abs(fitted_mean - true_mean) / true_mean)),
    )


@dataclass(frozen=True)
class SeededFit:
    """One candidate as a :mod:`sal.opt.budget` method.

    A dataclass rather than a closure so that
    :func:`sal.opt.budget.compare` can run its cells on a
    process pool.

    Parameters
    ----------
    name : str
        A key of :data:`SEEDINGS`, or the label ``seeding`` is compared under.
    at : ComponentsAt
        The seam every candidate ends at.
    seeding : Callable | None
        A rule of :data:`SEEDINGS`'s own signature, so a start outside that
        set is compared at the same budget through the same loop (issue
        #887). ``None`` runs the candidate ``name`` names. Under
        :func:`~sal.opt.budget.compare` on a process pool it
        must be importable by name, as every method there must.
    """

    name: str
    at: ComponentsAt
    seeding: Callable[..., Seeding[IndependentCountPair]] | None = None

    def __call__(
        self, instance: ProjectedCounts, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        """The negative log-likelihood the seeded fit reached, and the passes it spent.

        Returns
        -------
        Outcome
        """
        fitted = fit_projection(
            instance,
            self.name,
            self.at,
            budget,
            rng,
            seeding=None
            if self.seeding is None
            else self.seeding(instance, self.at, rng),
        )
        return Outcome(-fitted.log_likelihood, fitted.iterations)


@dataclass(frozen=True)
class ProjectedTrial(Trial):
    """One timed run of a start and its projected fit, as :class:`TimedFit` reports it.

    The seconds and the state's bytes are :class:`~sal.opt.starts.Trial`'s,
    read from the run's ``seconds`` series
    (:meth:`sal.track.TrackedOptimization.record_cost`).

    Parameters
    ----------
    fitted : Fitted
        The fit, its seeding and its fitted components.
    curve : tuple[float, ...]
        The ``log_likelihood`` series the run recorded, one value per
        iteration and one at the end.
    """

    fitted: Fitted
    curve: tuple[float, ...]


@dataclass(frozen=True)
class TimedFit:
    """One start as an :mod:`sal.opt.budget` method whose spend is seconds.

    The fit runs at ``passes``, the budget the starts are compared at; the
    budget :func:`~sal.opt.budget.compare` hands in is a
    ceiling in :attr:`~sal.cost.Cost.SECONDS`, and the spend
    reported against it is the wall clock rounded up. The seeding and the fit
    run inside one :func:`~sal.track.track` block, so the
    seconds cover both, and the whole record rides back on
    :attr:`~sal.opt.budget.Outcome.detail` as a :class:`ProjectedTrial`
    --- a process-pool cell records nowhere else (issue #891).

    Parameters
    ----------
    name : str
        The label the start is reported under.
    at : ComponentsAt
        The seam every candidate ends at.
    seeding : Callable
        A rule of :data:`SEEDINGS`'s signature; module-level where the cell
        runs on a process pool.
    passes : Budget
        The fit's own budget, in :attr:`~sal.cost.Cost.PASSES`.
    """

    name: str
    at: ComponentsAt
    seeding: Callable[..., Seeding[IndependentCountPair]]
    passes: Budget

    def __call__(
        self, instance: ProjectedCounts, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        """The negative log-likelihood reached, the whole seconds spent, and the :class:`ProjectedTrial`.

        Returns
        -------
        Outcome
        """
        del budget  # a ceiling compare checks the spend against, not an input
        with track(MemoryRun()) as tracked:
            fitted = fit_projection(
                instance,
                self.name,
                self.at,
                self.passes,
                rng,
                seeding=self.seeding(instance, self.at, rng),
            )
        run = tracked.run
        if not isinstance(run, MemoryRun):  # pragma: no cover - bound above
            msg = "the timed fit records into the MemoryRun it opened"
            raise TypeError(msg)
        seconds = float(run.last("seconds"))
        trial = ProjectedTrial(
            fitted=fitted,
            seconds=seconds,
            state_bytes=int(run.last("state_bytes")),
            curve=tuple(float(value) for _, value in run.series("log_likelihood")),
        )
        return Outcome(-fitted.log_likelihood, math.ceil(seconds), trial)


class ProjectedObjective(Objective):
    """One projected instance as an :class:`~sal.opt.objective.Objective`.

    What :class:`~sal.opt.starts.StartsBenchmark` needs of an
    instance to hand a start to a polisher (issue #894). ``theta`` is the
    ``M K - 1`` free weights of
    :func:`~sal.opt.constrain.log_simplex`, then the logs of
    every component's negative-binomial dispersion and mean and beta-binomial
    ``alpha`` and ``beta``, ``M K`` each; the trial count is the assay's and is
    held fixed. The value is the projected negative log-likelihood.

    Parameters
    ----------
    instance : ProjectedCounts
        The projected data and its truth.
    at : CountPairAt
        The seam every candidate ends at; :meth:`initial` places a component
        on the observations nearest the first channel's quantiles through it.
    """

    def __init__(self, instance: ProjectedCounts, at: CountPairAt) -> None:
        self._instance = instance
        self._at = at
        self._values = torch.as_tensor(instance.observations, dtype=torch.float64)

    @property
    def instance(self) -> ProjectedCounts:
        """The projected data and its truth."""
        return self._instance

    @property
    def observations(self) -> torch.Tensor:
        """The drawn pairs, shape ``(n_samples, 2)``."""
        return self._values

    def _blocks(self, theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``theta`` split into the free weights and the four log-parameter blocks."""
        size = self._instance.n_components
        return tuple(torch.split(theta, [size - 1, size, size, size, size]))

    def components(self, theta: torch.Tensor) -> IndependentCountPair:
        """The component family ``theta`` encodes."""
        _, dispersion, mean, alpha, beta = (
            positive(block) for block in self._blocks(theta)
        )
        return IndependentCountPair(
            NegativeBinomialEmission(dispersion, mean),
            BetaBinomialEmission(
                torch.full_like(alpha, self._instance.trials), alpha, beta
            ),
        )

    def theta_at(
        self, components: IndependentCountPair, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``theta`` for these components, at uniform weights unless ``weights`` are given."""
        if weights is None:
            weights = torch.full(
                (self._instance.n_components,),
                1.0 / self._instance.n_components,
                dtype=torch.float64,
            )
        return self.theta_from(
            {"log_weight": torch.log(weights), **components.named_parameters()}
        )

    def initial(self) -> torch.Tensor:
        """Uniform weights, and a component on the observation nearest each quantile of the totals."""
        return self.theta_at(
            seed_at_means(
                self._instance,
                quantile_locations(self._values[:, TOTAL], self._instance.n_components),
                self._at,
            )
        )

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The log weights, and the family's parameters under its own keys."""
        return {
            "log_weight": log_simplex(self._blocks(theta)[0]),
            **self.components(theta).named_parameters(),
        }

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``."""
        return torch.cat(
            [
                free_from_log_simplex(named["log_weight"].to(torch.float64)),
                *(
                    free_from_positive(named[key].reshape(-1).to(torch.float64))
                    for key in _PROJECTED_KEYS
                ),
            ]
        )

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The projected negative log-likelihood."""
        return -mixture_log_likelihood(
            self._values, log_simplex(self._blocks(theta)[0]), self.components(theta)
        )


#: The keys :meth:`IndependentCountPair.named_parameters` returns, in the
#: order :class:`ProjectedObjective` lays them out in ``theta``.
_PROJECTED_KEYS = (
    "total.dispersion",
    "total.mean",
    "successes.alpha",
    "successes.beta",
)


@dataclass(frozen=True)
class SeedingStart(Initializer):
    """A seeding rule of this module as an initializer of the projected objective.

    Each start draws the rule from ``rng``, one after another, so ``n_starts``
    starts are the draws :func:`~sal.opt.budget.restarts` of
    :class:`SeededFit` makes (issue #894). Records the passes the rule
    charged as ``seeding_passes``, which the seam keeps as a diagnostic.

    Parameters
    ----------
    name : str
        A key of :data:`SEEDINGS` or :data:`LOCATION_SEEDINGS`.
    at : CountPairAt
        The seam every candidate ends at.
    rng : np.random.Generator
        The cell's generator.
    n_starts : int
        Draws of the rule, each a start.
    """

    name: str
    at: CountPairAt
    rng: np.random.Generator
    n_starts: int = 1

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """``n_starts`` seedings of the objective's instance, at uniform weights.

        Returns
        -------
        list[torch.Tensor]

        Raises
        ------
        TypeError
            If the objective is not a :class:`ProjectedObjective`.
        """
        if not isinstance(objective, ProjectedObjective):
            msg = (
                f"a projected seeding seeds a ProjectedObjective, not "
                f"{type(objective).__name__}"
            )
            raise TypeError(msg)
        rule = {**SEEDINGS, **LOCATION_SEEDINGS}[self.name]
        drawn = [
            rule(objective.instance, self.at, self.rng) for _ in range(self.n_starts)
        ]
        current().record(0, seeding_passes=sum(seeding.passes for seeding in drawn))
        return [objective.theta_at(seeding.components) for seeding in drawn]


def _projected(objective: Objective) -> ProjectedObjective:
    """``objective`` as the projected objective, refusing anything else."""
    if not isinstance(objective, ProjectedObjective):
        msg = f"expected a ProjectedObjective, got {type(objective).__name__}"
        raise TypeError(msg)
    return objective


def polish_projected(
    objective: Objective, theta: torch.Tensor, budget: Budget
) -> PolishedPoint:
    """:func:`fit_projection`'s loop from ``theta``: the seam's polisher of experiment 009.

    Returns
    -------
    sal.opt.starts.PolishedPoint
        The last parameters, the negative log-likelihood there, and how the
        loop ended.
    """
    projected = _projected(objective)
    with torch.no_grad():
        weights = torch.exp(projected.constrain(theta)["log_weight"])
        components = projected.components(theta)
    run = _projected_em(projected.instance, weights, components, budget)
    return PolishedPoint(
        value=-run.trace[-1],
        termination=run.termination,
        theta=projected.theta_at(run.components, run.weights),
    )


def projected_recovery(objective: Objective, theta: torch.Tensor) -> float:
    """The fraction of observations assigned their generating component, up to renaming.

    Returns
    -------
    float
    """
    projected = _projected(objective)
    with torch.no_grad():
        weights = torch.exp(projected.constrain(theta)["log_weight"])
        return _scored(projected.instance, weights, projected.components(theta))[0]


def projected_mean_error(objective: Objective, theta: torch.Tensor) -> float:
    """The largest relative error in a component's negative-binomial mean, over that renaming.

    Returns
    -------
    float
    """
    projected = _projected(objective)
    with torch.no_grad():
        weights = torch.exp(projected.constrain(theta)["log_weight"])
        return _scored(projected.instance, weights, projected.components(theta))[1]
