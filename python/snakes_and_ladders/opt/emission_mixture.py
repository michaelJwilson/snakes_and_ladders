"""A mixture of count emissions: the Gaussian mixture with its family swapped.

:mod:`snakes_and_ladders.opt.mixture`'s mixture machinery --- the
log-likelihood, the responsibilities, the weights as their mean --- asks its
components only for a log-density and an M step, so it is reused here and
this module adds the loop around them and the start that seeds it.

**The M step is the family's, and nothing here reimplements one.** A count
family's is an optimization rather than a formula
(:class:`snakes_and_ladders.emissions.CountPairEmission` solves for a
dispersion and for a beta-binomial's two shape parameters) and reports whether
it settled. This loop propagates that report: an unconverged inner solve
reaching an outer likelihood is the fault ``likelihood/CLAUDE.md`` forbids.

**And it is where ``Emission_Mixture++`` finally has a model.** Issue #306
built the seeding rule --- k-means++ with a family's own Bregman divergence as
the distance --- and recorded that no problem used it.

Ground truth and data generation live in
:mod:`snakes_and_ladders.sim.emission_mixture`; this module draws no data.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.emissions import CountPairEmission, EmissionFamily
from snakes_and_ladders.enumeration import refuse_oversized
from snakes_and_ladders.opt.em import em_loop
from snakes_and_ladders.opt.mixture import (
    emission_mixture_plus_plus,
    mixture_log_likelihood,
    responsibilities,
    uniform_seeds,
)
from snakes_and_ladders.opt.termination import Termination

#: Builds a ``k``-state family centred on ``k`` observations, one per row. The
#: seam an initializer needs from a model whose parameters it cannot otherwise
#: interpret --- the counterpart of
#: :meth:`snakes_and_ladders.opt.mixture.GaussianMixtureObjective.theta_from_centres`
#: for a family that has no ``theta``.
type ComponentsAt = Callable[[np.ndarray], EmissionFamily]


@dataclass(frozen=True)
class EmissionMixtureFit:
    """What one expectation-maximization run produced.

    Parameters
    ----------
    weights : torch.Tensor
        Fitted mixing weights, shape ``(n_components,)``.
    components : EmissionFamily
        The fitted component family.
    responsibilities : torch.Tensor
        ``P(component | observation)`` at the fitted parameters, shape
        ``(n_samples, n_components)``. Carried on the result because it is the
        E step the last M step consumed; recomputing it from the returned
        parameters gives the next iteration's.
    log_likelihood : float
        The log-likelihood at the returned parameters. A probability, since
        every count family is discrete, so it is at most zero.
    iterations : int
        EM iterations run.
    at_boundary : bool
        Whether a component's M step reached the edge of the range this data
        identifies its parameter over --- a flat likelihood in a dispersion or
        a concentration, reported rather than treated as an error (issue
        #122).
    termination : Termination | None
        Whether the loop met its relative tolerance or ran out of iterations,
        in the form every result states it in (issue #860). ``iterations``
        stays: it is what this result has always been read by.
    """

    weights: torch.Tensor
    components: EmissionFamily
    responsibilities: torch.Tensor
    log_likelihood: float
    iterations: int
    at_boundary: bool
    termination: Termination | None = None


def expectation_maximization(
    observations: np.ndarray | torch.Tensor,
    weights: torch.Tensor,
    components: EmissionFamily,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> EmissionMixtureFit:
    """Fit a mixture of count emissions by EM.

    The E step is :func:`snakes_and_ladders.opt.mixture.responsibilities` and
    the M step is the family's own :meth:`reestimate`: independent
    observations carry no message between them, and the family receives the
    posterior an HMM's forward--backward pass would hand it. The alternation
    around the two is :func:`snakes_and_ladders.opt.em.em_loop`'s, which is
    the rung this one shares with
    :func:`snakes_and_ladders.opt.mixture.expectation_maximization` and its
    only one: the defaults, the result type and the oracle this is pinned
    against are here (issue #859).

    Parameters
    ----------
    observations : np.ndarray | torch.Tensor
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    weights : torch.Tensor
        Starting mixing weights, shape ``(n_components,)``.
    components : EmissionFamily
        Starting components.
    max_iterations : int
        Maximum EM iterations.
    tolerance : float
        Stop when the log-likelihood improves by less than this *relative* to
        its magnitude --- absolute would not transfer across data sizes
        (``DEV.md``, issue #111).

    Returns
    -------
    EmissionMixtureFit
        The fitted parameters, the responsibilities at them, and the
        log-likelihood.

    Raises
    ------
    ValueError
        If a component's M step did not converge. A number read off an inner
        solve that never settled is not an estimate, and returning it here
        would surface several iterations later as a non-monotone likelihood.
    """
    values = torch.as_tensor(observations, dtype=torch.float64)
    boundary = False
    attempt = 0

    def step(
        state: tuple[torch.Tensor, EmissionFamily, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, EmissionFamily, torch.Tensor], float]:
        """One E step, one M step, and the log-likelihood at the state given."""
        nonlocal boundary, attempt
        attempt += 1
        current, family, _ = state
        log_weight = torch.log(current)
        log_likelihood = float(mixture_log_likelihood(values, log_weight, family))
        posterior = responsibilities(values, log_weight, family)
        reestimated = family.reestimate(values, posterior)
        if not reestimated.converged:
            msg = (
                f"a component's M step did not settle at EM iteration "
                f"{attempt}: residual {reestimated.residual:.3e} after "
                f"{reestimated.iterations} inner iterations"
            )
            raise ValueError(msg)
        boundary = boundary or reestimated.at_boundary
        advanced = (posterior.mean(dim=0), reestimated.emissions, posterior)
        return advanced, log_likelihood

    # The responsibilities of the last E step are carried out of the loop, so
    # the result holds the posterior its final M step consumed. Before the
    # first step there is none, and an empty budget returns that.
    start = (
        weights,
        components,
        torch.empty((values.shape[0], components.n_states), dtype=torch.float64),
    )
    (weights, components, posterior), log_likelihood, termination = em_loop(
        step, start, tolerance=tolerance, max_iterations=max_iterations
    )
    return EmissionMixtureFit(
        weights=weights,
        components=components,
        responsibilities=posterior,
        log_likelihood=log_likelihood,
        iterations=termination.iterations,
        at_boundary=boundary,
        termination=termination,
    )


def enumerated_posterior(
    observations: np.ndarray | torch.Tensor,
    log_weight: torch.Tensor,
    components: EmissionFamily,
) -> torch.Tensor:
    """``P(component | observations)`` summed over every joint labelling.

    The independent answer :func:`snakes_and_ladders.opt.mixture.responsibilities`
    is refereed against, sharing no line with it: the responsibilities
    normalize each observation's row on its own, while this scores each of the
    ``K ** N`` labellings of the whole dataset, normalizes over all of them,
    and marginalizes back to one row per observation. Their agreement is what
    says the mixture's posterior factorizes across observations.

    Exponential in the number of observations, so affordable only over a
    handful. The observations being independent, the marginal at each of a
    handful is the quantity the whole dataset's carries, which is what makes a
    prefix a legitimate oracle.

    Parameters
    ----------
    observations : np.ndarray | torch.Tensor
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    log_weight : torch.Tensor
        Log mixing weights, shape ``(n_components,)``.
    components : EmissionFamily
        The component densities.

    Returns
    -------
    torch.Tensor
        Shape ``(n_samples, n_components)``, each row summing to one.

    Raises
    ------
    ValueError
        If ``n_components ** n_samples`` is past
        :data:`snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`.
    """
    values = torch.as_tensor(observations, dtype=torch.float64)
    n_samples = int(values.shape[0])
    n_components = int(log_weight.shape[0])
    refuse_oversized(
        n_components**n_samples,
        what=f"{n_components} ** {n_samples} component labellings",
    )
    scored = (components.log_density(values) + log_weight).numpy()

    labellings = list(itertools.product(range(n_components), repeat=n_samples))
    joint = np.array(
        [
            sum(scored[position, state] for position, state in enumerate(labelling))
            for labelling in labellings
        ]
    )
    probability = np.exp(joint - joint.max())
    probability /= probability.sum()

    marginal = np.zeros((n_samples, n_components))
    for weight, labelling in zip(probability, labellings, strict=True):
        for position, state in enumerate(labelling):
            marginal[position, state] += weight
    return torch.as_tensor(marginal)


@dataclass(frozen=True)
class CountPairSeeding:
    """Places a :class:`CountPairEmission` component on an observed pair.

    The pair says where a component starts: its depth is the component's
    negative-binomial mean, and its allele fraction, smoothed by the Jeffreys
    prior's half-count so ``(n, 0)`` does not seed a rate of exactly zero, is
    the beta-binomial rate. One observation carries no *shape*, so the
    dispersion and concentration start at declared values shared by every
    component.

    Parameters
    ----------
    dispersion : float
        Starting negative-binomial ``r`` for every component.
    concentration : float
        Starting beta-binomial ``a + b`` for every component.
    joint : bool
        Which form of the family to build. No default, for the reason
        :class:`CountPairEmission` gives.
    trials : float | None
        The independent form's fixed trial count, one number shared by every
        component: a property of the assay rather than of a component, and a
        per-component vector is unreadable when the seeding scores one
        candidate component against the observations. ``None`` in the joint
        form.

    Raises
    ------
    ValueError
        If the trial count is given in the joint form or omitted in the
        independent one, on the terms :class:`CountPairEmission` states.
    """

    dispersion: float
    concentration: float
    joint: bool
    trials: float | None = None

    def __post_init__(self) -> None:
        if self.joint and self.trials is not None:
            msg = "the joint form's trial count is the observed total"
            raise ValueError(msg)
        if not self.joint and self.trials is None:
            msg = "the independent form needs a fixed trial count"
            raise ValueError(msg)

    def __call__(self, rows: np.ndarray) -> CountPairEmission:
        """The family seeded on these pairs, one component per row.

        Parameters
        ----------
        rows : np.ndarray
            Observed pairs, shape ``(n_components, 2)``.

        Returns
        -------
        CountPairEmission
            A family with one state per row.

        """
        pairs = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
        totals, successes = pairs[:, 0], pairs[:, 1]
        trials = None if self.trials is None else np.full(pairs.shape[0], self.trials)
        over = np.maximum(totals, 1.0) if trials is None else trials
        rate = (successes + 0.5) / (over + 1.0)
        return CountPairEmission(
            np.full(pairs.shape[0], self.dispersion),
            np.maximum(totals, 1.0),
            rate * self.concentration,
            (1.0 - rate) * self.concentration,
            trials,
            joint=self.joint,
        )


def _seed_scores(
    observations: np.ndarray, at: ComponentsAt
) -> Callable[..., np.ndarray]:
    """``(index, indices) -> D_phi(y, the component seeded at that index)``.

    :func:`snakes_and_ladders.opt.mixture.emission_mixture_plus_plus` draws its
    seeds from the array it is given, so the array here is of *indices*: an
    observation is a pair, and a draw from a flattened array of pairs would
    seed a component on half of one.

    **The score is the family's Bregman divergence, not its negative log
    density** (issue #560). The two differ by ``log b_phi(y)``, the log density
    at the family's best member for that observation, which depends on the
    observation and not on the seed; D-squared sampling normalizes its scores
    rather than shifting them, so that term does not cancel --- it dilutes the
    rule toward uniform in proportion to its size against a typical divergence.
    Measured on a Gaussian, where the divergence is the squared Euclidean
    distance exactly: seeding under the divergence costs **1.8496** times the
    exact optimum, under the negative log density **3.9111**, and uniformly
    **4.3470** (``tests/regression/opt/test_opt_mixture_seeding.py``,
    ``docs/experiments/010``). The divergence is non-negative, as the sampling
    rule needs, and zero at the seed's own observation.
    """
    rows = np.asarray(observations, dtype=np.float64)

    def score(seed: float, candidates: np.ndarray) -> np.ndarray:
        family = at(rows[[int(seed)]])
        scored = family.bregman_divergence(
            torch.as_tensor(rows[candidates.astype(np.int64)], dtype=torch.float64)
        )
        return np.asarray(scored[:, 0].numpy())

    return score


def plus_plus_start(
    observations: np.ndarray,
    n_components: int,
    at: ComponentsAt,
    rng: np.random.Generator,
) -> EmissionFamily:
    """Seed the components by ``Emission_Mixture++`` (issue #306, ``eq:kmeanspp``).

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    n_components : int
        Components to seed.
    at : ComponentsAt
        Builds a family from the chosen observations.
    rng : np.random.Generator
        Generator, passed in rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    EmissionFamily
        The seeded components.
    """
    rows = np.asarray(observations, dtype=np.float64)
    indices = np.arange(rows.shape[0], dtype=np.float64)
    chosen = emission_mixture_plus_plus(
        indices, n_components, _seed_scores(rows, at), rng
    )
    return at(rows[chosen.astype(np.int64)])


def uniform_start(
    observations: np.ndarray,
    n_components: int,
    at: ComponentsAt,
    rng: np.random.Generator,
) -> EmissionFamily:
    """Seed the components on observations drawn uniformly, without replacement.

    The baseline :func:`plus_plus_start` is measured against, kept beside it
    on the terms :func:`snakes_and_ladders.opt.mixture.uniform_seeds` states.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, channels)``.
    n_components : int
        Components to seed.
    at : ComponentsAt
        Builds a family from the chosen observations.
    rng : np.random.Generator
        Generator, passed in.

    Returns
    -------
    EmissionFamily
        The seeded components.
    """
    rows = np.asarray(observations, dtype=np.float64)
    indices = np.arange(rows.shape[0], dtype=np.float64)
    chosen = uniform_seeds(indices, n_components, rng)
    return at(rows[chosen.astype(np.int64)])
