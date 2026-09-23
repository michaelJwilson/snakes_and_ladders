"""A Gaussian mixture: the reference instance whose start reads the data.

The third instance of ``Objective`` beside the Potts chain and the HMM,
testing two abstractions at once (issue #262); the problem is stated in
``sec:mixture``.

**It is the emission seam without the Markov chain.** Its M step for the
components *is*
:meth:`snakes_and_ladders.emissions.GaussianEmission.reestimate`, called with
responsibilities where an HMM passes state posteriors --- the evidence that
the seam extracted from an HMM was not shaped by one.

**And it is the model a data-dependent initializer was waiting for.** Issue
#251 built the ``Initializer`` protocol with nothing to initialize.
:class:`KMeansPlusPlus` lives here rather than in
:mod:`snakes_and_ladders.opt.initialize` for the reason #251 gave: a strategy
that reads the observations is specific to a model. It refuses an objective it
does not know rather than guessing what the parameter vector means.

**The likelihood is unbounded, exactly as the Gaussian HMM's is.** The floor
derived in :func:`snakes_and_ladders.emissions.pooled_variance_floor`
transfers unchanged, and reaching it is a refusal rather than a clamp.

Ground truth and data generation live in
:mod:`snakes_and_ladders.sim.mixture`; this module holds the fitting
objective, its EM oracle, and the seeding a start needs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.emissions import (
    EmissionFamily,
    GaussianEmission,
    pooled_variance_floor,
)
from snakes_and_ladders.opt.constrain import (
    free_from_log_simplex,
    free_from_positive,
    log_simplex,
    positive,
)
from snakes_and_ladders.opt.em import em_loop
from snakes_and_ladders.opt.initialize import Initializer, quantile_locations
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.termination import Termination


class GaussianMixtureObjective(Objective):
    """Negative log-likelihood of independent observations from a mixture.

    Parameters
    ----------
    observations : np.ndarray
        Observed values, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
        A trailing channel axis makes every component a product of independent
        Gaussians, one per channel, on the terms
        :class:`snakes_and_ladders.emissions.GaussianEmission` states.
    n_components : int
        Components in the mixture, at least 2 --- one component is a Gaussian,
        not a mixture, and the weight vector would be a constant.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default, since a
        finite-difference derivative check is meaningless in ``float32``.

    Raises
    ------
    ValueError
        If fewer than two components are asked for, or the observations carry
        more than one trailing axis.
    """

    def __init__(
        self,
        observations: np.ndarray,
        n_components: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n_components < 2:
            msg = f"n_components must be >= 2, got {n_components}"
            raise ValueError(msg)
        values = torch.as_tensor(observations, dtype=dtype)
        values = values.reshape(1) if values.ndim == 0 else values
        if values.ndim > 2:
            msg = (
                f"an observation is a value, or one value per channel; got "
                f"shape {tuple(values.shape)}"
            )
            raise ValueError(msg)
        self._observations = values
        self._n_channels = 1 if values.ndim == 1 else int(values.shape[1])
        self._n_components = n_components
        self._dtype = dtype
        self._variance_floor = pooled_variance_floor(np.asarray(observations))

    @property
    def n_components(self) -> int:
        """Components in the mixture."""
        return self._n_components

    @property
    def n_channels(self) -> int:
        """Entries an observation carries; ``1`` for a scalar observation."""
        return self._n_channels

    @property
    def observations(self) -> torch.Tensor:
        """The observations being fitted, of the shape they were given in."""
        return self._observations

    @property
    def variance_floor(self) -> float:
        """The floor the EM oracle refuses at, derived from these observations."""
        return self._variance_floor

    @property
    def n_parameters(self) -> int:
        """``(k - 1)`` free weights, then ``k`` means and ``k`` log scales, per channel."""
        return (self._n_components - 1) + 2 * self._n_components * self._n_channels

    @property
    def _block(self) -> int:
        return self._n_components * self._n_channels

    @property
    def _weight_slice(self) -> slice:
        return slice(0, self._n_components - 1)

    def _mean_slice(self) -> slice:
        start = self._n_components - 1
        return slice(start, start + self._block)

    def _log_scale_slice(self) -> slice:
        start = self._mean_slice().stop
        return slice(start, start + self._block)

    def _per_component(self, block: torch.Tensor) -> torch.Tensor:
        """One parameter block as the family reads it: ``(k,)``, or ``(k, channels)``."""
        if self._n_channels == 1:
            return block
        return block.reshape(self._n_components, self._n_channels)

    def components(self, theta: torch.Tensor) -> GaussianEmission:
        """The component family ``theta`` encodes, differentiable in ``theta``."""
        return GaussianEmission(
            self._per_component(theta[self._mean_slice()]),
            positive(self._per_component(theta[self._log_scale_slice()])),
            self._variance_floor,
        )

    def initial(self) -> torch.Tensor:
        """Uniform weights, means at quantiles of the data, pooled scales.

        The Gaussian HMM's construction, for its two reasons: equal means
        leave the components exchangeable and the gradient in that block
        exactly zero, and a mean far from every observation contributes a
        density that underflows, so the fit silently becomes one with fewer
        components.
        """
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        if self._n_channels == 1:
            theta[self._mean_slice()] = quantile_locations(
                self._observations, self._n_components
            )
            theta[self._log_scale_slice()] = free_from_positive(
                self._observations.std()
            )
            return theta
        # Per channel, since a quantile of the two channels pooled is a
        # location in neither and would seed every component off the data.
        theta[self._mean_slice()] = quantile_locations(
            self._observations, self._n_components, dim=0
        ).reshape(-1)
        theta[self._log_scale_slice()] = free_from_positive(
            self._observations.std(dim=0)
        ).repeat(self._n_components)
        return theta

    def theta_from_centres(self, centres: torch.Tensor) -> torch.Tensor:
        """A start whose component means are ``centres``.

        The seam an initializer that reads the data needs: it supplies
        locations, and the objective places them.

        Parameters
        ----------
        centres : torch.Tensor
            One location per component, shape ``(n_components,)`` or
            ``(n_components, n_channels)``.

        Returns
        -------
        torch.Tensor
            ``theta`` with uniform weights, these means, and pooled scales.

        Raises
        ------
        ValueError
            If the number of centres is not the number of components.
        """
        located = torch.as_tensor(centres, dtype=self._dtype).reshape(-1)
        if located.shape[0] != self._block:
            msg = (
                f"expected {self._n_components} centres of {self._n_channels} "
                f"channel(s), got {located.shape[0]} values"
            )
            raise ValueError(msg)
        theta = self.initial()
        theta[self._mean_slice()] = located
        return theta

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The mixing weights as log-probabilities, and the component parameters."""
        return {
            "log_weight": log_simplex(theta[self._weight_slice]),
            **self.components(theta).named_parameters(),
        }

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood, marginalizing the component of each point."""
        return -mixture_log_likelihood(
            self._observations,
            log_simplex(theta[self._weight_slice]),
            self.components(theta),
        )

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``.

        The inverse of the constraint map, keyed as :meth:`constrain`
        returns, so a fit produced by expectation-maximization — which never
        builds a ``theta`` — can be given an interval at the point it reached
        (issue #268).

        Parameters
        ----------
        named : Mapping[str, torch.Tensor]
            Constrained parameters, under :meth:`constrain`'s own keys.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns ``named``.
        """
        return torch.cat(
            [
                free_from_log_simplex(named["log_weight"].to(self._dtype)),
                named["mean"].reshape(-1).to(self._dtype),
                free_from_positive(named["scale"].reshape(-1).to(self._dtype)),
            ]
        )

    def theta_from_truth(
        self, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray
    ) -> torch.Tensor:
        """Place a known truth in the unconstrained coordinates.

        Parameters
        ----------
        weights : np.ndarray
            True mixing weights, shape ``(n_components,)``.
        mean, scale : np.ndarray
            True per-component mean and standard deviation.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns this truth.
        """
        return torch.cat(
            [
                free_from_log_simplex(
                    torch.log(torch.as_tensor(weights, dtype=self._dtype))
                ),
                torch.as_tensor(mean, dtype=self._dtype).reshape(-1),
                free_from_positive(torch.as_tensor(scale, dtype=self._dtype)).reshape(
                    -1
                ),
            ]
        )


def component_log_density(
    components: EmissionFamily,
    observations: torch.Tensor,
    covariate: torch.Tensor | None,
) -> torch.Tensor:
    """The components' log-density, conditioned on ``covariate`` where one is given.

    A family that takes no covariate is called as it always was, so a mixture
    without one is unchanged bitwise (issue #933).
    """
    if covariate is None:
        return components.log_density(observations)
    return components.log_density(observations, covariate)


def mixture_log_likelihood(
    observations: torch.Tensor,
    log_weight: torch.Tensor,
    components: EmissionFamily,
    *,
    covariate: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sum_i log sum_k w_k N(y_i; mu_k, s_k)``, in log space throughout (``eq:mixture``).

    Parameters
    ----------
    observations : torch.Tensor
        Observations, shape ``(n_samples,)``.
    log_weight : torch.Tensor
        Log mixing weights, shape ``(n_components,)``.
    components : EmissionFamily
        The component densities. Any family: the mixture asks its components
        for a log-density and nothing else, which lets
        :mod:`snakes_and_ladders.opt.emission_mixture` fit a mixture of count
        emissions through this function unchanged.
    covariate : torch.Tensor | None
        Per-observation covariate the components condition on, in the layout
        the family's ``log_density`` documents (issue #933).

    Returns
    -------
    torch.Tensor
        Scalar, differentiable with respect to every parameter. A **density**
        where the components are continuous, so it may be positive; a
        probability where they are discrete. The mixture inherits which from
        its components.
    """
    return torch.logsumexp(
        log_weight + component_log_density(components, observations, covariate), dim=-1
    ).sum()


def responsibilities(
    observations: torch.Tensor,
    log_weight: torch.Tensor,
    components: EmissionFamily,
    *,
    covariate: torch.Tensor | None = None,
) -> torch.Tensor:
    """``P(component | observation)``, shape ``(n_samples, n_components)`` (``eq:responsibilities``).

    The E step. An HMM's is a forward-backward recursion; a mixture's is one
    normalization, because independent observations carry no message between
    them. What the M step then receives is the same object either way, which
    is why the same emission family serves both. ``covariate`` is
    :func:`mixture_log_likelihood`'s.
    """
    joint = log_weight + component_log_density(components, observations, covariate)
    return torch.exp(joint - torch.logsumexp(joint, dim=-1, keepdim=True))


def e_step(
    observations: torch.Tensor,
    log_weight: torch.Tensor,
    components: EmissionFamily,
    *,
    covariate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`mixture_log_likelihood` and :func:`responsibilities` from one log-density pass (issue #924).

    An EM step needs both at the same parameters, and each of the two
    functions evaluates the components' log-density on every observation.
    Here the joint is formed once and its row normalizer taken once; the
    operations on it are the two functions' own, so both results are theirs
    bitwise.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The scalar log-likelihood, and the responsibilities, shape
        ``(n_samples, n_components)``.
    """
    joint = log_weight + component_log_density(components, observations, covariate)
    normalizer = torch.logsumexp(joint, dim=-1, keepdim=True)
    return normalizer.sum(), torch.exp(joint - normalizer)


@dataclass(frozen=True)
class MixtureMetrics:
    """What a mixture parameter vector means: its log-likelihood and its k-means cost (issue #778).

    A :class:`snakes_and_ladders.track.Metrics` over ``theta``, satisfied
    structurally.

    * ``log_likelihood`` is :func:`mixture_log_likelihood` at ``theta``'s
      weights and components, which
      :meth:`GaussianMixtureObjective.__call__` negates to minimize.
    * ``clustering_cost`` is :func:`clustering_cost` at the fitted component
      means: the summed squared distance of the observations to their nearest
      centre, which :func:`optimal_clustering_cost` bounds below in one
      dimension.

    **The assignment agreement the ticket listed is left out.** The only
    package function that produces one is
    :func:`snakes_and_ladders.likelihood.mixture_assignments.enumerate_mixture_assignments`,
    which sums ``k ** n`` assignments and refuses above
    ``MAX_ENUMERABLE_CONFIGURATIONS`` --- not a quantity to compute once a
    sweep --- and ``opt`` may not import ``likelihood``
    (``tests/regression/opt/test_opt_objective.py``).

    Parameters
    ----------
    objective : GaussianMixtureObjective
        The objective being fitted, which owns the observations and the
        constraint map.
    """

    objective: GaussianMixtureObjective

    names: tuple[str, ...] = ("log_likelihood", "clustering_cost")

    def __call__(self, theta: torch.Tensor) -> dict[str, float]:
        """The two metrics at ``theta``, computed without a graph."""
        with torch.no_grad():
            components = self.objective.components(theta)
            value = mixture_log_likelihood(
                self.objective.observations,
                self.objective.constrain(theta)["log_weight"],
                components,
            )
            return {
                "log_likelihood": float(value),
                "clustering_cost": clustering_cost(
                    self.objective.observations.numpy(),
                    components.mean.detach().numpy(),
                ),
            }


@dataclass(frozen=True)
class MixtureFit:
    """What one expectation-maximization run produced.

    Parameters
    ----------
    weights : torch.Tensor
        Fitted mixing weights, shape ``(n_components,)``.
    components : GaussianEmission
        The fitted component family.
    log_likelihood : float
        The final log-likelihood. A density, so possibly positive.
    iterations : int
        EM iterations run.
    at_boundary : bool
        Whether a component's M step reached the edge of the range this data
        identifies its parameter over --- reported rather than treated as an
        error (issue #122), as
        :attr:`snakes_and_ladders.opt.emission_mixture.EmissionMixtureFit.at_boundary`
        is. ``False`` for the closed-form Gaussian step, which has no such
        edge; the field carries what a family with one reports (issue #856).
    termination : Termination | None
        Whether the loop met its relative tolerance or ran out of iterations,
        in the form every result states it in (issue #860). ``iterations``
        stays: it is what this result has always been read by.
    """

    weights: torch.Tensor
    components: GaussianEmission
    log_likelihood: float
    iterations: int
    at_boundary: bool
    termination: Termination | None = None


def expectation_maximization(
    observations: np.ndarray,
    weights: torch.Tensor,
    components: GaussianEmission,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
) -> MixtureFit:
    """Fit a mixture by EM, with no autodiff involved.

    The independent oracle, on the footing ``baum_welch`` occupies for the
    HMM: no optimizer, parameterization or constraint map shared with ``fit``,
    only the model. **And its component M step is the HMM's**, since that step
    is the emission family's. The alternation around the two steps is
    :func:`snakes_and_ladders.opt.em.em_loop`'s (issue #859); this entry
    point's defaults, result type and oracle pin are its own.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)``.
    weights : torch.Tensor
        Starting mixing weights.
    components : GaussianEmission
        Starting components.
    max_iterations : int
        Maximum EM iterations.
    tolerance : float
        Stop when the log-likelihood improves by less than this *relative* to
        its magnitude — absolute would not transfer across data sizes
        (``DEV.md``, issue #111).

    Returns
    -------
    MixtureFit
        The fitted parameters, the final log-likelihood, and whether an M step
        reached a boundary.

    Raises
    ------
    ValueError
        If a component's re-estimated variance reaches its floor. The mixture
        likelihood is unbounded in that direction exactly as a Gaussian HMM's
        is, so this is an approach to a degenerate optimum rather than a
        convergence. Or if a component's M step did not converge: the loop
        reads the report its sibling
        :func:`snakes_and_ladders.opt.emission_mixture.expectation_maximization`
        reads, on the terms ``likelihood/CLAUDE.md`` states (issue #856).
    """
    values = torch.as_tensor(observations, dtype=torch.float64).reshape(-1)
    boundary = False
    attempt = 0

    def step(
        state: tuple[torch.Tensor, GaussianEmission],
    ) -> tuple[tuple[torch.Tensor, GaussianEmission], float]:
        """One E step, one M step, and the log-likelihood at the state given."""
        nonlocal boundary, attempt
        attempt += 1
        current, family = state
        log_weight = torch.log(current)
        evidence, posterior = e_step(values, log_weight, family)
        log_likelihood = float(evidence)
        reestimated = family.reestimate(
            values.reshape(1, -1), posterior.reshape(1, *posterior.shape)
        )
        if not reestimated.converged:
            msg = (
                f"a component's M step did not settle at EM iteration "
                f"{attempt}: residual {reestimated.residual:.3e} after "
                f"{reestimated.iterations} inner iterations"
            )
            raise ValueError(msg)
        boundary = boundary or reestimated.at_boundary
        return (posterior.mean(dim=0), reestimated.emissions), log_likelihood

    (weights, components), log_likelihood, termination = em_loop(
        step,
        (weights, components),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return MixtureFit(
        weights,
        components,
        log_likelihood,
        termination.iterations,
        boundary,
        termination,
    )


def clustering_cost(observations: np.ndarray, centres: np.ndarray) -> float:
    """The k-means objective: summed squared distance to the nearest centre.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    centres : np.ndarray
        Centres, of one row per centre in the observations' own shape.

    Returns
    -------
    float
        ``sum_i min_k ||y_i - c_k|| ** 2``, the squared distance summed over
        the channels.
    """
    values = np.atleast_2d(np.asarray(observations, dtype=np.float64).T).T
    located = np.atleast_2d(np.asarray(centres, dtype=np.float64).T).T
    squared = ((values[:, None, :] - located[None, :, :]) ** 2).sum(axis=-1)
    return float(squared.min(axis=1).sum())


def optimal_clustering_cost(observations: np.ndarray, n_centres: int) -> float:
    """The exact optimal k-means cost, by dynamic programming over runs.

    **Exact rather than approximate, and only in one dimension.** An optimal
    1-D k-means clustering partitions the *sorted* observations into
    contiguous runs, so the search is over ``k - 1`` cut positions and a
    dynamic program solves it in ``O(n**2 k)``. In two or more dimensions no
    such argument holds and the problem is NP-hard.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)``.
    n_centres : int
        Clusters, at least 1 and at most the number of observations.

    Returns
    -------
    float
        The minimum achievable ``clustering_cost``.

    Raises
    ------
    ValueError
        If ``n_centres`` is outside ``[1, n_samples]``.
    """
    values = np.sort(np.asarray(observations, dtype=np.float64).reshape(-1))
    n_samples = values.shape[0]
    if not 1 <= n_centres <= n_samples:
        msg = f"n_centres must lie in [1, {n_samples}], got {n_centres}"
        raise ValueError(msg)

    prefix = np.concatenate([[0.0], np.cumsum(values)])
    squares = np.concatenate([[0.0], np.cumsum(values**2)])

    def run_cost(start: int, stop: int) -> float:
        """Within-run sum of squares about the run's own mean, ``[start, stop)``."""
        count = stop - start
        total = prefix[stop] - prefix[start]
        return float(squares[stop] - squares[start] - total * total / count)

    best = np.full((n_centres + 1, n_samples + 1), np.inf)
    best[0, 0] = 0.0
    for centre in range(1, n_centres + 1):
        for stop in range(centre, n_samples + 1):
            best[centre, stop] = min(
                best[centre - 1, start] + run_cost(start, stop)
                for start in range(centre - 1, stop)
            )
    return float(best[n_centres, n_samples])


def kmeans_plus_plus(
    observations: np.ndarray, n_centres: int, rng: np.random.Generator
) -> np.ndarray:
    """Seed ``n_centres`` centres by D-squared sampling (Arthur & Vassilvitskii, 2007; ``eq:kmeanspp``).

    The first centre is drawn uniformly from the observations; each
    subsequent one with probability proportional to its squared distance from
    the nearest centre already chosen. The expected cost is within
    ``8 (ln k + 2)`` of optimal *before any refinement* --- the published
    bound a test here checks against.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    n_centres : int
        Centres to seed, at least 1 and at most the number of observations.
    rng : np.random.Generator
        Generator, passed in rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    np.ndarray
        The seeded centres, one per row in the observations' own shape, in the
        order chosen.

    Raises
    ------
    ValueError
        If ``n_centres`` is outside ``[1, n_samples]``.
    """
    values = np.asarray(observations, dtype=np.float64)
    if not 1 <= n_centres <= values.shape[0]:
        msg = f"n_centres must lie in [1, {values.shape[0]}], got {n_centres}"
        raise ValueError(msg)

    # `rng.choice` draws a row from a two-dimensional array and a value from a
    # one-dimensional one off the same integer draw, so the multi-channel form
    # is the scalar form's stream unchanged.
    chosen = [rng.choice(values)]
    nearest = _squared_distance(values, chosen[0])
    for _ in range(1, n_centres):
        total = float(nearest.sum())
        if total <= 0.0:
            # Every remaining point coincides with a centre, so no point can
            # reduce the cost and the distribution is undefined. Falling back
            # to uniform keeps the seeding total rather than raising on a
            # degenerate but legitimate dataset.
            chosen.append(rng.choice(values))
        else:
            chosen.append(rng.choice(values, p=nearest / total))
        nearest = np.minimum(nearest, _squared_distance(values, chosen[-1]))
    return np.array(chosen)


def _squared_distance(values: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """``||y_i - c|| ** 2`` per observation, summed over the channels if there are any."""
    difference = values - centre
    if difference.ndim == 1:
        return np.asarray(difference**2)
    return np.asarray((difference**2).sum(axis=-1))


def uniform_seeds(
    observations: np.ndarray, n_centres: int, rng: np.random.Generator
) -> np.ndarray:
    """Seed by drawing distinct observations uniformly.

    The baseline k-means++ is measured against, kept beside it rather than
    inside the test that wins the comparison.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    n_centres : int
        Centres to seed.
    rng : np.random.Generator
        Generator, passed in.

    Returns
    -------
    np.ndarray
        The seeded centres, one per row in the observations' own shape.
    """
    values = np.asarray(observations, dtype=np.float64)
    return np.asarray(rng.choice(values, size=n_centres, replace=False))


#: The published seeding guarantee: the *expected* cost of k-means++ is within
#: this factor of optimal, before any refinement (Arthur & Vassilvitskii,
#: 2007, theorem 1.1). An expectation, so a test checks a mean over replicates
#: rather than a single draw.
def seeding_guarantee(n_centres: int) -> float:
    """``8 (ln k + 2)``, the factor of optimal k-means++ is expected within (``eq:kmeanspp-bound``).

    Parameters
    ----------
    n_centres : int
        Clusters, at least 1.

    Returns
    -------
    float
        The factor.

    Raises
    ------
    ValueError
        If ``n_centres`` is below 1.
    """
    if n_centres < 1:
        msg = f"n_centres must be >= 1, got {n_centres}"
        raise ValueError(msg)
    return 8.0 * (math.log(n_centres) + 2.0)


class KMeansPlusPlus(Initializer):
    """Starting points seeded from the data by k-means++.

    The initializer #251 could not write, because nothing then read its
    observations. It satisfies
    :class:`snakes_and_ladders.opt.initialize.Initializer` unchanged and
    **refuses** an objective it cannot seed: guessing what an unknown
    parameter vector means is how an initializer silently returns nonsense.

    Parameters
    ----------
    n_starts : int
        Seedings to draw, at least 1. ``search/CLAUDE.md``'s budget rule
        applies: ``n_starts`` starts cost ``n_starts`` fits.
    rng : np.random.Generator
        Generator, passed in rather than seeded here, so a fit's dependence on
        randomness is declared by its caller (``sim/CLAUDE.md``, issue #240).

    Raises
    ------
    ValueError
        If fewer than one start is asked for.
    """

    def __init__(self, n_starts: int, rng: np.random.Generator) -> None:
        if n_starts < 1:
            msg = f"n_starts must be >= 1, got {n_starts}"
            raise ValueError(msg)
        self.n_starts = n_starts
        self.rng = rng

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """Seed each start by k-means++ on the objective's own observations.

        Returns
        -------
        list[torch.Tensor]
            ``n_starts`` points, in unconstrained coordinates.

        Raises
        ------
        TypeError
            If the objective is not one this initializer knows how to seed.
        """
        if not isinstance(objective, GaussianMixtureObjective | ClusteringObjective):
            msg = (
                f"k-means++ seeds a Gaussian mixture or a clustering and does not "
                f"know what {type(objective).__name__}'s parameters mean"
            )
            raise TypeError(msg)
        observations = objective.observations.numpy()
        return [
            objective.theta_from_centres(
                torch.as_tensor(
                    canonical_order(
                        kmeans_plus_plus(observations, objective.n_components, self.rng)
                    )
                )
            )
            for _ in range(self.n_starts)
        ]


class UniformSeeds(Initializer):
    """Starting points on distinct observations drawn uniformly: :func:`uniform_seeds` as an initializer.

    The baseline :class:`KMeansPlusPlus` is measured against, drawing from its
    generator the stream :func:`uniform_seeds` draws, so a benchmark of the
    two through `opt.starts` reproduces the seedings drawn by hand
    (issue #894).

    Parameters
    ----------
    n_starts : int
        Seedings to draw, at least 1.
    rng : np.random.Generator
        Generator, passed in rather than seeded here.

    Raises
    ------
    ValueError
        If fewer than one start is asked for.
    """

    def __init__(self, n_starts: int, rng: np.random.Generator) -> None:
        if n_starts < 1:
            msg = f"n_starts must be >= 1, got {n_starts}"
            raise ValueError(msg)
        self.n_starts = n_starts
        self.rng = rng

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """Seed each start on uniformly drawn observations.

        Returns
        -------
        list[torch.Tensor]
            ``n_starts`` points, in unconstrained coordinates.

        Raises
        ------
        TypeError
            If the objective is not one this initializer knows how to seed.
        """
        if not isinstance(objective, GaussianMixtureObjective | ClusteringObjective):
            msg = (
                f"uniform seeding places centres of a Gaussian mixture or a "
                f"clustering and does not know what {type(objective).__name__}'s "
                f"parameters mean"
            )
            raise TypeError(msg)
        observations = objective.observations.numpy()
        return [
            objective.theta_from_centres(
                torch.as_tensor(
                    canonical_order(
                        uniform_seeds(observations, objective.n_components, self.rng)
                    )
                )
            )
            for _ in range(self.n_starts)
        ]


class ClusteringObjective(Objective):
    """The k-means cost of ``k`` centres: what the k-means++ guarantee is stated against.

    ``theta`` is the centres themselves, unconstrained, one row per centre
    flattened; the value is :func:`clustering_cost`'s
    ``sum_i min_k ||y_i - c_k|| ** 2``, differentiable wherever the nearest
    centre of every observation is unique. The objective a seeding is scored
    on when the claim is the seeding's and not the mixture's
    (`qa.mixture_seeding`'s panel (b), issue #894).

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    n_components : int
        Centres, at least 1.

    Raises
    ------
    ValueError
        If fewer than one centre is asked for.
    """

    def __init__(self, observations: np.ndarray, n_components: int) -> None:
        if n_components < 1:
            msg = f"n_components must be >= 1, got {n_components}"
            raise ValueError(msg)
        self._observations = torch.as_tensor(observations, dtype=torch.float64)
        self._rows = self._observations.reshape(self._observations.shape[0], -1)
        self._n_components = n_components

    @property
    def n_components(self) -> int:
        """Centres."""
        return self._n_components

    @property
    def observations(self) -> torch.Tensor:
        """The observations, of the shape they were given in."""
        return self._observations

    def _centres(self, theta: torch.Tensor) -> torch.Tensor:
        return theta.reshape(self._n_components, self._rows.shape[1])

    def initial(self) -> torch.Tensor:
        """Centres at evenly spaced quantiles of each channel."""
        return self.theta_from_centres(
            quantile_locations(self._rows, self._n_components, dim=0)
        )

    def theta_from_centres(self, centres: torch.Tensor) -> torch.Tensor:
        """The point whose centres are ``centres``.

        Raises
        ------
        ValueError
            If there is not one centre per component.
        """
        located = torch.as_tensor(centres, dtype=torch.float64).reshape(-1)
        if located.shape[0] != self._n_components * self._rows.shape[1]:
            msg = (
                f"expected {self._n_components} centres of {self._rows.shape[1]} "
                f"channel(s), got {located.shape[0]} values"
            )
            raise ValueError(msg)
        return located

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The centres, one row each."""
        return {"centres": self._centres(theta)}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``."""
        return self.theta_from_centres(named["centres"])

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The summed squared distance of every observation to its nearest centre."""
        squared = (
            (self._rows[:, None, :] - self._centres(theta)[None, :, :]) ** 2
        ).sum(dim=-1)
        return squared.min(dim=1).values.sum()


def canonical_order(centres: np.ndarray) -> np.ndarray:
    """Centres sorted, so a relabelling of the same seeding is one seeding.

    A mixture is invariant under permuting its components, so two seedings
    differing only in the order they were drawn are the same start. Sorting on
    the first channel, ties broken by the next, is the order a multi-channel
    seeding is reported in; with one channel it is :func:`numpy.sort`.

    Parameters
    ----------
    centres : np.ndarray
        Shape ``(n_centres,)`` or ``(n_centres, n_channels)``.

    Returns
    -------
    np.ndarray
        The same rows, in the canonical order.
    """
    rows = np.asarray(centres, dtype=np.float64)
    if rows.ndim == 1:
        return np.sort(rows)
    return np.asarray(rows[np.lexsort(rows.T[::-1])])


def emission_mixture_plus_plus(
    observations: np.ndarray,
    n_centres: int,
    divergence: Callable[[float, np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """``Emission_Mixture++``: k-means++ with a family's Bregman divergence as the distance (issue #306).

    :func:`kmeans_plus_plus` draws each centre proportionally to squared
    Euclidean distance, which assumes a unit-normal emission. Here the
    distance is the family's: ``divergence(seed, observations)`` scores every
    observation against a family seeded at ``seed``, and the next seed is drawn
    proportionally to the smallest such score so far. With ``0.5 * ((x - seed)
    / scale) ** 2`` it is k-means++ exactly, the reduction the test pins ---
    and that expression *is* an isotropic Gaussian's divergence, so the two
    rules coincide there rather than resembling one another (issue #560).

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n_samples,)``; the seeds are drawn from these values.
    n_centres : int
        Seeds to draw, in ``[1, n_samples]``.
    divergence : Callable
        ``(seed, observations) -> scores``, non-negative, shape ``(n_samples,)``.
    rng : np.random.Generator
        Generator, passed in rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    np.ndarray
        The seeds, shape ``(n_centres,)``, in the order chosen.
    """
    values = np.asarray(observations).reshape(-1)
    if not 1 <= n_centres <= values.shape[0]:
        msg = f"n_centres must lie in [1, {values.shape[0]}], got {n_centres}"
        raise ValueError(msg)

    # The same draws in the same order as kmeans_plus_plus, so that with a
    # squared-distance score the two are one algorithm, draw for draw.
    chosen = [rng.choice(values)]
    nearest = np.asarray(divergence(float(chosen[0]), values), dtype=float)
    for _ in range(1, n_centres):
        total = float(nearest.sum())
        if total <= 0.0:
            chosen.append(rng.choice(values))
        else:
            chosen.append(rng.choice(values, p=nearest / total))
        nearest = np.minimum(
            nearest,
            np.asarray(divergence(float(chosen[-1]), values), dtype=float),
        )
    return np.array(chosen)
