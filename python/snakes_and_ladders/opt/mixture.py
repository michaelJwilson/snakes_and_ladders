"""A Gaussian mixture: the reference instance whose start reads the data.

The third instance of ``Objective`` beside the Potts chain and the HMM, and
the one that exists to test two abstractions at once (issue #262).

**It is the emission seam without the Markov chain.** Its M step for the
components *is*
:meth:`snakes_and_ladders.emissions.GaussianEmission.reestimate`, called with
responsibilities where an HMM passes state posteriors. Nothing is
reimplemented here, which is the evidence that the seam extracted from an HMM
was not shaped by one.

**And it is the model a data-dependent initializer was waiting for.** Issue
#251 built the ``Initializer`` protocol and recorded that it had nothing to
initialize, since there was no mixture in the repository. :class:`KMeansPlusPlus`
is that initializer, and it lives here rather than in
:mod:`snakes_and_ladders.opt.initialize` for the reason #251 gave: a strategy
that reads the observations is specific to a model, and it belongs beside the
objective it seeds. It refuses an objective it does not know rather than
guessing what the parameter vector means.

**The likelihood is unbounded, exactly as the Gaussian HMM's is.** Put a
component's mean on a single observation and let its scale go to zero and the
density there diverges. The floor derived in
:func:`snakes_and_ladders.emissions.pooled_variance_floor` transfers unchanged,
and reaching it is a refusal rather than a clamp.

Ground truth and data generation live in
:mod:`snakes_and_ladders.sim.mixture`; this module holds only the fitting
objective, its independent EM oracle, and the seeding a start needs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.emissions import GaussianEmission, pooled_variance_floor
from snakes_and_ladders.opt.constrain import free_from_log_simplex, log_simplex
from snakes_and_ladders.opt.objective import Objective


class GaussianMixtureObjective:
    """Negative log-likelihood of independent observations from a mixture.

    Parameters
    ----------
    observations : np.ndarray
        Observed values, shape ``(n_samples,)``.
    n_components : int
        Components in the mixture, at least 2 --- one component is a Gaussian,
        not a mixture, and the weight vector would be a constant.
    dtype : torch.dtype
        Precision of the computation; ``float64`` by default, since a
        finite-difference derivative check is meaningless in ``float32``.

    Raises
    ------
    ValueError
        If fewer than two components are asked for.
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
        self._observations = torch.as_tensor(observations, dtype=dtype).reshape(-1)
        self._n_components = n_components
        self._dtype = dtype
        self._variance_floor = pooled_variance_floor(np.asarray(observations))

    @property
    def n_components(self) -> int:
        """Components in the mixture."""
        return self._n_components

    @property
    def observations(self) -> torch.Tensor:
        """The observations being fitted, shape ``(n_samples,)``."""
        return self._observations

    @property
    def variance_floor(self) -> float:
        """The floor the EM oracle refuses at, derived from these observations."""
        return self._variance_floor

    @property
    def n_parameters(self) -> int:
        """``(k - 1)`` free weights, then ``k`` means and ``k`` log scales."""
        return (self._n_components - 1) + 2 * self._n_components

    @property
    def _weight_slice(self) -> slice:
        return slice(0, self._n_components - 1)

    def _mean_slice(self) -> slice:
        start = self._n_components - 1
        return slice(start, start + self._n_components)

    def _log_scale_slice(self) -> slice:
        start = self._mean_slice().stop
        return slice(start, start + self._n_components)

    def components(self, theta: torch.Tensor) -> GaussianEmission:
        """The component family ``theta`` encodes, differentiable in ``theta``."""
        return GaussianEmission(
            theta[self._mean_slice()],
            torch.exp(theta[self._log_scale_slice()]),
            self._variance_floor,
        )

    def initial(self) -> torch.Tensor:
        """Uniform weights, means at quantiles of the data, pooled scales.

        The same construction the Gaussian HMM uses, and for the same two
        reasons: equal means leave the components exchangeable and the
        gradient in that block exactly zero, and a mean far from every
        observation contributes a density that underflows, so the component
        is invisible to the E step and the fit silently becomes one with
        fewer components.
        """
        theta = torch.zeros(self.n_parameters, dtype=self._dtype)
        quantiles = (
            torch.arange(self._n_components, dtype=self._dtype) + 0.5
        ) / self._n_components
        theta[self._mean_slice()] = torch.quantile(self._observations, quantiles)
        theta[self._log_scale_slice()] = torch.log(self._observations.std())
        return theta

    def theta_from_centres(self, centres: torch.Tensor) -> torch.Tensor:
        """A start whose component means are ``centres``.

        The seam an initializer that reads the data needs: it supplies
        locations, and the objective — which alone knows what its parameter
        vector means — places them.

        Parameters
        ----------
        centres : torch.Tensor
            One location per component, shape ``(n_components,)``.

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
        if located.shape[0] != self._n_components:
            msg = f"expected {self._n_components} centres, got {located.shape[0]}"
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
                torch.log(torch.as_tensor(scale, dtype=self._dtype)).reshape(-1),
            ]
        )


def mixture_log_likelihood(
    observations: torch.Tensor,
    log_weight: torch.Tensor,
    components: GaussianEmission,
) -> torch.Tensor:
    """``sum_i log sum_k w_k N(y_i; mu_k, s_k)``, in log space throughout.

    Parameters
    ----------
    observations : torch.Tensor
        Observations, shape ``(n_samples,)``.
    log_weight : torch.Tensor
        Log mixing weights, shape ``(n_components,)``.
    components : GaussianEmission
        The component densities.

    Returns
    -------
    torch.Tensor
        Scalar, differentiable with respect to every parameter. A **density**,
        so it may be positive; the mixture inherits that from its components.
    """
    return torch.logsumexp(
        log_weight + components.log_density(observations), dim=-1
    ).sum()


def responsibilities(
    observations: torch.Tensor,
    log_weight: torch.Tensor,
    components: GaussianEmission,
) -> torch.Tensor:
    """``P(component | observation)``, shape ``(n_samples, n_components)``.

    The E step. An HMM's is a forward-backward recursion; a mixture's is one
    normalization, because independent observations carry no message between
    them. What the M step then receives is the same object either way, which
    is why the same emission family serves both.
    """
    joint = log_weight + components.log_density(observations)
    return torch.exp(joint - torch.logsumexp(joint, dim=-1, keepdim=True))


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
    """

    weights: torch.Tensor
    components: GaussianEmission
    log_likelihood: float
    iterations: int


def expectation_maximization(
    observations: np.ndarray,
    weights: torch.Tensor,
    components: GaussianEmission,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
) -> MixtureFit:
    """Fit a mixture by EM, with no autodiff involved.

    The independent oracle, on the footing ``baum_welch`` occupies for the
    HMM: it shares no optimizer, no parameterization and no constraint map
    with ``fit`` — only the model. **And it shares its component M step with
    the HMM's**, since that step is the emission family's rather than this
    module's.

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
        The fitted parameters and the final log-likelihood.

    Raises
    ------
    ValueError
        If a component's re-estimated variance reaches its floor. The mixture
        likelihood is unbounded in that direction exactly as a Gaussian HMM's
        is, so this is an approach to a degenerate optimum rather than a
        convergence.
    """
    values = torch.as_tensor(observations, dtype=torch.float64).reshape(-1)
    previous = -float("inf")
    log_likelihood = previous
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        log_weight = torch.log(weights)
        log_likelihood = float(mixture_log_likelihood(values, log_weight, components))
        posterior = responsibilities(values, log_weight, components)
        weights = posterior.mean(dim=0)
        components = components.reestimate(
            values.reshape(1, -1), posterior.reshape(1, *posterior.shape)
        ).emissions
        if abs(log_likelihood - previous) <= tolerance * abs(log_likelihood):
            break
        previous = log_likelihood
    return MixtureFit(weights, components, log_likelihood, iterations)


def clustering_cost(observations: np.ndarray, centres: np.ndarray) -> float:
    """The k-means objective: summed squared distance to the nearest centre.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)``.
    centres : np.ndarray
        Centres, shape ``(n_centres,)``.

    Returns
    -------
    float
        ``sum_i min_k (y_i - c_k) ** 2``.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1, 1)
    located = np.asarray(centres, dtype=np.float64).reshape(1, -1)
    return float(((values - located) ** 2).min(axis=1).sum())


def optimal_clustering_cost(observations: np.ndarray, n_centres: int) -> float:
    """The exact optimal k-means cost, by dynamic programming over runs.

    **Exact rather than approximate, and only in one dimension.** An optimal
    1-D k-means clustering partitions the *sorted* observations into
    contiguous runs — a point cannot belong to a cluster whose centre is
    further from it than another cluster's — so the search is over ``k - 1``
    cut positions and a dynamic program solves it in ``O(n**2 k)``. In two or
    more dimensions no such argument holds and the problem is NP-hard, which
    is why the guarantee this referees is worth having at all.

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
    """Seed ``n_centres`` centres by D-squared sampling (Arthur & Vassilvitskii, 2007).

    The first centre is drawn uniformly from the observations; each subsequent
    one is drawn with probability proportional to its squared distance from
    the nearest centre already chosen. The expected cost of the result is
    within ``8 (ln k + 2)`` of optimal *before any refinement*, which is the
    published bound a test here checks against rather than against "it looks
    better".

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)``.
    n_centres : int
        Centres to seed, at least 1 and at most the number of observations.
    rng : np.random.Generator
        Generator, passed in rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    np.ndarray
        The seeded centres, shape ``(n_centres,)``, in the order chosen.

    Raises
    ------
    ValueError
        If ``n_centres`` is outside ``[1, n_samples]``.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1)
    if not 1 <= n_centres <= values.shape[0]:
        msg = f"n_centres must lie in [1, {values.shape[0]}], got {n_centres}"
        raise ValueError(msg)

    chosen = [float(rng.choice(values))]
    for _ in range(1, n_centres):
        squared = (values[:, None] - np.array(chosen)[None, :]) ** 2
        nearest = squared.min(axis=1)
        total = float(nearest.sum())
        if total <= 0.0:
            # Every remaining point coincides with a centre, so no point can
            # reduce the cost and the distribution is undefined. Falling back
            # to uniform keeps the seeding total rather than raising on a
            # degenerate but legitimate dataset.
            chosen.append(float(rng.choice(values)))
            continue
        chosen.append(float(rng.choice(values, p=nearest / total)))
    return np.array(chosen)


def uniform_seeds(
    observations: np.ndarray, n_centres: int, rng: np.random.Generator
) -> np.ndarray:
    """Seed by drawing distinct observations uniformly.

    The baseline k-means++ is measured against. Kept beside it rather than
    written inside a test, since a comparison whose control lives only in the
    test that wins it is not a comparison.

    Parameters
    ----------
    observations : np.ndarray
        Observations, shape ``(n_samples,)``.
    n_centres : int
        Centres to seed.
    rng : np.random.Generator
        Generator, passed in.

    Returns
    -------
    np.ndarray
        The seeded centres, shape ``(n_centres,)``.
    """
    values = np.asarray(observations, dtype=np.float64).reshape(-1)
    return np.asarray(rng.choice(values, size=n_centres, replace=False))


#: The published seeding guarantee: the *expected* cost of k-means++ is within
#: this factor of optimal, before any refinement (Arthur & Vassilvitskii,
#: 2007, theorem 1.1). An expectation, so a test checks a mean over replicates
#: rather than a single draw.
def seeding_guarantee(n_centres: int) -> float:
    """``8 (ln k + 2)``, the factor of optimal k-means++ is expected within.

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


class KMeansPlusPlus:
    """Starting points seeded from the data by k-means++.

    The initializer #251 could not write, because nothing in the repository
    read its observations. It satisfies
    :class:`snakes_and_ladders.opt.initialize.Initializer` on the nose --- it
    takes an ``Objective`` --- and **refuses** an objective it cannot seed,
    because a strategy that reads the data is specific to a model and
    guessing what an unknown parameter vector means is how an initializer
    silently returns nonsense. The protocol needed no change; the refusal is
    where the model-specificity lives.

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
        if not isinstance(objective, GaussianMixtureObjective):
            msg = (
                f"k-means++ seeds a Gaussian mixture and does not know what "
                f"{type(objective).__name__}'s parameters mean"
            )
            raise TypeError(msg)
        observations = objective.observations.numpy()
        return [
            objective.theta_from_centres(
                torch.as_tensor(
                    np.sort(
                        kmeans_plus_plus(observations, objective.n_components, self.rng)
                    )
                )
            )
            for _ in range(self.n_starts)
        ]
