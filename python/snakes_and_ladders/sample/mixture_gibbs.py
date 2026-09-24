"""A conjugate Gibbs sampler for a Gaussian mixture (issue #964).

The data-augmentation sampler of Diebolt & Robert (1994): given the
allocations, the weights are Dirichlet and each component's mean and variance
normal--inverse-gamma, per channel; given those, each observation's
allocation is categorical with its responsibilities. Every conditional is
drawn exactly, so the chain's only approximation is its length.

**It is the chain label switching is defined on.** The prior is exchangeable
in the components, so the posterior is invariant under relabelling and a
chain that mixes visits every labelling; :mod:`snakes_and_ladders.sample.relabel`
undoes it. The chain records what every relabelling algorithm reads: the
parameters per draw, the allocations drawn, and the responsibilities they
were drawn from.

**Validated by the distribution it converges to.** On a handful of
observations the posterior over allocations is enumerable in closed form ---
a Dirichlet--multinomial times one normal--inverse-gamma marginal likelihood
per component and channel --- and the chain's allocation frequencies are
tested against it (``sample/CLAUDE.md``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp


@dataclass(frozen=True)
class GaussianMixturePrior:
    """An exchangeable conjugate prior for a Gaussian mixture, per channel.

    ``w ~ Dirichlet(concentration)``; per component and channel,
    ``sigma^2 ~ InvGamma(shape, rate)`` and ``mu | sigma^2 ~
    Normal(location, sigma^2 / precision_scale)``.

    Parameters
    ----------
    concentration : float
        The Dirichlet's parameter, shared by every component.
    location : np.ndarray
        The mean's prior centre, shape ``(n_channels,)``.
    precision_scale : float
        ``kappa_0``: the mean's prior is worth this many observations.
    shape : float
        ``a_0`` of the inverse-gamma.
    rate : np.ndarray
        ``b_0`` of the inverse-gamma, shape ``(n_channels,)``.
    """

    concentration: float
    location: np.ndarray
    precision_scale: float
    shape: float
    rate: np.ndarray

    @classmethod
    def weakly_informative(
        cls, observations: np.ndarray, n_components: int
    ) -> GaussianMixturePrior:
        """A prior centred on the data and worth 0.01 of an observation.

        The mean is centred on the sample mean with ``kappa_0 = 0.01``; the
        variance has ``a_0 = 2``, so its prior mean ``b_0 / (a_0 - 1)`` is
        the sample variance over ``K^2``, the spread one of ``K`` evenly
        spaced components would have; the weights are uniform, a Dirichlet
        of ones.
        """
        values = _channels(observations)
        variance = values.var(axis=0)
        return cls(
            concentration=1.0,
            location=values.mean(axis=0),
            precision_scale=0.01,
            shape=2.0,
            rate=variance / n_components**2,
        )


@dataclass(frozen=True)
class MixtureChain:
    """What the chain recorded, one row per kept draw.

    Parameters
    ----------
    weights : np.ndarray
        Shape ``(m, K)``.
    means : np.ndarray
        Shape ``(m, K, C)``.
    scales : np.ndarray
        Standard deviations, shape ``(m, K, C)``.
    allocations : np.ndarray
        The allocations drawn, shape ``(m, n)``.
    probabilities : np.ndarray
        The responsibilities each allocation was drawn from, shape
        ``(m, n, K)``.
    log_posterior : np.ndarray
        ``log p(x | w, mu, sigma) + log p(w, mu, sigma)`` up to a constant,
        shape ``(m,)``: the allocations marginalized, so its maximum is the
        maximum a posteriori draw a pivot is taken from.
    """

    weights: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    allocations: np.ndarray
    probabilities: np.ndarray
    log_posterior: np.ndarray


def _channels(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        msg = f"observations are (n,) or (n, channels); got {values.shape}"
        raise ValueError(msg)
    return values


def _log_normal(
    values: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """``log N(x_i | mu_k, sigma_k)`` summed over channels, shape ``(n, K)``."""
    z = (values[:, None, :] - means[None, :, :]) / scales[None, :, :]
    return np.asarray(
        (-0.5 * z * z - np.log(scales)[None, :, :] - 0.5 * math.log(2.0 * math.pi)).sum(
            axis=2
        )
    )


def _log_prior(
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    prior: GaussianMixturePrior,
) -> float:
    """``log p(w, mu, sigma^2)`` up to a constant."""
    alpha = prior.concentration
    log_weights = float(((alpha - 1.0) * np.log(weights)).sum())
    a, b, kappa = prior.shape, prior.rate[None, :], prior.precision_scale
    inverse_gamma = (-(a + 1.0) * np.log(variances) - b / variances).sum()
    normal = (
        -0.5 * np.log(variances / kappa)
        - 0.5 * kappa * (means - prior.location[None, :]) ** 2 / variances
    ).sum()
    return log_weights + float(inverse_gamma + normal)


def gibbs_gaussian_mixture(
    observations: np.ndarray,
    n_components: int,
    prior: GaussianMixturePrior,
    rng: np.random.Generator,
    n_draws: int,
    *,
    burn_in: int = 0,
    thin: int = 1,
    start: np.ndarray | None = None,
) -> MixtureChain:
    """Draw from a Gaussian mixture's posterior by conjugate Gibbs sweeps.

    One sweep draws the weights, then every component's variance and mean
    per channel, from their conditionals given the allocations, then every
    allocation from its responsibilities at those parameters.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n,)`` or ``(n, C)``.
    n_components : int
        ``K``, at least 2.
    prior : GaussianMixturePrior
        The exchangeable prior.
    rng : np.random.Generator
        The one stream every draw comes from.
    n_draws : int
        Draws kept.
    burn_in : int
        Sweeps discarded first.
    thin : int
        Sweeps per kept draw.
    start : np.ndarray | None
        Initial allocations ``(n,)``; uniform draws when omitted.

    Returns
    -------
    MixtureChain
    """
    if n_components < 2:
        msg = f"a mixture has at least two components, got {n_components}"
        raise ValueError(msg)
    if thin < 1 or n_draws < 1 or burn_in < 0:
        msg = "thin and n_draws are at least 1 and burn_in at least 0"
        raise ValueError(msg)
    values = _channels(observations)
    n, channels = values.shape
    k = n_components
    labels = (
        rng.integers(0, k, n) if start is None else np.asarray(start, dtype=np.int64)
    )
    kept: dict[str, list[np.ndarray | float]] = {
        name: []
        for name in (
            "weights",
            "means",
            "scales",
            "allocations",
            "probabilities",
            "log_posterior",
        )
    }
    for sweep in range(burn_in + n_draws * thin):
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        weights = rng.dirichlet(prior.concentration + counts)
        sums = np.zeros((k, channels))
        np.add.at(sums, labels, values)
        centred = np.where(
            counts[:, None] > 0, sums / np.maximum(counts, 1.0)[:, None], 0.0
        )
        squares = np.zeros((k, channels))
        np.add.at(squares, labels, (values - centred[labels]) ** 2)
        kappa = prior.precision_scale + counts[:, None]
        location = (prior.precision_scale * prior.location[None, :] + sums) / kappa
        shape = prior.shape + 0.5 * counts[:, None]
        rate = (
            prior.rate[None, :]
            + 0.5 * squares
            + 0.5
            * prior.precision_scale
            * counts[:, None]
            * (centred - prior.location[None, :]) ** 2
            / kappa
        )
        variances = rate / rng.gamma(shape, 1.0, size=(k, channels))
        means = rng.normal(location, np.sqrt(variances / kappa))
        scales = np.sqrt(variances)
        joint = np.log(weights)[None, :] + _log_normal(values, means, scales)
        normalizer = logsumexp(joint, axis=1, keepdims=True)
        probabilities = np.exp(joint - normalizer)
        cumulative = np.cumsum(probabilities, axis=1)
        uniforms = np.asarray(rng.random(n), dtype=np.float64)
        labels = (cumulative[:, :-1] < uniforms[:, None]).sum(axis=1)
        if sweep >= burn_in and (sweep - burn_in) % thin == thin - 1:
            kept["weights"].append(weights)
            kept["means"].append(means)
            kept["scales"].append(scales)
            kept["allocations"].append(labels.copy())
            kept["probabilities"].append(probabilities)
            kept["log_posterior"].append(
                float(normalizer.sum()) + _log_prior(weights, means, variances, prior)
            )
    return MixtureChain(
        weights=np.array(kept["weights"]),
        means=np.array(kept["means"]),
        scales=np.array(kept["scales"]),
        allocations=np.array(kept["allocations"], dtype=np.int64),
        probabilities=np.array(kept["probabilities"]),
        log_posterior=np.array(kept["log_posterior"]),
    )


def exact_allocation_posterior(
    observations: np.ndarray, n_components: int, prior: GaussianMixturePrior
) -> tuple[np.ndarray, np.ndarray]:
    """Every labelled allocation of a handful of observations and its posterior probability.

    ``p(z | x)`` is proportional to the Dirichlet--multinomial ``p(z)``
    times, per component and channel, the normal--inverse-gamma marginal
    likelihood of the observations allocated to it,
    ``Gamma(a_n) b_0^a_0 / (Gamma(a_0) b_n^a_n) sqrt(kappa_0 / kappa_n)
    (2 pi)^(-n_k / 2)``. Enumerated over all ``K^n`` allocations, so a
    handful of observations only.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The allocations ``(K^n, n)`` in lexicographic order, and their
        probabilities ``(K^n,)``.
    """
    values = _channels(observations)
    n = values.shape[0]
    k = n_components
    if k**n > 200_000:
        msg = f"{k}^{n} allocations is past the enumeration limit"
        raise ValueError(msg)
    allocations = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int64)
    alpha = prior.concentration
    a0, b0, kappa0, m0 = prior.shape, prior.rate, prior.precision_scale, prior.location
    log_mass = np.empty(len(allocations))
    for row, labels in enumerate(allocations):
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        value = float(
            gammaln(k * alpha)
            - gammaln(k * alpha + n)
            + (gammaln(alpha + counts) - gammaln(alpha)).sum()
        )
        for component in range(k):
            members = values[labels == component]
            size = members.shape[0]
            if size == 0:
                continue
            mean = members.mean(axis=0)
            kappa = kappa0 + size
            shape = a0 + 0.5 * size
            rate = (
                b0
                + 0.5 * ((members - mean) ** 2).sum(axis=0)
                + 0.5 * kappa0 * size * (mean - m0) ** 2 / kappa
            )
            value += float(
                (
                    gammaln(shape)
                    - gammaln(a0)
                    + a0 * np.log(b0)
                    - shape * np.log(rate)
                    + 0.5 * np.log(kappa0 / kappa)
                    - 0.5 * size * math.log(2.0 * math.pi)
                ).sum()
            )
        log_mass[row] = value
    probability = np.exp(log_mass - logsumexp(log_mass))
    return allocations, probability
