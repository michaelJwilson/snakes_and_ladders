"""The beta-binomial from rising factorials, one construction on every route (issues #1064, #1332).

Under a per-observation trial count ``n`` the beta-binomial's log pmf is

    ``log p(z | a, b, n) = log C(n, z) + R(a, z) + R(b, n - z) - R(a + b, n)``,

``R(x, j) = lgamma(x + j) - lgamma(x)`` the log rising factorial,
:func:`~sal.emissions.rising.log_rising`, summed in that order. No term is a
difference of two ``lgamma`` of size ``tau log tau``, ``tau = a + b``, so the
pmf holds its digits as ``tau`` grows, and at ``tau = inf`` it is the
binomial. Each term is a function of one integer, so a caller scoring many
trial counts tabulates each once (:func:`trial_tables`), and the tables
summed in that order are :func:`beta_binomial_log_pmf` bit for bit. The
torch :meth:`~sal.emissions.counts.BetaBinomialEmission.log_density` keeps
its own ``lgamma`` arithmetic and agrees within 1e-14 ``max(|f|, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from sal.emissions.counts import BetaBinomialEmission
from sal.emissions.rising import log_rising


@dataclass(frozen=True)
class TrialTables:
    """The per-state terms of the factored beta-binomial, each by its own integer.

    Parameters
    ----------
    success : np.ndarray
        ``U[z, k] = R(a_k, z)``, shape ``(successes_extent, K)``.
    failure : np.ndarray
        ``V[j, k] = R(b_k, j)``, shape ``(trials_extent, K)``.
    trial : np.ndarray
        ``W[n, k] = R(a_k + b_k, n)``, shape ``(trials_extent, K)``.
    """

    success: NDArray[np.float64]
    failure: NDArray[np.float64]
    trial: NDArray[np.float64]


def trial_tables(
    family: BetaBinomialEmission, successes_extent: int, trials_extent: int
) -> TrialTables:
    """``U``, ``V`` and ``W``, each a :func:`~sal.emissions.rising.log_rising` table.

    Built once per M step; ``log C + U[z] + V[n - z] - W[n]`` in that order
    is :func:`beta_binomial_log_pmf` bit for bit.

    Parameters
    ----------
    family : BetaBinomialEmission
        The family tabulated.
    successes_extent : int
        One past the largest success count ``U`` is indexed by.
    trials_extent : int
        One past the largest trial count ``V`` and ``W`` are indexed by.

    Returns
    -------
    TrialTables
    """
    successes = np.arange(successes_extent, dtype=np.float64)[:, None]
    trials = np.arange(trials_extent, dtype=np.float64)[:, None]
    alpha = family.alpha.detach().numpy()
    beta = family.beta.detach().numpy()
    return TrialTables(
        success=log_rising(alpha, successes),
        failure=log_rising(beta, trials),
        trial=log_rising(alpha + beta, trials),
    )


def beta_binomial_log_pmf(
    successes: ArrayLike, trials: ArrayLike, alpha: ArrayLike, beta: ArrayLike
) -> NDArray[np.float64]:
    """``log C(n, k) + R(a, k) + R(b, n - k) - R(a + b, n)``, in that order; broadcasts.

    The beta-binomial on NumPy arrays (issue #1332). ``a = inf`` or ``b = inf``
    with a finite rate is not a beta-binomial; use :func:`binomial_log_pmf`,
    its ``concentration = inf`` limit. ``k`` outside ``[0, n]`` scores
    ``-inf``.

    Parameters
    ----------
    successes, trials : ArrayLike
        ``k`` and ``n``, non-negative integers.
    alpha, beta : ArrayLike
        ``a`` and ``b``, the Beta prior's shapes, positive.

    Returns
    -------
    np.ndarray
        The broadcast shape of the four inputs.
    """
    k_, n_, a_, b_ = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (successes, trials, alpha, beta))
    )
    inside = (k_ >= 0) & (k_ <= n_)
    kk, nn = np.where(inside, k_, 0.0), np.where(inside, n_, 0.0)
    out = (
        (_log_choose(nn, kk) + log_rising(a_, kk)) + log_rising(b_, nn - kk)
    ) - log_rising(a_ + b_, nn)
    return np.where(inside, out, -np.inf)


def binomial_log_pmf(
    successes: ArrayLike, trials: ArrayLike, p: ArrayLike
) -> NDArray[np.float64]:
    """``log C(n, k) + k log p + (n - k) log1p(-p)``; broadcasts.

    :func:`beta_binomial_log_pmf` at ``concentration = inf``, its rate ``p``.
    A zero count against a zero probability scores ``0``; ``k`` outside
    ``[0, n]`` scores ``-inf``.
    """
    k_, n_, p_ = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (successes, trials, p))
    )
    inside = (k_ >= 0) & (k_ <= n_)
    kk, nn = np.where(inside, k_, 0.0), np.where(inside, n_, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = np.where(kk == 0, 0.0, kk * np.log(p_))
        misses = np.where(nn == kk, 0.0, (nn - kk) * np.log1p(-p_))
    out = (_log_choose(nn, kk) + hits) + misses
    return np.where(inside, out, -np.inf)


def _log_choose(n: NDArray[np.float64], k: NDArray[np.float64]) -> NDArray[np.float64]:
    """``lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)``, :func:`log_factorial`'s order."""
    result: NDArray[np.float64] = (gammaln(n + 1.0) - gammaln(k + 1.0)) - gammaln(
        n - k + 1.0
    )
    return result


def log_factorial(extent: int) -> NDArray[np.float64]:
    """``lgamma(j + 1)`` for ``j < extent``: the state-free terms, each by its own integer.

    :func:`beta_binomial_log_pmf`'s ``log C`` takes the same ``gammaln``, so
    the tables and the pmf agree bit for bit.

    Returns
    -------
    np.ndarray
        Shape ``(extent,)``.
    """
    result: NDArray[np.float64] = gammaln(np.arange(extent, dtype=np.float64) + 1.0)
    return result
