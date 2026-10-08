"""The beta-binomial from scaled rising factorials, one construction on every route (issues #1064, #1332).

Under a per-observation trial count ``n`` the beta-binomial's log pmf is

    ``log p(z | a, b, n) = ((((log C(n, z) + z log p) + (n - z) log q)``
    ``+ S(a, z)) + S(b, n - z)) - S(a + b, n)``,

summed in that order, with ``p = a / (a + b)``, ``q = b / (a + b)`` and
``S(x, m) = lgamma(x + m) - lgamma(x) - m log x`` the scaled log rising
factorial, :func:`~sal.emissions.rising.scaled_rising_array`. The rising
factorials' ``m log x`` parts are collected into the binomial's two
logarithms of rates, each a ``log`` of one rounded quotient, so no term of
size ``n log tau`` is formed and cancelled: ``S`` goes to ``0`` as
``tau = a + b`` grows and the pmf to :func:`binomial_log_pmf`. Each term is a
function of one integer and the state, so a caller scoring many trial counts
tabulates each once (:func:`trial_tables`), and the tables summed in that
order are :func:`beta_binomial_log_pmf` bit for bit. The torch
:meth:`~sal.emissions.counts.BetaBinomialEmission.log_density` keeps its own
arithmetic and agrees within 1e-14 ``max(|f|, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from sal.emissions.counts import BetaBinomialEmission
from sal.emissions.rising import scaled_rising_array


@dataclass(frozen=True)
class TrialTables:
    """The per-state terms of the factored beta-binomial, each by its own integer.

    Parameters
    ----------
    success : np.ndarray
        ``U[z, k] = S(a_k, z)``, shape ``(successes_extent, K)``.
    failure : np.ndarray
        ``V[j, k] = S(b_k, j)``, shape ``(trials_extent, K)``.
    trial : np.ndarray
        ``W[n, k] = S(a_k + b_k, n)``, shape ``(trials_extent, K)``.
    log_rate : np.ndarray
        ``log p_k`` and ``log q_k``, shape ``(2, K)``.
    """

    success: NDArray[np.float64]
    failure: NDArray[np.float64]
    trial: NDArray[np.float64]
    log_rate: NDArray[np.float64]


def trial_tables(
    family: BetaBinomialEmission, successes_extent: int, trials_extent: int
) -> TrialTables:
    """``U``, ``V`` and ``W``, each a :func:`~sal.emissions.rising.scaled_rising_array` table, and the log rates.

    Built once per M step; summed in the module's order they are
    :func:`beta_binomial_log_pmf` bit for bit.

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
        success=scaled_rising_array(alpha, successes),
        failure=scaled_rising_array(beta, trials),
        trial=scaled_rising_array(alpha + beta, trials),
        log_rate=np.stack(_log_rates(alpha, beta)),
    )


def _log_rates(
    alpha: NDArray[np.float64], beta: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``log(a / (a + b))`` and ``log(b / (a + b))``, each the ``log`` of one rounded quotient."""
    total = alpha + beta
    return np.log(alpha / total), np.log(beta / total)


def density_table(family: BetaBinomialEmission, extent: int) -> NDArray[np.float64]:
    """:func:`beta_binomial_log_pmf` at every count below ``extent`` and the family's own trials, ``(extent, K)``.

    The table a caller without a per-observation trial count reads (issue
    #1332); successes past a state's trials score ``-inf``.
    """
    return beta_binomial_log_pmf(
        np.arange(extent, dtype=np.float64)[:, None],
        family.trials.detach().numpy(),
        family.alpha.detach().numpy(),
        family.beta.detach().numpy(),
    )


def beta_binomial_log_pmf(
    successes: ArrayLike, trials: ArrayLike, alpha: ArrayLike, beta: ArrayLike
) -> NDArray[np.float64]:
    """The beta-binomial's log pmf, summed in the module's order; broadcasts.

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
    log_p, log_q = _log_rates(a_, b_)
    binomial = (_log_choose(nn, kk) + kk * log_p) + (nn - kk) * log_q
    out = (
        (binomial + scaled_rising_array(a_, kk)) + scaled_rising_array(b_, nn - kk)
    ) - scaled_rising_array(a_ + b_, nn)
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
