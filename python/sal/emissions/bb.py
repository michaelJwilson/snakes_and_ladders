"""The beta-binomial's trial count, factored out of its density (issue #1064).

Under a per-observation trial count ``n`` the density of
:class:`~sal.emissions.counts.BetaBinomialEmission` is nine ``lgamma`` terms,
each a function of one integer or of none:

    ``log p(z | k, n) = [lgamma(n + 1) - lgamma(z + 1) - lgamma(n - z + 1)]``
    ``+ U_k(z) + V_k(n - z) - W_k(n) + lgamma(a_k + b_k) - lgamma(a_k) - lgamma(b_k)``,

with ``U_k(j) = lgamma(j + a_k)``, ``V_k(j) = lgamma(j + b_k)`` and
``W_k(j) = lgamma(j + a_k + b_k)``. The bracket names no state and is
:func:`log_factorial` at three integers. So a caller scoring many trial counts
tabulates each term once by its own integer, and no table is indexed by a
``(successes, trial count)`` pair. Summed in the order above --- the order of
:meth:`~sal.emissions.counts.BetaBinomialEmission.log_density` --- the terms
reproduce it bit for bit, since each is the same ``lgamma`` of the same
``float64`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sal.emissions.counts import BetaBinomialEmission, lgamma_shifted


@dataclass(frozen=True)
class TrialTables:
    """The per-state terms of the factored beta-binomial, each by its own integer.

    Parameters
    ----------
    success : torch.Tensor
        ``U[z, k] = lgamma(z + a_k)``, shape ``(successes_extent, K)``.
    failure : torch.Tensor
        ``V[j, k] = lgamma(j + b_k)``, shape ``(trials_extent, K)``.
    trial : torch.Tensor
        ``W[n, k] = lgamma(n + a_k + b_k)``, shape ``(trials_extent, K)``.
    log_beta : torch.Tensor
        ``lgamma(a + b)``, ``lgamma(a)`` and ``lgamma(b)``, shape ``(3, K)``.
    """

    success: torch.Tensor
    failure: torch.Tensor
    trial: torch.Tensor
    log_beta: torch.Tensor


def trial_tables(
    family: BetaBinomialEmission, successes_extent: int, trials_extent: int
) -> TrialTables:
    """``U``, ``V``, ``W`` and the Beta function's three terms, through the family's arithmetic.

    Each is the ``lgamma`` of the argument ``log_density`` forms --- ``z + a``,
    ``(n - z) + b``, ``n + (a + b)`` --- with the integer as a ``float64``, so
    each entry is the family's own term to the bit.

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
    successes = torch.arange(successes_extent, dtype=torch.float64).unsqueeze(-1)
    trials = torch.arange(trials_extent, dtype=torch.float64).unsqueeze(-1)
    alpha, beta = family.alpha, family.beta
    total = alpha + beta
    return TrialTables(
        success=lgamma_shifted(successes, alpha),
        failure=lgamma_shifted(trials, beta),
        trial=lgamma_shifted(trials, total),
        log_beta=torch.stack(
            [torch.lgamma(total), torch.lgamma(alpha), torch.lgamma(beta)]
        ),
    )


def log_factorial(extent: int) -> torch.Tensor:
    """``lgamma(j + 1)`` for ``j < extent``: the state-free terms, each by its own integer.

    Returns
    -------
    torch.Tensor
        Shape ``(extent,)``.
    """
    return torch.lgamma(torch.arange(extent, dtype=torch.float64) + 1.0)
