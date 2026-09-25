"""The negative binomial's exposure, factored out of its density (issue #1064).

Under a per-observation exposure ``c`` the density of
:class:`~sal.emissions.counts.NegativeBinomialEmission` is

    ``log p(y | k, c) = A_k(y) + r_k log(r_k / t) + y log(mu_k c / t)``,
    ``t = r_k + mu_k c``, ``A_k(y) = lgamma(y + r_k) - lgamma(r_k) - lgamma(y + 1)``.

``A`` is a function of the count alone, so a caller scoring many exposures
tabulates it once by count and forms the exposure terms per observation, which
need logarithms and no ``lgamma``. A table by count *and* distinct exposure
would instead have as many rows as there are observations.
"""

from __future__ import annotations

import torch

from sal.emissions.counts import NegativeBinomialEmission, lgamma_shifted


def count_log_factor(
    family: NegativeBinomialEmission, observations: torch.Tensor
) -> torch.Tensor:
    """``A_k(y)``, the part of ``log_density`` that no exposure reaches.

    These are :meth:`~sal.emissions.counts.NegativeBinomialEmission.log_density`'s
    first three terms, in its order, so a caller completing them in that order
    reproduces it to the rounding of its logarithm.

    Parameters
    ----------
    family : NegativeBinomialEmission
        The family whose dispersions ``r`` enter.
    observations : torch.Tensor
        Counts, any shape.

    Returns
    -------
    torch.Tensor
        Shape ``(..., n_states)``.
    """
    counts = observations.unsqueeze(-1).to(family.mean.dtype)
    return (
        lgamma_shifted(counts, family.dispersion)
        - torch.lgamma(family.dispersion)
        - torch.lgamma(counts + 1.0)
    )


def exposure_table(family: NegativeBinomialEmission, extent: int) -> torch.Tensor:
    """``B_k(y) = A_k(y) + r_k log r_k + y log mu_k`` for every count ``y < extent``.

    The exposure-free terms of ``B_k(y) + y log c - (y + r_k) log t``, the form
    the Rust coupled kernel completes per observation with one logarithm per
    score (``src/coupled.rs``). Moving ``r log r + y log mu`` into the table
    saves that logarithm and costs the cancellation of ``y log mu`` against
    ``-y log t``: 262.9 ulp relative to ``log_density`` at the ci instance of
    ``spatio_sequential_counts_covariate``, against 2.3 in the family's order.

    Parameters
    ----------
    family : NegativeBinomialEmission
        The family tabulated.
    extent : int
        One past the largest count tabulated.

    Returns
    -------
    torch.Tensor
        Shape ``(extent, n_states)``, row ``y`` the count ``y``.
    """
    counts = torch.arange(extent, dtype=torch.float64)
    return (
        count_log_factor(family, counts)
        + family.dispersion * torch.log(family.dispersion)
        + counts.unsqueeze(-1) * torch.log(family.mean)
    )
