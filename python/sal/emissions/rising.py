"""Rising factorials at large shape, without the cancellation (issue #1136).

Both count families form ``lgamma(x + m) - lgamma(x)``: a difference of two
terms of size ``x log x`` whose value is of size ``m log x``. In ``float64``
the difference keeps an absolute error of about ``eps x log x``.
Measured against ``mpmath`` at 50 digits, over ``m`` in ``[1, 1000]``:

=========  =====================
``x``      worst error (nats)
=========  =====================
1e2        5.7e-14
1e3        9.1e-13
1e4        5.0e-12
1e6        7.2e-10
=========  =====================

and at ``x = 1e16`` the negative binomial scores nonsense (+12.6 nats at a
count of 3) and the beta-binomial's pmf sums to about ``e^132``. A shape that
large is the Poisson or binomial limit, where a fit's line search and a
dispersion's identifiability bound both go.

**The fix differences Stirling's series rather than its sum.** With
``t = m / x``,

    ``lgamma(x + m) - lgamma(x) - m log x = m h(t) + (m - 1/2) log1p(t) + S(x + m) - S(x)``,

``h(t) = (log1p(t) - t) / t`` and ``S`` the series' tail; every term is of
the size of the answer, and the limit ``x -> inf`` is ``0`` exactly. Autograd
differentiates this form, so the gradient --- the digamma rise --- is as
accurate. The M steps' own digamma rises are not here: the default route
(``src/count_mstep.rs``) sums them as reciprocals, which does not cancel.

**Above :data:`LARGE_SHAPE` only.** Below it the plain difference is within
6e-14 of exact and every caller keeps its arithmetic bit for bit, the factored
tables of :mod:`sal.emissions.nb` and :mod:`sal.emissions.bb` included.
"""

from __future__ import annotations

import torch

#: The shape at and above which the differenced series replaces ``lgamma``.
#: Below it the plain difference is within 6e-14 of ``mpmath``; the series
#: with :data:`_LGAMMA_SERIES`'s eight terms is within 1e-15 relative from
#: ``x = 10`` up.
LARGE_SHAPE = 100.0

#: ``B_2k / (2k (2k - 1))``, the coefficients of ``x^(1 - 2k)`` in
#: ``lgamma(x) - ((x - 1/2) log x - x + log(2 pi) / 2)``, ``k = 1..8``.
_LGAMMA_SERIES = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
    -3617.0 / 122400.0,
)

#: Below this ``|t|``, ``h(t)`` is its Taylor series: the direct form
#: ``(log1p(t) - t) / t`` cancels to about ``eps / t`` relative there.
_SMALL_T = 1e-3


def broadcast(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``tensors`` expanded to their common shape: ``torch.broadcast_tensors``, typed once here."""
    expanded: tuple[torch.Tensor, ...] = torch.broadcast_tensors(*tensors)  # type: ignore[no-untyped-call]
    return expanded


def _h(t: torch.Tensor) -> torch.Tensor:
    """``(log1p(t) - t) / t``, with ``h(0) = 0``."""
    small = t.abs() < _SMALL_T
    safe = torch.where(small, torch.ones_like(t), t)
    direct = (torch.log1p(safe) - safe) / safe
    series = t * (-0.5 + t * (1.0 / 3.0 + t * (-0.25 + t * 0.2)))
    return torch.where(small, series, direct)


def log1p_over(u: torch.Tensor) -> torch.Tensor:
    """``log1p(u) / u``, with its limit ``1`` at ``u = 0``.

    Written ``1 + h(u)`` so that below ``|u| = 1e-3`` it, and its derivative,
    are the Taylor series: the direct quotient's derivative cancels to a
    relative ``eps / u``, 7e-9 at ``r = 1e8`` in the negative binomial's
    gradient.
    """
    return 1.0 + _h(u)


def _inverse_powers(x: torch.Tensor) -> list[torch.Tensor]:
    """``x^-(2k - 1)`` for ``k = 1..8``; zero at ``x = inf``."""
    inverse = 1.0 / x
    square = inverse * inverse
    term = inverse
    powers = []
    for _ in range(len(_LGAMMA_SERIES)):
        powers.append(term)
        term = term * square
    return powers


def _series(x: torch.Tensor) -> torch.Tensor:
    """Stirling's tail ``S(x)``, summed smallest term first."""
    total = torch.zeros_like(x)
    for coefficient, power in reversed(
        list(zip(_LGAMMA_SERIES, _inverse_powers(x), strict=True))
    ):
        total = total + coefficient * power
    return total


def log_rising_scaled(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``lgamma(x + m) - lgamma(x) - m log x`` for ``x >= 10``, by the differenced series.

    Exact in the limit: ``0`` at ``x = inf`` for every finite ``m``, and at
    ``m = 0``. Broadcasts.
    """
    x, m = broadcast(x, m)
    t = m / x
    return m * _h(t) + (m - 0.5) * torch.log1p(t) + (_series(x + m) - _series(x))


def scaled_rising(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``lgamma(x + m) - lgamma(x) - m log x``: plain below :data:`LARGE_SHAPE`, the series above.

    Each branch is evaluated on shapes it is finite at, so neither carries a
    ``nan`` into the other's gradient.
    """
    x, m = broadcast(x, m)
    large = x >= LARGE_SHAPE
    small_x = torch.where(large, torch.ones_like(x), x)
    big_x = torch.where(large, x, torch.full_like(x, LARGE_SHAPE))
    plain = torch.lgamma(small_x + m) - torch.lgamma(small_x) - m * torch.log(small_x)
    return torch.where(large, log_rising_scaled(big_x, m), plain)
