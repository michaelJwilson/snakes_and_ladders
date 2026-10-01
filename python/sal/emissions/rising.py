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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

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


def on_distinct(
    values: torch.Tensor, table: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """``table(values)``, evaluated on the distinct values only and gathered back.

    ``values`` ends in a singleton axis, ``(..., 1)``, and ``table`` maps a
    ``(D, 1)`` column of them to ``(D, K)``: a count takes a few hundred
    values across thousands of observations, so the series is formed once
    per (distinct count, state) rather than once per (observation, state),
    the rule :func:`~sal.emissions.counts.lgamma_shifted` follows (issue
    #924). Elementwise, so the gathered entries are the direct ones bit for
    bit. Too few values, or no singleton axis, and ``table`` takes them
    directly.
    """
    if values.dim() == 0 or values.shape[-1] != 1 or values.numel() < 64:
        return table(values)
    distinct, inverse = distinct_values(values)
    rows = table(distinct.reshape(-1, 1))
    return rows[inverse.reshape(-1)].reshape(*values.shape[:-1], rows.shape[-1])


def by_state(
    large: torch.Tensor,
    above: Callable[[torch.Tensor], torch.Tensor],
    below: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Each state from the path its shape takes, and each state computed once.

    ``large`` marks the states at or above :data:`LARGE_SHAPE`; ``above`` and
    ``below`` take the indices of their states and return their columns,
    ``(..., len(index))``. Neither path is evaluated on the other's states,
    and a state below the threshold is the plain arithmetic bit for bit.
    """
    upper = torch.nonzero(large).reshape(-1)
    lower = torch.nonzero(~large).reshape(-1)
    if lower.numel() == 0:
        return above(upper)
    if upper.numel() == 0:
        return below(lower)
    columns = torch.cat([above(upper), below(lower)], dim=-1)
    order = torch.argsort(torch.cat([upper, lower]))
    return columns.index_select(-1, order)


#: Whether a gradient taken here may be summed over distinct counts.
_GRADIENT_ON_DISTINCT: ContextVar[bool] = ContextVar(
    "gradient_on_distinct", default=False
)


@contextmanager
def gradients_on_distinct() -> Iterator[None]:
    """Let the count densities tabulate on distinct counts while autograd tracks them (issue #1136).

    Off by default: the gather's backward sums each parameter's gradient over
    the distinct counts rather than the observations, which is the same
    gradient to rounding and not bit for bit, and an observed-information
    Hessian pinned bitwise would move (#924). A sampler that only follows the
    gradient --- the Hamiltonian, annealing and tempering mixture starts ---
    takes it inside this block. Per thread and per task, as a
    :class:`contextvars.ContextVar` is.
    """
    token = _GRADIENT_ON_DISTINCT.set(True)
    try:
        yield
    finally:
        _GRADIENT_ON_DISTINCT.reset(token)


def tracked(*parameters: torch.Tensor) -> bool:
    """Whether autograd tracks a parameter and the gradient must be summed per observation."""
    return (
        not _GRADIENT_ON_DISTINCT.get()
        and torch.is_grad_enabled()
        and any(p.requires_grad for p in parameters)
    )


#: ``torch.unique(values, return_inverse=True)`` for values seen before in
#: this context, or ``None`` outside :func:`reusing_distinct`.
type DistinctCache = dict[
    tuple[tuple[int, ...], torch.dtype, float],
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]

_DISTINCT_CACHE: ContextVar[DistinctCache | None] = ContextVar(
    "distinct_cache", default=None
)


@contextmanager
def reusing_distinct(cache: DistinctCache) -> Iterator[None]:
    """Keep the distinct values of the counts scored inside, in ``cache``, across calls (issue #1136).

    A sampler scores the same observations at every step, and each step's
    ``torch.unique`` of them is the last step's: 4.1 s of a 23.9 s chain at
    the stress mixture. The caller owns ``cache`` --- an objective holds its
    own --- and installs it only for its own calls, so no state is shared
    between threads or tasks. A hit is verified by ``torch.equal`` against
    the stored values, so it is the recomputation exactly.
    """
    token = _DISTINCT_CACHE.set(cache)
    try:
        yield
    finally:
        _DISTINCT_CACHE.reset(token)


def distinct_values(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.unique(values, return_inverse=True)``, from the installed cache where it holds them."""
    cache = _DISTINCT_CACHE.get()
    if cache is None:
        found: tuple[torch.Tensor, torch.Tensor] = torch.unique(
            values, return_inverse=True
        )
        return found
    key = (tuple(values.shape), values.dtype, float(values.sum()))
    for held, distinct, inverse in cache.get(key, []):
        if torch.equal(held, values):
            return distinct, inverse
    distinct, inverse = torch.unique(values, return_inverse=True)
    cache.setdefault(key, []).append((values.detach().clone(), distinct, inverse))
    return distinct, inverse

