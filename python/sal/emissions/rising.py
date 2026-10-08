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

**On NumPy arrays (issue #1300),** for a caller that takes no derivative:
:func:`log_rising` and its derivative in ``x``, :func:`digamma_rising`.
:func:`log_rising` takes the plain ``lgamma`` difference wherever its own
error bound meets the promise (issue #1329). Otherwise neither subtracts
two ``lgamma``, two ``digamma`` or two of Stirling's tails: each power's
difference is a positive sum times ``1 / (x + m) - 1 / x``, and below
:data:`_SERIES_FROM` the recurrence shifts ``x`` up to the series. Measured
against ``mpmath`` at 50 digits, ``x`` from 1e-3 to 1e16 and ``m`` from 0 to
1e6, the error over ``max(|f|, 1)`` is 6.4e-15 for :func:`log_rising`
against 5.6e-14 for :func:`scaled_rising`'s plain route below
:data:`LARGE_SHAPE` (1.1e-8 relative at ``x = 99.9``, ``m = 1e-6``), and
:func:`digamma_rising` is within 4.0e-16 relative. The torch route keeps
its own arithmetic, so no caller moves; at and above :data:`LARGE_SHAPE`
the two agree to 5.9e-16 on the same scale.
"""

from __future__ import annotations

import ctypes
import functools
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import scipy.special.cython_special
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

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


#: The shape from which :func:`log_rising` and :func:`digamma_rising` take the
#: differenced series; below it the recurrence shifts ``x`` up to it, by at most
#: ten steps. :data:`_LGAMMA_SERIES` is within 1e-15 relative from here up.
_SERIES_FROM = 10.0

#: ``B_2k / (2k)``, the coefficients of ``y^(-2k)`` in
#: ``log y - 1 / (2 y) - digamma(y)``, ``k = 1..8``: :data:`_LGAMMA_SERIES`
#: differentiated.
_DIGAMMA_SERIES = (
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
    1.0 / 12.0,
    -3617.0 / 8160.0,
)


def _h_array(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """:func:`_h` on an array: ``(log1p(t) - t) / t``, with ``h(0) = 0``."""
    small = np.abs(t) < _SMALL_T
    safe = np.where(small, 1.0, t)
    direct = (np.log1p(safe) - safe) / safe
    series = t * (-0.5 + t * (1.0 / 3.0 + t * (-0.25 + t * 0.2)))
    return np.where(small, series, direct)


#: The relative size below which a series term is dropped: a thousandth of
#: :func:`digamma_rising`'s 1e-15 promise.
_TERM_FLOOR = 1e-18


def _terms(inverse: NDArray[np.float64]) -> int:
    """How many of the eight series terms the largest ``1 / x`` in ``inverse`` needs.

    Term ``k`` of either difference is at most ``|c_k| 2k m x^(-2k)`` over an
    answer of at least ``m / x``, so it is dropped once
    ``|c_k| 2k x^(1 - 2k)`` is below :data:`_TERM_FLOOR` at the smallest
    ``x``, with every later term smaller still from ``x = 10`` up: four terms
    from ``x = 100`` (issue #1329).
    """
    if inverse.size == 0:
        return 1
    largest = float(inverse.max())
    for k, coefficient in enumerate(_DIGAMMA_SERIES, start=1):
        if abs(coefficient) * 2.0 * k * largest ** (2 * k - 1) < _TERM_FLOOR:
            return k
    return len(_DIGAMMA_SERIES)


def _log_tail(
    inverse: NDArray[np.float64], gap: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``S(x + m) - S(x) = sum_k c_k (u^(2k - 1) - v^(2k - 1))``, without subtracting.

    ``inverse`` is ``v = 1 / x``, ``gap`` is ``u - v = -m / (x (x + m))``
    and ``u = 1 / (x + m)``. Each difference is ``(u - v) P_j`` with
    ``P_j = sum_i u^i v^(j - 1 - i)``, a sum of positive terms, and
    ``P_(j + 2) = u^2 P_j + v^j (u + v)``: at ``x = 1.4616``, ``m = 1e-6``,
    shifted to 11.46, the subtraction would leave 1e-18 nats of an answer of
    3e-11, a relative 3.5e-8. The recurrence runs in place, with no list of
    powers (issue #1329).
    """
    upper = inverse + gap
    upper_square = upper * upper
    inverse_square = inverse * inverse
    both = upper + inverse
    power = inverse.copy()
    p = np.ones_like(inverse)
    total = _LGAMMA_SERIES[0] * p
    for coefficient in _LGAMMA_SERIES[1 : _terms(inverse)]:
        p *= upper_square
        p += power * both
        total += coefficient * p
        power *= inverse_square
    total *= gap
    return total


def _log_rising_scaled_array(
    x: NDArray[np.float64], m: NDArray[np.float64]
) -> NDArray[np.float64]:
    """:func:`log_rising_scaled` on arrays, with Stirling's tail differenced term by term."""
    t = m / x
    step = np.log1p(t)
    return m * _h_array(t) + (m - 0.5) * step + _log_tail(1.0 / x, -t / (x + m))


def _digamma_rising_series(
    x: NDArray[np.float64], m: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``digamma(x + m) - digamma(x)`` for ``x >= 10``, by the differenced series.

    With ``U = (x + m)^-2`` and ``V = x^-2``, each power's difference
    ``U^k - V^k`` is ``(U - V) Q_k``, ``Q_(k + 1) = U Q_k + V^k`` a sum of
    positive terms and ``U - V = (u - v)(u + v)`` formed from
    ``u - v = -m / (x (x + m))``, so every term is of the size of the answer
    (issue #1329).
    """
    t = m / x
    step = np.log1p(t)
    inverse = 1.0 / x
    gap = -t / (x + m)
    upper = inverse + gap
    upper_square = upper * upper
    inverse_square = inverse * inverse
    power = inverse_square.copy()
    q = np.ones_like(x)
    total = _DIGAMMA_SERIES[0] * q
    for coefficient in _DIGAMMA_SERIES[1 : _terms(inverse)]:
        q *= upper_square
        q += power
        total += coefficient * q
        power *= inverse_square
    total *= gap * (upper + inverse)
    return step - np.expm1(-step) / (2.0 * x) - total


def _shifted(
    x: ArrayLike,
    m: ArrayLike,
    series: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    term: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """``series(x, m)``, with ``x`` below :data:`_SERIES_FROM` first shifted up to it.

    ``f(x, m) = f(x + 1, m) + term(x, m)`` is the recurrence; each step's
    ``term`` is added for the entries still below, so at most ten steps
    are taken and only on those entries.
    """
    x_, m_ = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(m, dtype=np.float64)
    )
    out = np.empty(x_.shape, dtype=np.float64)
    below = x_ < _SERIES_FROM
    # From LARGE_SHAPE up the series needs four terms of eight: split there.
    large = x_ >= LARGE_SHAPE
    out[large] = series(x_[large], m_[large])
    middle = ~(below | large)
    if middle.any():
        out[middle] = series(x_[middle], m_[middle])
    if below.any():
        y, n = x_[below], m_[below]
        recurrence = np.zeros_like(y)
        for _ in range(int(_SERIES_FROM)):
            short = y < _SERIES_FROM
            if not short.any():
                break
            recurrence[short] += term(y[short], n[short])
            y = np.where(short, y + 1.0, y)
        out[below] = series(y, n) + recurrence
    return out


def _log_rising_series(
    x: NDArray[np.float64], m: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``lgamma(x + m) - lgamma(x)`` for ``x >= 10``: the scaled series plus ``m log x``."""
    return _log_rising_scaled_array(x, m) + m * np.log(x)


#: :func:`log_rising`'s promise: its error over ``max(|f|, 1)``.
_LOG_PROMISE = 1e-14

#: The error of ``scipy.special.gammaln(y)`` taken as ``4 eps max(|lgamma y|, 1)``,
#: the difference's included: against ``mpmath`` at 40 digits over 6,000 draws,
#: ``x`` log-uniform from 1e-3 to 1e4 and ``m`` from 1e-6 to 1e6, the worst
#: is 2.19 eps over the sum of the two ``max(|lgamma|, 1)``.
_PLAIN_ERROR = 4.0 * float(np.finfo(np.float64).eps)


def _log_rising_numpy(x: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """``lgamma(x + m) - lgamma(x)``, the log rising factorial, on NumPy arrays; broadcasts.

    For ``x > 0`` and ``m >= 0``; ``0`` at ``m = 0``. Where the plain
    ``gammaln(x + m) - gammaln(x)`` is within the promise by its own error
    bound, ``_PLAIN_ERROR (max(|lgamma(x + m)|, 1) + max(|lgamma x|, 1))
    <= 1e-14 max(|f|, 1)``, evaluated per entry, it is that: small ``x``, or
    ``m`` not small against ``x``. Elsewhere, from :data:`_SERIES_FROM` up,
    it is the differenced series plus ``m log x``; below, the recurrence
    ``f(x, m) = f(x + 1, m) - log1p(m / x)`` shifts ``x`` up to it
    (issue #1329). Within 1e-14 of ``mpmath`` over ``max(|f|, 1)``: near a
    zero of ``(x)_m`` --- ``x = 1``, ``m = 1`` --- the terms are of size one
    and a relative bound is not attainable.
    """
    x_, m_ = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(m, dtype=np.float64)
    )
    rise, shape = gammaln(x_ + m_), gammaln(x_)
    out: NDArray[np.float64] = rise - shape
    error = np.maximum(np.abs(rise), 1.0)
    error += np.maximum(np.abs(shape), 1.0)
    error *= _PLAIN_ERROR / _LOG_PROMISE
    rest = error > np.maximum(np.abs(out), 1.0)
    rest &= m_ != 0.0
    if rest.any():
        out[rest] = _shifted(
            x_[rest], m_[rest], _log_rising_series, lambda y, n: -np.log1p(n / y)
        )
    return out


def _digamma_rising_numpy(x: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """``digamma(x + m) - digamma(x)``, the derivative of :func:`log_rising` in ``x``; broadcasts.

    For ``x > 0`` and ``m >= 0``. Neither ``digamma`` is formed: from
    :data:`_SERIES_FROM` up the asymptotic series is differenced term by
    term, and below it the recurrence adds
    ``1 / x - 1 / (x + m) = m / (x (x + m))``, a sum of positive terms. So
    it does not cancel at any ``x``: within 1e-15 relative of ``mpmath``
    from ``x = 1e-3`` to ``1e16``, where two ``digamma`` are 2.8e-8 off at
    ``x = 99``, ``m = 1e-6``.
    """
    return _shifted(x, m, _digamma_rising_series, lambda y, n: n / (y * (y + n)))


#: ``scipy.special.gammaln``'s own Cephes routine, bound on first use by
#: :func:`_kernels`: the compiled plain route is the NumPy oracle's ``gammaln``
#: bit for bit, where ``math.lgamma`` (``libm``) differs in the last places
#: and took 2.55 ms against 1.82 ms at 25 x 2,829 pairs.
_gammaln: Callable[[float], float] = math.lgamma


def _cython_function(name: str) -> Callable[[float], float]:
    """A ``double -> double`` function of :mod:`scipy.special.cython_special`, callable from ``njit``."""
    capsule = scipy.special.cython_special.__pyx_capi__[name]
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    address = get_pointer(capsule, get_name(capsule))
    return ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(address)


def _series_terms(inverse: float, floor: float) -> int:
    """How many series terms ``1 / y = inverse`` needs: :func:`_terms` at one ``y``, in scalars."""
    terms = 1
    while (
        terms < 8
        and abs(_DIGAMMA_SERIES[terms - 1]) * 2.0 * terms * inverse ** (2 * terms - 1)
        >= floor
    ):
        terms += 1
    return terms


def _series_route(
    xi: float,
    mi: float,
    series_from: float,
    small_t: float,
    floor: float,
    scaled: bool,
    known_terms: int,
) -> float:
    """The series route of :func:`_log_rising_kernel` at one pair: the recurrence up to ``series_from``, then the differenced series.

    One body for :func:`_log_rising_kernel` and :func:`_scaled_rising_table_kernel`,
    so the two take the series in the same arithmetic (issue #1341).
    ``known_terms`` is :func:`_series_terms` at ``1 / x`` where the caller has
    it, for ``x`` at or above ``series_from``; ``0`` has it counted here.
    """
    y = xi
    recurrence = 0.0
    while y < series_from:
        recurrence += -math.log1p(mi / y)
        y += 1.0
    t = mi / y
    step = math.log1p(t)
    if abs(t) < small_t:
        h = t * (-0.5 + t * (1.0 / 3.0 + t * (-0.25 + t * 0.2)))
    else:
        h = (step - t) / t
    inverse = 1.0 / y
    gap = -t / (y + mi)
    upper = inverse + gap
    upper_square = upper * upper
    inverse_square = inverse * inverse
    both = upper + inverse
    power = inverse
    p = 1.0
    total = _LGAMMA_SERIES[0] * p
    terms = known_terms if known_terms > 0 else _series_count(inverse, floor)
    for k in range(1, terms):
        p = p * upper_square + power * both
        total += _LGAMMA_SERIES[k] * p
        power *= inverse_square
    series = mi * h + (mi - 0.5) * step + total * gap
    if scaled:
        # No shift when `x` is already in the series' range: `log1p(0)`
        # is 0, and at `x = inf` the ratio would be `inf / inf`.
        moved = mi * math.log1p((y - xi) / xi) if y != xi else 0.0
        return (series + moved) + recurrence
    return (series + mi * math.log(y)) + recurrence


#: :func:`_series_route`, compiled by :func:`_kernels` on first use, as the
#: kernels call it.
_series_step: Callable[[float, float, float, float, float, bool, int], float] = (
    _series_route
)

#: :func:`_series_terms`, compiled by :func:`_kernels` on first use.
_series_count: Callable[[float, float], int] = _series_terms


def _log_rising_kernel(
    x: NDArray[np.float64],
    m: NDArray[np.float64],
    out: NDArray[np.float64],
    series_from: float,
    small_t: float,
    plain_error: float,
    promise: float,
    floor: float,
    scaled: bool,
) -> None:
    """``out[i] = lgamma(x[i] + m[i]) - lgamma(x[i])`` over flat, contiguous arrays.

    With ``scaled``, ``out[i]`` is that less ``m[i] log x[i]``. The plain
    difference less ``m log x`` is taken where its bound --- the two
    ``lgamma`` and ``m log x`` each to ``plain_error`` relative --- meets the
    promise over ``max(|out|, 1)``, which holds wherever ``m log x`` does not
    cancel against the rise (small ``x``, or ``m`` large beside ``x``); the
    series elsewhere, where ``m log x`` is never formed (issue #1332).
    """
    for i in range(x.size):
        xi = x[i]
        mi = m[i]
        if mi == 0.0:
            out[i] = 0.0
            continue
        rise = _gammaln(xi + mi)
        base = _gammaln(xi)
        plain = rise - base
        bound = plain_error * (max(abs(rise), 1.0) + max(abs(base), 1.0))
        if scaled:
            shift = mi * math.log(xi)
            value = plain - shift
            if bound + plain_error * abs(shift) <= promise * max(abs(value), 1.0):
                out[i] = value
                continue
        elif bound <= promise * max(abs(plain), 1.0):
            out[i] = plain
            continue
        out[i] = _series_step(xi, mi, series_from, small_t, floor, scaled, 0)


def _scaled_rising_table_kernel(
    shapes: NDArray[np.float64],
    counts: NDArray[np.float64],
    out: NDArray[np.float64],
    series_from: float,
    small_t: float,
    plain_error: float,
    promise: float,
    floor: float,
) -> None:
    """``out[i, j] = S(shapes[i], counts[j])``, :func:`_log_rising_kernel`'s scaled route per pair.

    ``gammaln(x)``, its bound term, ``log x`` and, from ``series_from`` up,
    the series' term count are formed once per shape and read across the row; the route, the bound and the series are the
    per-element kernel's, so each entry is it bit for bit (issue #1341).
    """
    for i in range(shapes.size):
        xi = shapes[i]
        base = _gammaln(xi)
        base_size = max(abs(base), 1.0)
        log_x = math.log(xi)
        # From `series_from` up no shift is taken, so `y = x` in every column.
        row_terms = _series_count(1.0 / xi, floor) if xi >= series_from else 0
        for j in range(counts.size):
            mi = counts[j]
            if mi == 0.0:
                value = 0.0
            else:
                rise = _gammaln(xi + mi)
                plain = rise - base
                bound = plain_error * (max(abs(rise), 1.0) + base_size)
                shift = mi * log_x
                value = plain - shift
                # `not <=`, so a NaN bound, as at `x = inf`, takes the series.
                if not bound + plain_error * abs(shift) <= promise * max(
                    abs(value), 1.0
                ):
                    value = _series_step(
                        xi, mi, series_from, small_t, floor, True, row_terms
                    )
            out[i, j] = value


def _digamma_rising_kernel(
    x: NDArray[np.float64],
    m: NDArray[np.float64],
    out: NDArray[np.float64],
    series_from: float,
    floor: float,
) -> None:
    """``out[i] = digamma(x[i] + m[i]) - digamma(x[i])`` over flat, contiguous arrays."""
    for i in range(x.size):
        xi = x[i]
        mi = m[i]
        y = xi
        recurrence = 0.0
        while y < series_from:
            recurrence += mi / (y * (y + mi))
            y += 1.0
        t = mi / y
        step = math.log1p(t)
        inverse = 1.0 / y
        gap = -t / (y + mi)
        upper = inverse + gap
        upper_square = upper * upper
        inverse_square = inverse * inverse
        power = inverse_square
        q = 1.0
        total = _DIGAMMA_SERIES[0] * q
        terms = 1
        while (
            terms < 8
            and abs(_DIGAMMA_SERIES[terms - 1])
            * 2.0
            * terms
            * inverse ** (2 * terms - 1)
            >= floor
        ):
            terms += 1
        for k in range(1, terms):
            q = q * upper_square + power
            total += _DIGAMMA_SERIES[k] * q
            power *= inverse_square
        total *= gap * (upper + inverse)
        out[i] = (step - math.expm1(-step) / (2.0 * y) - total) + recurrence


@functools.cache
def _kernels() -> tuple[Callable[..., None], Callable[..., None], Callable[..., None]]:
    """:func:`_log_rising_kernel`, :func:`_digamma_rising_kernel` and :func:`_scaled_rising_table_kernel`, compiled on first use (issue #1329).

    Compiled here rather than at import, so that importing this module does
    not import ``numba``: about 0.7 s and 0.2 s in the first call of a
    process, as the ``ctypes`` pointer to ``gammaln`` keeps the log kernel
    out of ``numba``'s cache. Both touch no Python object and are
    ``nogil=True``. The kernels are the NumPy oracles' arithmetic, one pass
    per element; the plain route is the oracle bit for bit, while the series
    counts its terms per element where the oracle counts them per subset (a
    difference below :data:`_TERM_FLOOR` relative) and rounds in scalars.
    """
    from numba import njit

    global _gammaln, _series_count, _series_step  # noqa: PLW0603 - numba reads these as globals at compile time
    _gammaln = _cython_function("gammaln")
    _series_count = njit(nogil=True)(_series_terms)
    _series_step = njit(nogil=True)(_series_route)
    return (
        njit(nogil=True)(_log_rising_kernel),
        njit(nogil=True)(_digamma_rising_kernel),
        njit(nogil=True)(_scaled_rising_table_kernel),
    )


def _flat(x: ArrayLike, m: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """``x`` and ``m`` broadcast, as flat contiguous ``float64``, and an output buffer."""
    x_, m_ = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(m, dtype=np.float64)
    )
    out = np.empty(x_.shape, dtype=np.float64)
    return np.ascontiguousarray(x_).ravel(), np.ascontiguousarray(m_).ravel(), out


def log_rising(x: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """``lgamma(x + m) - lgamma(x)``, the log rising factorial, on NumPy arrays; broadcasts.

    :func:`_log_rising_numpy`'s arithmetic, one compiled pass per element
    (:func:`_log_rising_kernel`, issue #1329); that
    function is the oracle, and the docstring there states the routes and
    the promise: within 1e-14 of ``mpmath`` over ``max(|f|, 1)``.
    """
    x_, m_, out = _flat(x, m)
    _kernels()[0](
        x_,
        m_,
        out.reshape(-1),
        _SERIES_FROM,
        _SMALL_T,
        _PLAIN_ERROR,
        _LOG_PROMISE,
        _TERM_FLOOR,
        False,
    )
    return out


def log_rising_into(
    x: NDArray[np.float64], m: NDArray[np.float64], out: NDArray[np.float64]
) -> None:
    """:func:`log_rising` written into ``out``, allocating nothing (issue #1332).

    ``x``, ``m`` and ``out`` are ``float64``, C-contiguous and of one shape;
    no broadcast is taken, since a broadcast allocates. A compiled loop that
    needs rising factorials takes them from tables built once per M step
    with this, such as :func:`sal.emissions.bb.trial_tables`; no scalar
    ``@njit`` form is public. ``out`` is :func:`log_rising` bit for bit.
    """
    for name, array in (("x", x), ("m", m), ("out", out)):
        if array.dtype != np.float64 or not array.flags.c_contiguous:
            msg = f"{name} is C-contiguous float64, got {array.dtype}"
            raise ValueError(msg)
    if not x.shape == m.shape == out.shape:
        msg = f"x, m and out share one shape, got {x.shape}, {m.shape}, {out.shape}"
        raise ValueError(msg)
    _kernels()[0](
        x.reshape(-1),
        m.reshape(-1),
        out.reshape(-1),
        _SERIES_FROM,
        _SMALL_T,
        _PLAIN_ERROR,
        _LOG_PROMISE,
        _TERM_FLOOR,
        False,
    )


def scaled_rising_array(x: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """``lgamma(x + m) - lgamma(x) - m log x``, :func:`scaled_rising` on NumPy arrays; broadcasts.

    :func:`log_rising`'s compiled kernel. The plain ``lgamma`` difference less
    ``m log x`` where its error bound meets the 1e-14 promise over
    ``max(|f|, 1)`` --- no cancellation, as at small ``x`` --- and otherwise
    the series with ``m log x`` removed inside it, never formed: ``0`` at
    ``x = inf`` and at ``m = 0``. The beta-binomial's tables are these
    (issue #1332).
    """
    x_, m_, out = _flat(x, m)
    _kernels()[0](
        x_,
        m_,
        out.reshape(-1),
        _SERIES_FROM,
        _SMALL_T,
        _PLAIN_ERROR,
        _LOG_PROMISE,
        _TERM_FLOOR,
        True,
    )
    return out


def scaled_rising_table(shapes: ArrayLike, counts: ArrayLike) -> NDArray[np.float64]:
    """``S(x, m) = lgamma(x + m) - lgamma(x) - m log x`` at every shape and count, ``(len(shapes), len(counts))``.

    :func:`scaled_rising_array` of ``shapes[:, None]`` and ``counts[None, :]``
    bit for bit, with ``gammaln(x)`` and ``log x`` formed once per shape
    rather than once per pair (issue #1341). ``shapes`` and ``counts`` are
    1-D; the result is ``float64`` and C-contiguous, and ``0`` at
    ``x = inf`` and at ``m = 0``. The beta-binomial's
    :func:`~sal.emissions.bb.trial_tables` and the negative binomial's
    :func:`~sal.emissions.nb.count_log_factor` are built with it.
    """
    x = np.ascontiguousarray(shapes, dtype=np.float64)
    m = np.ascontiguousarray(counts, dtype=np.float64)
    if x.ndim != 1 or m.ndim != 1:
        msg = f"shapes and counts are 1-D, got {x.shape} and {m.shape}"
        raise ValueError(msg)
    out = np.empty((x.size, m.size), dtype=np.float64)
    _kernels()[2](
        x, m, out, _SERIES_FROM, _SMALL_T, _PLAIN_ERROR, _LOG_PROMISE, _TERM_FLOOR
    )
    return out


def digamma_rising(x: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """``digamma(x + m) - digamma(x)``, the derivative of :func:`log_rising` in ``x``; broadcasts.

    :func:`_digamma_rising_numpy`'s arithmetic, one compiled pass per element
    (:func:`_digamma_rising_kernel`, issue #1329);
    that function is the oracle: within 1e-15 relative of ``mpmath``.
    """
    x_, m_, out = _flat(x, m)
    _kernels()[1](x_, m_, out.reshape(-1), _SERIES_FROM, _TERM_FLOOR)
    return out


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


def on_distinct_array(
    values: NDArray[np.float64],
    table: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """:func:`on_distinct` on NumPy arrays, for a caller that takes no derivative (issue #1332).

    ``values`` ends in a singleton axis, ``(..., 1)``, and ``table`` maps a
    ``(D, 1)`` column of them to ``(D, K)``. Elementwise, so the gathered
    entries are ``table(values)`` bit for bit. Too few values, or no
    singleton axis, and ``table`` takes them directly.
    """
    if values.ndim == 0 or values.shape[-1] != 1 or values.size < 64:
        return table(values)
    distinct, inverse = np.unique(values, return_inverse=True)
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
