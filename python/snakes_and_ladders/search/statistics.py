"""Two statistics the Monte Carlo move sets are judged by, and no more.

A chi-square tail probability, so a sampler's realized visit frequencies can
be tested against the exact Boltzmann distribution at a declared significance
rather than eyeballed; and an integrated autocorrelation time, so the reason
for having a cluster update is reported as a number rather than asserted.

Both are written here because the repository carries no `scipy` and root
``CLAUDE.md`` requires a dependency be justified rather than reached for. The
chi-square tail is pinned against textbook critical values, which is an
independent source in the sense root ``CLAUDE.md`` means: not another
implementation of the same series.
"""

from __future__ import annotations

import math

import numpy as np

# `_regularized_upper_gamma` switches between a series and a continued
# fraction at the standard crossover; both converge quickly on their own side
# and slowly on the other (Numerical Recipes, sec. 6.2).
_MAX_TERMS = 1_000
_EPSILON = 1e-15
_TINY = 1e-300


def chi_square_p_value(
    observed: np.ndarray, expected: np.ndarray, *, degrees_of_freedom: int | None = None
) -> float:
    """Tail probability of a chi-square goodness-of-fit statistic.

    Parameters
    ----------
    observed : np.ndarray
        Realized counts per category.
    expected : np.ndarray
        Expected counts per category, same shape and the same total.
    degrees_of_freedom : int | None
        ``None`` uses ``len(observed) - 1``, correct when the expected counts
        come from a distribution with no parameters estimated from the
        sample. Every caller here is that case: the Boltzmann distribution is
        known exactly from the fixture, not fitted to the chain.

    Returns
    -------
    float
        ``P(X > statistic)``. A sampler that is right returns a value uniform
        on ``[0, 1]``; a sampler with a broken accept step returns one
        indistinguishable from zero.

    Raises
    ------
    ValueError
        If the shapes disagree, or any expected count is not positive. A zero
        expected count makes the statistic infinite for any observation, so
        the caller has chosen a category the model cannot produce.
    """
    if observed.shape != expected.shape:
        msg = f"observed {observed.shape} and expected {expected.shape} disagree"
        raise ValueError(msg)
    if not (expected > 0).all():
        msg = "every expected count must be positive"
        raise ValueError(msg)

    statistic = float((((observed - expected) ** 2) / expected).sum())
    dof = len(observed) - 1 if degrees_of_freedom is None else degrees_of_freedom
    return _regularized_upper_gamma(dof / 2.0, statistic / 2.0)


def integrated_autocorrelation_time(series: np.ndarray, *, window: int = 5) -> float:
    """Sweeps a chain must run to buy one independent sample of ``series``.

    ``tau = 1 + 2 * sum_t rho(t)``, truncated by Sokal's automatic window: the
    sum is cut at the first ``t`` with ``t >= window * tau``. Truncation is not
    optional --- the estimator of ``rho(t)`` has roughly constant variance in
    ``t`` while the signal decays, so summing the whole series adds noise
    without adding signal and the result grows with the chain length instead
    of converging.

    Parameters
    ----------
    series : np.ndarray
        A scalar observable, one entry per sweep.
    window : int
        Sokal's ``c``. 5 is the standard choice.

    Returns
    -------
    float
        ``tau``, in sweeps. Never below 0.5, which is the value for a series
        of independent draws.

    Raises
    ------
    ValueError
        If the series is shorter than two entries.
    """
    if series.shape[0] < 2:
        msg = (
            f"an autocorrelation time needs at least two sweeps, got {series.shape[0]}"
        )
        raise ValueError(msg)

    centred = series - series.mean()
    variance = float((centred**2).mean())
    if variance == 0.0:
        # A constant chain has no correlation to measure. It is also a chain
        # that never moved, which is a sampler failure rather than a fast one,
        # so this returns the floor and lets the caller's own test see it.
        return 0.5

    length = centred.shape[0]
    padded = int(2 ** math.ceil(math.log2(2 * length)))
    spectrum = np.fft.rfft(centred, n=padded)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=padded)[:length]
    correlation = correlation / correlation[0]

    total = 0.0
    for lag in range(1, length):
        total += float(correlation[lag])
        tau = 0.5 + total
        if lag >= window * tau:
            break
    return max(0.5 + total, 0.5)


def _regularized_upper_gamma(shape: float, value: float) -> float:
    """``Q(a, x)``, the regularized upper incomplete gamma function."""
    if value <= 0.0:
        return 1.0
    if value < shape + 1.0:
        return 1.0 - _lower_series(shape, value)
    return _upper_continued_fraction(shape, value)


def _lower_series(shape: float, value: float) -> float:
    """``P(a, x)`` by its series, which converges for ``x < a + 1``."""
    term = 1.0 / shape
    total = term
    power = shape
    for _ in range(_MAX_TERMS):
        power += 1.0
        term *= value / power
        total += term
        if abs(term) < abs(total) * _EPSILON:
            break
    return total * math.exp(-value + shape * math.log(value) - math.lgamma(shape))


def _upper_continued_fraction(shape: float, value: float) -> float:
    """``Q(a, x)`` by Lentz's continued fraction, for ``x >= a + 1``.

    The two recurrences advance the numerator and denominator of successive
    convergents and are *not* interchangeable: ``d`` is updated from its own
    previous value, ``c`` from the current ``b``. Writing one in terms of the
    other overflows within a few iterations rather than converging, which is
    how the first draft of this function was caught.
    """
    b = value + 1.0 - shape
    c = 1.0 / _TINY
    d = 1.0 / b
    result = d
    for index in range(1, _MAX_TERMS):
        offset = -index * (index - shape)
        b += 2.0
        d = offset * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + offset / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        step = d * c
        result *= step
        if abs(step - 1.0) < _EPSILON:
            break
    return result * math.exp(-value + shape * math.log(value) - math.lgamma(shape))
