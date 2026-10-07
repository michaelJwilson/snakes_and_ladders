"""The NumPy log rising factorial and its digamma rise, against `mpmath` and torch (issue #1300).

:func:`sal.emissions.rising.log_rising` and
:func:`~sal.emissions.rising.digamma_rising` are judged against ``mpmath`` at
50 digits from ``x = 1e-3`` to ``1e16`` and ``m`` from 0 to ``1e6``, against
the torch route at and above :data:`~sal.emissions.rising.LARGE_SHAPE`, and
against ``m = 0`` and the recurrence ``(x)_(m + 1) = (x)_m (x + m)``.
"""

from __future__ import annotations

import mpmath  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from sal.emissions.rising import (
    LARGE_SHAPE,
    digamma_rising,
    log_rising,
    scaled_rising,
)

mpmath.mp.dps = 50

#: Shapes judged: small, either side of the gamma function's minimum and
#: of the shift, either side of :data:`LARGE_SHAPE`, and up to the limit.
SHAPES = [1e-3, 0.1, 0.5, 0.618, 1.0, 1.4616, 2.0, 9.5, 9.99, 10.0, 10.5, 50.0]
SHAPES += [99.0, 99.9, LARGE_SHAPE, 101.0, 1e3, 1e4, 1e6, 1e8, 1e12, 1e16]

#: Counts judged, fractional included.
COUNTS = [0.0, 1e-6, 1e-3, 0.5, 1.0, 2.0, 3.0, 7.5, 10.0, 40.0, 200.0, 1e3]
COUNTS += [1e4, 1e6]

#: The log rising factorial's error over ``max(|f|, 1)``. Measured worst:
#: 6.4e-15 at ``x = 1e-3``, ``m = 7.5``. Near a zero of ``(x)_m`` the
#: terms are of size one, so a relative bound is not attainable there.
_LOG_TOLERANCE = 1e-14

#: The digamma rise's relative error. Measured worst: 4.0e-16.
_DIGAMMA_TOLERANCE = 1e-15

#: NumPy against torch at and above :data:`LARGE_SHAPE`, over
#: ``max(|f|, 1)``: the two difference Stirling's tail differently. Measured
#: worst: 5.9e-16, and 98 of the 112 pairs bit for bit: the plain route
#: is taken at large ``m`` (issue #1329); 7.6e-20 before it.
_TORCH_TOLERANCE = 1e-15


def _grid() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    shape, count = np.meshgrid(SHAPES, COUNTS, indexing="ij")
    return shape.ravel(), count.ravel()


def _exact(
    function: str, shape: NDArray[np.float64], count: NDArray[np.float64]
) -> NDArray[np.float64]:
    exact = getattr(mpmath, function)
    return np.array(
        [
            float(exact(mpmath.mpf(x) + mpmath.mpf(m)) - exact(mpmath.mpf(x)))
            for x, m in zip(shape, count, strict=True)
        ]
    )


@pytest.mark.oracle
def test_the_log_rising_factorial_is_exact() -> None:
    shape, count = _grid()
    want = _exact("loggamma", shape, count)

    got = log_rising(shape, count)

    error = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
    assert float(error.max()) <= _LOG_TOLERANCE


@pytest.mark.oracle
def test_the_digamma_rise_is_exact() -> None:
    shape, count = _grid()
    want = _exact("digamma", shape, count)

    got = digamma_rising(shape, count)

    error = np.abs(got - want) / np.where(want == 0.0, 1.0, np.abs(want))
    assert float(error.max()) <= _DIGAMMA_TOLERANCE


@pytest.mark.oracle
def test_numpy_agrees_with_the_torch_route_at_large_shape() -> None:
    shape, count = _grid()
    large = shape >= LARGE_SHAPE
    shape, count = shape[large], count[large]
    x, m = torch.from_numpy(shape), torch.from_numpy(count)
    torch_route = (scaled_rising(x, m) + m * torch.log(x)).numpy()

    got = log_rising(shape, count)

    error = np.abs(got - torch_route) / np.maximum(np.abs(torch_route), 1.0)
    assert float(error.max()) <= _TORCH_TOLERANCE


@pytest.mark.analytic
def test_no_count_is_no_rise() -> None:
    shape = np.array(SHAPES)

    assert np.all(log_rising(shape, 0.0) == 0.0)
    assert np.all(digamma_rising(shape, 0.0) == 0.0)


@pytest.mark.analytic
def test_the_recurrence_holds() -> None:
    shape, count = _grid()

    step = log_rising(shape, count + 1.0) - log_rising(shape, count)

    error = np.abs(step - np.log(shape + count))
    scale = np.maximum(np.abs(log_rising(shape, count + 1.0)), 1.0)
    assert float((error / scale).max()) <= 2.0 * _LOG_TOLERANCE


@pytest.mark.analytic
def test_the_digamma_rise_is_the_derivative() -> None:
    """A central difference of :func:`log_rising` in ``x``, at step ``1e-5 x``."""
    shape = np.array([0.5, 3.0, 50.0, 1e3, 1e6])
    count = np.array([1.0, 7.5, 40.0, 200.0, 1e4])
    step = 1e-5 * shape

    difference = (log_rising(shape + step, count) - log_rising(shape - step, count)) / (
        2.0 * step
    )

    np.testing.assert_allclose(digamma_rising(shape, count), difference, rtol=1e-8)


def _plain_margin(
    shape: NDArray[np.float64], count: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The plain difference's error bound over the promise: at most 1 takes the plain route."""
    from sal.emissions.rising import _LOG_PROMISE, _PLAIN_ERROR
    from scipy.special import gammaln

    rise, base = gammaln(shape + count), gammaln(shape)
    bound = _PLAIN_ERROR * (
        np.maximum(np.abs(rise), 1.0) + np.maximum(np.abs(base), 1.0)
    )
    margin: NDArray[np.float64] = bound / (
        _LOG_PROMISE * np.maximum(np.abs(rise - base), 1.0)
    )
    return margin


@pytest.mark.oracle
def test_the_log_rising_factorial_is_exact_either_side_of_the_plain_route() -> None:
    # The 16 pairs of a dense grid nearest the plain route's bound on each
    # side (issue #1329), judged against mpmath at the same tolerance.
    grid = np.meshgrid(
        np.geomspace(1e-3, 1e4, 120), np.geomspace(1e-6, 1e6, 120), indexing="ij"
    )
    shape, count = grid[0].ravel(), grid[1].ravel()
    margin = _plain_margin(shape, count)
    plain = np.flatnonzero(margin <= 1.0)
    series = np.flatnonzero(margin > 1.0)
    pick = np.concatenate(
        [
            plain[np.argsort(-margin[plain])[:16]],
            series[np.argsort(margin[series])[:16]],
        ]
    )
    shape, count = shape[pick], count[pick]
    want = _exact("loggamma", shape, count)

    got = log_rising(shape, count)

    error = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
    assert float(error.max()) <= _LOG_TOLERANCE
