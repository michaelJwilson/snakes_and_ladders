"""The NumPy log rising factorial against the torch route, on 10^6 pairs (issue #1300).

A record, not a claimed speed-up: :func:`sal.emissions.rising.log_rising`
beside :func:`~sal.emissions.rising.scaled_rising` plus ``m log x``, and
:func:`~sal.emissions.rising.digamma_rising`, at shapes log-uniform from
1e-3 to 1e16 and counts from 0 to 1000. ``tests/regression/test_emissions_rising_numpy.py``
pins them against ``mpmath``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from sal.emissions.rising import digamma_rising, log_rising, scaled_rising

#: Pairs per call.
N_PAIRS = 1_000_000


@pytest.fixture(scope="module")
def pairs() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shapes log-uniform over 19 decades and integer counts up to 1000."""
    rng = np.random.default_rng(1300)
    shape = 10.0 ** rng.uniform(-3.0, 16.0, N_PAIRS)
    return shape, rng.integers(0, 1001, N_PAIRS).astype(np.float64)


@pytest.mark.release
@pytest.mark.benchmark
def test_log_rising_numpy(
    benchmark: object, pairs: tuple[NDArray[np.float64], NDArray[np.float64]]
) -> None:
    benchmark(log_rising, *pairs)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_log_rising_torch(
    benchmark: object, pairs: tuple[NDArray[np.float64], NDArray[np.float64]]
) -> None:
    shape, count = (torch.from_numpy(p) for p in pairs)
    benchmark(lambda: scaled_rising(shape, count) + count * torch.log(shape))  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_digamma_rising_numpy(
    benchmark: object, pairs: tuple[NDArray[np.float64], NDArray[np.float64]]
) -> None:
    benchmark(digamma_rising, *pairs)  # type: ignore[operator]
