"""`Ragged.reduce` and `Ragged.floored` at 10^6 positions (issue #1141).

10^4 segments is the issue's size; 10^5 segments of ten positions is the
stress case for `floored`, whose greedy pass is one Python step per segment.
`tests/regression/test_ragged.py` referees both against written-out loops.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.ragged import Ragged

POSITIONS = 1_000_000


def _batch(n_segments: int) -> Ragged:
    """`n_segments` segments of random lengths tiling 10^6 standard normals."""
    rng = np.random.default_rng(1141)
    cuts = np.sort(
        rng.choice(np.arange(2, POSITIONS - 1, 2), n_segments - 1, replace=False)
    )
    lengths = np.diff(np.r_[0, cuts, POSITIONS])
    return Ragged(rng.standard_normal(POSITIONS), tuple(int(x) for x in lengths))


@pytest.mark.benchmark
@pytest.mark.parametrize("n_segments", [10_000, 100_000], ids=str)
def test_reduce(benchmark: object, n_segments: int) -> None:
    batch = _batch(n_segments)
    sums = benchmark(batch.reduce)  # type: ignore[operator]
    assert sums.shape == (n_segments,)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_segments", [10_000, 100_000], ids=str)
def test_floored(benchmark: object, n_segments: int) -> None:
    batch = _batch(n_segments)
    rng = np.random.default_rng(1142)
    weight = rng.exponential(1.0, n_segments)
    groups = np.sort(rng.integers(0, 100, n_segments))
    min_length = 4.0 * POSITIONS / n_segments
    merged, parent = benchmark(  # type: ignore[operator]
        lambda: batch.floored(min_length, weight=weight, min_weight=2.0, groups=groups)
    )
    assert parent.shape == (n_segments,)
    assert merged.n_segments < n_segments
