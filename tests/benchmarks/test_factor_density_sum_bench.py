"""The ordered sum is the fast one, so its bitwise pin costs nothing (#651).

`kernels.factor_graph_log_density` accumulates factors left to right in graph
order, which reproduces the oracle bitwise (#563). #651 listed that as a site
where `CLAUDE.md`'s bitwise-to-tolerance rule might admit a vectorized sum,
since vectorization is the repository's own optimization rule and the pin reads
like a constraint being paid for.

**It is not being paid for.** Gathering the offsets into an array and calling
``np.sum`` is *slower* at every size measured -- the extra array and the second
pass over it cost more than fusing the accumulate into the offset loop saves.
There is no performance to admit, so the rule never comes into it.

Measured on the reference host, minimum of three runs of each:

| factors | ordered | offsets + ``np.sum`` | ratio |
| --- | --- | --- | --- |
| 1,000 | 6.3 us | 10.5 us | 0.60x |
| 10,000 | 62.5 us | 96.7 us | 0.65x |
| 100,000 | 614.7 us | 997.6 us | 0.62x |
| 1,000,000 | 10,796 us | 14,090 us | 0.77x |

The two also agreed bitwise at all four sizes on this data, ``np.sum``'s
pairwise order having coincided with the sequential one. That is a property of
these inputs and not a guarantee, which is why the pin stays; what it is not is
a cost.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit
from snakes_and_ladders.search.kernels import factor_graph_log_density

SEED = 20260916
#: Past the lattice fixtures, where a per-factor cost is visible.
N_FACTORS = 100_000
N_VARIABLES, N_STATES = 5041, 3


@njit
def _gathered_sum(
    state: np.ndarray,
    tables: np.ndarray,
    factor_start: np.ndarray,
    factor_offsets: np.ndarray,
    factor_column: np.ndarray,
    factor_stride: np.ndarray,
) -> float:
    """The vectorized alternative: every offset, then one ``np.sum``."""
    count = factor_start.shape[0]
    offsets = np.empty(count, dtype=np.int64)
    for factor in range(count):
        offset = factor_start[factor]
        for axis in range(factor_offsets[factor], factor_offsets[factor + 1]):
            offset += state[factor_column[axis]] * factor_stride[axis]
        offsets[factor] = offset
    return float(np.sum(tables[offsets]))


@pytest.fixture(scope="module")
def layout() -> tuple[np.ndarray, ...]:
    """A two-axis factor graph at the size above, seeded."""
    rng = np.random.default_rng(SEED)
    return (
        rng.integers(0, N_STATES, N_VARIABLES).astype(np.int64),
        rng.normal(0.0, 1.0, N_FACTORS * N_STATES * N_STATES),
        (np.arange(N_FACTORS) * N_STATES * N_STATES).astype(np.int64),
        (np.arange(N_FACTORS + 1) * 2).astype(np.int64),
        rng.integers(0, N_VARIABLES, 2 * N_FACTORS).astype(np.int64),
        np.tile(np.array([N_STATES, 1], dtype=np.int64), N_FACTORS),
    )


@pytest.mark.release
@pytest.mark.structural
def test_the_vectorized_sum_is_the_slower_one(layout: tuple[np.ndarray, ...]) -> None:
    # The finding, asserted rather than left in a pull request body. If this
    # ever fails, gathering has become cheaper than fusing and #651's second
    # candidate is worth re-reading -- at which point the bitwise question
    # becomes live for the first time.
    import time

    factor_graph_log_density(*layout)
    _gathered_sum(*layout)

    def elapsed(function: object) -> float:
        start = time.perf_counter()
        for _ in range(20):
            function(*layout)  # type: ignore[operator]
        return time.perf_counter() - start

    ordered = min(elapsed(factor_graph_log_density) for _ in range(3))
    gathered = min(elapsed(_gathered_sum) for _ in range(3))

    assert ordered < gathered, (
        f"the ordered accumulate took {ordered:.4f}s against the gathered sum's "
        f"{gathered:.4f}s; the vectorized form has become the faster one and "
        "#651 candidate 2 should be re-read"
    )


@pytest.mark.release
@pytest.mark.benchmark
def test_factor_graph_log_density_ordered(
    benchmark: object, layout: tuple[np.ndarray, ...]
) -> None:
    """The form the kernel uses."""
    factor_graph_log_density(*layout)
    benchmark(factor_graph_log_density, *layout)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_factor_graph_log_density_gathered(
    benchmark: object, layout: tuple[np.ndarray, ...]
) -> None:
    """The vectorized alternative, as the reference the other is read against."""
    _gathered_sum(*layout)
    benchmark(_gathered_sum, *layout)  # type: ignore[operator]
