"""Benchmarks for the Rust categorical sampler against its NumPy oracle.

See `tests/regression/test_numerics_rust.py` for correctness. What is timed
here is the *Python-visible* cost, which is the number that decides whether
the port is worth having and is smaller than the kernel's own speedup: the
arrays have to cross the FFI boundary and the oracle's do not.
`benches/oxi_snakes_and_ladders_bench.rs` measures the kernel alone, and reporting only
that one would overstate what a caller gets.

The sizes are the ones issue #181's audit profiled, where `sample_rows` was
94-96% of `simulate_alignment`'s self time.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.numerics import sample_rows as oracle
from snakes_and_ladders.numerics_rust import sample_rows as accelerated

N_CATEGORIES = 4
SEED = 20260904


def _inputs(n_draws: int) -> tuple[np.ndarray, np.ndarray]:
    distributions = np.random.default_rng(1).dirichlet(
        np.ones(N_CATEGORIES), size=N_CATEGORIES
    )
    rows = np.random.default_rng(2).integers(N_CATEGORIES, size=n_draws)
    return distributions, rows


@pytest.mark.parametrize("n_draws", [200_000, 2_000_000])
def test_numpy_sample_rows_benchmark(benchmark: BenchmarkFixture, n_draws: int) -> None:
    """The oracle, at the profiled sizes."""
    distributions, rows = _inputs(n_draws)

    sampled = benchmark(oracle, np.random.default_rng(SEED), distributions, rows)

    assert sampled.shape == (n_draws,)


@pytest.mark.parametrize("n_draws", [200_000, 2_000_000])
def test_rust_sample_rows_benchmark(benchmark: BenchmarkFixture, n_draws: int) -> None:
    """The same call through `oxi_snakes_and_ladders`, including the boundary it crosses."""
    distributions, rows = _inputs(n_draws)

    sampled = benchmark(accelerated, np.random.default_rng(SEED), distributions, rows)

    assert sampled.shape == (n_draws,)
