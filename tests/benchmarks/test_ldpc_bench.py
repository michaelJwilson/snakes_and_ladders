"""Benchmarks for the vectorized decoder and the ensemble draw.

See tests/regression/likelihood/test_ldpc.py for correctness. Each decoding
cell runs a fixed 50 iterations with the syndrome stop off, so the number
is a per-iteration cost over the edges and not a function of the noise;
the (3,6) code at 19,998 bits is the ticket's size, 59,994 edges.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.ldpc import DecodingAlgorithm, decode
from snakes_and_ladders.sim.ldpc import (
    BinarySymmetricChannel,
    all_zero_transmission,
    gallager_code,
)

ITERATIONS = 50


@pytest.mark.parametrize("n_bits", [996, 19_998])
@pytest.mark.parametrize("algorithm", list(DecodingAlgorithm))
def test_decode_benchmark(
    benchmark: BenchmarkFixture, n_bits: int, algorithm: DecodingAlgorithm
) -> None:
    """Fifty flooding iterations over every edge of a (3,6) code."""
    code = gallager_code(n_bits, 3, 6, np.random.default_rng(3))
    llr = all_zero_transmission(
        code, BinarySymmetricChannel(0.05), np.random.default_rng(0)
    )

    result = benchmark(
        decode,
        code,
        llr,
        algorithm=algorithm,
        max_iterations=ITERATIONS,
        early_stop=False,
    )

    # Benchmarks assert finiteness only; correctness is pinned in the regression suite.
    assert result.iterations == ITERATIONS
    assert np.all(np.isfinite(result.posterior_llr))


def test_gallager_code_benchmark(benchmark: BenchmarkFixture) -> None:
    """Drawing the 19,998-bit ensemble member: three permutations and one sort."""
    result = benchmark(gallager_code, 19_998, 3, 6, np.random.default_rng(3))

    assert result.n_edges == 3 * 19_998
