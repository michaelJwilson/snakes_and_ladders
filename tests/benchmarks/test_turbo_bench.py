"""Benchmarks for the BCJR pass and the turbo iteration that runs two of them.

See tests/regression/likelihood/test_convolutional.py and test_turbo.py for
correctness. The two cells answer the two questions that decide where a port
would pay: one BCJR pass at the waterfall's block length, which is the whole
of the inner loop, and eight turbo iterations, which is sixteen of those
passes plus the interleaving. A fixed iteration count with the early stop off,
so the number is a cost per iteration and not a function of the noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.convolutional import bcjr
from snakes_and_ladders.likelihood.turbo import (
    decode_turbo,
    noise_scale,
    split_streams,
)
from snakes_and_ladders.sim.convolutional import TurboCode, turbo_code

ITERATIONS = 8


def _instance(message_length: int) -> tuple[TurboCode, np.ndarray]:
    """The code and one received word at 1 dB, under fixed seeds."""
    code = turbo_code(0o7, 0o5, 2, message_length, np.random.default_rng(233))
    sigma = noise_scale(1.0, code.rate)
    rng = np.random.default_rng(9)
    received = 1.0 + sigma * rng.standard_normal(code.block_length)
    return code, np.asarray(2.0 * received / sigma**2)


@pytest.mark.parametrize("message_length", [256, 1024])
def test_bcjr_benchmark(benchmark: BenchmarkFixture, message_length: int) -> None:
    """One forward-backward pass over `K + m` steps of a four-state trellis."""
    code, llr = _instance(message_length)
    streams = split_streams(code, llr)

    result = benchmark(bcjr, code.trellis, streams.systematic, streams.parity_first)

    # Benchmarks assert finiteness only; correctness is pinned in the suite.
    assert np.all(np.isfinite(result.posterior_llr))


@pytest.mark.parametrize("message_length", [256, 1024])
def test_decode_turbo_benchmark(
    benchmark: BenchmarkFixture, message_length: int
) -> None:
    """Eight iterations: sixteen BCJR passes plus the interleaving between them."""
    code, llr = _instance(message_length)

    result = benchmark(decode_turbo, code, llr, iterations=ITERATIONS)

    assert result.iterations == ITERATIONS
    assert np.all(np.isfinite(result.posterior_llr))
