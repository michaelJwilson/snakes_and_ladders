"""Benchmarks for the CSS layer: the enumeration oracle and the criterion.

See tests/regression/likelihood/test_css.py and tests/regression/sim/test_css.py
for correctness. Two costs are new here and neither is the decoder's, which
`test_ldpc_bench.py` already times. The oracle enumerates ``2 ** n`` errors
rather than ``2 ** k`` codewords, so its cost is set by the block length alone
and it is timed at the 16-qubit instance the fixture declares. The criterion
runs one dense GF(2) elimination per decode, so it is timed at both lengths:
it is paid per trial of every rate measured, and a rate over 2,000 trials pays
it 2,000 times.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.css import decode_succeeds, error_cosets
from snakes_and_ladders.sim.css import CssCode
from snakes_and_ladders.sim.ldpc import bicycle_code

FLIP = 0.05


def _code(n_bits: int, n_checks: int, seed: int) -> CssCode:
    return CssCode.from_parity_check(
        bicycle_code(n_bits, n_checks, 3, np.random.default_rng(seed))
    )


def test_error_cosets_benchmark(benchmark: BenchmarkFixture) -> None:
    """Partitioning all 65,536 errors of the 16-qubit instance by coset."""
    code = _code(16, 7, 1)

    result = benchmark(error_cosets, code, FLIP)

    # Benchmarks assert finiteness only; correctness is pinned in the suite.
    assert np.isfinite(result.degenerate_logical_error_rate)


@pytest.mark.parametrize(("n_bits", "n_checks"), [(16, 7), (96, 40)])
def test_decode_succeeds_benchmark(
    benchmark: BenchmarkFixture, n_bits: int, n_checks: int
) -> None:
    """One rank test over the parity-check matrix with the residual appended."""
    code = _code(n_bits, n_checks, 1 if n_bits == 16 else 361)
    error = np.zeros(n_bits, dtype=np.uint8)
    error[[0, 3]] = 1

    result = benchmark(decode_succeeds, code, error, error ^ code.logical_x[0])

    assert result is False


@pytest.mark.parametrize(("n_bits", "n_checks"), [(16, 7), (96, 40)])
def test_css_code_benchmark(
    benchmark: BenchmarkFixture, n_bits: int, n_checks: int
) -> None:
    """Building the code: one rank, one null space, and the greedy extension."""
    checks = bicycle_code(
        n_bits, n_checks, 3, np.random.default_rng(1 if n_bits == 16 else 361)
    )

    result = benchmark(CssCode.from_parity_check, checks)

    assert result.n_logical > 0
