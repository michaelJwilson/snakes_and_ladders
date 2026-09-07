"""Benchmarks for `fit` on surfaces with no model behind them.

Correctness is pinned in `tests/regression/opt/`, per the repo's division of
labor between the two directories.

What these isolate is the optimizer's own cost. Every other `fit` benchmark
measures a likelihood evaluation as well, so a change in the driver and a
change in the model are confounded in the number; here the objective is three
lines of arithmetic and what remains is L-BFGS.
"""

from __future__ import annotations

import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.testfunctions import Himmelblau, Rastrigin, Rosenbrock


@pytest.mark.parametrize("dimension", [2, 10, 50])
def test_rosenbrock_fit_benchmark(benchmark: BenchmarkFixture, dimension: int) -> None:
    # Scaling in dimension on a badly conditioned surface: the limited-memory
    # Hessian approximation is what this is really measuring.
    objective = Rosenbrock(dimension=dimension)

    result = benchmark(fit, objective)

    assert result.converged


def test_rastrigin_fit_benchmark(benchmark: BenchmarkFixture) -> None:
    result = benchmark(fit, Rastrigin(dimension=10))

    assert result.converged


def test_himmelblau_fit_benchmark(benchmark: BenchmarkFixture) -> None:
    result = benchmark(fit, Himmelblau())

    assert result.converged
