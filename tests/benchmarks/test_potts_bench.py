"""Benchmarks for `snakes_and_ladders.sim.potts.simulate_potts`.

Correctness is pinned in `tests/regression/sim/test_potts_simulate.py`
against exhaustive enumeration; these measure cost. Both sampling paths are
timed because the ratio between them is the number that matters: the exact
open-chain recursion draws independent configurations in one pass, and Gibbs
pays a sweep loop per sample, so the gap says what the general path costs and
what a future exact lattice sampler would be worth.
"""

from __future__ import annotations

import math

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.potts import simulate_potts

_FIELD = np.array([0.40, -0.10, -0.30])
_SEED = 20260903


def test_the_exact_open_chain_sampler(benchmark: BenchmarkFixture) -> None:
    graph = lattice_graph((50,), boundary=BoundaryCondition.OPEN, coupling=0.6)

    result = benchmark(
        simulate_potts, graph, _FIELD, seed=_SEED, n_samples=200, burn_in=0
    )

    # Shape and finiteness only: this file measures, it does not validate.
    assert result.configurations.shape == (200, 50)
    assert math.isfinite(float(result.configurations.sum()))


def test_the_gibbs_sampler_on_a_two_dimensional_lattice(
    benchmark: BenchmarkFixture,
) -> None:
    graph = lattice_graph((6, 6), boundary=BoundaryCondition.OPEN, coupling=0.5)

    result = benchmark(
        simulate_potts, graph, _FIELD, seed=_SEED, n_samples=50, burn_in=50
    )

    assert result.configurations.shape == (50, 36)
    assert math.isfinite(float(result.configurations.sum()))
