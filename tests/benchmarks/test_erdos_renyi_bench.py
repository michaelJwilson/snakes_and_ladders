"""Benchmarks for the random-graph generator and belief propagation on it.

Correctness is pinned in `tests/regression/sim/`, per the repo's division of
labor between the two directories.

The generator is `O(n^2)` regardless of density, because every pair is drawn
whether or not it becomes an edge. That is deliberate — one vectorized draw
beats a sparse construction at the sizes here — and the benchmark is where
that choice stays visible if the sizes ever grow.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.belief_propagation import belief_propagation
from snakes_and_ladders.sim.graph import erdos_renyi_graph


@pytest.mark.parametrize("n_nodes", [50, 200, 800])
def test_erdos_renyi_graph_benchmark(benchmark: BenchmarkFixture, n_nodes: int) -> None:
    rng = np.random.default_rng(1)

    graph = benchmark(erdos_renyi_graph, n_nodes, 3.0 / n_nodes, 0.7, rng)

    assert graph.n_nodes == n_nodes


@pytest.mark.parametrize("n_nodes", [20, 60])
def test_belief_propagation_on_a_random_graph_benchmark(
    benchmark: BenchmarkFixture, n_nodes: int
) -> None:
    # Sparse and locally tree-like, which is the regime BP converges fastest
    # in — the complement of the lattice case `test_belief_propagation_bench`
    # measures, where the short cycles slow it down.
    rng = np.random.default_rng(n_nodes)
    graph = erdos_renyi_graph(n_nodes, 2.0 / n_nodes, 0.5, rng)

    result = benchmark(belief_propagation, graph, np.array([0.3, -0.7, 0.15]))

    assert result.residual <= 1e-12
