"""Benchmarks for Goemans-Williamson, and for the enumeration it replaces.

Correctness is pinned in `tests/regression/search/`, per the repo's division
of labor between the two directories.

Both are run because the interesting quantity is where one stops being
possible. Enumeration is `2 ** (n - 1)` assignments and reaches about twenty
nodes; the relaxation is a fixed number of gradient steps on an `n x n` Gram
matrix and does not care. That crossover is what the ticket is about, and a
timing on one alone would not show it.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.max_cut import enumerate_max_cut, goemans_williamson
from snakes_and_ladders.sim.graph import PottsGraph


def _random_graph(n_nodes: int, density: float, seed: int) -> PottsGraph:
    rng = np.random.default_rng(seed)
    edges = tuple(
        (first, second)
        for first in range(n_nodes)
        for second in range(first + 1, n_nodes)
        if rng.random() < density
    )
    return PottsGraph(n_nodes=n_nodes, edges=edges, coupling=(1.0,) * len(edges))


@pytest.mark.parametrize("n_nodes", [16, 40, 100])
def test_goemans_williamson_benchmark(
    benchmark: BenchmarkFixture, n_nodes: int
) -> None:
    # 100 nodes is `2 ** 99` assignments: enumeration is not merely slow there,
    # it is impossible, and the relaxation is the only thing that returns a
    # number with a certificate attached.
    graph = _random_graph(n_nodes, 0.3, n_nodes)

    result = benchmark(goemans_williamson, graph, 1)

    assert result.value >= 0.0


@pytest.mark.parametrize("n_nodes", [12, 16, 20])
def test_enumerate_max_cut_benchmark(benchmark: BenchmarkFixture, n_nodes: int) -> None:
    graph = _random_graph(n_nodes, 0.3, n_nodes)

    _, value = benchmark(enumerate_max_cut, graph)

    assert value >= 0.0
