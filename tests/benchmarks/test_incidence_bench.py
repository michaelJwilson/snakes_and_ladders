"""Benchmarks for the one incidence layout, and the three stores that hold it.

Correctness is pinned in `tests/regression/test_incidence.py`, per the repo's
division of labor between the two directories.

Issue #586 lifted one compressed-row layout out of three classes that each
derived their own. Root `CLAUDE.md` asks that such a change be at least as
fast at every consumer, so the consumers are timed rather than the structure:
a Wolff cluster move, which asked the graph to rebuild its adjacency once per
move; the factor graph's two neighbourhood queries, which scanned every factor
per query; and the parity-check build, the one consumer that already had the
layout and so had nothing to gain.

The parity check is the honest case. Routing it through the structure costs a
fixed ~30 us of extra calls, which is 25 per cent at 120 bits, 10 at 600,
nothing by 3,000, and nothing at the 19,998-bit instance the ticket names ---
a build paid once against a decode paid per iteration. It is timed here so
that stays visible.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.spatio_sequential import _wolff_update
from snakes_and_ladders.sim.factor_graph import Factor, FactorGraph, Variable
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.ldpc import gallager_code


@pytest.mark.parametrize("extent", [16, 32, 64])
def test_compressed_adjacency_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    # What every caller asks for. Derived once per graph now, so this is the
    # cost of handing back three arrays; it was 2.5 ms at 64x64.
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 1.0)
    graph.compressed_adjacency()

    offsets, neighbours, _ = benchmark(graph.compressed_adjacency)

    assert int(offsets[-1]) == neighbours.size == 2 * len(graph.edges)


@pytest.mark.parametrize("extent", [16, 32, 64])
def test_wolff_cluster_move_benchmark(benchmark: BenchmarkFixture, extent: int) -> None:
    # The consumer the per-call derivation cost most: a cluster move touches
    # one cluster and rebuilt the whole layout to do it.
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 1.0)
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 3, size=graph.n_nodes)
    field = rng.normal(size=(graph.n_nodes, 3))

    benchmark(lambda: _wolff_update(labels, graph, field, 0.2, rng))

    assert set(np.unique(labels)) <= {0, 1, 2}


@pytest.mark.parametrize("n_variables", [200, 800])
def test_factor_graph_neighbourhoods_benchmark(
    benchmark: BenchmarkFixture, n_variables: int
) -> None:
    # Quadratic before: every query scanned every factor, so asking each
    # variable for its own was `O(V F)`.
    graph = FactorGraph(
        [Variable(f"v{index}", 2) for index in range(n_variables)],
        [
            Factor(f"f{index}", (f"v{index}", f"v{index + 1}"), np.zeros((2, 2)))
            for index in range(n_variables - 1)
        ],
    )

    degrees = benchmark(
        lambda: [graph.degree(variable.name) for variable in graph.variables]
    )

    assert degrees == [1, *[2] * (n_variables - 2), 1]


@pytest.mark.parametrize("n_bits", [120, 3000])
def test_parity_check_build_benchmark(benchmark: BenchmarkFixture, n_bits: int) -> None:
    # The consumer with nothing to gain: it had the layout already, and now
    # holds it through the structure. The small size carries the fixed cost.
    code = benchmark(lambda: gallager_code(n_bits, 3, 6, np.random.default_rng(5)))

    assert code.n_edges == 3 * n_bits
