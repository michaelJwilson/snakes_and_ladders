"""Benchmarks for max flow, and the port decision they justify.

Correctness is pinned in `tests/regression/search/`, per the repo's division
of labor between the two directories.

Root `CLAUDE.md` requires a candidate be benchmarked against the reference
before a Rust port is committed to, and this is that measurement kept rather
than discarded: the Python and Rust paths run the same problem at the same
sizes, so the ratio is visible in one table rather than asserted in prose.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search import maxflow_rust
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

# The Python blocking flow recurses to the depth of the level graph; the Rust
# one uses an explicit stack. This raise is itself part of what the port buys.
sys.setrecursionlimit(50_000)


def _problem(extent: int) -> tuple[object, np.ndarray]:
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    rng = np.random.default_rng(extent)
    return graph, rng.normal(size=(graph.n_nodes, 2))


@pytest.mark.parametrize("extent", [16, 32, 64])
def test_python_ising_ground_state_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    # The oracle, and the baseline the port is measured against. A random
    # per-node field is the case with content: a uniform one makes the
    # ferromagnetic ground state trivially all-aligned.
    graph, field_values = _problem(extent)

    _, energy = benchmark(ising_ground_state, graph, field_values)

    assert np.isfinite(energy)


@pytest.mark.parametrize("extent", [16, 32, 64])
def test_rust_ising_ground_state_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    graph, field_values = _problem(extent)

    _, energy = benchmark(maxflow_rust.ising_ground_state, graph, field_values)

    assert np.isfinite(energy)
