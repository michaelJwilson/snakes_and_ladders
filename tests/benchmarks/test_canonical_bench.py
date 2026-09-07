"""Benchmarks for the canonical constructions and the path enumeration.

Correctness is pinned in `tests/regression/`, per the repo's division of
labor between the two directories.

The one number worth watching here is the path enumeration: it is `k ** T`
paths times `T` sites, so its cost doubles with every observation added and
`MAX_ENUMERABLE_PATHS` is the wall that stops a test from being killed for
memory instead of failing with a stated limit.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.hmm_paths import enumerate_hidden_paths
from snakes_and_ladders.sim.canonical import (
    AMBIGUOUS_OBSERVATIONS,
    ambiguous_hmm,
    frustrated_triangular_lattice,
    planted_spin_glass,
)


@pytest.mark.parametrize("extent", [8, 24, 64])
def test_frustrated_triangular_lattice_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    graph = benchmark(frustrated_triangular_lattice, (extent, extent))

    assert len(graph.edges) == 3 * graph.n_nodes


@pytest.mark.parametrize("n_nodes", [50, 200, 800])
def test_planted_spin_glass_benchmark(
    benchmark: BenchmarkFixture, n_nodes: int
) -> None:
    # Dominated by `erdos_renyi_graph`'s `O(n^2)` pair draw, not by the
    # per-edge sign assignment, so this tracks that choice rather than the
    # planting.
    rng = np.random.default_rng(1)

    instance = benchmark(planted_spin_glass, n_nodes, 4.0, 0.2, rng)

    assert instance.graph.n_nodes == n_nodes


@pytest.mark.parametrize("length", [5, 9, 13])
def test_enumerate_hidden_paths_benchmark(
    benchmark: BenchmarkFixture, length: int
) -> None:
    # Doubling with each added site is the point of measuring it: the fixture
    # runs at length 5, and the growth here is what justifies refusing a
    # longer sequence rather than attempting one.
    params = ambiguous_hmm()
    observations = np.resize(AMBIGUOUS_OBSERVATIONS, length)

    result = benchmark(enumerate_hidden_paths, params, observations)

    assert result.viterbi.shape == (length,)
