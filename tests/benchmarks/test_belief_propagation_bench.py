"""Benchmarks for belief propagation and its two exact oracles.

Correctness is pinned in `tests/regression/likelihood/`, per the repo's
division of labor between the two directories.

The three together are the reason BP exists as more than an exercise. Each
scales differently in the same lattice, and the benchmark is where that shows:
enumeration is exponential in the *site count*, the strip transfer matrix is
exponential in the *width* alone, and BP is linear in the edge count times the
sweeps it needs. Ranking them in seconds on CI hardware is what `DEV.md`
forbids in a caption, not what a benchmark is for.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.belief_propagation import belief_propagation
from snakes_and_ladders.likelihood.potts import enumerate_potts, strip_log_partition
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

FIELD = np.array([0.3, -0.7, 0.15])


@pytest.mark.parametrize(
    ("shape", "coupling"),
    [
        ((6, 4), 0.25),  # weak coupling: converges in tens of sweeps
        ((6, 4), 0.875),  # at the Bethe error peak: the slow case
        ((10, 6), 0.5),  # beyond what enumeration can reach at all
    ],
)
def test_belief_propagation_benchmark(
    benchmark: BenchmarkFixture, shape: tuple[int, int], coupling: float
) -> None:
    graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)

    result = benchmark(belief_propagation, graph, FIELD)

    assert result.residual <= 1e-12


@pytest.mark.parametrize("shape", [(6, 4), (10, 6), (20, 6)])
def test_strip_transfer_matrix_benchmark(
    benchmark: BenchmarkFixture, shape: tuple[int, int]
) -> None:
    # Cost is `k ** (2 * width)` per column and linear in the column count,
    # so the third case costs the same per step as the second and runs twice
    # as many steps -- the shape of the scaling this measures.
    realized = benchmark(strip_log_partition, shape, BoundaryCondition.OPEN, 0.5, FIELD)

    assert np.isfinite(realized)


@pytest.mark.parametrize("shape", [(3, 3), (5, 2)])
def test_enumeration_benchmark(
    benchmark: BenchmarkFixture, shape: tuple[int, int]
) -> None:
    graph = lattice_graph(shape, BoundaryCondition.OPEN, 0.5)

    exact = benchmark(enumerate_potts, graph, FIELD)

    assert np.isfinite(exact.log_partition)
