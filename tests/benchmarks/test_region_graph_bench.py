"""What the plaquette regions cost, beside what `belief_propagation` costs.

Correctness is pinned in `tests/regression/likelihood/test_region_graph.py`,
per the repository's division of labour between the two directories.

The point of measuring both here is that the accuracy claim of issue #689 has
two halves and only one of them is a ratio. A plaquette region holds
``q ** 4 = 81`` entries against a pair's ``q ** 2 = 9``, and it takes more
sweeps to settle, so the accuracy is bought at a price --- and where the price
is unbounded, because the messages do not settle at all, the module refuses.
Both cases are parameterized below so a regression in either shows.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.region_graph import (
    RegionGraph,
    bethe_region_graph,
    generalized_belief_propagation,
    lattice_plaquettes,
    region_graph,
)
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

FIELD = np.array([0.3, -0.7, 0.15])


def _regions(shape: tuple[int, int], coupling: float, plaquette: bool) -> RegionGraph:
    """The region graph of one lattice, pairwise or by unit cell."""
    gauge = FIELD - np.log(float(np.exp(FIELD).sum()))
    graph = from_potts(lattice_graph(shape, BoundaryCondition.OPEN, coupling), gauge)
    if plaquette:
        return region_graph(graph, lattice_plaquettes(shape))
    return bethe_region_graph(graph)


@pytest.mark.parametrize(
    ("shape", "coupling", "plaquette", "damping"),
    [
        ((3, 3), 0.6, False, 0.5),  # the pairwise baseline on the enumerable case
        ((3, 3), 0.6, True, 0.7),  # 3.5x the wall clock for 1,300x the accuracy
        ((6, 4), 0.25, False, 0.5),  # at size, where enumeration cannot reach
        ((6, 4), 0.25, True, 0.7),  # the weakest coupling the plaquettes settle at
    ],
)
def test_generalized_belief_propagation_benchmark(
    benchmark: BenchmarkFixture,
    shape: tuple[int, int],
    coupling: float,
    plaquette: bool,
    damping: float,
) -> None:
    regions = _regions(shape, coupling, plaquette)

    settled = benchmark(
        generalized_belief_propagation, regions, damping=damping, max_iterations=4_000
    )

    assert settled.residual <= 1e-12
