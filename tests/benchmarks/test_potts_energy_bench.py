"""The one Potts energy, at the size the consolidation had to not slow down.

Correctness is pinned in `tests/regression/sim/test_potts_energy.py`; this
measures. Four implementations scored a labelling before issue #277 and the
fastest of them --- `search.maxflow.energy`'s gather and dot product over the
edges --- is the one that survived, so both entries below run the same
arithmetic and the second adds one call and a field widening to it. The
sampler's entry is the one that moved: it evaluated
`likelihood.potts.log_weights`, a Python-level term per edge.

The cell is `profile_hotpaths.py`'s: 32x32 periodic at two states, 64
configurations, which is where issue #341 measured the edge loop at 92% of
the self time.
"""

from __future__ import annotations

import math

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.maxflow import energy as cut_energy
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.potts import energies

EXTENT = 32
CONFIGURATIONS = 64
FIELD = np.array([0.4, -0.4])


def _problem() -> tuple[object, np.ndarray]:
    graph = lattice_graph((EXTENT, EXTENT), BoundaryCondition.PERIODIC, 0.4)
    states = np.random.default_rng(17).integers(
        0, 2, size=(CONFIGURATIONS, graph.n_nodes)
    )
    return graph, states


def test_the_block_energy(benchmark: BenchmarkFixture) -> None:
    graph, states = _problem()

    realized = benchmark(energies, graph, FIELD, states)

    # Shape and finiteness only: this file measures, it does not validate.
    assert realized.shape == (CONFIGURATIONS,)
    assert math.isfinite(float(realized.sum()))


def test_the_two_state_entry_point(benchmark: BenchmarkFixture) -> None:
    graph, states = _problem()

    realized = benchmark(cut_energy, graph, FIELD, states)

    assert realized.shape == (CONFIGURATIONS,)
    assert math.isfinite(float(realized.sum()))
