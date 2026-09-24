"""The one Potts energy, at the size the consolidation had to not slow down.

Correctness is pinned in `tests/regression/sim/test_potts_energy.py`; this
measures. Four implementations scored a labelling before issue #277 and the
fastest of them --- the gather and dot product over the edges that was
`search.maxflow.energy`'s and is `sim.potts.energies` (issue #717), a pairwise
sum outside BLAS since issue #1044 --- is the one that survived, so both
entries below run the same arithmetic and the second adds one call and a
field widening to it. The sampler's entry is the one that moved: it evaluated
`likelihood.potts.log_weights`, a Python-level term per edge.

The cell is `profile_hotpaths.py`'s: 32x32 periodic at two states, 64
configurations, which is where issue #341 measured the edge loop at 92% of
the self time. The second cell is the release-q10 rung (5,041 sites, ten
states), where issue #1044 measured the edge term as a threaded gemv at 92
to 98% of an annealed run's wall: one labelling, as the annealed runs score
it after every step, and six, the tempering ladder's block.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.search.ground_state import N_REPLICAS
from snakes_and_ladders.search.potts_starts import spatio_rung
from snakes_and_ladders.sim.fixtures import fixture
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

    realized = benchmark(energies, graph, FIELD, states)

    assert realized.shape == (CONFIGURATIONS,)
    assert math.isfinite(float(realized.sum()))


@pytest.mark.parametrize("block", [1, N_REPLICAS])
def test_the_energy_at_the_release_rung(
    benchmark: BenchmarkFixture, block: int
) -> None:
    # The rung `docs/nb/potts_starts.ipynb` runs every solver on; `block`
    # labellings drawn from a fixed seed.
    rung = spatio_rung(fixture("spatio_only", "release").params, "release-q10")
    states = np.random.default_rng(1044).integers(
        0, rung.n_states, size=(block, rung.n_nodes)
    )

    realized = benchmark(energies, rung.graph, rung.field, states)

    assert realized.shape == (block,)
    assert math.isfinite(float(realized.sum()))


@pytest.mark.parametrize("extent", [8, 16, 32])
def test_log_weights_on_one_configuration_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    """One configuration, the shape the annealing loop scores per sweep (#598).

    :func:`~snakes_and_ladders.sample.potts_mcmc.anneal_potts` calls this once
    a step, so a per-edge Python loop here is paid once per edge per sweep.
    """
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 1.0)
    rng = np.random.default_rng(598)
    field = rng.normal(size=(graph.n_nodes, 3))
    state = rng.integers(0, 3, size=graph.n_nodes)[None]

    scored = benchmark(log_weights, graph, field, state)

    assert scored.shape == (1,)
