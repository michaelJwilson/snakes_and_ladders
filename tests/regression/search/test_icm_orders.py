"""ICM's checkerboard and residual sweep orders (issue #1073).

Referees: each order stops at a single-site local minimum, checked by trying
every single-site change; the checkerboard order is a colouring (no class
holds an edge) and runs bitwise on the compiled kernel and the Python sweep;
the residual order, which reorders from each sweep's labels, is the Python
route's alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.search.icm import (
    SweepOrder,
    colour_order,
    colouring,
    iterated_conditional_modes,
)
from sal.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    erdos_renyi_graph,
    lattice_graph,
    triangular_lattice_graph,
)
from sal.sim.potts import energy

GRAPH = triangular_lattice_graph((7, 6), BoundaryCondition.OPEN, 0.7)
FIELD = np.random.default_rng(1073).normal(0.0, 1.0, (GRAPH.n_nodes, 4))


def _local_minimum(labelling: np.ndarray) -> bool:
    base = energy(GRAPH, FIELD, labelling)
    for node in range(GRAPH.n_nodes):
        for state in range(4):
            moved = labelling.copy()
            moved[node] = state
            if energy(GRAPH, FIELD, moved) < base - 1e-12:
                return False
    return True


@pytest.mark.oracle
@pytest.mark.parametrize("order", list(SweepOrder))
def test_every_order_stops_at_a_single_site_local_minimum(order: SweepOrder) -> None:
    run = iterated_conditional_modes(
        GRAPH,
        FIELD,
        n_states=4,
        rng=np.random.default_rng(0),
        sweep_order=order,
        stop_when_clean=order is not SweepOrder.RANDOM,
        backend=Backend.PYTHON,
    )

    assert run.termination is not None
    assert run.termination.converged
    assert _local_minimum(run.labelling)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("graph", "colours"),
    [
        (lattice_graph((6, 5), BoundaryCondition.OPEN, 1.0), 2),
        (GRAPH, 4),
        (erdos_renyi_graph(60, 0.15, 1.0, np.random.default_rng(1073)), 6),
    ],
)
def test_the_checkerboard_classes_hold_no_edge(graph: PottsGraph, colours: int) -> None:
    classes = colouring(graph)
    edges = graph.edge_index
    order = colour_order(graph)

    assert int(classes.max()) + 1 == colours
    assert not bool(np.any(classes[edges[:, 0]] == classes[edges[:, 1]]))
    assert np.array_equal(np.sort(order), np.arange(graph.n_nodes))
    assert bool(np.all(np.diff(classes[order]) >= 0))
    assert np.array_equal(classes, colouring(graph, backend=Backend.PYTHON))


@pytest.mark.oracle
def test_the_checkerboard_order_is_bitwise_on_both_routes() -> None:
    runs = [
        iterated_conditional_modes(
            GRAPH,
            FIELD,
            n_states=4,
            rng=np.random.default_rng(3),
            sweep_order=SweepOrder.CHECKERBOARD,
            backend=backend,
        )
        for backend in (Backend.NUMBA, Backend.PYTHON)
    ]

    assert np.array_equal(runs[0].labelling, runs[1].labelling)
    assert runs[0].sweeps == runs[1].sweeps


@pytest.mark.smoke
def test_the_residual_order_refuses_the_compiled_kernel() -> None:
    with pytest.raises(ValueError, match="needs python"):
        iterated_conditional_modes(
            GRAPH,
            FIELD,
            n_states=4,
            rng=np.random.default_rng(0),
            sweep_order=SweepOrder.RESIDUAL,
        )
