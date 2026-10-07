"""The package's minimum cut against PyMaxflow's, run in a subprocess (issue #973).

Kolmogorov's Boykov--Kolmogorov code, sharing none with Dinic or
`src/maxflow.rs`. `lattice_rung(side, 2, seed=973)` at 10², 71², 142²: same
configuration from both backends, so the same energy bitwise. Directed
asymmetric networks: flow within 1e-12 relative, source side node for node.
Control: doubled capacities double the value; dropped back arcs match the
Rust kernel. With labels forbidden (#1139, #1274), PyMaxflow's own grid
expansion, `aexpansion_grid`, on the finite stand-in `_forbidden.stand_in`
states: neither expansion holds a forbidden label, PyMaxflow's labelling is a
fixed point of the package's move, and on 13 enumerable instances both reach
the constrained minimum within 1e-12. Runtime goal: `test_goals.py`.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from sal.backend import Backend
from sal.search import maxflow
from sal.search.ground_state import lattice_rung
from sal.search.maxflow import rust as maxflow_rust
from sal.sim.graph import PottsGraph
from sal.validation import pymaxflow

from tests._frameworks import requires
from tests._rows import every_row, every_value
from tests.validation._forbidden import check_expansion, enumerable, lattices

pytestmark = [
    pytest.mark.validation,
    requires("pymaxflow"),
]

#: Relative agreement of two flow values summed in different orders.
FLOW_RTOL = 1e-12


def _network(seed: int, n_nodes: int, n_edges: int) -> maxflow.FlowNetwork:
    """A random directed network, terminals 0 and 1, arcs at both terminals."""
    rng = np.random.default_rng(seed)
    tail = rng.integers(0, n_nodes, size=n_edges)
    head = rng.integers(0, n_nodes, size=n_edges)
    keep = tail != head
    tail, head = tail[keep], head[keep]
    capacity = rng.exponential(1.0, size=tail.size)
    reverse = np.where(
        rng.random(tail.size) < 0.5, 0.0, rng.exponential(1.0, tail.size)
    )
    return maxflow.FlowNetwork.from_arcs(n_nodes, tail, head, capacity, reverse)


@pytest.mark.oracle
def test_the_ground_state_is_pymaxflows_node_for_node() -> None:
    def check(side: int) -> None:
        rung = lattice_rung(side, 2, seed=973)
        theirs, cut = pymaxflow.ising_ground_state(rung.graph, rung.field)
        backends = [Backend.RUST] if side > 71 else [Backend.PYTHON, Backend.RUST]
        for backend in backends:
            ours = maxflow.ising_ground_state(rung.graph, rung.field, backend=backend)
            assert np.array_equal(ours.configuration, theirs.configuration), backend
            assert ours.energy == theirs.energy, backend
        # The cut's value is the energy's reduction, read back through the
        # package's own arithmetic.
        reduced = maxflow.cut_energy(rung.graph, rung.field, cut.value)
        assert reduced == pytest.approx(theirs.energy, rel=FLOW_RTOL)

    every_value([10, 71, 142], check)


@pytest.mark.oracle
def test_a_directed_network_cuts_to_the_same_value_and_side() -> None:
    def check(seed: int) -> None:
        network = _network(seed, 60, 400)
        theirs = pymaxflow.min_cut(network, 0, 1)
        rust = maxflow_rust.min_cut(network, 0, 1)
        dinic = maxflow.max_flow(_network(seed, 60, 400), 0, 1, backend=Backend.PYTHON)
        for ours in (rust, dinic):
            assert ours.value == pytest.approx(theirs.value, rel=FLOW_RTOL)
            assert np.array_equal(ours.source_side, theirs.source_side)
        assert theirs.value > 0.0

    every_value([0, 1, 2, 3], check)


@pytest.mark.oracle
def test_the_adapter_reads_the_capacities_it_is_given() -> None:
    # Doubling every capacity doubles the flow, exactly: each term of the sum
    # is doubled without rounding. Dropping the back arcs moves it, against
    # the Rust kernel on the same network, so the reverse capacities are read.
    network = _network(4, 60, 400)
    before = pymaxflow.min_cut(network, 0, 1).value
    paired = network.as_arrays()
    ends = paired.arcs.reshape(-1, 2)
    doubled = maxflow.FlowNetwork.from_arcs(
        network.n_nodes,
        ends[:, 0],
        ends[:, 1],
        2.0 * paired.capacity,
        2.0 * paired.reverse,
    )
    assert pymaxflow.min_cut(doubled, 0, 1).value == 2.0 * before
    one_way = maxflow.FlowNetwork.from_arcs(
        network.n_nodes,
        ends[:, 0],
        ends[:, 1],
        paired.capacity,
        np.zeros_like(paired.reverse),
    )
    theirs = pymaxflow.min_cut(one_way, 0, 1).value
    assert theirs == pytest.approx(
        maxflow_rust.min_cut(one_way, 0, 1).value, rel=FLOW_RTOL
    )
    assert theirs < before


def _pymaxflow(graph: PottsGraph, field: np.ndarray, n_states: int) -> np.ndarray:
    return pymaxflow.alpha_expansion(graph, field, n_states).labelling


@pytest.mark.oracle
def test_with_forbidden_labels_both_expansions_reach_the_constrained_minimum() -> None:
    # Issue #1274, on #1139's forbidden labels. Measured 2026-10-06: all 13
    # instances reach the enumerated minimum, both, with the same labelling.
    every_row(enumerable(), partial(check_expansion, theirs=_pymaxflow, exact=True))


@pytest.mark.experiment
def test_with_forbidden_labels_pymaxflows_labelling_is_a_fixed_point_of_the_package_move() -> (
    None
):
    # Same start, the per-site cheapest label, and the same label order. Measured
    # 2026-10-06: the labelling is the package's at 16², and at 71² its energy
    # is 0.03% (q = 3) and 0.13% (q = 10) above the package's.
    every_row(lattices(), partial(check_expansion, theirs=_pymaxflow, exact=False))
