"""The package's minimum cut against PyMaxflow's, run in a subprocess (issue #973).

PyMaxflow wraps Kolmogorov's own Boykov--Kolmogorov code and shares none with
`search.maxflow` (Dinic) or `src/maxflow.rs`. Referees:

- the two-state ferromagnet `lattice_rung(side, 2, seed=973)` at 10², 71² and
  142²: the same configuration node for node, from both backends, and so the
  same energy bitwise, since both are `sim.potts.energy` on one labelling;
- directed networks with asymmetric capacities and arcs at both terminals:
  the flow value within 1e-12 relative and the source side node for node,
  against the Python Dinic and the Rust kernel;
- a control: every capacity doubled doubles PyMaxflow's value exactly, and
  the back arcs dropped lowers it to the Rust kernel's value on the same
  network, so the adapter reads both capacities it is handed.

The runtime goal PyMaxflow sets is in `test_goals.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search import maxflow, maxflow_rust
from snakes_and_ladders.search.ground_state import lattice_rung
from snakes_and_ladders.validation import pymaxflow

from tests._frameworks import requires
from tests._rows import every_value

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
        dinic = maxflow.max_flow(_network(seed, 60, 400), 0, 1)
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
