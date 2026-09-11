"""Max flow, and the exact ground state it certifies.

Three kinds of reference, which is the point of the ticket: this is the first
place in the repository where a discrete optimum is *proved* rather than
enumerated, and enumeration alone would leave it capped at twenty sites.

1. Exhaustive enumeration, where it fits, at exact equality --- a ground state
   is a combinatorial minimum, not a float comparison.
2. Two analytic corners, at sizes far past what enumeration reaches: with no
   field the ground state is all-aligned at energy `-J |E|`, and with no
   coupling every site independently takes its better state.
3. The max-flow min-cut theorem, as a self-check rather than a second
   implementation: the flow value must equal the capacity of the cut induced
   by residual reachability.

The Rust kernel is pinned against the Python oracle at exact equality of
energy. The *configuration* may legitimately differ where the minimum is
degenerate, so the energy is what is compared.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest
from snakes_and_ladders.search import maxflow_rust
from snakes_and_ladders.search.maxflow import (
    FlowNetwork,
    energy,
    ising_ground_state,
    max_flow,
    site_field,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

# The Python blocking flow recurses to the depth of the level graph. The Rust
# port uses an explicit stack and needs no such raise, which is one of the two
# reasons it exists.
sys.setrecursionlimit(50_000)

FIELD = np.array([0.35, -0.25])


def _enumerated_minimum(graph: PottsGraph, field_values: np.ndarray) -> float:
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    return float(energy(graph, field_values, configurations).min())


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 3), (4, 4)])
@pytest.mark.parametrize("coupling", [0.0, 0.4, 1.2])
def test_the_cut_finds_the_enumerated_minimum_with_a_per_node_field(
    shape: tuple[int, int], coupling: float
) -> None:
    # A per-node field is the case with content. Under the *uniform* field the
    # rest of the repository uses, every coupling favours agreement and every
    # site prefers the same state, so the ferromagnetic ground state is
    # `argmax(h)` everywhere and a cut is an expensive way to say so.
    rng = np.random.default_rng(0)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)

    for _ in range(3):
        field_values = rng.normal(size=(graph.n_nodes, 2))
        _, realized = ising_ground_state(graph, field_values)

        assert realized == pytest.approx(
            _enumerated_minimum(graph, field_values), abs=1e-12
        )


@pytest.mark.oracle
def test_the_uniform_field_energy_is_the_negated_model_log_weight() -> None:
    # The per-node generalization must reduce to the model the rest of the
    # repository fits, or this is solving a different problem accurately.
    from snakes_and_ladders.likelihood.potts import log_weights

    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)
    configurations = np.array(
        list(itertools.product(range(2), repeat=9)), dtype=np.int64
    )

    realized = energy(graph, FIELD, configurations)

    # Relative rather than exact: `energy` sums its edge terms in one ``dot``
    # (issue #341) where `log_weights` adds them left to right in edge order,
    # so the two differ by rounding and not by model.
    np.testing.assert_allclose(
        realized, -log_weights(graph, FIELD, configurations), rtol=1e-12, atol=0.0
    )


@pytest.mark.oracle
def test_a_zero_field_ground_state_is_aligned_at_the_analytic_energy() -> None:
    # Past enumeration: 16 sites is 65,536 configurations, but the answer is
    # known in closed form, so the check does not depend on the size.
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, 0.7)

    configuration, realized = ising_ground_state(graph, np.zeros(2))

    assert len(set(configuration.tolist())) == 1
    assert realized == pytest.approx(-0.7 * len(graph.edges), abs=1e-12)


@pytest.mark.oracle
def test_a_zero_coupling_ground_state_follows_the_field_site_by_site() -> None:
    # With no bonds the sites decouple, so each independently takes its better
    # state and the answer is `argmax` per node -- again with no enumeration.
    rng = np.random.default_rng(3)
    graph = lattice_graph((6, 6), BoundaryCondition.OPEN, 0.0)
    field_values = rng.normal(size=(graph.n_nodes, 2))

    configuration, realized = ising_ground_state(graph, field_values)

    np.testing.assert_array_equal(configuration, field_values.argmax(axis=1))
    assert realized == pytest.approx(-field_values.max(axis=1).sum(), abs=1e-12)


@pytest.mark.mathematical
def test_the_flow_value_equals_the_capacity_of_the_cut_it_induces() -> None:
    # The max-flow min-cut theorem, as a self-check. Nothing here searches for
    # a cut: the source side is residual reachability on termination, and the
    # theorem says that set *is* a minimum cut. If it were not, this equality
    # would fail without any second implementation to compare against.
    rng = np.random.default_rng(11)
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 0.6)
    field_values = rng.normal(size=(graph.n_nodes, 2))
    values = site_field(graph, field_values)

    source, sink = graph.n_nodes, graph.n_nodes + 1
    network = FlowNetwork(n_nodes=graph.n_nodes + 2)
    cost = -values
    offsets = cost.min(axis=1)
    original: list[tuple[int, int, float]] = []
    for node in range(graph.n_nodes):
        network.add_edge(source, node, float(cost[node, 1] - offsets[node]))
        original.append((source, node, float(cost[node, 1] - offsets[node])))
        network.add_edge(node, sink, float(cost[node, 0] - offsets[node]))
        original.append((node, sink, float(cost[node, 0] - offsets[node])))
    for (first, second), coupling in zip(graph.edges, graph.coupling, strict=True):
        network.add_edge(first, second, coupling, reverse=coupling)
        original.append((first, second, coupling))
        original.append((second, first, coupling))

    cut = max_flow(network, source, sink)

    crossing = sum(
        capacity
        for tail, head, capacity in original
        if cut.source_side[tail] and not cut.source_side[head]
    )
    assert crossing == pytest.approx(cut.value, rel=1e-11)


@pytest.mark.oracle
@pytest.mark.parametrize("extent", [4, 8, 12])
def test_the_rust_kernel_reproduces_the_python_oracle_exactly(extent: int) -> None:
    # Exact equality of energy, not a tolerance: this is a combinatorial
    # minimum. The configuration itself may differ where the minimum is
    # degenerate, which is why the energy is what is compared.
    rng = np.random.default_rng(extent)
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    field_values = rng.normal(size=(graph.n_nodes, 2))

    _, expected = ising_ground_state(graph, field_values)
    _, realized = maxflow_rust.ising_ground_state(graph, field_values)

    assert realized == pytest.approx(expected, abs=1e-12)


@pytest.mark.oracle
def test_the_rust_min_cut_reproduces_a_hand_computed_value_and_cut() -> None:
    # Two disjoint paths carry 2 each; the cross edge carries a third unit a
    # greedy first path would have blocked. The minimum cut is the two arcs
    # out of the source, 3 + 2 = 5, and it leaves only the source reachable.
    network = FlowNetwork(n_nodes=4)
    for (tail, head), capacity in zip(
        [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)],
        [3.0, 2.0, 2.0, 3.0, 1.0],
        strict=True,
    ):
        network.add_edge(tail, head, capacity)

    cut = maxflow_rust.min_cut(network, 0, 3)

    assert cut.value == pytest.approx(5.0)
    assert cut.source_side.tolist() == [True, False, False, False]


@pytest.mark.oracle
@pytest.mark.parametrize("n_nodes", [6, 12, 24])
@pytest.mark.parametrize("seed", [528, 529])
def test_the_rust_min_cut_returns_the_python_cut_on_seeded_networks(
    n_nodes: int, seed: int
) -> None:
    # The claim issue #528 rests on: the side the binding now returns is the
    # side the oracle computes, arc for arc, on a random network with back
    # capacities -- not merely a set of the same capacity. Two networks are
    # built from the same draws because the pure solver consumes the one it
    # is given, turning its capacities into residual capacities.
    rng = np.random.default_rng(seed)
    tails = rng.integers(0, n_nodes, size=4 * n_nodes)
    heads = rng.integers(0, n_nodes, size=4 * n_nodes)
    forward = rng.uniform(0.1, 3.0, size=4 * n_nodes)
    reverse = rng.uniform(0.0, 1.0, size=4 * n_nodes)
    networks = [FlowNetwork(n_nodes=n_nodes) for _ in range(2)]
    for network in networks:
        for tail, head, capacity, back in zip(
            tails, heads, forward, reverse, strict=True
        ):
            if tail != head:
                network.add_edge(int(tail), int(head), float(capacity), float(back))

    expected = max_flow(networks[0], 0, n_nodes - 1)
    realized = maxflow_rust.min_cut(networks[1], 0, n_nodes - 1)

    assert realized.value == pytest.approx(expected.value, rel=1e-12)
    assert realized.source_side.tolist() == expected.source_side.tolist()


@pytest.mark.edge_case
def test_the_rust_min_cut_leaves_the_network_it_was_given_alone() -> None:
    # The one contract that differs from the pure implementation, so it is
    # asserted rather than only documented: the residual graph lives in Rust.
    network = FlowNetwork(n_nodes=3)
    network.add_edge(0, 1, 2.0)
    network.add_edge(1, 2, 1.0)
    before = list(network.capacity)

    assert maxflow_rust.min_cut(network, 0, 2).value == pytest.approx(1.0)
    assert network.capacity == before


@pytest.mark.edge_case
def test_a_negative_coupling_is_refused_by_both_implementations() -> None:
    # The submodularity boundary. Past it the ground state is NP-hard and no
    # cut computes it, so both return nothing rather than a lattice-shaped
    # wrong answer.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.5)

    with pytest.raises(ValueError, match="non-submodular"):
        ising_ground_state(graph, FIELD)
    with pytest.raises(ValueError, match="non-negative"):
        maxflow_rust.ising_ground_state(graph, FIELD)


@pytest.mark.edge_case
def test_more_than_two_states_is_refused_and_names_alpha_expansion() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match="two-state case only"):
        ising_ground_state(graph, np.zeros(3))


@pytest.mark.edge_case
def test_a_per_node_field_of_the_wrong_shape_is_refused() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match=r"must be \(9, 2\)"):
        ising_ground_state(graph, np.zeros((4, 2)))


@pytest.mark.edge_case
def test_coincident_terminals_are_refused() -> None:
    network = FlowNetwork(n_nodes=3)

    with pytest.raises(ValueError, match="source and sink must differ"):
        max_flow(network, 1, 1)


@pytest.mark.edge_case
def test_a_negative_capacity_is_refused() -> None:
    network = FlowNetwork(n_nodes=2)

    with pytest.raises(ValueError, match="capacities must be non-negative"):
        network.add_edge(0, 1, -1.0)


@pytest.mark.edge_case
def test_a_disconnected_sink_carries_no_flow() -> None:
    network = FlowNetwork(n_nodes=3)
    network.add_edge(0, 1, 4.0)

    cut = max_flow(network, 0, 2)

    assert cut.value == pytest.approx(0.0)
    assert not bool(cut.source_side[2])
