"""`PottsGraph` against `rustworkx`: a round trip, and the generators as oracles (issue #322).

`rustworkx` is the graph backend #389 scopes for the hot paths
(#242). What it can referee now is structure: its ``grid_graph`` and
``path_graph`` are independent constructions of the open lattices ours
builds, and its `G(n, p)` generator draws the same distribution ours does
from a different generator. What it cannot referee is an s-t cut: the
installed API carries a global ``stoer_wagner_min_cut`` and no maximum flow,
so `search.maxflow` is pinned here against `networkx`'s instead, which
arrives transitively through `torch`.

Everything here skips without the ``frameworks`` extra.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np
import pytest
from snakes_and_ladders.search.maxflow import (
    cut_energy,
    ising_ground_state,
)
from snakes_and_ladders.sim.canonical import planted_spin_glass
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    erdos_renyi_graph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.potts import energies, energy, site_field

rustworkx = pytest.importorskip("rustworkx")


def _pairs(edges: Iterable[tuple[int, int]]) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in edges}


def _graphs() -> list[PottsGraph]:
    return [
        lattice_graph((2, 2), BoundaryCondition.PERIODIC, 0.5),  # doubled bonds
        lattice_graph((4, 5), BoundaryCondition.OPEN, 0.3),
        lattice_graph((3, 3, 2), BoundaryCondition.PERIODIC, 0.7),
        triangular_lattice_graph((3, 4), BoundaryCondition.OPEN, -0.4),
        planted_spin_glass(40, 4.0, 0.2, np.random.default_rng(1)).graph,
    ]


# --- the round trip -------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "graph", _graphs(), ids=["2x2-periodic", "4x5", "3x3x2", "triangular", "glass"]
)
def test_a_graph_survives_the_round_trip_with_its_edge_order(graph: PottsGraph) -> None:
    converted = graph.to_rustworkx()
    assert converted.num_nodes() == graph.n_nodes
    assert converted.num_edges() == len(graph.edges)
    assert converted.multigraph
    back = PottsGraph.from_rustworkx(converted)
    assert back.n_nodes == graph.n_nodes
    assert back.edges == graph.edges
    assert back.coupling == graph.coupling
    # A PyGraph has no slot for these; a lattice read back is a graph.
    assert back.shape is None
    assert back.boundary is None


@pytest.mark.smoke
def test_the_converted_graph_carries_the_coupling_as_edge_data() -> None:
    graph = PottsGraph(
        n_nodes=3, edges=((0, 1), (1, 2), (0, 1)), coupling=(0.5, 0.7, 0.9)
    )
    converted = graph.to_rustworkx()
    assert list(converted.nodes()) == [0, 1, 2]
    assert list(converted.weighted_edge_list()) == [
        (0, 1, 0.5),
        (1, 2, 0.7),
        (0, 1, 0.9),
    ]


@pytest.mark.smoke
def test_a_graph_with_a_removed_node_is_refused() -> None:
    converted = lattice_graph((3, 3), BoundaryCondition.OPEN, 1.0).to_rustworkx()
    converted.remove_node(4)
    with pytest.raises(ValueError, match="without holes"):
        PottsGraph.from_rustworkx(converted)


@pytest.mark.smoke
def test_an_edge_carrying_no_coupling_is_refused() -> None:
    converted = rustworkx.PyGraph()
    converted.add_nodes_from(range(2))
    converted.add_edge(0, 1, "unweighted")
    with pytest.raises(ValueError, match="not a coupling"):
        PottsGraph.from_rustworkx(converted)


# --- the generators, refereed ---------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(2, 2), (3, 4), (5, 3), (6, 6)])
def test_the_open_square_lattice_is_rustworkx_s_grid_graph(
    shape: tuple[int, int],
) -> None:
    # Both index sites row-major, so the edge sets agree as unordered pairs
    # of the same integers, not merely up to isomorphism.
    ours = lattice_graph(shape, BoundaryCondition.OPEN, 1.0)
    theirs = rustworkx.generators.grid_graph(*shape)
    assert theirs.num_nodes() == ours.n_nodes
    assert _pairs(theirs.edge_list()) == _pairs(ours.edges)
    assert len(ours.edges) == theirs.num_edges()


@pytest.mark.oracle
@pytest.mark.parametrize("length", [2, 5, 9])
def test_the_open_chain_is_rustworkx_s_path_graph(length: int) -> None:
    ours = lattice_graph((length,), BoundaryCondition.OPEN, 1.0)
    theirs = rustworkx.generators.path_graph(length)
    assert _pairs(theirs.edge_list()) == _pairs(ours.edges)
    assert len(ours.edges) == theirs.num_edges()


@pytest.mark.oracle
def test_the_periodic_lattice_is_the_grid_plus_its_wraparound_edges() -> None:
    # rustworkx has no torus generator, so the periodic case is checked as
    # the open grid plus exactly the wraparound edges, which is what the
    # boundary condition adds and nothing else.
    shape = (4, 5)
    periodic = lattice_graph(shape, BoundaryCondition.PERIODIC, 1.0)
    grid = _pairs(rustworkx.generators.grid_graph(*shape).edge_list())
    rows, columns = shape
    wraparound = {
        frozenset((row * columns + columns - 1, row * columns)) for row in range(rows)
    } | {
        frozenset(((rows - 1) * columns + column, column)) for column in range(columns)
    }
    assert _pairs(periodic.edges) == grid | wraparound
    assert len(periodic.edges) == len(grid) + len(wraparound)


@pytest.mark.oracle
def test_the_random_graph_agrees_with_rustworkx_s_at_both_ends_of_p() -> None:
    rng = np.random.default_rng(0)
    for probability in (0.0, 1.0):
        ours = erdos_renyi_graph(7, probability, 1.0, rng)
        theirs = rustworkx.undirected_gnp_random_graph(7, probability, seed=0)
        assert _pairs(ours.edges) == _pairs(theirs.edge_list())


@pytest.mark.oracle
def test_the_random_graph_draws_the_edge_count_rustworkx_s_generator_draws() -> None:
    # Two generators of one distribution, seeded independently: the mean
    # edge counts of 400 draws each differ by less than four standard errors
    # of the difference. Deterministic under the seeds; a bound rather than
    # an equality because the streams are different by construction.
    n_nodes, probability, draws = 12, 0.3, 400
    rng = np.random.default_rng(3)
    ours = np.array(
        [
            len(erdos_renyi_graph(n_nodes, probability, 1.0, rng).edges)
            for _ in range(draws)
        ]
    )
    theirs = np.array(
        [
            rustworkx.undirected_gnp_random_graph(
                n_nodes, probability, seed=seed
            ).num_edges()
            for seed in range(draws)
        ]
    )
    pairs = n_nodes * (n_nodes - 1) // 2
    standard_error = np.sqrt(2 * pairs * probability * (1 - probability) / draws)
    assert abs(ours.mean() - theirs.mean()) < 4 * standard_error
    assert abs(ours.mean() - pairs * probability) < 4 * standard_error


# --- the cut, refereed by networkx where rustworkx has no s-t flow -------


@pytest.mark.smoke
def test_rustworkx_offers_no_s_t_flow_which_is_why_networkx_referees_it() -> None:
    # The guard for the docstring's claim: the day rustworkx grows a maximum
    # flow, this fails and the oracle below should move to it (#242).
    assert hasattr(rustworkx, "stoer_wagner_min_cut")
    assert not any("flow" in name.lower() for name in dir(rustworkx))


@pytest.mark.oracle
@pytest.mark.parametrize("extent", [8, 12, 16])
def test_the_ground_state_energy_matches_networkx_s_minimum_cut(extent: int) -> None:
    # The same reduction (`ising_ground_state`'s docstring) built as a
    # networkx flow network and cut by its preflow-push, against our Dinic:
    # the cut values agree, so the energies do, and networkx's partition
    # scores the same energy when evaluated by our energy function.
    nx = pytest.importorskip("networkx")
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    field_values = np.random.default_rng(extent).normal(size=(graph.n_nodes, 2))
    values = site_field(field_values, graph.n_nodes)
    cost = -values
    offsets = cost.min(axis=1)
    source, sink = "s", "t"
    network = nx.DiGraph()
    for node in range(graph.n_nodes):
        network.add_edge(source, node, capacity=float(cost[node, 1] - offsets[node]))
        network.add_edge(node, sink, capacity=float(cost[node, 0] - offsets[node]))
    for (first, second), coupling in graph.weighted_edges():
        network.add_edge(first, second, capacity=coupling)
        network.add_edge(second, first, capacity=coupling)
    cut_value, (source_side, _) = nx.minimum_cut(network, source, sink)
    configuration, ours = ising_ground_state(graph, field_values)
    theirs = np.array(
        [0 if node in source_side else 1 for node in range(graph.n_nodes)],
        dtype=np.int64,
    )
    assert ours == pytest.approx(cut_energy(graph, field_values, cut_value), abs=1e-9)
    assert ours == pytest.approx(float(energy(graph, values, theirs)), abs=1e-9)
    assert ours <= float(energy(graph, values, configuration)) + 1e-12


@pytest.mark.oracle
def test_the_networkx_cut_is_the_enumerated_minimum_where_enumeration_fits() -> None:
    # The referee is itself refereed once, at a size enumeration reaches, so
    # the pin above is against something known to be right and not merely
    # against a second library.
    nx = pytest.importorskip("networkx")
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.4)
    field_values = np.random.default_rng(9).normal(size=(graph.n_nodes, 2))
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    enumerated = float(energies(graph, field_values, configurations).min())
    _, ours = ising_ground_state(graph, field_values)
    assert ours == enumerated
    values = site_field(field_values, graph.n_nodes)
    cost = -values
    offsets = cost.min(axis=1)
    network = nx.DiGraph()
    for node in range(graph.n_nodes):
        network.add_edge("s", node, capacity=float(cost[node, 1] - offsets[node]))
        network.add_edge(node, "t", capacity=float(cost[node, 0] - offsets[node]))
    for (first, second), coupling in graph.weighted_edges():
        network.add_edge(first, second, capacity=coupling)
        network.add_edge(second, first, capacity=coupling)
    cut_value, _ = nx.minimum_cut(network, "s", "t")
    assert cut_energy(graph, field_values, cut_value) == pytest.approx(
        enumerated, abs=1e-9
    )


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(2, 2), (3, 4), (5, 3), (6, 6)])
def test_a_lattice_read_in_from_rustworkx_s_own_grid_is_the_one_we_build(
    shape: tuple[int, int],
) -> None:
    # `from_rustworkx` against a graph this repository did not construct:
    # `rustworkx.generators.grid_graph` builds the open square lattice, its
    # edges are weighted with the coupling, and what comes back must be the
    # `lattice_graph` of the same shape -- same node count, same edges as
    # unordered pairs, same coupling on each. The referee is the second
    # implementation, so a reader that transposed the node indexing or
    # dropped an edge fails here and not on a round trip through itself.
    coupling = 0.3
    theirs = rustworkx.generators.grid_graph(*shape)
    weighted = rustworkx.PyGraph(multigraph=True)
    weighted.add_nodes_from(range(theirs.num_nodes()))
    weighted.add_edges_from(
        [(first, second, coupling) for first, second in theirs.edge_list()]
    )

    read = PottsGraph.from_rustworkx(weighted)
    ours = lattice_graph(shape, BoundaryCondition.OPEN, coupling)

    assert read.n_nodes == ours.n_nodes
    assert _pairs(read.edges) == _pairs(ours.edges)
    assert set(read.coupling) == {coupling}
    assert len(read.edges) == len(ours.edges)
    # And the closed-form counts the lattice is defined by, read off the
    # graph that came back: (rows - 1) * columns + rows * (columns - 1)
    # edges, and every interior node of degree 4.
    rows, columns = shape
    assert len(read.edges) == (rows - 1) * columns + rows * (columns - 1)
    degree = [0] * read.n_nodes
    for first, second in read.edges:
        degree[first] += 1
        degree[second] += 1
    assert degree == [
        (2 if row in (0, rows - 1) else 3) + (0 if column in (0, columns - 1) else 1)
        for row in range(rows)
        for column in range(columns)
    ]


@pytest.mark.oracle
def test_the_converted_triangular_lattice_is_the_degree_six_torus_it_claims() -> None:
    # `to_rustworkx` refereed by rustworkx's own reading of what it produced,
    # against the closed form the construction states: a periodic triangular
    # lattice has three edges per site and every node of degree 6, and the
    # open one drops exactly the edges leaving the grid. Counted through
    # `rustworkx.PyGraph.degree`, which walks its own adjacency and not ours.
    rows, columns = 4, 5
    periodic = triangular_lattice_graph(
        (rows, columns), BoundaryCondition.PERIODIC, -0.6
    ).to_rustworkx()

    assert periodic.num_nodes() == rows * columns
    assert periodic.num_edges() == 3 * rows * columns
    assert [periodic.degree(node) for node in range(rows * columns)] == [6] * (
        rows * columns
    )

    opened = triangular_lattice_graph(
        (rows, columns), BoundaryCondition.OPEN, -0.6
    ).to_rustworkx()
    closed_form = (
        rows * (columns - 1) + (rows - 1) * columns + (rows - 1) * (columns - 1)
    )

    assert opened.num_edges() == closed_form
    # What the boundary condition adds, counted separately: one wrap per row,
    # one per column, and one diagonal per cell on either far edge.
    assert periodic.num_edges() - opened.num_edges() == rows + columns + (
        rows + columns - 1
    )
    assert max(opened.degree(node) for node in range(rows * columns)) == 6
    assert opened.degree(0) == 3
