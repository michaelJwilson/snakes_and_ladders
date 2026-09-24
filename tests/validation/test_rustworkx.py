"""Graph construction, topology equality and cluster labelling against rustworkx (issue #976).

In a subprocess (`validation.rustworkx`; #322, #376). Open lattice =
`grid_graph`, chain = `path_graph`, periodic = grid plus wraparound;
`erdos_renyi_graph` against `undirected_gnp_random_graph` at both ends of `p`
and on 400 draws' mean edges within four errors; multigraph order and
couplings, degree 6 on the periodic triangular lattice; Robinson--Foulds
equality iff isomorphism; union-find against `connected_components` at 16²,
71², 142², three temperatures. Dropped: `from_rustworkx` (gone) and the
networkx cut checks (PyMaxflow pins the cut, #973).
"""

from __future__ import annotations

import itertools
from itertools import product

import numpy as np
import pytest
from snakes_and_ladders.sample.potts_mcmc import find_root, union_roots
from snakes_and_ladders.sample.potts_mcmc.sweeps import bond_probability
from snakes_and_ladders.sim.canonical import planted_spin_glass
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    erdos_renyi_graph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.potts import critical_coupling
from snakes_and_ladders.sim.topology import (
    Topology,
    enumerate_topologies,
    nni_neighbours,
    random_topology,
    robinson_foulds,
)
from snakes_and_ladders.sim.tree import edges as tree_edges
from snakes_and_ladders.sim.tree import preorder
from snakes_and_ladders.validation import rustworkx

from tests._frameworks import requires
from tests._rows import every_row, every_value

pytestmark = [
    pytest.mark.validation,
    requires("rustworkx"),
]

NAMES = ("t1", "t2", "t3", "t4", "t5", "t6")


def _pairs(edges: np.ndarray | tuple[tuple[int, int], ...]) -> set[frozenset[int]]:
    return {frozenset((int(a), int(b))) for a, b in edges}


@pytest.mark.oracle
def test_the_open_square_lattice_is_rustworkxs_grid() -> None:
    def check(shape: tuple[int, int]) -> None:
        ours = lattice_graph(shape, BoundaryCondition.OPEN, 1.0)
        theirs = rustworkx.grid_edges(shape)
        assert _pairs(theirs) == _pairs(ours.edges)
        assert len(ours.edges) == theirs.shape[0]

    every_value([(2, 2), (3, 4), (5, 3), (6, 6)], check)


@pytest.mark.oracle
def test_the_open_chain_is_rustworkxs_path() -> None:
    def check(length: int) -> None:
        ours = lattice_graph((length,), BoundaryCondition.OPEN, 1.0)
        theirs = rustworkx.path_edges(length)
        assert _pairs(theirs) == _pairs(ours.edges)
        assert len(ours.edges) == theirs.shape[0]

    every_value([2, 5, 9], check)


@pytest.mark.oracle
def test_the_periodic_lattice_is_the_grid_plus_its_wraparound_edges() -> None:
    rows, columns = 4, 5
    periodic = lattice_graph((rows, columns), BoundaryCondition.PERIODIC, 1.0)
    grid = _pairs(rustworkx.grid_edges((rows, columns)))
    wraparound = {
        frozenset((row * columns + columns - 1, row * columns)) for row in range(rows)
    } | {
        frozenset(((rows - 1) * columns + column, column)) for column in range(columns)
    }
    assert _pairs(periodic.edges) == grid | wraparound
    assert len(periodic.edges) == len(grid) + len(wraparound)


@pytest.mark.oracle
def test_the_random_graph_agrees_with_rustworkxs_at_both_ends_and_in_mean() -> None:
    rng = np.random.default_rng(0)
    for probability in (0.0, 1.0):
        ours = erdos_renyi_graph(7, probability, 1.0, rng)
        _, theirs = rustworkx.gnp(7, probability, [0])
        assert _pairs(ours.edges) == _pairs(theirs)
    n_nodes, probability, draws = 12, 0.3, 400
    rng = np.random.default_rng(3)
    ours_counts = np.array(
        [
            len(erdos_renyi_graph(n_nodes, probability, 1.0, rng).edges)
            for _ in range(draws)
        ]
    )
    theirs_counts, _ = rustworkx.gnp(n_nodes, probability, list(range(draws)))
    pairs = n_nodes * (n_nodes - 1) // 2
    error = np.sqrt(2 * pairs * probability * (1 - probability) / draws)
    assert abs(ours_counts.mean() - theirs_counts.mean()) < 4 * error
    assert abs(ours_counts.mean() - pairs * probability) < 4 * error


def _graphs() -> list[PottsGraph]:
    return [
        lattice_graph((2, 2), BoundaryCondition.PERIODIC, 0.5),  # doubled bonds
        lattice_graph((4, 5), BoundaryCondition.OPEN, 0.3),
        lattice_graph((3, 3, 2), BoundaryCondition.PERIODIC, 0.7),
        triangular_lattice_graph((3, 4), BoundaryCondition.OPEN, -0.4),
        planted_spin_glass(40, 4.0, 0.2, np.random.default_rng(1)).graph,
    ]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "graph", _graphs(), ids=["2x2-periodic", "4x5", "3x3x2", "triangular", "glass"]
)
def test_a_multigraph_reads_back_our_edges_in_order(graph: PottsGraph) -> None:
    read = rustworkx.multigraph(graph.n_nodes, graph.edge_index, graph.edge_coupling)
    assert np.array_equal(read.edges, graph.edge_index)
    assert np.array_equal(read.weight, graph.edge_coupling)
    degree = np.bincount(graph.edge_index.reshape(-1), minlength=graph.n_nodes)
    assert np.array_equal(read.degree, degree)


@pytest.mark.oracle
def test_the_periodic_triangular_lattice_has_degree_six_by_rustworkx() -> None:
    rows, columns = 4, 5
    periodic = triangular_lattice_graph(
        (rows, columns), BoundaryCondition.PERIODIC, -0.6
    )
    read = rustworkx.multigraph(
        periodic.n_nodes, periodic.edge_index, periodic.edge_coupling
    )
    assert read.edges.shape[0] == 3 * rows * columns
    assert np.array_equal(read.degree, np.full(rows * columns, 6))


def _labelled(topology: Topology) -> tuple[np.ndarray, np.ndarray]:
    """The tree as node labels (taxon index, or -1 inside) and edges, in preorder."""
    nodes = list(preorder(topology))
    index = {node.name: position for position, node in enumerate(nodes)}
    labels = np.array(
        [NAMES.index(node.name) if node.is_leaf else -1 for node in nodes]
    )
    ends = np.array(
        [
            (index[parent.name], index[child.name])
            for parent, child in tree_edges(topology)
        ]
    )
    return labels, ends


@pytest.mark.oracle
def test_topology_equality_is_labelled_isomorphism() -> None:
    def check(n_taxa: int) -> None:
        topologies = list(enumerate_topologies(NAMES[:n_taxa]))
        pairs = list(itertools.combinations(range(len(topologies)), 2))
        theirs = rustworkx.isomorphic([_labelled(t) for t in topologies], pairs)
        ours = [robinson_foulds(topologies[a], topologies[b]) == 0 for a, b in pairs]
        assert list(theirs) == ours
        assert not any(ours)

    every_value([4, 5], check)


@pytest.mark.oracle
def test_a_topology_and_its_nni_neighbours_by_isomorphism() -> None:
    first = random_topology(NAMES, np.random.default_rng(376))
    second = random_topology(NAMES, np.random.default_rng(376))
    neighbours = list(nni_neighbours(first))
    graphs = [_labelled(t) for t in (first, second, *neighbours)]
    same = rustworkx.isomorphic(
        graphs, [(0, 1)] + [(k, k) for k in range(2, len(graphs))]
    )
    differ = rustworkx.isomorphic(graphs, [(0, k) for k in range(2, len(graphs))])
    assert robinson_foulds(first, second) == 0
    assert same.all()
    assert not differ.any()
    assert all(robinson_foulds(first, neighbour) > 0 for neighbour in neighbours)
    # Without the label matcher every six-taxon tree is one shape.
    assert rustworkx.isomorphic(graphs, [(0, 2)], match_labels=False).all()


def _bond_mask(side: int, beta: float, seed: int) -> tuple[PottsGraph, np.ndarray]:
    """The bonds a Swendsen--Wang step activates from a random 3-state labelling."""
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, critical_coupling(3))
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 3, graph.n_nodes)
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < bond_probability(graph, beta))
    return graph, graph.edge_index[active]


def _union_find(n_nodes: int, bonds: np.ndarray) -> np.ndarray:
    """The pass's own labelling, then each node's smallest fellow member."""
    parent = np.arange(n_nodes)
    for first, second in bonds.tolist():
        union_roots(parent, first, second)
    roots = np.array([find_root(parent, node) for node in range(n_nodes)])
    smallest = np.full(n_nodes, n_nodes)
    np.minimum.at(smallest, roots, np.arange(n_nodes))
    return np.asarray(smallest[roots])


@pytest.mark.oracle
def test_the_bond_pass_partitions_as_connected_components() -> None:
    def check(side: int, beta: float) -> None:
        graph, bonds = _bond_mask(side, beta, 976)
        ours = _union_find(graph.n_nodes, bonds)
        theirs = rustworkx.components(graph.n_nodes, bonds)
        assert np.array_equal(ours, theirs.labels)
        assert 1 < np.unique(ours).size < graph.n_nodes

    every_row(product([16, 71, 142], [0.5, 1.0, 1.5]), check)
