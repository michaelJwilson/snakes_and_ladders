"""Regression tests for :mod:`phylo.sim.graph`.

Node and edge counts are checked against closed-form combinatorics -- not
against a second traversal of the same graph -- across 1-D, 2-D and 3-D
shapes and both boundary conditions.
"""

from __future__ import annotations

import pytest
from phylo.sim.graph import BoundaryCondition, PottsGraph, lattice_graph


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(4,), (5,), (3, 3), (3, 4), (2, 2, 2), (2, 3, 4)])
def test_open_lattice_node_and_edge_counts_match_the_closed_form(
    shape: tuple[int, ...],
) -> None:
    graph = lattice_graph(shape, boundary=BoundaryCondition.OPEN, coupling=0.5)
    n_nodes = 1
    for extent in shape:
        n_nodes *= extent
    assert graph.n_nodes == n_nodes

    expected_edges = sum(
        (shape[dim] - 1) * (n_nodes // shape[dim]) for dim in range(len(shape))
    )
    assert len(graph.edges) == expected_edges
    assert len(graph.coupling) == expected_edges
    assert set(graph.coupling) == {0.5}


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(4,), (5,), (3, 3), (3, 4), (3, 3, 3)])
def test_periodic_lattice_node_and_edge_counts_match_the_closed_form(
    shape: tuple[int, ...],
) -> None:
    # Extents are kept >= 3 here: at extent 2 a periodic dimension's "+1" and
    # "-1" neighbour coincide, so the count below (ndim * n_nodes) still
    # holds, but as a doubled bond rather than as len(set(edges)) -- a
    # distinct claim tested separately.
    graph = lattice_graph(shape, boundary=BoundaryCondition.PERIODIC, coupling=1.0)
    n_nodes = 1
    for extent in shape:
        n_nodes *= extent
    assert graph.n_nodes == n_nodes
    assert len(graph.edges) == len(shape) * n_nodes


@pytest.mark.edge_case
def test_a_periodic_dimension_of_extent_two_doubles_the_bond() -> None:
    graph = lattice_graph((2,), boundary=BoundaryCondition.PERIODIC, coupling=1.0)
    assert graph.n_nodes == 2
    assert graph.edges == ((0, 1), (1, 0))


@pytest.mark.oracle
def test_every_node_appears_in_the_expected_number_of_edges() -> None:
    # Interior nodes of an open 3x3 grid have degree 4; corners have degree 2.
    graph = lattice_graph((3, 3), boundary=BoundaryCondition.OPEN, coupling=1.0)
    degree = [0] * graph.n_nodes
    for a, b in graph.edges:
        degree[a] += 1
        degree[b] += 1
    assert degree[4] == 4, "the centre of a 3x3 grid has 4 neighbours"
    assert degree[0] == 2, "a corner of a 3x3 grid has 2 neighbours"


@pytest.mark.structural
def test_a_1d_open_chain_is_recognized_and_a_ring_is_not() -> None:
    assert lattice_graph(
        (5,), boundary=BoundaryCondition.OPEN, coupling=0.5
    ).is_open_chain()
    assert not lattice_graph(
        (5,), boundary=BoundaryCondition.PERIODIC, coupling=0.5
    ).is_open_chain()
    assert not lattice_graph(
        (3, 3), boundary=BoundaryCondition.OPEN, coupling=0.5
    ).is_open_chain()


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((), "at least one dimension"),
        ((1, 3), "must be >= 2"),
    ],
)
def test_an_invalid_lattice_specification_is_refused(
    shape: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        lattice_graph(shape, boundary=BoundaryCondition.OPEN, coupling=1.0)


@pytest.mark.edge_case
def test_a_graph_whose_coupling_does_not_match_its_edges_is_refused() -> None:
    # Every consumer indexes the coupling array by edge position, so a
    # mismatch is a silently wrong energy rather than an IndexError.
    with pytest.raises(ValueError, match="one per edge"):
        PottsGraph(n_nodes=3, edges=((0, 1), (1, 2)), coupling=(0.5,))


@pytest.mark.edge_case
def test_a_graph_whose_edge_names_a_missing_node_is_refused() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        PottsGraph(n_nodes=2, edges=((0, 2),), coupling=(0.5,))
