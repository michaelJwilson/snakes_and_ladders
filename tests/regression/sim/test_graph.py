"""Regression tests for :mod:`snakes_and_ladders.sim.graph`.

Node and edge counts are checked against closed-form combinatorics -- not
against a second traversal of the same graph -- across 1-D, 2-D and 3-D
shapes and both boundary conditions.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph


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


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_derived_arrays_are_the_edges_and_couplings_the_graph_declares() -> None:
    """`edge_index` and `edge_coupling` restate the graph, in array form.

    The conversion every consumer used to write per call, done once. Asserted
    against the declaration rather than against a stored expectation, so the
    test cannot drift from what a `PottsGraph` says about itself.
    """
    graph = lattice_graph((4, 3), BoundaryCondition.OPEN, 0.75)

    assert graph.edge_index.shape == (len(graph.edges), 2)
    assert graph.edge_index.dtype == np.int64
    assert [tuple(pair) for pair in graph.edge_index.tolist()] == list(graph.edges)
    assert graph.edge_coupling.dtype == np.float64
    assert graph.edge_coupling.tolist() == list(graph.coupling)


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_graph_with_no_edges_still_has_two_columns() -> None:
    """Empty is ``(0, 2)``, not ``(0,)``.

    A caller indexing `edge_index[:, 0]` must not need a special case for the
    graph that happens to have no edges; `np.asarray(())` would give it one.
    """
    graph = PottsGraph(n_nodes=3, edges=(), coupling=())

    assert graph.edge_index.shape == (0, 2)
    assert graph.edge_index[:, 0].shape == (0,)
    assert graph.edge_coupling.shape == (0,)


@pytest.mark.critical
@pytest.mark.structural
def test_a_consumer_cannot_write_through_the_cached_arrays() -> None:
    """The cache is handed out, so it is handed out unwritable.

    A `cached_property` returns the same array to every caller, and the graph
    is frozen: a consumer writing through one would corrupt every later call
    with nothing to say it had. The flag makes that a `ValueError` at the
    write instead.
    """
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 1.0)

    with pytest.raises(ValueError, match="read-only"):
        graph.edge_index[0, 0] = 99
    with pytest.raises(ValueError, match="read-only"):
        graph.edge_coupling[0] = -1.0


@pytest.mark.critical
@pytest.mark.structural
def test_the_arrays_are_derived_once_and_handed_back() -> None:
    """The same object, not an equal one: that is the whole optimization.

    604 rebuilds of these two arrays were 81% of a 200-step `anneal_potts` at
    32x32 (issue #608). Identity is what this change buys, so identity is what
    is asserted.
    """
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 1.0)

    assert graph.edge_index is graph.edge_index
    assert graph.edge_coupling is graph.edge_coupling
