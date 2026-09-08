"""Topology equality against `rustworkx.is_isomorphic` (issue #376).

`search.topology` has no canonical serialization: two topologies are the same
unrooted tree when their leaf-set bipartitions agree, which is
`leaf_bipartitions`, and the Robinson--Foulds distance between them is zero.
That is a claim about graphs, and `rustworkx.is_isomorphic` decides it
independently --- on the labelled tree, with leaves matched by taxon name and
internal nodes matched to internal nodes, since the internal names are
synthetic and an unlabelled comparison would call every binary tree on the
same leaf count isomorphic.

What this establishes is that our equality is graph isomorphism on the
labelled tree, in both directions: every pair the code calls equal is
isomorphic, and every pair it calls different is not. What it does not
establish is a canonical form --- issue #73's Newick key, which fixes one
serialization rather than deciding a pair --- and it says nothing about
rooting, which the bipartitions deliberately discard.

Everything here skips without the ``frameworks`` extra.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    nni_neighbours,
    random_topology,
    robinson_foulds,
)
from snakes_and_ladders.sim.tree import edges, preorder

rustworkx = pytest.importorskip("rustworkx")

NAMES = ("t1", "t2", "t3", "t4", "t5", "t6")


def _as_graph(topology: Topology) -> object:
    """The topology as an undirected graph, leaves carrying their taxon name.

    An internal node carries ``None`` rather than its name: the names are
    synthetic and differ between two topologies that are the same tree, so
    matching on them would call every pair different.
    """
    graph = rustworkx.PyGraph()
    index = {
        node.name: graph.add_node(node.name if node.is_leaf else None)
        for node in preorder(topology)
    }
    for parent, child in edges(topology):
        graph.add_edge(index[parent.name], index[child.name], None)
    return graph


def _isomorphic(first: Topology, second: Topology) -> bool:
    return bool(
        rustworkx.is_isomorphic(
            _as_graph(first), _as_graph(second), lambda x, y: x == y
        )
    )


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", [4, 5])
def test_two_topologies_our_code_calls_equal_are_isomorphic(n_taxa: int) -> None:
    # Every topology against a re-rooted, re-ordered copy of itself: the
    # enumeration builds each by a different insertion order, so pairs with
    # equal bipartitions and different node names occur without contriving
    # them. Both directions are asserted on the same pairs, which is what
    # makes this an equality check rather than one implication.
    topologies = list(enumerate_topologies(NAMES[:n_taxa]))
    assert len(topologies) > 1

    for first, second in itertools.combinations(topologies, 2):
        ours = robinson_foulds(first, second) == 0
        assert ours == (leaf_bipartitions(first) == leaf_bipartitions(second))
        assert ours == _isomorphic(first, second), (first, second)


@pytest.mark.oracle
def test_a_topology_is_isomorphic_to_itself_however_it_was_built() -> None:
    # The positive half at a size the enumeration above does not reach: a
    # topology drawn twice from the same seed, and every NNI neighbour of it
    # against itself, each rebuilt through `_from_adjacency` and therefore
    # carrying different synthetic internal names.
    first = random_topology(NAMES, np.random.default_rng(376))
    second = random_topology(NAMES, np.random.default_rng(376))
    assert robinson_foulds(first, second) == 0
    assert _isomorphic(first, second)

    for neighbour in nni_neighbours(first):
        assert robinson_foulds(neighbour, neighbour) == 0
        assert _isomorphic(neighbour, neighbour)


@pytest.mark.oracle
def test_every_nni_neighbour_differs_from_its_parent_and_is_not_isomorphic() -> None:
    # The negative half. An NNI move changes exactly one bipartition, so no
    # neighbour is the parent; a construction that silently returned the
    # parent would pass a test that only checked the positive direction.
    parent = random_topology(NAMES, np.random.default_rng(1))
    neighbours = list(nni_neighbours(parent))
    assert len(neighbours) == 2 * (len(NAMES) - 3)

    for neighbour in neighbours:
        assert robinson_foulds(parent, neighbour) > 0
        assert not _isomorphic(parent, neighbour)


@pytest.mark.structural
def test_an_unlabelled_comparison_would_call_every_topology_the_same() -> None:
    # The guard for the docstring's reason for matching on names: without the
    # matcher, rustworkx calls two different topologies on six taxa
    # isomorphic, because they are the same tree shape.
    topologies = list(enumerate_topologies(NAMES[:5]))
    first, second = topologies[0], topologies[1]

    assert robinson_foulds(first, second) > 0
    assert rustworkx.is_isomorphic(_as_graph(first), _as_graph(second))
