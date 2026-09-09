"""Swendsen-Wang cluster labelling: the vectorized roots against the union-find walk (issue #389).

The sweep no longer walks :func:`_find` once per site, and no longer scans
``labels == root`` once per cluster. Both replacements are exact rather than
equivalent-up-to-something, and both are the kind of change that would be
invisible in a diff if it were wrong: a labelling that merged two clusters
still samples, and a grouping that reordered the clusters still samples --
from a different chain. So the roots are pinned against the walk they
replaced, and the grouping against the ``unique``/``flatnonzero`` scan it
replaced, on seeded bond sets.

`rustworkx` and `scipy.sparse.csgraph` referee the partition from outside.
Neither replaced the labelling: issue #389 measured `rustworkx` slower than
this module's own code on the whole sweep at every extent, and `scipy` faster
only past extent 16 and only by renumbering the clusters, which composes a
different chain from the same seed. So what they are here for is the partition
itself, which each computes by a different algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from snakes_and_ladders.search.potts_mcmc import _bonds_of, _find, _roots, _union
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

EXTENTS = (4, 8, 16)
N_DRAWS = 200


def _seeded_bond_sets(extent: int, n_draws: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(parent, active)`` for ``n_draws`` seeded bond activations on a lattice."""
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    bonds = _bonds_of(graph)
    rng = np.random.default_rng(extent)
    drawn = []
    for _ in range(n_draws):
        state = rng.integers(0, 3, size=graph.n_nodes)
        like = state[bonds.first] == state[bonds.second]
        active = like & (rng.random(bonds.first.shape[0]) < bonds.activation)
        parent = np.arange(graph.n_nodes)
        for edge in np.flatnonzero(active):
            _union(parent, int(bonds.first[edge]), int(bonds.second[edge]))
        drawn.append((parent, active))
    return drawn


@pytest.mark.oracle
@pytest.mark.parametrize("extent", EXTENTS)
def test_the_vectorized_roots_equal_the_per_site_union_find_walk(extent: int) -> None:
    # Entry for entry, not as a partition: `_roots` claims to give the same
    # root array `_find` gives, and a weaker claim would not catch a fixed
    # point reached one squaring early.
    for parent, _ in _seeded_bond_sets(extent, N_DRAWS):
        walked = np.array([_find(parent.copy(), node) for node in range(parent.size)])

        assert np.array_equal(_roots(parent.copy()), walked)


@pytest.mark.oracle
@pytest.mark.parametrize("extent", EXTENTS)
def test_the_sorted_grouping_yields_the_clusters_in_the_scan_s_order(
    extent: int,
) -> None:
    # The order is what the recolouring's draws are keyed on, so the test is
    # on the sequence of member arrays and not on the set of clusters.
    for parent, _ in _seeded_bond_sets(extent, N_DRAWS):
        labels = _roots(parent.copy())
        scanned = [np.flatnonzero(labels == root) for root in np.unique(labels)]

        order = np.argsort(labels, kind="stable")
        grouped = labels[order]
        starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
        sorted_groups = [
            order[begin:end]
            for begin, end in zip(starts, np.r_[starts[1:], labels.size], strict=True)
        ]

        assert len(sorted_groups) == len(scanned)
        for ours, theirs in zip(sorted_groups, scanned, strict=True):
            assert np.array_equal(ours, theirs)


def _partition(labels: np.ndarray) -> frozenset[frozenset[int]]:
    """The labelling as a set of blocks, so two label conventions compare."""
    return frozenset(
        frozenset(np.flatnonzero(labels == value).tolist())
        for value in np.unique(labels)
    )


@pytest.mark.oracle
@pytest.mark.parametrize("extent", EXTENTS)
def test_scipy_connected_components_finds_the_same_clusters(extent: int) -> None:
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    bonds = _bonds_of(graph)
    for parent, active in _seeded_bond_sets(extent, N_DRAWS):
        rows, cols = bonds.first[active], bonds.second[active]
        adjacency = coo_matrix(
            (np.ones(rows.shape[0], dtype=np.int8), (rows, cols)),
            shape=(graph.n_nodes, graph.n_nodes),
        ).tocsr()
        _, theirs = connected_components(adjacency, directed=False)

        assert _partition(_roots(parent.copy())) == _partition(theirs)


@pytest.mark.oracle
@pytest.mark.parametrize("extent", EXTENTS)
def test_rustworkx_connected_components_finds_the_same_clusters(extent: int) -> None:
    rustworkx = pytest.importorskip("rustworkx")
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    bonds = _bonds_of(graph)
    for parent, active in _seeded_bond_sets(extent, N_DRAWS):
        theirs = rustworkx.PyGraph(multigraph=False)
        theirs.add_nodes_from(range(graph.n_nodes))
        theirs.add_edges_from_no_data(
            [
                (int(bonds.first[edge]), int(bonds.second[edge]))
                for edge in np.flatnonzero(active)
            ]
        )
        blocks = frozenset(
            frozenset(int(node) for node in component)
            for component in rustworkx.connected_components(theirs)
        )

        assert _partition(_roots(parent.copy())) == blocks
