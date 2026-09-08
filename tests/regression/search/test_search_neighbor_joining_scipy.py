"""`search.neighbor_joining` against SciPy's UPGMA where the two must agree (issue #376).

On an *ultrametric* additive matrix --- one whose leaves are all the same
distance from the root --- average-linkage agglomeration and neighbor joining
return the same tree, and both return the generating one. That is the regime
where `scipy.cluster.hierarchy.linkage(method="average")` is an independent
implementation of the answer, written from a different algorithm: it joins
the closest pair by mean distance where neighbor joining joins the pair
minimizing ``(n-2) d_ij - r_i - r_j``.

What this establishes is the joining, on the matrices where the two criteria
coincide. What it does not establish is anything outside that regime, and the
second test here is the reason the distinction is stated rather than assumed:
on an additive matrix that is *not* ultrametric --- two long branches that are
not a cherry --- UPGMA returns the wrong tree and neighbor joining the right
one, which is Atteson's guarantee doing work no clustering method has.

``scipy`` is not a declared dependency of this repository, so this skips
unless it is installed; whether to declare it is the open question issue #376
leaves standing.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.search.neighbor_joining import neighbor_joining
from snakes_and_ladders.search.topology import leaf_bipartitions
from snakes_and_ladders.sim.tree import Node

hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
distance = pytest.importorskip("scipy.spatial.distance")

#: An ultrametric tree on six taxa: ((t1,t2),(t3,t4)) joined to (t5,t6), with
#: every leaf 1.0 from the root. A pair's distance is twice the height of the
#: node joining them, which is what makes the matrix ultrametric.
ULTRAMETRIC_NAMES = ("t1", "t2", "t3", "t4", "t5", "t6")
ULTRAMETRIC_HEIGHTS = {
    frozenset({"t1", "t2"}): 0.5,
    frozenset({"t3", "t4"}): 0.7,
    frozenset({"t5", "t6"}): 1.1,
}
#: The generating tree's three internal splits, canonicalized as
#: `leaf_bipartitions` canonicalizes: the side without the smallest name.
ULTRAMETRIC_SPLITS = frozenset(
    {
        frozenset({"t3", "t4", "t5", "t6"}),
        frozenset({"t3", "t4"}),
        frozenset({"t5", "t6"}),
    }
)


def _ultrametric_matrix() -> np.ndarray:
    """Twice the height of the node joining each pair."""
    size = len(ULTRAMETRIC_NAMES)
    matrix = np.zeros((size, size))
    for i, first in enumerate(ULTRAMETRIC_NAMES):
        for j, second in enumerate(ULTRAMETRIC_NAMES):
            if i == j:
                continue
            pair = frozenset({first, second})
            if pair in ULTRAMETRIC_HEIGHTS:
                height = ULTRAMETRIC_HEIGHTS[pair]
            elif {first, second} <= {"t1", "t2", "t3", "t4"}:
                height = 1.4
            else:
                height = 2.0
            matrix[i, j] = 2.0 * height
    return matrix


def _upgma_clusters(names: tuple[str, ...], matrix: np.ndarray) -> list[frozenset[str]]:
    """The leaf sets SciPy's average linkage agglomerates, largest last."""
    linkage = hierarchy.linkage(distance.squareform(matrix, checks=False), "average")
    members: dict[int, frozenset[str]] = {
        index: frozenset({name}) for index, name in enumerate(names)
    }
    joined: list[frozenset[str]] = []
    for step, row in enumerate(linkage):
        first, second = int(row[0]), int(row[1])
        members[len(names) + step] = members[first] | members[second]
        joined.append(members[len(names) + step])
    return joined


def _splits(
    clusters: list[frozenset[str]], names: tuple[str, ...]
) -> frozenset[frozenset[str]]:
    """Clusters as bipartitions, in `leaf_bipartitions`' canonical form."""
    everything = frozenset(names)
    anchor = min(everything)
    found: set[frozenset[str]] = set()
    for cluster in clusters:
        side = cluster if anchor not in cluster else everything - cluster
        if 2 <= len(side) <= len(names) - 2:
            found.add(side)
    return frozenset(found)


def _internal_splits(
    topology: Node, names: tuple[str, ...]
) -> frozenset[frozenset[str]]:
    return frozenset(
        split
        for split in leaf_bipartitions(topology)
        if 2 <= len(split) <= len(names) - 2
    )


@pytest.mark.oracle
def test_the_two_return_the_same_tree_on_an_ultrametric_matrix() -> None:
    matrix = _ultrametric_matrix()

    ours = neighbor_joining(ULTRAMETRIC_NAMES, matrix)
    theirs = _upgma_clusters(ULTRAMETRIC_NAMES, matrix)

    assert _internal_splits(ours, ULTRAMETRIC_NAMES) == ULTRAMETRIC_SPLITS
    assert _splits(theirs, ULTRAMETRIC_NAMES) == ULTRAMETRIC_SPLITS


@pytest.mark.edge_case
def test_upgma_returns_the_wrong_tree_where_the_matrix_is_not_ultrametric() -> None:
    # Two long branches that are not a cherry: the true tree is
    # ((t1,t2),(t3,t4)) with t1 and t3 long, so the two shortest branches are
    # t2 and t4 and the closest pair by distance is t2 with t4 --- which is
    # what average linkage joins first, and it is wrong. The matrix is exactly
    # additive, so this is the criterion failing and not noise in an estimate.
    names = ("t1", "t2", "t3", "t4")
    matrix = np.array(
        [
            [0.0, 0.8, 1.5, 0.9],
            [0.8, 0.0, 0.9, 0.3],
            [1.5, 0.9, 0.0, 0.8],
            [0.9, 0.3, 0.8, 0.0],
        ]
    )
    truth = frozenset({frozenset({"t3", "t4"})})

    assert _internal_splits(neighbor_joining(names, matrix), names) == truth
    assert _splits(_upgma_clusters(names, matrix), names) == frozenset(
        {frozenset({"t2", "t4"})}
    )
