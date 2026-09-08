"""Neighbor joining: the tree a distance matrix implies, and whether it is tree-like.

The second half of the tree's moment estimator (issue #364; the first is
:mod:`snakes_and_ladders.likelihood.distance`). Saitou and Nei's algorithm
(1987), ``alg:neighbor-joining`` of ``docs/tex/textbook.tex``, joins at
each of ``n - 3`` steps the pair minimizing ``(n - 2) d_ij - r_i - r_j``
with ``r`` the row sums, places the new node by the three-point formula,
and reduces the matrix by one --- ``O(n^2)`` a step and ``O(n^3)`` in all. On an additive matrix, one whose entries are the path
lengths of some tree, it returns that tree with every branch length exact,
which is what the test pins to ``1e-12``. On an estimated matrix, Atteson
(1999) gives the guarantee: the topology is the true one whenever every
entry's error is below half the shortest branch, :func:`atteson_radius`.
That radius, with the variance the distance module reports, is what lets a
caller state whether a start is inside the guarantee rather than hope so.

Whether a matrix is additive at all is the four-point condition (Buneman;
Felsenstein, *Inferring Phylogenies*, ch. 11): of the three sums ``d_ij +
d_kl``, ``d_ik + d_jl``, ``d_il + d_jk`` over any four taxa, the two largest
are equal. :func:`four_point_violation` reports the largest gap over every
quartet, zero exactly on an additive matrix.

The tree comes back as the repository's topology --- ``Node`` in the
trifurcating-root convention :mod:`snakes_and_ladders.search.topology` walks
--- with branch lengths attached, so it can start a search directly or, by
:func:`split_lengths`, seed a fit on any topology that shares its splits.
Estimated distances can give a negative branch length; it is returned as
computed, and the caller that needs a positive start floors it and says so
(:mod:`snakes_and_ladders.search.initialize`).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from snakes_and_ladders.search.topology import NodeId, _from_adjacency, _leaf_names
from snakes_and_ladders.sim.tree import Node, edges

_MIN_TAXA = 3


def _check_matrix(names: Sequence[str], distances: np.ndarray) -> None:
    n = len(names)
    if n < _MIN_TAXA:
        msg = f"neighbor joining needs at least {_MIN_TAXA} taxa, got {n}"
        raise ValueError(msg)
    if len(set(names)) != n:
        msg = "taxon names must be distinct"
        raise ValueError(msg)
    if distances.shape != (n, n):
        msg = f"distances have shape {distances.shape} for {n} taxa"
        raise ValueError(msg)
    if not np.allclose(distances, distances.T) or not np.allclose(
        np.diag(distances), 0.0
    ):
        msg = "distances must be symmetric with a zero diagonal"
        raise ValueError(msg)


def neighbor_joining(names: Sequence[str], distances: np.ndarray) -> Node:
    """The neighbor-joining tree of a distance matrix (Saitou and Nei 1987).

    Parameters
    ----------
    names : Sequence[str]
        Taxon names, at least 3, distinct, indexing ``distances``.
    distances : np.ndarray
        Symmetric ``(n, n)`` matrix with a zero diagonal.

    Returns
    -------
    Node
        An unrooted binary tree in the trifurcating-root convention, every
        non-root node carrying the branch length the algorithm assigned;
        internal nodes are named ``n<index>``. On an additive matrix the
        lengths are those of the generating tree.

    Raises
    ------
    ValueError
        If fewer than three taxa are given, the names repeat, or the matrix
        is not symmetric with a zero diagonal.
    """
    _check_matrix(names, distances)
    matrix = np.array(distances, dtype=np.float64)
    active: list[NodeId] = list(names)
    adjacency: dict[NodeId, list[NodeId]] = {name: [] for name in names}
    lengths: dict[frozenset[NodeId], float] = {}
    next_id = 1

    def connect(first: NodeId, second: NodeId, length: float) -> None:
        adjacency[first].append(second)
        adjacency[second].append(first)
        lengths[frozenset((first, second))] = length

    while len(active) > _MIN_TAXA:
        n = len(active)
        row_sums = matrix.sum(axis=1)
        criterion = (n - 2) * matrix - row_sums[:, np.newaxis] - row_sums[np.newaxis, :]
        np.fill_diagonal(criterion, np.inf)
        i, j = np.unravel_index(int(np.argmin(criterion)), criterion.shape)
        i, j = (i, j) if i < j else (j, i)
        to_i = 0.5 * matrix[i, j] + (row_sums[i] - row_sums[j]) / (2.0 * (n - 2))
        to_j = matrix[i, j] - to_i
        new_node = next_id
        next_id += 1
        adjacency[new_node] = []
        connect(active[i], new_node, float(to_i))
        connect(active[j], new_node, float(to_j))
        to_new = 0.5 * (matrix[i] + matrix[j] - matrix[i, j])
        keep = [index for index in range(n) if index not in (i, j)]
        reduced = np.empty((n - 1, n - 1))
        reduced[:-1, :-1] = matrix[np.ix_(keep, keep)]
        reduced[:-1, -1] = reduced[-1, :-1] = to_new[keep]
        reduced[-1, -1] = 0.0
        matrix = reduced
        active = [active[index] for index in keep] + [new_node]

    a, b, c = active
    root: NodeId = 0
    adjacency[root] = []
    connect(a, root, float(0.5 * (matrix[0, 1] + matrix[0, 2] - matrix[1, 2])))
    connect(b, root, float(0.5 * (matrix[0, 1] + matrix[1, 2] - matrix[0, 2])))
    connect(c, root, float(0.5 * (matrix[0, 2] + matrix[1, 2] - matrix[0, 1])))
    return _with_lengths(_from_adjacency(adjacency, root), adjacency, root, lengths)


def _with_lengths(
    topology: Node,
    adjacency: dict[NodeId, list[NodeId]],
    root: NodeId,
    lengths: dict[frozenset[NodeId], float],
) -> Node:
    """Attach the edge lengths to the ``Node`` tree ``_from_adjacency`` built.

    ``_from_adjacency`` names an internal node ``n<id>``, so the id is read
    back from the name to find the edge; a leaf keeps its name as its id.
    """

    def node_id(node: Node) -> NodeId:
        return node.name if node.is_leaf else int(node.name[1:])

    def rebuild(node: Node, parent: NodeId | None) -> Node:
        own = node_id(node)
        length = None if parent is None else lengths[frozenset((parent, own))]
        return Node(
            name=node.name,
            branch_length=length,
            children=tuple(rebuild(child, own) for child in node.children),
        )

    assert node_id(topology) == root
    assert set(adjacency[root]) == {node_id(child) for child in topology.children}
    return rebuild(topology, None)


def split_lengths(tau: Node) -> dict[frozenset[str], float]:
    """Branch length per split, keyed as :func:`~snakes_and_ladders.search.topology.branch_splits` keys them.

    A branch is the bipartition of the leaves it separates, canonicalized to
    the side without the lexicographically smallest leaf, so lengths can be
    matched between trees whose internal names differ. On a rooted binary
    tree the two branches below the root induce one split and are summed,
    which is the estimable quantity
    (:mod:`snakes_and_ladders.likelihood.objective`).

    Parameters
    ----------
    tau : Node
        A tree with a branch length on every non-root node.

    Returns
    -------
    dict[frozenset[str], float]
        One entry per split, pendant splits included.

    Raises
    ------
    ValueError
        If a non-root node carries no branch length.
    """
    all_leaves = _leaf_names(tau)
    anchor = min(all_leaves)
    found: dict[frozenset[str], float] = {}
    for _, child in edges(tau):
        if child.branch_length is None:
            msg = f"non-root node {child.name!r} has no branch_length"
            raise ValueError(msg)
        below = _leaf_names(child)
        split = below if anchor not in below else all_leaves - below
        found[split] = found.get(split, 0.0) + child.branch_length
    return found


def atteson_radius(tau: Node) -> float:
    """Half the shortest branch: the error every distance may carry and neighbor joining still recover ``tau``.

    Atteson (1999): if ``max_ij |D_ij - T_ij| < mu / 2`` for ``T`` the
    additive matrix of ``tau`` and ``mu`` its shortest branch, neighbor
    joining on ``D`` returns ``tau``'s topology. The two branches below a
    rooted binary root count as their sum, since the unrooted tree has one
    branch there.

    Parameters
    ----------
    tau : Node
        A tree with a branch length on every non-root node.

    Returns
    -------
    float
        ``mu / 2``.
    """
    return 0.5 * min(split_lengths(tau).values())


def four_point_violation(distances: np.ndarray) -> float:
    """The largest four-point-condition gap over every quartet.

    For taxa ``i, j, k, l`` the three sums ``d_ij + d_kl``, ``d_ik + d_jl``
    and ``d_il + d_jk`` are formed; on an additive matrix the two largest
    are equal, and the gap between them is the violation. The maximum over
    quartets is zero exactly when the matrix is tree-like. ``O(n^4)``, a
    check and not a step of the algorithm.

    Parameters
    ----------
    distances : np.ndarray
        Symmetric ``(n, n)`` matrix, ``n >= 4``.

    Returns
    -------
    float
        The largest gap, in the matrix's units, non-negative.

    Raises
    ------
    ValueError
        If fewer than four taxa are given.
    """
    n = distances.shape[0]
    if n < 4:
        msg = f"the four-point condition needs at least 4 taxa, got {n}"
        raise ValueError(msg)
    quartets = np.array(list(combinations(range(n), 4)))
    i, j, k, ll = quartets.T
    sums = np.stack(
        [
            distances[i, j] + distances[k, ll],
            distances[i, k] + distances[j, ll],
            distances[i, ll] + distances[j, k],
        ],
        axis=1,
    )
    ordered = np.sort(sums, axis=1)
    return float(np.max(ordered[:, 2] - ordered[:, 1]))
