"""NNI and SPR neighbourhood generators over unrooted binary topologies.

Topologies are ``snakes_and_ladders.sim.tree.Node`` in the trifurcating-root convention
(``snakes_and_ladders.sim.newick.validate_unrooted_newick``): the root has 3 children,
every other internal node has 2. Branch lengths and internal labels carry no
information here -- this is pure combinatorics over topology (issue #79),
not likelihood -- so generated neighbours have ``branch_length=None`` and
synthetic internal names.

Both generators share one interface, ``Topology -> Iterator[Topology]``,
per the module ``CLAUDE.md``'s "one interface" for move sets.

Move definitions follow Felsenstein, *Inferring Phylogenies*, ch. 4. Both
operate on an undirected adjacency view of the topology (each internal node
degree 3, each leaf degree 1) rather than on the rooted ``Node`` directly,
because NNI and SPR are edge operations and the trifurcating root is not
otherwise a distinguished point of the tree.

**Completeness.** NNI is a strict subset of SPR (``nni_neighbours(t)`` is
contained in ``spr_neighbours(t)`` as sets of topologies, checked by
``tests/regression/test_search_topology.py``): every NNI move is realizable
as an SPR prune-regraft of one of the two subtrees adjacent to the shared
internal edge. Neither is complete in a single step; both are complete in
the sense that their transitive closure (repeated application) reaches
every topology on the same leaf set -- proven for NNI by connectivity of the
associated graph (Robinson 1971) and inherited by SPR since it dominates
NNI. Per-step cost: ``nni_neighbours`` is ``O(n)`` (one pass to build the
adjacency, ``O(1)`` work per internal edge); ``spr_neighbours`` is
``O(n^2)`` (a prune-and-regraft candidate per (edge, edge) pair, before
dedup).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from snakes_and_ladders.sim.tree import Node

Topology = Node

NodeId = int | str


def enumerate_topologies(leaf_names: Sequence[str]) -> Iterator[Topology]:
    """Every unrooted binary topology on ``leaf_names``, each exactly once.

    The same stepwise insertion :func:`random_topology` samples from, taken
    exhaustively: three leaves meet at the root, then each remaining leaf is
    inserted at every edge in turn. Each topology is produced once, so the
    count is ``(2n-5)!!`` --- which is what pins this function, since a
    generator that double-counted or missed trees would make a search look
    better or worse than it is.

    This is the oracle for search quality. Below ``n = 8`` it is the only
    independent statement available about whether hill climbing found the
    best tree or merely a good one.

    Parameters
    ----------
    leaf_names : Sequence[str]
        Taxon names, at least 3, all distinct.

    Returns
    -------
    Iterator[Topology]
        Every unrooted binary topology on that leaf set, with
        ``branch_length=None`` throughout.

    Raises
    ------
    ValueError
        If fewer than 3 names are given, or any two are equal.
    """
    if len(leaf_names) < 3:
        msg = f"need at least 3 leaves for an unrooted topology, got {len(leaf_names)}"
        raise ValueError(msg)
    if len(set(leaf_names)) != len(leaf_names):
        msg = "leaf names must be distinct"
        raise ValueError(msg)

    seed_adjacency: dict[NodeId, list[NodeId]] = {0: list(leaf_names[:3])}
    for name in leaf_names[:3]:
        seed_adjacency[name] = [0]

    def grow(
        adjacency: dict[NodeId, list[NodeId]], remaining: Sequence[str], next_id: int
    ) -> Iterator[Topology]:
        if not remaining:
            yield _from_adjacency(adjacency, 0)
            return
        name, rest = remaining[0], remaining[1:]
        for first, second in sorted(
            _edges(adjacency), key=lambda edge: (str(edge[0]), str(edge[1]))
        ):
            grown = {node: list(neighbours) for node, neighbours in adjacency.items()}
            grown[first].remove(second)
            grown[second].remove(first)
            grown[next_id] = [first, second, name]
            grown[first].append(next_id)
            grown[second].append(next_id)
            grown[name] = [next_id]
            yield from grow(grown, rest, next_id + 1)

    yield from grow(seed_adjacency, list(leaf_names[3:]), 1)


def random_topology(leaf_names: Sequence[str], rng: np.random.Generator) -> Topology:
    """Draw an unrooted binary topology on ``leaf_names``, uniformly by construction.

    Stepwise insertion: three leaves meet at the root, then each remaining
    leaf subdivides a uniformly chosen edge. Every topology is reachable, and
    the construction is the standard one behind the ``(2n-5)!!`` count --- at
    step ``i`` there are ``2i - 5`` edges to choose from, and the product of
    those choices is that count, so no topology is favoured.

    A search needs a starting point that is not the answer. Reading one from
    a fixture would make every run start beside the truth and measure
    nothing.

    Parameters
    ----------
    leaf_names : Sequence[str]
        Taxon names, at least 3, all distinct.
    rng : np.random.Generator
        Seeded generator, so a search is reproducible from its seed.

    Returns
    -------
    Topology
        An unrooted binary topology in the trifurcating-root convention,
        with ``branch_length=None`` throughout --- lengths belong to the
        objective that fits them, not to the topology.

    Raises
    ------
    ValueError
        If fewer than 3 names are given, or any two are equal.
    """
    if len(leaf_names) < 3:
        msg = f"need at least 3 leaves for an unrooted topology, got {len(leaf_names)}"
        raise ValueError(msg)
    if len(set(leaf_names)) != len(leaf_names):
        msg = "leaf names must be distinct"
        raise ValueError(msg)

    adjacency: dict[NodeId, list[NodeId]] = {0: []}
    for name in leaf_names[:3]:
        adjacency[0].append(name)
        adjacency[name] = [0]

    next_id = 1
    for name in leaf_names[3:]:
        edges = sorted(_edges(adjacency), key=lambda edge: (str(edge[0]), str(edge[1])))
        first, second = edges[int(rng.integers(len(edges)))]
        internal = next_id
        next_id += 1
        adjacency[first].remove(second)
        adjacency[second].remove(first)
        adjacency[internal] = [first, second, name]
        adjacency[first].append(internal)
        adjacency[second].append(internal)
        adjacency[name] = [internal]

    return _from_adjacency(adjacency, 0)


def leaf_bipartitions(topology: Topology) -> frozenset[frozenset[str]]:
    """The set of leaf-set bipartitions induced by every edge of ``topology``.

    An unrooted binary topology is exactly determined, independent of
    rooting and of child order, by the bipartitions its edges induce on the
    leaf set (Felsenstein, *Inferring Phylogenies*, ch. 3). Two topologies
    (possibly rooted at different points, possibly with children in a
    different order) denote the same unrooted tree iff this set agrees.

    This is a search-internal equality/hash key for move-set validation and
    deduplication (symmetry, containment, "differs from its parent",
    enumeration cross-checks) -- not the canonical Newick key of issue #73,
    which fixes a single serialization rather than an unordered set of
    bipartitions, and is not exposed as such.

    Parameters
    ----------
    topology : Topology
        An unrooted binary topology (trifurcating root).

    Returns
    -------
    frozenset[frozenset[str]]
        One ``frozenset`` of leaf names per edge, each canonicalized to the
        side not containing the lexicographically smallest leaf name, so
        that a bipartition and its complement collapse to one entry.
    """
    all_leaves = _leaf_names(topology)
    anchor = min(all_leaves)
    splits: set[frozenset[str]] = set()
    _collect_splits(topology, all_leaves, anchor, splits)
    return frozenset(splits)


def _leaf_names(node: Node) -> frozenset[str]:
    if node.is_leaf:
        return frozenset((node.name,))
    result: frozenset[str] = frozenset()
    for child in node.children:
        result |= _leaf_names(child)
    return result


def _collect_splits(
    node: Node, all_leaves: frozenset[str], anchor: str, out: set[frozenset[str]]
) -> frozenset[str]:
    if node.is_leaf:
        return frozenset((node.name,))
    child_leaf_sets = [
        _collect_splits(child, all_leaves, anchor, out) for child in node.children
    ]
    union: frozenset[str] = frozenset()
    for child_side in child_leaf_sets:
        side = child_side if anchor not in child_side else (all_leaves - child_side)
        out.add(side)
        union |= child_side
    return union


def _to_adjacency(topology: Topology) -> tuple[dict[NodeId, list[NodeId]], int]:
    """Build an undirected adjacency map: leaves keyed by name, internal
    nodes keyed by a fresh ``int`` id starting at 0 for the root.
    """
    adjacency: dict[NodeId, list[NodeId]] = {}
    next_id = 0

    def add_edge(a: NodeId, b: NodeId) -> None:
        adjacency[a].append(b)
        adjacency[b].append(a)

    def visit(node: Node) -> NodeId:
        nonlocal next_id
        if node.is_leaf:
            adjacency.setdefault(node.name, [])
            return node.name
        node_id = next_id
        next_id += 1
        adjacency[node_id] = []
        for child in node.children:
            child_id = visit(child)
            add_edge(node_id, child_id)
        return node_id

    root_id = visit(topology)
    assert isinstance(root_id, int)
    return adjacency, root_id


def _from_adjacency(adjacency: dict[NodeId, list[NodeId]], root_id: NodeId) -> Node:
    def build(node_id: NodeId, parent_id: NodeId | None) -> Node:
        if isinstance(node_id, str):
            return Node(name=node_id, branch_length=None)
        children_ids = [nb for nb in adjacency[node_id] if nb != parent_id]
        children = tuple(build(cid, node_id) for cid in children_ids)
        return Node(name=f"n{node_id}", branch_length=None, children=children)

    return build(root_id, None)


def _internal_edges(adjacency: dict[NodeId, list[NodeId]]) -> Iterator[tuple[int, int]]:
    for u, neighbours in adjacency.items():
        if not isinstance(u, int):
            continue
        for v in neighbours:
            if isinstance(v, int) and u < v:
                yield u, v


def _rewired(
    adjacency: dict[NodeId, list[NodeId]],
    drop: list[tuple[NodeId, NodeId]],
    add: list[tuple[NodeId, NodeId]],
) -> dict[NodeId, list[NodeId]]:
    new_adjacency = {
        node_id: list(neighbours) for node_id, neighbours in adjacency.items()
    }
    for a, b in drop:
        new_adjacency[a].remove(b)
        new_adjacency[b].remove(a)
    for a, b in add:
        new_adjacency[a].append(b)
        new_adjacency[b].append(a)
    return new_adjacency


def nni_neighbours(topology: Topology) -> Iterator[Topology]:
    """Generate the nearest-neighbour-interchange neighbourhood of ``topology``.

    For each of the ``n - 3`` internal edges ``(u, v)``, the 4 subtrees
    hanging off it -- ``u``'s other 2 neighbours ``a1, a2`` and ``v``'s
    other 2 ``b1, b2`` -- are regrouped the 2 other ways: swap ``a1`` with
    ``b1``, or swap ``a1`` with ``b2``, yielding exactly ``2 * (n - 3)``
    neighbours (Felsenstein, *Inferring Phylogenies*, ch. 4).

    Parameters
    ----------
    topology : Topology
        An unrooted binary topology with at least 4 leaves.

    Returns
    -------
    Iterator[Topology]
        The ``2 * (n - 3)`` NNI neighbours, each a valid unrooted binary
        topology on the same leaf set, differing from ``topology``.
    """
    adjacency, root_id = _to_adjacency(topology)
    for u, v in _internal_edges(adjacency):
        a1 = next(nb for nb in adjacency[u] if nb != v)
        for b in adjacency[v]:
            if b == u:
                continue
            new_adjacency = _rewired(
                adjacency, drop=[(u, a1), (v, b)], add=[(v, a1), (u, b)]
            )
            yield _from_adjacency(new_adjacency, root_id)


def spr_neighbours(topology: Topology) -> Iterator[Topology]:
    """Generate the subtree-prune-and-regraft neighbourhood of ``topology``.

    For every edge ``(u, v)`` with ``u`` internal, prune the component
    containing ``v`` (suppressing ``u``'s now-degree-2 node in the
    remainder), then regraft the pruned piece onto every edge of the
    remainder by subdividing it with a fresh internal node (Felsenstein,
    *Inferring Phylogenies*, ch. 4). Regrafts that reconstruct ``topology``
    (the vacated edge) or a topology already yielded are skipped, per the
    module ``CLAUDE.md``'s "differs from its parent" validity rule.

    Parameters
    ----------
    topology : Topology
        An unrooted binary topology with at least 4 leaves.

    Returns
    -------
    Iterator[Topology]
        Distinct SPR neighbours, each a valid unrooted binary topology on
        the same leaf set, differing from ``topology``. Matches the
        closed-form count ``2 * (n - 3) * (2 * n - 7)``.
    """
    adjacency, _root_id = _to_adjacency(topology)
    seen: set[frozenset[frozenset[str]]] = {leaf_bipartitions(topology)}

    for u in list(adjacency):
        if not isinstance(u, int):
            continue
        for v in list(adjacency[u]):
            remainder, pruned = _prune(adjacency, u, v)
            for x, y in _edges(remainder):
                candidate_adjacency, new_id = _regraft(remainder, pruned, v, x, y)
                candidate = _from_adjacency(candidate_adjacency, new_id)
                key = leaf_bipartitions(candidate)
                if key in seen:
                    continue
                seen.add(key)
                yield candidate


def _prune(
    adjacency: dict[NodeId, list[NodeId]], u: int, v: NodeId
) -> tuple[dict[NodeId, list[NodeId]], dict[NodeId, list[NodeId]]]:
    """Remove edge ``(u, v)``: the ``u``-side component, with ``u``
    suppressed (now degree 2), is the remainder; the ``v``-side component,
    unchanged but for the removed edge, is the pruned subtree.
    """
    cut = _rewired(adjacency, drop=[(u, v)], add=[])
    remainder = {node_id: list(cut[node_id]) for node_id in _component(cut, u)}
    pruned = {node_id: list(cut[node_id]) for node_id in _component(cut, v)}

    x, y = remainder[u]
    del remainder[u]
    remainder[x].remove(u)
    remainder[y].remove(u)
    remainder[x].append(y)
    remainder[y].append(x)
    return remainder, pruned


def _component(adjacency: dict[NodeId, list[NodeId]], start: NodeId) -> set[NodeId]:
    seen = {start}
    stack = [start]
    while stack:
        node_id = stack.pop()
        for neighbour in adjacency[node_id]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def _edges(adjacency: dict[NodeId, list[NodeId]]) -> Iterator[tuple[NodeId, NodeId]]:
    done: set[frozenset[NodeId]] = set()
    for a, neighbours in adjacency.items():
        for b in neighbours:
            edge = frozenset((a, b))
            if edge not in done:
                done.add(edge)
                yield a, b


def _regraft(
    remainder: dict[NodeId, list[NodeId]],
    pruned: dict[NodeId, list[NodeId]],
    pruned_root: NodeId,
    x: NodeId,
    y: NodeId,
) -> tuple[dict[NodeId, list[NodeId]], NodeId]:
    used_internal_ids = [
        node_id for node_id in (*remainder, *pruned) if isinstance(node_id, int)
    ]
    new_id = (max(used_internal_ids) + 1) if used_internal_ids else 0
    merged = {**remainder, **pruned, new_id: []}
    grafted = _rewired(
        merged, drop=[(x, y)], add=[(x, new_id), (y, new_id), (new_id, pruned_root)]
    )
    return grafted, new_id


def robinson_foulds(first: Topology, second: Topology) -> int:
    """Robinson-Foulds distance: splits in one topology and not the other.

    The symmetric difference of the two split sets, which is the standard
    definition for unrooted trees (Robinson & Foulds 1981; Felsenstein,
    *Inferring Phylogenies*, ch. 30). Zero exactly when the topologies are
    the same tree, so it refines the equality `leaf_bipartitions` already
    gives into a distance.

    Parameters
    ----------
    first, second : Topology
        Unrooted binary topologies over the same leaf set.

    Returns
    -------
    int
        The number of splits present in one and not the other. Even, since a
        split missing from one tree is matched by one missing from the other
        when both are binary over the same leaves.

    Raises
    ------
    ValueError
        If the two topologies do not carry the same leaves. A distance
        between trees over different taxa is not defined.
    """
    if _leaf_names(first) != _leaf_names(second):
        msg = "cannot compare topologies over different leaf sets"
        raise ValueError(msg)
    return len(leaf_bipartitions(first) ^ leaf_bipartitions(second))


def normalized_robinson_foulds(first: Topology, second: Topology) -> float:
    """Robinson-Foulds distance scaled to ``[0, 1]``.

    Divided by the number of **internal** splits the two trees hold between
    them, so the figure is comparable across taxon counts -- which is what
    makes a bound like ``ROADMAP.md``'s ``<= 0.05`` mean the same thing at 6
    taxa and at 60.

    The trivial splits are excluded from the denominator deliberately. Every
    tree over the same leaves induces all of them, so they never contribute
    to the numerator; leaving them in the denominator would shrink every
    distance by a factor that depends on the taxon count, and would weaken
    the bound above without saying so. For two binary unrooted trees on ``n``
    leaves the denominator is ``2(n - 3)``, and the distance reaches ``1.0``
    when the trees share no internal split.

    Returns
    -------
    float
        ``0.0`` for the same topology, ``1.0`` when the two share no internal
        split. Trees with no internal edge at all -- fewer than four taxa --
        score ``0.0`` rather than dividing by zero.
    """
    internal = sum(
        1
        for topology in (first, second)
        for split in leaf_bipartitions(topology)
        if 1 < len(split) < len(_leaf_names(topology)) - 1
    )
    if internal == 0:
        return 0.0
    return robinson_foulds(first, second) / internal
