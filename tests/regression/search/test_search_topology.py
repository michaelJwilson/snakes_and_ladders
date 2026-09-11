"""Regression tests for ``snakes_and_ladders.search.topology``'s NNI and SPR generators.

Per root ``CLAUDE.md`` ("Pin to Independent Sources") and the module
``CLAUDE.md``'s local rules, every property here is checked exhaustively at
sizes where brute-force enumeration is a feasible oracle (``n <= 10`` per
``DEV.md``'s CI budget), rather than on a handful of hand-picked trees:

* the closed-form neighbourhood counts, ``2 * (n - 3)`` for NNI and
  ``2 * (n - 3) * (2 * n - 7)`` for SPR (issue #79), against every one of
  the ``count_topologies(n - 1)`` distinct unrooted topologies on ``n``
  taxa;
* neighbour validity (same leaf set, valid grammar, differs from parent);
* symmetry, ``tau' in N(tau) <=> tau in N(tau')``;
* NNI-neighbour containment in the SPR neighbourhood.

The exhaustive sweep runs at ``n = 5, 6, 7`` per PR; the same sweep at
``n = 8`` (10395 topologies) is marked ``release`` (``DEV.md``'s Release-Gated
budget line), taking ~2.5 minutes. The enumeration itself is cheap even at
``n = 8`` (~1s) and stays in the per-PR suite as a check on the brute-force
oracle.

A random-walk connectivity test is deferred: it needs #73's canonical Newick
key to identify "every topology" visited (see issue #79's plan comment).
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from itertools import combinations, pairwise

import numpy as np
import pytest
from snakes_and_ladders.search import topology as topology_module
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    nni_neighbours,
    random_topology,
    spr_neighbours,
)
from snakes_and_ladders.sim.newick import (
    count_topologies,
    to_newick,
    validate_unrooted_newick,
)
from snakes_and_ladders.sim.tree import Node

# Sizes at which exhaustive enumeration is the oracle. The per-pull-request
# budget holds 5 and 6; 7 costs about 8 s across the two neighbourhood tests
# below and runs under `stress` (`DEV.md`, CI & Performance Budget). The claim
# is identical at every size.
EXHAUSTIVE_SIZES = (5, 6)
STRESS_SIZES = (7,)
EXHAUSTIVE_PARAMS = [
    *(pytest.param(size) for size in EXHAUSTIVE_SIZES),
    *(pytest.param(size, marks=pytest.mark.stress) for size in STRESS_SIZES),
]


def _enumerate_rooted(taxa: tuple[str, ...]) -> Iterator[Node]:
    """Brute-force-enumerate every rooted binary topology on ``taxa``.

    Identical construction to ``tests/regression/test_newick.py``'s
    ``_enumerate_topologies``, duplicated locally to keep this module
    independent.
    """
    if len(taxa) == 1:
        yield Node(name=taxa[0], branch_length=None)
        return

    first, rest = taxa[0], taxa[1:]
    for size in range(1, len(rest) + 1):
        for right_taxa in combinations(rest, size):
            right_set = set(right_taxa)
            left_taxa = (first, *(t for t in rest if t not in right_set))
            for left in _enumerate_rooted(left_taxa):
                for right in _enumerate_rooted(right_taxa):
                    yield Node(
                        name="internal", branch_length=None, children=(left, right)
                    )


def _enumerate_unrooted(n_taxa: int) -> Iterator[Topology]:
    """Brute-force-enumerate every unrooted binary topology on ``n_taxa`` leaves.

    Bijection with rooted binary topologies on ``n_taxa - 1`` leaves: a rooted
    tree's root has no incoming edge, so grafting one more leaf onto it as a
    third child recreates the trifurcating-root convention, one-to-one with
    attaching that leaf via every edge of the corresponding unrooted tree.
    Hence ``count_topologies(n_taxa - 1)`` is the oracle count (module docstring
    of ``snakes_and_ladders.sim.newick``).
    """
    taxa = tuple(f"t{i}" for i in range(n_taxa))
    rooted_taxa, outgroup = taxa[:-1], taxa[-1]
    for rooted in _enumerate_rooted(rooted_taxa):
        yield Node(
            name="root",
            branch_length=None,
            children=(*rooted.children, Node(name=outgroup, branch_length=None)),
        )


@cache
def _all_unrooted(n_taxa: int) -> tuple[Topology, ...]:
    return tuple(_enumerate_unrooted(n_taxa))


@pytest.mark.oracle
@pytest.mark.parametrize(
    "n_taxa", [*EXHAUSTIVE_PARAMS, pytest.param(8, marks=pytest.mark.stress)]
)
def test_enumeration_matches_count_topologies(n_taxa: int) -> None:
    topologies = _all_unrooted(n_taxa)
    assert len(topologies) == count_topologies(n_taxa - 1)
    assert len({leaf_bipartitions(t) for t in topologies}) == count_topologies(
        n_taxa - 1
    )


def _leaves(node: Node) -> frozenset[str]:
    if node.is_leaf:
        return frozenset((node.name,))
    result: frozenset[str] = frozenset()
    for child in node.children:
        result |= _leaves(child)
    return result


def _assert_valid_neighbourhood(
    topology: Topology, neighbours: list[Topology], expected_count: int
) -> None:
    assert len(neighbours) == expected_count

    parent_leaves = _leaves(topology)
    parent_key = leaf_bipartitions(topology)
    keys = set()
    for neighbour in neighbours:
        assert validate_unrooted_newick(to_newick(neighbour))
        assert _leaves(neighbour) == parent_leaves, (
            "neighbour must share the parent's leaf set"
        )
        key = leaf_bipartitions(neighbour)
        assert key != parent_key, "neighbour must differ from its parent"
        keys.add(key)
    assert len(keys) == expected_count, "neighbours must be pairwise distinct"


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", EXHAUSTIVE_PARAMS)
def test_nni_neighbour_count_and_validity(n_taxa: int) -> None:
    expected = 2 * (n_taxa - 3)
    for topology in _all_unrooted(n_taxa):
        _assert_valid_neighbourhood(topology, list(nni_neighbours(topology)), expected)


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", EXHAUSTIVE_PARAMS)
def test_spr_neighbour_count_and_validity(n_taxa: int) -> None:
    expected = 2 * (n_taxa - 3) * (2 * n_taxa - 7)
    for topology in _all_unrooted(n_taxa):
        _assert_valid_neighbourhood(topology, list(spr_neighbours(topology)), expected)


@pytest.mark.mathematical
@pytest.mark.parametrize("n_taxa", EXHAUSTIVE_PARAMS)
def test_nni_neighbours_are_symmetric(n_taxa: int) -> None:
    neighbour_keys = {
        leaf_bipartitions(t): {leaf_bipartitions(n) for n in nni_neighbours(t)}
        for t in _all_unrooted(n_taxa)
    }
    for key, neighbours in neighbour_keys.items():
        for neighbour_key in neighbours:
            assert key in neighbour_keys[neighbour_key], (
                "NNI neighbourhood must be symmetric"
            )


@pytest.mark.mathematical
@pytest.mark.parametrize("n_taxa", EXHAUSTIVE_PARAMS)
def test_nni_neighbours_are_spr_neighbours(n_taxa: int) -> None:
    for topology in _all_unrooted(n_taxa):
        nni_keys = {leaf_bipartitions(n) for n in nni_neighbours(topology)}
        spr_keys = {leaf_bipartitions(n) for n in spr_neighbours(topology)}
        assert nni_keys <= spr_keys, "every NNI neighbour must also be an SPR neighbour"


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("n_taxa", [8])
def test_nni_and_spr_exhaustive_at_n8(n_taxa: int) -> None:
    """The same properties as the per-PR sweep, at the next size up.

    Marked ``release`` (``DEV.md``'s CI & Performance Budget): ~2.5 minutes for
    10395 topologies.
    """
    nni_expected = 2 * (n_taxa - 3)
    spr_expected = 2 * (n_taxa - 3) * (2 * n_taxa - 7)
    nni_map = {}
    for topology in _all_unrooted(n_taxa):
        nni_neighbour_list = list(nni_neighbours(topology))
        _assert_valid_neighbourhood(topology, nni_neighbour_list, nni_expected)
        nni_map[leaf_bipartitions(topology)] = {
            leaf_bipartitions(n) for n in nni_neighbour_list
        }

    for key, neighbours in nni_map.items():
        for neighbour_key in neighbours:
            assert key in nni_map[neighbour_key]

    for topology in _all_unrooted(n_taxa):
        spr_neighbour_list = list(spr_neighbours(topology))
        _assert_valid_neighbourhood(topology, spr_neighbour_list, spr_expected)
        spr_keys = {leaf_bipartitions(n) for n in spr_neighbour_list}
        assert nni_map[leaf_bipartitions(topology)] <= spr_keys


# --- the deduplication key, and the neighbourhood it deduplicates ---------------


@pytest.mark.oracle
def test_the_bitmask_split_key_is_leaf_bipartitions_on_every_topology() -> None:
    # `spr_neighbours` deduplicates on `_split_key` since #264; it must name the
    # same splits `leaf_bipartitions` does, on every one of the 105 six-leaf
    # topologies, with the anchor convention applied identically.
    names = [f"t{i}" for i in range(6)]
    checked = 0
    for candidate in enumerate_topologies(names):
        adjacency, root_id = topology_module._to_adjacency(candidate)
        bit_of = topology_module._leaf_bits(adjacency)
        expected = frozenset(
            sum(1 << bit_of[name] for name in side)
            for side in leaf_bipartitions(candidate)
        )

        assert topology_module._split_key(adjacency, root_id, bit_of) == expected
        checked += 1
    assert checked == 105


def _spr_by_definition(start: Topology) -> list[frozenset[frozenset[str]]]:
    """Every prune-and-regraft, deduplicated on `leaf_bipartitions`, in order.

    The definition `spr_neighbours` implemented before #264, kept here as the
    oracle for what it implements now: the same candidates in the same order
    with the same first-seen rule, keyed the slow way.
    """
    adjacency, _ = topology_module._to_adjacency(start)
    seen = {leaf_bipartitions(start)}
    keys = []
    for u in list(adjacency):
        if not isinstance(u, int):
            continue
        for v in list(adjacency[u]):
            remainder, pruned, _ = topology_module._prune(adjacency, u, v)
            for x, y in topology_module._edges(remainder):
                grafted, new_id = topology_module._regraft(remainder, pruned, v, x, y)
                key = leaf_bipartitions(
                    topology_module._from_adjacency(grafted, new_id)
                )
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
    return keys


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", [5, 7, 9])
def test_spr_neighbours_are_the_definition_in_the_definition_order(n_taxa: int) -> None:
    # Order matters as well as membership: a hill climb takes the first of
    # equally good neighbours, so a faster generator that permuted them would
    # change which tree a search lands on while every count test passed.
    start = random_topology(
        [f"x{i}" for i in range(n_taxa)], np.random.default_rng(n_taxa)
    )

    generated = [leaf_bipartitions(neighbour) for neighbour in spr_neighbours(start)]

    assert generated == _spr_by_definition(start)
    assert len(generated) == 2 * (n_taxa - 3) * (2 * n_taxa - 7)


# --- the bounded regraft (issue #408) -------------------------------------


@pytest.mark.structural
@pytest.mark.parametrize("n_taxa", [5, 6, 7, 8, 9])
def test_spr_radius_at_the_leaf_count_is_the_unbounded_neighbourhood(
    n_taxa: int,
) -> None:
    # At a radius no edge can exceed, a bounded search is the unbounded one
    # candidate for candidate. Order is asserted as well as membership, a hill
    # climb taking the first of equally good neighbours.
    start = random_topology(
        [f"x{i}" for i in range(n_taxa)], np.random.default_rng(n_taxa)
    )

    unbounded = [leaf_bipartitions(neighbour) for neighbour in spr_neighbours(start)]
    bounded = [
        leaf_bipartitions(neighbour)
        for neighbour in spr_neighbours(start, radius=n_taxa)
    ]

    assert bounded == unbounded
    assert len(bounded) == 2 * (n_taxa - 3) * (2 * n_taxa - 7)


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", [5, 6, 7, 8])
def test_spr_radius_one_is_the_nni_neighbourhood(n_taxa: int) -> None:
    # The reduction the module CLAUDE.md asks for: at radius 1 the general
    # construction must reproduce the simpler one already validated. A
    # mis-measured radius yields a merely *smaller* neighbourhood, which no
    # count test catches on its own.
    start = random_topology(
        [f"x{i}" for i in range(n_taxa)], np.random.default_rng(n_taxa)
    )

    bounded = {
        leaf_bipartitions(neighbour) for neighbour in spr_neighbours(start, radius=1)
    }

    assert bounded == {
        leaf_bipartitions(neighbour) for neighbour in nni_neighbours(start)
    }
    assert len(bounded) == 2 * (n_taxa - 3)


@pytest.mark.mathematical
@pytest.mark.parametrize("n_taxa", [6, 8])
def test_spr_radius_nests(n_taxa: int) -> None:
    # A radius is a bound on one distance, so the neighbourhoods are nested
    # and saturate. A distance defined on the wrong endpoint would still
    # grow with the radius but would not nest.
    start = random_topology(
        [f"x{i}" for i in range(n_taxa)], np.random.default_rng(n_taxa)
    )

    sets = [
        {
            leaf_bipartitions(neighbour)
            for neighbour in spr_neighbours(start, radius=radius)
        }
        for radius in range(1, n_taxa + 1)
    ]

    for smaller, larger in pairwise(sets):
        assert smaller <= larger
    assert sets[-1] == {
        leaf_bipartitions(neighbour) for neighbour in spr_neighbours(start)
    }


@pytest.mark.edge_case
def test_spr_radius_below_one_is_refused() -> None:
    # Radius 0 admits only the vacated edge, which reconstructs the parent,
    # so the neighbourhood is empty and a search from it cannot move.
    start = random_topology([f"x{i}" for i in range(6)], np.random.default_rng(6))

    with pytest.raises(ValueError, match="radius must be at least 1"):
        list(spr_neighbours(start, radius=0))
