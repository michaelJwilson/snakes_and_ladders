"""Regression tests for ``phylo.search.topology``'s NNI and SPR generators.

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
``n = 8`` (10395 topologies) is marked ``release`` (``DEV.md``'s
Release-Gated budget line) since it takes ~2.5 minutes and is redundant
evidence once ``n <= 7`` passes -- run it with ``pytest -m release``. The
enumeration itself (no neighbour generation) is cheap even at ``n = 8``
(~1s) and stays in the per-PR suite as a check on the brute-force oracle.

A random-walk connectivity test (every topology reachable from a seeded
NNI-only walk) is deferred: it needs #73's canonical Newick key to identify
"every topology" visited, and lands once #73 merges (see issue #79's plan
comment).
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from itertools import combinations

import pytest
from phylo.search.topology import (
    Topology,
    leaf_bipartitions,
    nni_neighbours,
    spr_neighbours,
)
from phylo.sim.newick import count_topologies, to_newick, validate_unrooted_newick
from phylo.sim.tree import Node

# Sizes at which exhaustive enumeration is the oracle. The per-pull-request
# budget holds 5 and 6; 7 costs about 8 s across the two neighbourhood tests
# below and is the size that shows the count formula holding as the
# neighbourhood grows, so it runs under `stress` (`DEV.md`, CI & Performance
# Budget). The claim asserted is identical at every size.
EXHAUSTIVE_SIZES = (5, 6)
STRESS_SIZES = (7,)
EXHAUSTIVE_PARAMS = [
    *(pytest.param(size) for size in EXHAUSTIVE_SIZES),
    *(pytest.param(size, marks=pytest.mark.stress) for size in STRESS_SIZES),
]


def _enumerate_rooted(taxa: tuple[str, ...]) -> Iterator[Node]:
    """Brute-force-enumerate every rooted binary topology on ``taxa``.

    Identical construction to ``tests/regression/test_newick.py``'s
    ``_enumerate_topologies``, duplicated locally to keep this test module
    independent (root ``CLAUDE.md``'s test-isolation convention, implicit
    in every other regression test file here).
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

    Bijection with rooted binary topologies on ``n_taxa - 1`` leaves: root
    a rooted tree's own root has no incoming edge, so grafting one more
    leaf directly onto it as a 3rd child recreates exactly the
    trifurcating-root convention, one-to-one with attaching that leaf via
    every possible edge of the corresponding unrooted tree. This is why
    ``count_topologies(n_taxa - 1)`` is the right oracle count (module
    docstring of ``phylo.sim.newick``).
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


@pytest.mark.parametrize("n_taxa", EXHAUSTIVE_PARAMS)
def test_nni_neighbour_count_and_validity(n_taxa: int) -> None:
    expected = 2 * (n_taxa - 3)
    for topology in _all_unrooted(n_taxa):
        _assert_valid_neighbourhood(topology, list(nni_neighbours(topology)), expected)


@pytest.mark.parametrize("n_taxa", EXHAUSTIVE_PARAMS)
def test_spr_neighbour_count_and_validity(n_taxa: int) -> None:
    expected = 2 * (n_taxa - 3) * (2 * n_taxa - 7)
    for topology in _all_unrooted(n_taxa):
        _assert_valid_neighbourhood(topology, list(spr_neighbours(topology)), expected)


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


@pytest.mark.parametrize("n_taxa", EXHAUSTIVE_PARAMS)
def test_nni_neighbours_are_spr_neighbours(n_taxa: int) -> None:
    for topology in _all_unrooted(n_taxa):
        nni_keys = {leaf_bipartitions(n) for n in nni_neighbours(topology)}
        spr_keys = {leaf_bipartitions(n) for n in spr_neighbours(topology)}
        assert nni_keys <= spr_keys, "every NNI neighbour must also be an SPR neighbour"


@pytest.mark.release
@pytest.mark.parametrize("n_taxa", [8])
def test_nni_and_spr_exhaustive_at_n8(n_taxa: int) -> None:
    """The same properties as the per-PR sweep, at the next size up.

    Marked ``release`` (``DEV.md``'s CI & Performance Budget): ~2.5 minutes
    for 10395 topologies, run on release rather than per PR.
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
