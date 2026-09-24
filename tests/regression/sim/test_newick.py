"""Regression tests for ``snakes_and_ladders.sim.newick``.

Per root ``CLAUDE.md`` ("Pin to Independent Sources"), ``count_topologies``
is checked against an independent brute-force enumeration rather than
re-deriving its own closed form, and ``validate_newick`` is checked against
both strings it must accept (round-tripped simulated trees) and strings it
must reject (malformed by construction, not by inspection of the code under
test).
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pytest
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.sim.newick import (
    _parse_newick,
    _parse_unrooted_newick,
    count_topologies,
    to_newick,
    validate_newick,
    validate_unrooted_newick,
)
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulator import simulate_tree
from snakes_and_ladders.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "tree_jc/stress.yaml"

# tree_jc/stress.yaml's root has 3 children (A, B, ancestor_CD): the
# common "unrooted tree drawn at a trifurcating root" convention, correctly
# rejected by validate_newick as not strictly binary. The 8-taxon fixture is
# binary at every node including the root, so it exercises validate_newick
# and the state-labelled round trip below.
BINARY_FIXTURE = FIXTURES_DIR / "tree_jc/release.yaml"


def _enumerate_topologies(taxa: tuple[str, ...]) -> Iterator[Node]:
    """Brute-force-enumerate every rooted binary topology on ``taxa``.

    Fixes ``taxa[0]`` into the left bipartition of every split, which visits
    each unordered topology exactly once: this is the standard construction
    behind the ``(2n-3)!!`` count (see ``newick.py``'s module docstring), so
    counting the topologies it yields is an independent check on
    ``count_topologies``, not a restatement of it.
    """
    if len(taxa) == 1:
        yield Node(name=taxa[0], branch_length=1.0)
        return

    first, rest = taxa[0], taxa[1:]
    for size in range(1, len(rest) + 1):
        for right_taxa in combinations(rest, size):
            right_set = set(right_taxa)
            left_taxa = (first, *(t for t in rest if t not in right_set))
            for left in _enumerate_topologies(left_taxa):
                for right in _enumerate_topologies(right_taxa):
                    yield Node(
                        name="internal", branch_length=1.0, children=(left, right)
                    )


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", [1, 2, 3, 4, 5, 6])
def test_count_topologies_matches_brute_force_enumeration(n_taxa: int) -> None:
    taxa = tuple(f"t{i}" for i in range(n_taxa))
    brute_force_count = sum(1 for _ in _enumerate_topologies(taxa))

    assert count_topologies(n_taxa) == brute_force_count


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("n_taxa", "expected"),
    [(1, 1), (2, 1), (3, 3), (4, 15), (5, 105), (6, 945)],
)
def test_count_topologies_matches_known_values(n_taxa: int, expected: int) -> None:
    assert count_topologies(n_taxa) == expected


@pytest.mark.smoke
def test_count_topologies_rejects_non_positive_n_taxa() -> None:
    with pytest.raises(ValueError, match="n_taxa"):
        count_topologies(0)


@pytest.mark.smoke
def test_validate_newick_accepts_a_simulated_binary_tree() -> None:
    params = load_params(BINARY_FIXTURE, SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=10)

    assert validate_newick(dataset.newick)


@pytest.mark.smoke
def test_validate_newick_rejects_a_trifurcating_root() -> None:
    params = load_params(FIXTURE, SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=10)

    assert not validate_newick(dataset.newick)


@pytest.mark.oracle
def test_to_newick_with_node_states_round_trips_ancestor_labels() -> None:
    params = load_params(BINARY_FIXTURE, SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=10)

    labelled = to_newick(dataset.tau, dataset.node_states, site=0)

    assert validate_newick(labelled)
    for node in preorder(dataset.tau):
        expected_state = int(dataset.node_states[node.name][0])
        assert f"[&state={expected_state}]" in labelled


@pytest.mark.smoke
@pytest.mark.parametrize(
    "malformed",
    [
        "(A,B",  # unbalanced: missing closing paren and ';'
        "(A,B))",  # unbalanced: extra closing paren
        "(A,B,C);",  # non-binary: three children
        "(A);",  # non-binary: one child
        "(A,B)",  # missing terminating ';'
        "(A,B);extra",  # trailing characters after ';'
        "(,B);",  # empty leaf label
        "(A:notanumber,B);",  # invalid branch length
        "(A[unterminated,B);",  # unterminated comment
        "",  # empty string
    ],
)
def test_validate_newick_rejects_malformed_strings(malformed: str) -> None:
    assert not validate_newick(malformed)


@pytest.mark.smoke
def test_validate_newick_accepts_a_single_leaf() -> None:
    assert validate_newick("A;")


@pytest.mark.smoke
def test_validate_newick_accepts_branch_lengths_and_internal_labels() -> None:
    assert validate_newick("(A:0.1,(B:0.2,C:0.3)anc:0.05)root;")


@pytest.mark.smoke
def test_validate_unrooted_newick_accepts_a_trifurcating_root() -> None:
    params = load_params(FIXTURE, SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites=10)

    assert validate_unrooted_newick(dataset.newick)


@pytest.mark.smoke
def test_validate_unrooted_newick_rejects_a_strictly_binary_root() -> None:
    # A rooted binary tree (2 children at the root) is not the trifurcating-
    # root convention: validate_newick and validate_unrooted_newick partition
    # well-formed Newick strings into disjoint sets.
    assert not validate_unrooted_newick("(A,(B,C)anc);")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "malformed",
    [
        "(A,B,C",  # unbalanced: missing closing paren and ';'
        "(A,B,C,D);",  # 4 children at the root, not 3
        "(A);",  # 1 child at the root: missing ',' after the first
        "(A,B);",  # 2 children at the root, not 3 (validate_newick's shape)
        "(A,B,C))",  # unbalanced: extra closing paren
        "(A,B,C)",  # missing terminating ';'
        "(A,B,C);extra",  # trailing characters after ';'
        "(,B,C);",  # empty leaf label
        "",  # empty string
    ],
)
def test_validate_unrooted_newick_rejects_malformed_strings(malformed: str) -> None:
    assert not validate_unrooted_newick(malformed)


@pytest.mark.smoke
def test_validate_unrooted_newick_accepts_binary_subtrees_under_the_root() -> None:
    assert validate_unrooted_newick("(A,B,(C,D)anc:0.1)root;")


# The two trees the Newick format's own documentation carries (Felsenstein,
# PHYLIP's `newicktree.html`; the six-species primate tree of Olsen's
# description of the format). They are written here as the literature states
# them, to the digit, so what the parser returns is read against a source
# outside this repository rather than against a tree this repository built.
PUBLISHED_ROOTED = "(((One:0.2,Two:0.3):0.3,(Three:0.5,Four:0.3):0.2):0.3,Five:0.7);"
PUBLISHED_UNROOTED = (
    "(Bovine:0.69395,(Gibbon:0.36079,(Orang:0.33636,(Gorilla:0.17147,"
    "(Chimp:0.19268,Human:0.11927):0.08386):0.06124):0.15057):0.54939,"
    "Mouse:1.21460);"
)

#: The primate tree's branch lengths, leaf by leaf, as the published string
#: writes them. Read back by name, so a permuted parse fails here and not on
#: a count.
PUBLISHED_UNROOTED_LEAVES = {
    "Bovine": 0.69395,
    "Gibbon": 0.36079,
    "Orang": 0.33636,
    "Gorilla": 0.17147,
    "Chimp": 0.19268,
    "Human": 0.11927,
    "Mouse": 1.21460,
}


@pytest.mark.end2end
def test_the_published_rooted_tree_round_trips_byte_for_byte() -> None:
    # Parse and write against a string this repository did not produce: the
    # rooted five-taxon tree of the Newick format's documentation. Byte
    # equality is the tolerance -- every length in it is a float whose
    # shortest repr is the published digits -- so a reordered child, a
    # dropped internal branch or a rounded length all fail.
    parsed = _parse_newick(PUBLISHED_ROOTED)

    assert validate_newick(PUBLISHED_ROOTED)
    assert to_newick(parsed) == PUBLISHED_ROOTED
    assert [node.name for node in preorder(parsed) if node.is_leaf] == [
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
    ]
    # In preorder, so the two unnamed internal nodes are told apart by where
    # they sit rather than by a name the format does not give them.
    assert [(node.name, node.branch_length) for node in preorder(parsed)] == [
        ("", None),
        ("", 0.3),
        ("", 0.3),
        ("One", 0.2),
        ("Two", 0.3),
        ("", 0.2),
        ("Three", 0.5),
        ("Four", 0.3),
        ("Five", 0.7),
    ]


@pytest.mark.end2end
def test_the_published_primate_tree_reads_back_its_topology_and_lengths() -> None:
    # The trifurcating-root convention, on the six-species primate tree the
    # format's description carries. The topology is read back as the nested
    # leaf sets of the root's three subtrees and every leaf's length by name,
    # exactly: these are decimal literals a double represents to 1e-17, so
    # the comparison is equality and not a tolerance.
    parsed = _parse_unrooted_newick(PUBLISHED_UNROOTED)

    assert validate_unrooted_newick(PUBLISHED_UNROOTED)
    assert not validate_newick(PUBLISHED_UNROOTED)
    assert len(parsed.children) == 3

    def leaves(node: Node) -> frozenset[str]:
        return frozenset(one.name for one in preorder(node) if one.is_leaf)

    assert [leaves(child) for child in parsed.children] == [
        frozenset({"Bovine"}),
        frozenset({"Gibbon", "Orang", "Gorilla", "Chimp", "Human"}),
        frozenset({"Mouse"}),
    ]
    # Chimp and Human are sisters, and that clade sits under Gorilla: the
    # nesting the string states, checked as a set rather than as a string.
    inner = parsed.children[1].children[1].children[1]
    assert leaves(inner) == frozenset({"Gorilla", "Chimp", "Human"})
    assert leaves(inner.children[1]) == frozenset({"Chimp", "Human"})

    read = {node.name: node.branch_length for node in preorder(parsed) if node.is_leaf}
    assert read == PUBLISHED_UNROOTED_LEAVES
    # Written back, the one difference from the published string is the
    # trailing zero of `1.21460`, which a float's shortest repr drops.
    assert to_newick(parsed) == PUBLISHED_UNROOTED.replace("1.21460", "1.2146")


@pytest.mark.oracle
def test_a_published_length_survives_a_comment_between_it_and_its_label() -> None:
    # The state-labelled form this module writes is the published grammar's
    # comment, so a published tree carrying one must parse to the same tree
    # with the same lengths. Checked against the uncommented parse, which is
    # the literature's string and not a second run of the annotator.
    annotated = PUBLISHED_ROOTED.replace("One:0.2", "One[&state=2]:0.2").replace(
        "Five:0.7", "Five[&state=0]:0.7"
    )

    assert validate_newick(annotated)
    assert to_newick(_parse_newick(annotated)) == PUBLISHED_ROOTED
