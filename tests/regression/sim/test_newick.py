"""Regression tests for ``phylo.sim.newick``.

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

import pytest
from phylo.sim.newick import (
    count_topologies,
    to_newick,
    validate_newick,
    validate_unrooted_newick,
)
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params.yaml"

# simulation_params.yaml's root has 3 children (A, B, ancestor_CD): the
# common "unrooted tree drawn at a trifurcating root" convention, correctly
# rejected by validate_newick as not strictly binary. The 8-taxon fixture is
# binary at every node including the root, so it exercises validate_newick
# and the state-labelled round trip below.
BINARY_FIXTURE = FIXTURES_DIR / "simulation_params_8taxa.yaml"


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


@pytest.mark.edge_case
def test_count_topologies_rejects_non_positive_n_taxa() -> None:
    with pytest.raises(ValueError, match="n_taxa"):
        count_topologies(0)


@pytest.mark.structural
def test_validate_newick_accepts_a_simulated_binary_tree() -> None:
    params = load_simulation_params(BINARY_FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=10
    )

    assert validate_newick(dataset.newick)


@pytest.mark.edge_case
def test_validate_newick_rejects_a_trifurcating_root() -> None:
    params = load_simulation_params(FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=10
    )

    assert not validate_newick(dataset.newick)


@pytest.mark.oracle
def test_to_newick_with_node_states_round_trips_ancestor_labels() -> None:
    params = load_simulation_params(BINARY_FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=10
    )

    labelled = to_newick(dataset.tau, dataset.node_states, site=0)

    assert validate_newick(labelled)
    for node in preorder(dataset.tau):
        expected_state = int(dataset.node_states[node.name][0])
        assert f"[&state={expected_state}]" in labelled


@pytest.mark.edge_case
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


@pytest.mark.edge_case
def test_validate_newick_accepts_a_single_leaf() -> None:
    assert validate_newick("A;")


@pytest.mark.structural
def test_validate_newick_accepts_branch_lengths_and_internal_labels() -> None:
    assert validate_newick("(A:0.1,(B:0.2,C:0.3)anc:0.05)root;")


@pytest.mark.structural
def test_validate_unrooted_newick_accepts_a_trifurcating_root() -> None:
    params = load_simulation_params(FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=10
    )

    assert validate_unrooted_newick(dataset.newick)


@pytest.mark.edge_case
def test_validate_unrooted_newick_rejects_a_strictly_binary_root() -> None:
    # A rooted binary tree (2 children at the root) is not the trifurcating-
    # root convention: validate_newick and validate_unrooted_newick partition
    # well-formed Newick strings into disjoint sets.
    assert not validate_unrooted_newick("(A,(B,C)anc);")


@pytest.mark.edge_case
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


@pytest.mark.structural
def test_validate_unrooted_newick_accepts_binary_subtrees_under_the_root() -> None:
    assert validate_unrooted_newick("(A,B,(C,D)anc:0.1)root;")
