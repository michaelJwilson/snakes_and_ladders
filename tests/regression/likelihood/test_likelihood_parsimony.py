"""Fitch and Sankoff parsimony, and the region where being wrong is the prediction.

Two kinds of test, and the second is the reason the module exists.

The algorithms are pinned against exhaustive enumeration over internal-node
labellings, sharing no traversal with it --- the same relationship
`brute_force_log_likelihood` has to the pruning recursion. The score is an
integer, so agreement is equality and not a tolerance. Sankoff is pinned
twice: by reduction to Fitch under the unit step matrix on every topology of
the five-taxon fixture, and against a brute force written here, over the
directed edge costs of a planted asymmetric matrix.

Then the zones. In the *Felsenstein zone* parsimony is statistically
inconsistent: its error rate converges to 1, not 0, as sites increase. That is
a theorem, so a test can assert it as a prediction rather than discover it as
a defect. The *Farris zone* is the control that makes the first
interpretable --- move the same two long branches to be adjacent and parsimony
becomes correct and fast. An implementation that is simply broken fails both.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.likelihood.parsimony import (
    brute_force_parsimony_score,
    fitch_score,
    sankoff_score,
    unit_step_matrix,
)
from snakes_and_ladders.search.infer import score_topology
from snakes_and_ladders.search.topology import enumerate_topologies, leaf_bipartitions
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node, edges, preorder

from tests._fixtures import (
    FARRIS_ZONE,
    FELSENSTEIN_ZONE,
    FOUR_TAXA_LEAVES,
    ZONE_TRUE_SPLIT,
    balanced_four_taxa,
    load_fixture,
)

FIVE_TAXA = "tree_search/ci.yaml"
UNIFORM = np.full(4, 0.25)

# Transitions (within {0, 1} and within {2, 3}) cost 1, transversions 2: a
# metric, and the weighting a phylogeneticist actually uses.
TRANSITION_TRANSVERSION = np.array(
    [[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 1], [2, 2, 1, 0]], dtype=np.float64
)

# Planted so that no two directions cost the same: W[i, j] != W[j, i] for
# every pair, which is what makes the rooted brute force below a check of
# the direction Sankoff reads the step matrix in.
ASYMMETRIC = np.array(
    [[0, 1, 3, 4], [2, 0, 1, 3], [4, 2, 0, 1], [1, 3, 2, 0]], dtype=np.float64
)


def _recovery_rates(tau: Node, n_sites: int, replicates: int) -> tuple[int, int]:
    """How often each criterion picks the true topology out of all three."""
    parsimony_correct = likelihood_correct = 0
    for replicate in range(replicates):
        dataset = simulate_alignment(
            tau, 4, UNIFORM, np.random.default_rng(1000 + replicate), n_sites
        )
        alignment = dataset.alignment
        topologies = list(enumerate_topologies(FOUR_TAXA_LEAVES))

        parsimony = [fitch_score(topology, alignment, 4) for topology in topologies]
        likelihood = [score_topology(topology, alignment, 4) for topology in topologies]

        best_parsimony = topologies[int(np.argmin(parsimony))]
        best_likelihood = topologies[int(np.argmax(likelihood))]
        parsimony_correct += leaf_bipartitions(best_parsimony) == ZONE_TRUE_SPLIT
        likelihood_correct += leaf_bipartitions(best_likelihood) == ZONE_TRUE_SPLIT
    return parsimony_correct, likelihood_correct


@pytest.mark.oracle
def test_fitch_matches_exhaustive_enumeration_over_internal_labellings() -> None:
    # The oracle assigns states to internal nodes directly and counts
    # disagreeing edges; Fitch intersects state sets in one post-order pass.
    # No traversal is shared, so agreement is evidence rather than a tautology.
    rng = np.random.default_rng(0)
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)

    for _ in range(8):
        alignment = {name: rng.integers(0, 4, size=12) for name in FOUR_TAXA_LEAVES}

        assert fitch_score(tau, alignment, 4) == brute_force_parsimony_score(
            tau, alignment, 4
        )


@pytest.mark.oracle
def test_fitch_matches_a_score_worked_out_by_hand() -> None:
    # Three sites chosen so each exercises a different branch of the
    # recursion: an informative split, a constant site, and a site where every
    # taxon differs.
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {
        "A": np.array([0, 0, 0]),
        "B": np.array([0, 0, 1]),
        "C": np.array([1, 0, 2]),
        "D": np.array([1, 0, 3]),
    }

    # 1 change for the AA|BB split, 0 for the constant site, 3 for the site
    # with four distinct states on four taxa.
    assert fitch_score(tau, alignment, 4) == 1 + 0 + 3


@pytest.mark.edge_case
def test_a_constant_alignment_needs_no_changes() -> None:
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(20, dtype=np.int64) for name in FOUR_TAXA_LEAVES}

    assert fitch_score(tau, alignment, 4) == 0


@pytest.mark.edge_case
def test_a_missing_leaf_is_refused() -> None:
    # Silently scoring the subtree it can reach would return a smaller number
    # for the wrong reason, and smaller is better under this criterion.
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(5, dtype=np.int64) for name in ("A", "B", "C")}

    with pytest.raises(ValueError, match="not in the alignment"):
        fitch_score(tau, alignment, 4)


@pytest.mark.edge_case
def test_sequences_of_different_lengths_are_refused() -> None:
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(5, dtype=np.int64) for name in FOUR_TAXA_LEAVES}
    alignment["D"] = np.zeros(4, dtype=np.int64)

    with pytest.raises(ValueError, match="differ in length"):
        fitch_score(tau, alignment, 4)


@pytest.mark.edge_case
@pytest.mark.parametrize("k", [1, 64])
def test_a_state_count_a_bitmask_cannot_hold_is_refused(k: int) -> None:
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(4, dtype=np.int64) for name in FOUR_TAXA_LEAVES}

    with pytest.raises(ValueError, match=r"k must be in \[2, 63\]"):
        fitch_score(tau, alignment, k)


def _directed_brute_force(
    tau: Node, alignment: dict[str, np.ndarray], step_matrix: np.ndarray
) -> float:
    """Minimum over every internal labelling of the summed directed edge costs.

    Independent of :func:`sankoff_score`: it assigns states to internal nodes
    and reads ``step_matrix[parent, child]`` on every edge of the rooted
    tree, sharing no recursion with the dynamic programme.
    """
    internal = [node.name for node in preorder(tau) if not node.is_leaf]
    edge_list = [(parent.name, child.name) for parent, child in edges(tau)]
    n_sites = next(iter(alignment.values())).shape[0]
    total = 0.0
    for site in range(n_sites):
        observed = {name: int(states[site]) for name, states in alignment.items()}
        best = np.inf
        for labelling in itertools.product(range(4), repeat=len(internal)):
            state = dict(zip(internal, labelling, strict=True)) | observed
            cost = sum(step_matrix[state[p], state[c]] for p, c in edge_list)
            best = min(best, cost)
        total += best
    return total


@pytest.mark.oracle
def test_sankoff_with_the_unit_matrix_is_fitch_on_every_five_taxon_topology() -> None:
    # The reduction: with every change costing 1, the min-plus recursion must
    # return the set recursion's count, and exactly -- 15 of 15 topologies of
    # the five-taxon fixture at 1,200 sites, equality not tolerance.
    params = load_fixture(FIVE_TAXA)
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(params.seed), 1200
    )
    alignment = dict(dataset.alignment)
    unit = unit_step_matrix(params.k)

    for topology in enumerate_topologies(sorted(alignment)):
        assert sankoff_score(topology, alignment, unit) == fitch_score(
            topology, alignment, params.k
        )


@pytest.mark.oracle
def test_sankoff_matches_a_brute_force_minimum_under_an_asymmetric_matrix() -> None:
    # 4 ** 3 labellings of the three internal nodes per site, every edge read
    # parent to child. An implementation that transposed the matrix, or that
    # took the root's state from a child rather than minimizing over it,
    # agrees with the symmetric cases and fails this one.
    rng = np.random.default_rng(2)
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)

    for _ in range(6):
        alignment = {name: rng.integers(0, 4, size=8) for name in FOUR_TAXA_LEAVES}

        assert sankoff_score(tau, alignment, ASYMMETRIC) == _directed_brute_force(
            tau, alignment, ASYMMETRIC
        )


@pytest.mark.oracle
def test_sankoff_matches_a_weighted_score_worked_out_by_hand() -> None:
    # The same three sites as the Fitch case under transition/transversion
    # weights: 1 for the AA|BB split (one transition), 0 for the constant
    # site, and 4 for the site with four distinct states -- one transition
    # inside each cherry and one transversion across the root -- where Fitch
    # counts 3.
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {
        "A": np.array([0, 0, 0]),
        "B": np.array([0, 0, 1]),
        "C": np.array([1, 0, 2]),
        "D": np.array([1, 0, 3]),
    }

    assert sankoff_score(tau, alignment, TRANSITION_TRANSVERSION) == 1 + 0 + 4
    assert fitch_score(tau, alignment, 4) == 1 + 0 + 3


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("step_matrix", "message"),
    [
        (np.zeros((4, 3)), "must be square"),
        (np.zeros((1, 1)), "must be square"),
        (-unit_step_matrix(4), "finite and non-negative"),
        (np.full((4, 4), np.inf), "finite and non-negative"),
    ],
)
def test_an_unusable_step_matrix_is_refused(
    step_matrix: np.ndarray, message: str
) -> None:
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(4, dtype=np.int64) for name in FOUR_TAXA_LEAVES}

    with pytest.raises(ValueError, match=message):
        sankoff_score(tau, alignment, step_matrix)


@pytest.mark.edge_case
def test_a_state_the_step_matrix_does_not_cover_is_refused() -> None:
    # Indexing past the matrix would raise an IndexError from inside the
    # recursion, naming nothing; a state of 4 under a 4-state matrix is an
    # alignment error and is reported as one.
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(4, dtype=np.int64) for name in FOUR_TAXA_LEAVES}
    alignment["C"] = np.array([0, 4, 0, 0])

    with pytest.raises(ValueError, match=r"leaf 'C' carries a state outside \[0, 4\)"):
        sankoff_score(tau, alignment, unit_step_matrix(4))


@pytest.mark.edge_case
def test_sankoff_refuses_what_fitch_refuses() -> None:
    tau = balanced_four_taxa(0.1, 0.1, 0.1, 0.1)
    unit = unit_step_matrix(4)

    with pytest.raises(ValueError, match="not in the alignment"):
        sankoff_score(tau, {n: np.zeros(5, dtype=np.int64) for n in "ABC"}, unit)
    alignment = {name: np.zeros(5, dtype=np.int64) for name in FOUR_TAXA_LEAVES}
    alignment["D"] = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="differ in length"):
        sankoff_score(tau, alignment, unit)


@pytest.mark.simulated_truth
def test_parsimony_is_inconsistent_in_the_felsenstein_zone() -> None:
    # The theorem, as a prediction. Parsimony does not merely do badly here --
    # more data does not help, because the systematic pull toward grouping the
    # two long branches grows with the data exactly as the true signal does.
    # Measured over 12 replicates: 0/12 correct at 200, 1000 and 5000 sites.
    few_parsimony, _ = _recovery_rates(FELSENSTEIN_ZONE, n_sites=200, replicates=6)
    many_parsimony, _ = _recovery_rates(FELSENSTEIN_ZONE, n_sites=2000, replicates=6)

    assert few_parsimony == 0
    assert many_parsimony == 0


@pytest.mark.simulated_truth
def test_likelihood_is_consistent_in_the_felsenstein_zone() -> None:
    # The other half of the same claim, and the reason the zone is the
    # canonical argument for the criterion this repository actually uses.
    # Measured: 10/12 at 200 sites, 12/12 at 1000 and 5000.
    _, few_likelihood = _recovery_rates(FELSENSTEIN_ZONE, n_sites=200, replicates=6)
    _, many_likelihood = _recovery_rates(FELSENSTEIN_ZONE, n_sites=2000, replicates=6)

    assert many_likelihood == 6
    assert many_likelihood >= few_likelihood


@pytest.mark.simulated_truth
def test_parsimony_is_correct_and_fast_in_the_farris_zone() -> None:
    # The control. Without it, "parsimony got the Felsenstein zone wrong" is
    # indistinguishable from "this parsimony implementation is broken".
    # Measured: 12/12 at every site count, and likelihood 4/12, 6/12, 10/12 --
    # so in this zone parsimony is the *faster* of the two, which is the
    # published result and not an artifact.
    parsimony, _ = _recovery_rates(FARRIS_ZONE, n_sites=200, replicates=6)

    assert parsimony == 6


@pytest.mark.edge_case
@pytest.mark.mathematical
def test_a_zero_length_internal_branch_leaves_the_three_topologies_tied() -> None:
    # An analytic corner: with no internal branch there is no split to detect,
    # so no topology should be preferred and a strict preference would be
    # reading noise as signal.
    #
    # Asserted across seeds rather than by a spread threshold on one. A single
    # replicate always has a winner -- measured spreads of 0.5% to 3.3% -- so
    # a threshold either passes trivially or fails on an unlucky draw. What
    # says "no signal" is that the winner is *uniform*: over 30 seeds each of
    # the three topologies won 12, 9 and 9 times.
    star = Node(
        "root",
        None,
        (
            Node("i1", 0.0, (Node("A", 0.3), Node("B", 0.3))),
            Node("i2", 0.0, (Node("C", 0.3), Node("D", 0.3))),
        ),
    )
    topologies = list(enumerate_topologies(FOUR_TAXA_LEAVES))

    winners = set()
    for seed in range(15):
        dataset = simulate_alignment(
            star, 4, UNIFORM, np.random.default_rng(5000 + seed), 400
        )
        scores = [
            fitch_score(topology, dataset.alignment, 4) for topology in topologies
        ]
        winners.add(int(np.argmin(scores)))

    # Each topology winning at least once over 15 draws has probability 0.993
    # under a uniform winner, and is essentially impossible under a systematic
    # preference for any one of them.
    assert winners == {0, 1, 2}
