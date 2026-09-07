"""Fitch parsimony, and the region where being wrong is the prediction.

Two kinds of test, and the second is the reason the module exists.

The algorithm is pinned against exhaustive enumeration over internal-node
labellings, sharing no traversal with it --- the same relationship
`brute_force_log_likelihood` has to the pruning recursion. The score is an
integer, so agreement is equality and not a tolerance.

Then the zones. In the *Felsenstein zone* parsimony is statistically
inconsistent: its error rate converges to 1, not 0, as sites increase. That is
a theorem, so a test can assert it as a prediction rather than discover it as
a defect. The *Farris zone* is the control that makes the first
interpretable --- move the same two long branches to be adjacent and parsimony
becomes correct and fast. An implementation that is simply broken fails both.
"""

from __future__ import annotations

import numpy as np
import pytest
from phylo.likelihood.parsimony import brute_force_parsimony_score, fitch_score
from phylo.search.infer import score_topology
from phylo.search.topology import enumerate_topologies, leaf_bipartitions
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node

LEAVES = ("A", "B", "C", "D")

# A long branch is long enough for convergent change to be common; the
# internal branch is short enough that little signal supports the true split.
# These are the standard proportions for the zone, not values tuned until the
# effect appeared.
LONG, SHORT, INTERNAL = 0.75, 0.02, 0.02


def _balanced(first: float, second: float, third: float, fourth: float) -> Node:
    """``((A,B),(C,D))`` with the four pendant branch lengths given."""
    return Node(
        "root",
        None,
        (
            Node("i1", INTERNAL, (Node("A", first), Node("B", second))),
            Node("i2", INTERNAL, (Node("C", third), Node("D", fourth))),
        ),
    )


# Felsenstein: the long branches are A and C, which are *not* a cherry in the
# true tree, so grouping them is the mistake convergent change invites.
FELSENSTEIN = _balanced(LONG, SHORT, LONG, SHORT)
# Farris: the long branches are A and B, which *are* a cherry.
FARRIS = _balanced(LONG, LONG, SHORT, SHORT)

TRUE_SPLIT = leaf_bipartitions(_balanced(0.1, 0.1, 0.1, 0.1))
UNIFORM = np.full(4, 0.25)


def _recovery_rates(tau: Node, n_sites: int, replicates: int) -> tuple[int, int]:
    """How often each criterion picks the true topology out of all three."""
    parsimony_correct = likelihood_correct = 0
    for replicate in range(replicates):
        dataset = simulate_alignment(tau, 4, UNIFORM, 1000 + replicate, n_sites)
        alignment = dataset.alignment
        topologies = list(enumerate_topologies(LEAVES))

        parsimony = [fitch_score(topology, alignment, 4) for topology in topologies]
        likelihood = [score_topology(topology, alignment, 4) for topology in topologies]

        best_parsimony = topologies[int(np.argmin(parsimony))]
        best_likelihood = topologies[int(np.argmax(likelihood))]
        parsimony_correct += leaf_bipartitions(best_parsimony) == TRUE_SPLIT
        likelihood_correct += leaf_bipartitions(best_likelihood) == TRUE_SPLIT
    return parsimony_correct, likelihood_correct


@pytest.mark.oracle
def test_fitch_matches_exhaustive_enumeration_over_internal_labellings() -> None:
    # The oracle assigns states to internal nodes directly and counts
    # disagreeing edges; Fitch intersects state sets in one post-order pass.
    # No traversal is shared, so agreement is evidence rather than a tautology.
    rng = np.random.default_rng(0)
    tau = _balanced(0.1, 0.1, 0.1, 0.1)

    for _ in range(8):
        alignment = {name: rng.integers(0, 4, size=12) for name in LEAVES}

        assert fitch_score(tau, alignment, 4) == brute_force_parsimony_score(
            tau, alignment, 4
        )


@pytest.mark.oracle
def test_fitch_matches_a_score_worked_out_by_hand() -> None:
    # Three sites chosen so each exercises a different branch of the
    # recursion: an informative split, a constant site, and a site where every
    # taxon differs.
    tau = _balanced(0.1, 0.1, 0.1, 0.1)
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
    tau = _balanced(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(20, dtype=np.int64) for name in LEAVES}

    assert fitch_score(tau, alignment, 4) == 0


@pytest.mark.edge_case
def test_a_missing_leaf_is_refused() -> None:
    # Silently scoring the subtree it can reach would return a smaller number
    # for the wrong reason, and smaller is better under this criterion.
    tau = _balanced(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(5, dtype=np.int64) for name in ("A", "B", "C")}

    with pytest.raises(ValueError, match="not in the alignment"):
        fitch_score(tau, alignment, 4)


@pytest.mark.edge_case
def test_sequences_of_different_lengths_are_refused() -> None:
    tau = _balanced(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(5, dtype=np.int64) for name in LEAVES}
    alignment["D"] = np.zeros(4, dtype=np.int64)

    with pytest.raises(ValueError, match="differ in length"):
        fitch_score(tau, alignment, 4)


@pytest.mark.edge_case
@pytest.mark.parametrize("k", [1, 64])
def test_a_state_count_a_bitmask_cannot_hold_is_refused(k: int) -> None:
    tau = _balanced(0.1, 0.1, 0.1, 0.1)
    alignment = {name: np.zeros(4, dtype=np.int64) for name in LEAVES}

    with pytest.raises(ValueError, match=r"k must be in \[2, 63\]"):
        fitch_score(tau, alignment, k)


@pytest.mark.simulated_truth
def test_parsimony_is_inconsistent_in_the_felsenstein_zone() -> None:
    # The theorem, as a prediction. Parsimony does not merely do badly here --
    # more data does not help, because the systematic pull toward grouping the
    # two long branches grows with the data exactly as the true signal does.
    # Measured over 12 replicates: 0/12 correct at 200, 1000 and 5000 sites.
    few_parsimony, _ = _recovery_rates(FELSENSTEIN, n_sites=200, replicates=6)
    many_parsimony, _ = _recovery_rates(FELSENSTEIN, n_sites=2000, replicates=6)

    assert few_parsimony == 0
    assert many_parsimony == 0


@pytest.mark.simulated_truth
def test_likelihood_is_consistent_in_the_felsenstein_zone() -> None:
    # The other half of the same claim, and the reason the zone is the
    # canonical argument for the criterion this repository actually uses.
    # Measured: 10/12 at 200 sites, 12/12 at 1000 and 5000.
    _, few_likelihood = _recovery_rates(FELSENSTEIN, n_sites=200, replicates=6)
    _, many_likelihood = _recovery_rates(FELSENSTEIN, n_sites=2000, replicates=6)

    assert many_likelihood == 6
    assert many_likelihood >= few_likelihood


@pytest.mark.simulated_truth
def test_parsimony_is_correct_and_fast_in_the_farris_zone() -> None:
    # The control. Without it, "parsimony got the Felsenstein zone wrong" is
    # indistinguishable from "this parsimony implementation is broken".
    # Measured: 12/12 at every site count, and likelihood 4/12, 6/12, 10/12 --
    # so in this zone parsimony is the *faster* of the two, which is the
    # published result and not an artifact.
    parsimony, _ = _recovery_rates(FARRIS, n_sites=200, replicates=6)

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
    topologies = list(enumerate_topologies(LEAVES))

    winners = set()
    for seed in range(15):
        dataset = simulate_alignment(star, 4, UNIFORM, 5000 + seed, 400)
        scores = [
            fitch_score(topology, dataset.alignment, 4) for topology in topologies
        ]
        winners.add(int(np.argmin(scores)))

    # Each topology winning at least once over 15 draws has probability 0.993
    # under a uniform winner, and is essentially impossible under a systematic
    # preference for any one of them.
    assert winners == {0, 1, 2}
