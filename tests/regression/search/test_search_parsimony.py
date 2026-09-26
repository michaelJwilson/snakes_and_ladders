"""Large parsimony against enumeration, and the zone where its optimum is the wrong tree.

`parsimony_search` is the hill climb of `infer` with one parsimony pass in
place of a fit (issue #335). Two things are asserted about it. Below eight
taxa every topology can be scored, so whether the climb found the minimum has
an answer: from every one of the 15 starts at five taxa and the 105 at six,
under NNI and SPR, under the Fitch score and a weighted one, it must reach
the enumerated minimum. And on the Felsenstein-zone fixture the minimum it
reaches is the wrong tree --- the theorem the small-parsimony tests state per
criterion, restated here for the search: a correct search returns exactly the
tree the theorem says it will, and the likelihood optimum is the other one.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from sal.likelihood.parsimony import (
    fitch_score,
    sankoff_score,
    unit_step_matrix,
)
from sal.search.infer import (
    ParsimonyInference,
    parsimony_search,
    score_topology,
)
from sal.sim.newick import count_topologies
from sal.sim.simulate import simulate_alignment
from sal.sim.simulator import simulate_tree
from sal.sim.topology import (
    MoveSet,
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
)

from tests._fixtures import (
    EIGHT_TAXA,
    FELSENSTEIN_ZONE,
    FOUR_TAXA_LEAVES,
    ZONE_TRUE_SPLIT,
    load_fixture,
)

FIVE_TAXA = "tree_search/ci.yaml"
SIX_TAXA = "tree_search/stress.yaml"

# Transitions cost 1, transversions 2: a metric, so the unrooted score is
# well defined, and a weighting under which the search is a different
# surface from Fitch's rather than a rescaling of it.
TRANSITION_TRANSVERSION = np.array(
    [[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 1], [2, 2, 1, 0]], dtype=np.float64
)

Key = frozenset[frozenset[str]]


def _alignment(
    name: str, n_sites: int | None = None
) -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(name)
    dataset = simulate_tree(
        params,
        np.random.default_rng(params.seed),
        n_sites=params.n_sites if n_sites is None else n_sites,
    )
    return dict(dataset.alignment), params.k


def _score(
    topology: Topology,
    alignment: Mapping[str, np.ndarray],
    k: int,
    step_matrix: np.ndarray | None,
) -> float:
    if step_matrix is None:
        return float(fitch_score(topology, alignment, k))
    return sankoff_score(topology, alignment, step_matrix)


def _enumerated(
    alignment: Mapping[str, np.ndarray], k: int, step_matrix: np.ndarray | None
) -> tuple[float, dict[Key, float]]:
    scores = {
        leaf_bipartitions(topology): _score(topology, alignment, k, step_matrix)
        for topology in enumerate_topologies(sorted(alignment))
    }
    return min(scores.values()), scores


def _reaches_the_minimum(
    result: ParsimonyInference, best: float, scores: Mapping[Key, float]
) -> bool:
    # Both the score and the topology: a search that reported the minimum
    # score on a topology that does not have it would be a bookkeeping error
    # the score alone cannot see.
    return result.score == best and scores[leaf_bipartitions(result.topology)] == best


# --- every start reaches the enumerated minimum -----------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
@pytest.mark.parametrize(
    "step_matrix", [None, TRANSITION_TRANSVERSION], ids=["fitch", "weighted"]
)
def test_every_start_reaches_the_enumerated_minimum_at_five_taxa(
    moves: MoveSet, step_matrix: np.ndarray | None
) -> None:
    # 15 starts: 15 of 15 per combination; unique minimum 1163 (Fitch), 1901
    # (weighted) at 1,200 sites, the generating tree; median 8 (NNI), 14 (SPR).
    alignment, k = _alignment(FIVE_TAXA)
    best, scores = _enumerated(alignment, k, step_matrix)
    assert sum(value == best for value in scores.values()) == 1

    for start in enumerate_topologies(sorted(alignment)):
        result = parsimony_search(
            alignment,
            k,
            step_matrix=step_matrix,
            start=start,
            moves=moves,
            rng=np.random.default_rng(0),
        )
        assert result.converged
        assert _reaches_the_minimum(result, best, scores), (
            f"{moves.value} from {leaf_bipartitions(start)} stopped at {result.score}"
        )


@pytest.mark.oracle
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
@pytest.mark.parametrize(
    "step_matrix", [None, TRANSITION_TRANSVERSION], ids=["fitch", "weighted"]
)
def test_every_start_reaches_the_enumerated_minimum_at_six_taxa(
    moves: MoveSet, step_matrix: np.ndarray | None
) -> None:
    # 105 starts: 105 of 105; unique minimum 1792 / 2920 at 1,500 sites, 37
    # ahead; median 17 (NNI), 61 (SPR). Per PR: 0.15 ms a pass against 250 ms a fit.
    alignment, k = _alignment(SIX_TAXA)
    best, scores = _enumerated(alignment, k, step_matrix)
    assert sum(value == best for value in scores.values()) == 1

    for start in enumerate_topologies(sorted(alignment)):
        result = parsimony_search(
            alignment,
            k,
            step_matrix=step_matrix,
            start=start,
            moves=moves,
            max_evaluations=500,
            rng=np.random.default_rng(0),
        )
        assert result.converged
        assert _reaches_the_minimum(result, best, scores), (
            f"{moves.value} from {leaf_bipartitions(start)} stopped at {result.score}"
        )


@pytest.mark.oracle
def test_spr_escapes_the_local_minima_nni_stops_in_at_eight_taxa() -> None:
    # Where the neighbourhoods separate: 10,395 topologies, unique minimum 1607
    # (9 ahead). From 12 starts NNI reached it 9 times, SPR 12 (median 289
    # candidates against 60). Asserted at the margin.
    alignment, k = _alignment(EIGHT_TAXA, n_sites=1000)
    best, scores = _enumerated(alignment, k, None)
    assert sum(value == best for value in scores.values()) == 1

    hits = {}
    for moves in MoveSet:
        hits[moves] = sum(
            _reaches_the_minimum(
                parsimony_search(
                    alignment,
                    k,
                    rng=np.random.default_rng(seed),
                    moves=moves,
                    max_evaluations=1000,
                ),
                best,
                scores,
            )
            for seed in range(12)
        )

    assert hits[MoveSet.SPR] >= hits[MoveSet.NNI], hits
    assert hits[MoveSet.SPR] >= 9, hits
    assert hits[MoveSet.NNI] >= 6, hits


# --- the loop --------------------------------------------------------------


@pytest.mark.analytic
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_every_accepted_move_strictly_improves(moves: MoveSet) -> None:
    alignment, k = _alignment(SIX_TAXA)

    result = parsimony_search(alignment, k, rng=np.random.default_rng(1), moves=moves)

    assert list(result.trace) == sorted(result.trace, reverse=True)
    assert len(set(result.trace)) == len(result.trace)
    assert result.score == result.trace[-1] == min(result.trace)
    assert result.score == fitch_score(result.topology, alignment, k)


@pytest.mark.oracle
def test_the_unit_step_matrix_walks_the_same_path_as_fitch() -> None:
    # The reduction at the level of the search: the same start, the same
    # accepted moves, the same trace, so a weighted search differs from the
    # unit one only through the matrix it is given.
    alignment, k = _alignment(SIX_TAXA)
    for seed in range(3):
        unit = parsimony_search(
            alignment,
            k,
            step_matrix=unit_step_matrix(k),
            rng=np.random.default_rng(seed),
            moves=MoveSet.SPR,
        )
        fitch = parsimony_search(
            alignment, k, rng=np.random.default_rng(seed), moves=MoveSet.SPR
        )
        assert unit.trace == fitch.trace
        assert unit.evaluations == fitch.evaluations
        assert leaf_bipartitions(unit.topology) == leaf_bipartitions(fitch.topology)


@pytest.mark.oracle
def test_no_topology_is_scored_twice() -> None:
    # `(2n - 5)!!`: 15 topologies at five taxa, so at most 14 candidates;
    # realized 14, so the bound binds.
    alignment, k = _alignment(FIVE_TAXA)

    result = parsimony_search(
        alignment, k, rng=np.random.default_rng(1), moves=MoveSet.SPR
    )

    assert result.converged
    assert result.evaluations <= count_topologies(len(alignment) - 1) - 1 == 14


@pytest.mark.smoke
def test_the_budget_is_respected_and_reported_unconverged() -> None:
    alignment, k = _alignment(SIX_TAXA)

    result = parsimony_search(
        alignment, k, rng=np.random.default_rng(1), max_evaluations=1
    )

    assert result.evaluations <= 1
    assert not result.converged


@pytest.mark.smoke
def test_a_zero_budget_scores_the_start_and_nothing_else() -> None:
    alignment, k = _alignment(SIX_TAXA)
    params = load_fixture(SIX_TAXA)

    result = parsimony_search(
        alignment, k, start=params.tau, max_evaluations=0, rng=np.random.default_rng(0)
    )

    assert result.converged
    assert result.evaluations == 0
    assert result.trace == (float(fitch_score(params.tau, alignment, k)),)


@pytest.mark.smoke
def test_a_search_is_reproducible_from_its_seed() -> None:
    alignment, k = _alignment(SIX_TAXA)

    first = parsimony_search(alignment, k, rng=np.random.default_rng(5))
    second = parsimony_search(alignment, k, rng=np.random.default_rng(5))

    assert first.trace == second.trace
    assert leaf_bipartitions(first.topology) == leaf_bipartitions(second.topology)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("step_matrix", "message"),
    [
        (np.ones((3, 3)) - np.eye(3), "must have shape"),
        (
            np.array([[0, 1, 3, 4], [2, 0, 1, 3], [4, 2, 0, 1], [1, 3, 2, 0]]),
            "must be symmetric",
        ),
        (np.ones((4, 4)), "zero diagonal"),
        (
            # 0 -> 2 costs 10 directly and 2 through state 1: a degree-2 root
            # on that edge would take the detour, and the score would then
            # depend on where the tree was rooted.
            np.array([[0, 1, 10, 1], [1, 0, 1, 1], [10, 1, 0, 1], [1, 1, 1, 0]]),
            "triangle inequality",
        ),
    ],
    ids=["shape", "asymmetric", "diagonal", "triangle"],
)
def test_a_step_matrix_without_a_root_free_score_is_refused(
    step_matrix: np.ndarray, message: str
) -> None:
    # The search keys topologies on their bipartitions, which forget the
    # rooting, so it may only run under a matrix every rooting scores alike.
    # An asymmetric matrix is legitimate input to `sankoff_score`, which
    # scores a rooted tree; here it is refused rather than scored as spelled.
    alignment, k = _alignment(FIVE_TAXA)

    with pytest.raises(ValueError, match=message):
        parsimony_search(
            alignment, k, step_matrix=step_matrix, rng=np.random.default_rng(0)
        )


@pytest.mark.smoke
def test_too_few_taxa_and_a_missing_rng_are_refused() -> None:
    # The rng is required at the signature since #1091, so its absence is a
    # TypeError at the call rather than a ValueError from inside.
    with pytest.raises(ValueError, match="at least 4 taxa"):
        parsimony_search(
            {name: np.zeros(5, dtype=np.int64) for name in "ABC"},
            4,
            rng=np.random.default_rng(0),
        )
    alignment, k = _alignment(FIVE_TAXA)
    with pytest.raises(TypeError, match="rng"):
        parsimony_search(alignment, k)  # type: ignore[call-arg]


# --- the Felsenstein zone ---------------------------------------------------


@pytest.mark.end2end
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_large_parsimony_returns_the_wrong_tree_in_the_felsenstein_zone(
    moves: MoveSet,
) -> None:
    # The search is right, the criterion wrong. At 2,000 sites (#209, seed
    # 1000) every start returns AC|BD at Fitch 1869 (AB|CD 1982, AD|BC 2005);
    # weighted 3013 against 3273. ML puts AB|CD first, -8325.16 against
    # -8338.15 (12.99). At 200 sites: 1.71 log units, 10 changes, same signs.
    n_sites = 2000
    dataset = simulate_alignment(
        FELSENSTEIN_ZONE, 4, np.full(4, 0.25), np.random.default_rng(1000), n_sites
    )
    alignment = dict(dataset.alignment)
    topologies = list(enumerate_topologies(FOUR_TAXA_LEAVES))
    long_branch_pair = frozenset({"A", "C"})

    best, scores = _enumerated(alignment, 4, None)
    most_parsimonious = [key for key, value in scores.items() if value == best]
    assert len(most_parsimonious) == 1
    assert most_parsimonious[0] != ZONE_TRUE_SPLIT
    assert long_branch_pair in most_parsimonious[0] or (
        frozenset({"B", "D"}) in most_parsimonious[0]
    )
    assert scores[ZONE_TRUE_SPLIT] - best > 0

    for start in topologies:
        result = parsimony_search(
            alignment, 4, start=start, moves=moves, rng=np.random.default_rng(0)
        )
        assert result.converged
        assert _reaches_the_minimum(result, best, scores)
    weighted_best, weighted = _enumerated(alignment, 4, TRANSITION_TRANSVERSION)
    assert weighted[most_parsimonious[0]] == weighted_best

    likelihoods = {
        leaf_bipartitions(topology): score_topology(topology, alignment, 4)
        for topology in topologies
    }
    assert max(likelihoods, key=lambda key: likelihoods[key]) == ZONE_TRUE_SPLIT
    assert likelihoods[ZONE_TRUE_SPLIT] - likelihoods[most_parsimonious[0]] > 0
