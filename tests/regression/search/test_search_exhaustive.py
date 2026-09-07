"""Search quality against exhaustive enumeration.

This is the first independent oracle this project has for the *discrete*
half. Everything else about search is checked against closed-form neighbour
counts, which say the move sets are what they claim to be and nothing about
whether hill climbing finds the best tree. Below `n = 8` every unrooted
topology can be scored, so "did it find the maximum" has an answer rather
than an opinion.

The expensive study is release-gated: 105 topologies at 6 taxa is 50 s of
fitting, and the per-PR suite should not pay that. What runs per PR is the
same question at 5 taxa, where 15 topologies cost a few seconds, plus the
enumeration's own correctness -- which is cheap and is what the study rests
on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phylo.search.infer import MoveSet, infer, score_topology
from phylo.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
)
from phylo.sim.newick import count_topologies, to_newick, validate_unrooted_newick
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params_6taxa.yaml"

# Agreement between a search's optimum and the enumerated maximum. Both are
# the same objective fitted by the same optimizer, so they agree to the
# optimizer's own convergence, not to machine precision.
_LIKELIHOOD_TOLERANCE = 1e-5


def _alignment(path: Path = FIXTURE) -> tuple[dict[str, np.ndarray], int, Topology]:
    params = load_simulation_params(path)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    return dict(dataset.alignment), params.k, params.tau


def _exhaustive_maximum(
    alignment: dict[str, np.ndarray], k: int
) -> tuple[float, dict[frozenset[frozenset[str]], float]]:
    scores = {
        leaf_bipartitions(topology): score_topology(topology, alignment, k)
        for topology in enumerate_topologies(sorted(alignment))
    }
    return max(scores.values()), scores


@pytest.fixture(scope="module")
def six_taxon() -> tuple[
    dict[str, np.ndarray], int, Topology, float, dict[frozenset[frozenset[str]], float]
]:
    """The full 6-taxon alignment and its 105 scored topologies, computed once.

    50 s of fitting, which is why every release-gated test below shares one
    sweep rather than repeating it.
    """
    alignment, k, truth = _alignment()
    best, scores = _exhaustive_maximum(alignment, k)
    return alignment, k, truth, best, scores


@pytest.fixture(scope="module")
def five_taxon() -> tuple[
    dict[str, np.ndarray], int, float, dict[frozenset[frozenset[str]], float]
]:
    """The 5-taxon alignment and its 15 scored topologies, computed once.

    Module-scoped because the sweep is the expensive part and every test
    below asks the same question of it. Recomputing it per test tripled the
    module's wall clock for no additional evidence.
    """
    full, k, _ = _alignment()
    alignment = {name: full[name] for name in sorted(full)[:5]}
    best, scores = _exhaustive_maximum(alignment, k)
    return alignment, k, best, scores


# --- the enumeration itself ---------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("n_taxa", [3, 4, 5, 6, 7])
def test_enumeration_produces_every_topology_exactly_once(n_taxa: int) -> None:
    # Checked against the closed form, not against a second enumeration. A
    # generator that double-counted would make hill climbing look better
    # than it is; one that missed trees would make it look worse.
    names = [f"t{index}" for index in range(n_taxa)]
    produced = list(enumerate_topologies(names))

    assert len(produced) == count_topologies(n_taxa - 1)
    assert len({leaf_bipartitions(topology) for topology in produced}) == len(produced)


@pytest.mark.structural
@pytest.mark.parametrize("n_taxa", [4, 6])
def test_every_enumerated_topology_is_well_formed(n_taxa: int) -> None:
    names = [f"t{index}" for index in range(n_taxa)]
    for topology in enumerate_topologies(names):
        assert validate_unrooted_newick(to_newick(topology))


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("names", "message"),
    [(["A", "B"], "at least 3 leaves"), (["A", "A", "B"], "must be distinct")],
)
def test_enumeration_refuses_unusable_leaf_sets(names: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        list(enumerate_topologies(names))


@pytest.mark.structural
@pytest.mark.release
def test_enumeration_scales_to_the_size_cap() -> None:
    # DEV.md caps exhaustive topological tests at n <= 10; 8 taxa is 10395
    # topologies and is where that cap starts to bite.
    names = [f"t{index}" for index in range(8)]
    produced = list(enumerate_topologies(names))

    assert len(produced) == count_topologies(7) == 10395
    assert len({leaf_bipartitions(topology) for topology in produced}) == 10395


# --- search quality, per PR at 5 taxa ------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_hill_climbing_reaches_the_enumerated_maximum(
    moves: MoveSet,
    five_taxon: tuple[
        dict[str, np.ndarray], int, float, dict[frozenset[frozenset[str]], float]
    ],
) -> None:
    # 15 topologies, so the maximum is known rather than assumed. One taxon
    # is dropped from the fixture to reach 5: the question here is whether
    # the search finds the best tree for an alignment, which needs no truth.
    alignment, k, best, _ = five_taxon

    for seed in range(2):
        result = infer(alignment, k, seed=seed, moves=moves, max_evaluations=200)
        assert result.converged
        assert result.log_likelihood <= best + _LIKELIHOOD_TOLERANCE
        assert result.log_likelihood >= best - _LIKELIHOOD_TOLERANCE, (
            f"{moves.value} from seed {seed} stopped at {result.log_likelihood:.4f}, "
            f"below the enumerated maximum {best:.4f}"
        )


@pytest.mark.oracle
def test_the_enumerated_maximum_is_reached_by_no_topology_twice_over(
    five_taxon: tuple[
        dict[str, np.ndarray], int, float, dict[frozenset[frozenset[str]], float]
    ],
) -> None:
    # Sanity on the oracle: distinct topologies must have distinct scores on
    # real data, or "the maximum" would be ambiguous and a search could be
    # right and look wrong.
    _, _, _, scores = five_taxon

    ordered = sorted(scores.values(), reverse=True)
    assert ordered[0] - ordered[1] > _LIKELIHOOD_TOLERANCE


# --- the full study, release-gated ---------------------------------------


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("moves", [MoveSet.NNI, MoveSet.SPR])
def test_hill_climbing_success_rate_at_six_taxa(
    moves: MoveSet,
    six_taxon: tuple[
        dict[str, np.ndarray],
        int,
        Topology,
        float,
        dict[frozenset[frozenset[str]], float],
    ],
) -> None:
    # The measured rate is the deliverable, not the pass. On this fixture
    # both move sets reached the enumerated maximum from 12 of 12 starts and
    # recovered the generating topology every time, at a median of 14 fits
    # for NNI against 48 for SPR -- so SPR's larger neighbourhood costs 3.4
    # times as much here and buys nothing. That is a statement about the
    # problem, not about SPR: the optimum leads the runner-up by 41.6 log
    # units, so hill climbing is not being challenged at this size.
    #
    # The assertion is deliberately weaker than the observation. A move set
    # that fails sometimes is a true result about a weak neighbourhood, and
    # a threshold tuned to what was measured would hide exactly that.
    alignment, k, _, best, _ = six_taxon

    successes = 0
    trials = 12
    for seed in range(trials):
        result = infer(alignment, k, seed=seed, moves=moves, max_evaluations=500)
        assert result.converged
        if abs(result.log_likelihood - best) <= _LIKELIHOOD_TOLERANCE:
            successes += 1

    assert successes >= trials // 2, (
        f"{moves.value} reached the enumerated maximum {successes}/{trials} times"
    )


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_maximum_likelihood_tree_is_the_generating_tree_here(
    six_taxon: tuple[
        dict[str, np.ndarray],
        int,
        Topology,
        float,
        dict[frozenset[frozenset[str]], float],
    ],
) -> None:
    # Not guaranteed in general -- at finite data the ML tree need not be the
    # tree that generated the alignment, and treating that as a failure would
    # measure the sample rather than the method. On this fixture at 1500
    # sites they coincide, which is what makes it usable for a recovery
    # study: a search that finds the ML tree has also found the truth.
    _, _, truth, best, scores = six_taxon

    assert abs(scores[leaf_bipartitions(truth)] - best) <= _LIKELIHOOD_TOLERANCE


@pytest.mark.simulated_truth
@pytest.mark.release
def test_search_recovers_the_generating_topology(
    six_taxon: tuple[
        dict[str, np.ndarray],
        int,
        Topology,
        float,
        dict[frozenset[frozenset[str]], float],
    ],
) -> None:
    alignment, k, truth, _, _ = six_taxon
    truth_key = leaf_bipartitions(truth)

    recovered = sum(
        leaf_bipartitions(
            infer(
                alignment, k, seed=seed, moves=MoveSet.SPR, max_evaluations=500
            ).topology
        )
        == truth_key
        for seed in range(8)
    )

    assert recovered >= 4, f"recovered the generating topology {recovered}/8 times"
