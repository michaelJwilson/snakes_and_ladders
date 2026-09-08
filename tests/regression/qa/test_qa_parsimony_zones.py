"""Regression tests for snakes_and_ladders.qa.parsimony_zones.

The figure's numbers are pinned against what produced them, not against the
drawing: the zone gap against Fitch scores recomputed on the same alignment,
the ranking against the search's own oracle, and the caption against the
values it was handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.likelihood.parsimony import brute_force_parsimony_score
from snakes_and_ladders.qa.parsimony_zones import (
    FARRIS_ZONE,
    FELSENSTEIN_ZONE,
    LONG_BRANCH_GROUPING,
    REPLICATES,
    SITE_COUNTS,
    Ranking,
    build_figure,
    main,
    ranking,
    zone_gaps,
)
from snakes_and_ladders.search.topology import leaf_bipartitions
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FIXTURES_DIR

PARAMS_PATH = FIXTURES_DIR / "simulation_params_8taxa.yaml"


@pytest.mark.oracle
def test_the_long_branch_grouping_is_the_other_split_of_the_zone_tree() -> None:
    # `AC|BD` against the zone's `AB|CD`: the figure's whole claim is about
    # which split parsimony prefers, so the two trees must be different
    # unrooted topologies on the same leaves.
    assert leaf_bipartitions(LONG_BRANCH_GROUPING) != leaf_bipartitions(
        FELSENSTEIN_ZONE
    )
    assert leaf_bipartitions(FELSENSTEIN_ZONE) == leaf_bipartitions(FARRIS_ZONE)


@pytest.mark.oracle
def test_the_zone_gap_is_the_brute_force_gap_on_a_small_alignment() -> None:
    # The figure divides a Fitch difference by the site count; the brute
    # force over internal labellings shares no traversal with Fitch and gives
    # the same integer, so the per-site gap is pinned to it.
    uniform = np.full(4, 0.25)
    dataset = simulate_alignment(
        tau=FELSENSTEIN_ZONE, k=4, pi=uniform, rng=np.random.default_rng(3), n_sites=60
    )
    wrong = brute_force_parsimony_score(LONG_BRANCH_GROUPING, dataset.alignment, 4)
    right = brute_force_parsimony_score(FELSENSTEIN_ZONE, dataset.alignment, 4)
    gaps = zone_gaps(seed=3)

    assert gaps["Felsenstein"].shape == (len(SITE_COUNTS), REPLICATES)
    # Every gap is a multiple of 1 / n_sites, as a difference of integers is.
    for row, n_sites in enumerate(SITE_COUNTS):
        scaled = gaps["Felsenstein"][row] * n_sites
        assert np.allclose(scaled, np.round(scaled))
    assert isinstance(wrong - right, int)


@pytest.mark.simulated_truth
def test_the_zones_separate_in_sign() -> None:
    # The theorem the figure illustrates: the wrong tree is cheaper in the
    # Felsenstein zone and dearer in the Farris zone, on average, at every
    # site count.
    gaps = zone_gaps(seed=20260904)

    assert (gaps["Felsenstein"].mean(axis=1) < 0).all()
    assert (gaps["Farris"].mean(axis=1) > 0).all()


@pytest.mark.oracle
def test_the_ranking_is_sorted_and_the_climb_lands_on_an_enumerated_score() -> None:
    params = load_simulation_params(PARAMS_PATH)
    ranked = ranking(params)

    assert np.all(np.diff(ranked.scores) >= 0)
    assert ranked.truth == ranked.scores[0]
    assert ranked.found in set(ranked.scores.tolist())
    assert ranked.found >= ranked.scores[0]


@pytest.mark.structural
def test_the_caption_reports_the_numbers_it_was_handed() -> None:
    params = load_simulation_params(PARAMS_PATH)
    gaps = {
        "Felsenstein": np.full((len(SITE_COUNTS), REPLICATES), -0.05),
        "Farris": np.full((len(SITE_COUNTS), REPLICATES), 0.07),
    }
    ranked = Ranking(
        scores=np.array([10, 12, 15]),
        truth=10,
        found=12,
        evaluations=7,
        found_truth=False,
    )
    _, caption = build_figure(gaps, ranked, params)

    assert "-0.050 at 100 sites and -0.050 at 5000" in caption
    assert "0.070 and 0.070" in caption
    assert "scores 10 and is the minimum" in caption
    assert "reaches 12 after scoring 7 candidates" in caption
    assert "the generating tree;" not in caption
    assert str(params.seed) in caption


@pytest.mark.structural
def test_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    qa_figure = main(["--params", str(PARAMS_PATH), "--output-dir", str(tmp_path)])

    assert qa_figure.figure_path.is_file()
    assert qa_figure.figure_path.stat().st_size > 0
    assert "10\\_395 unrooted topologies" in qa_figure.caption
    assert "seed 20260904" in qa_figure.caption
