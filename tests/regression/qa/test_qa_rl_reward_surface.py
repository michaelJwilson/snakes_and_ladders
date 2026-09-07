"""The reward-surface comparison, and the statistic it is reported with.

The figure's claim is that a cheap reward can stand in for an expensive one,
and that claim is a *ranking* claim, so the rank correlation it rests on is
pinned here against hand-computed values rather than trusted. The comparison
itself is checked at 5 taxa per PR and at 6 behind the release gate: the
fitted surface costs one optimization per topology, and 105 of them is half a
minute the per-PR suite should not pay.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.qa import rl_reward_surface
from phylo.qa.figure import pearson_correlation
from phylo.qa.rl_reward_surface import (
    BRANCH_LENGTHS,
    build_figure,
    mean_branch_length,
    reward_surfaces,
)
from phylo.sim.params import load_simulation_params

from tests._fixtures import FIXTURES_DIR

# Both surfaces, the generating topology's index, the default branch length,
# and the sweep's correlations and argmax agreements -- what `reward_surfaces`
# returns, named once so every test that consumes it can say so.
Surfaces = tuple[
    np.ndarray, np.ndarray, int, float, dict[float, float], dict[float, bool]
]

FIVE_TAXA = FIXTURES_DIR / "simulation_params_5taxa.yaml"
SIX_TAXA = FIXTURES_DIR / "simulation_params_6taxa.yaml"


@pytest.fixture(scope="module")
def five_taxon() -> Surfaces:
    """Both surfaces at 5 taxa, computed once for every test that reads them."""
    return reward_surfaces(load_simulation_params(FIVE_TAXA))


# --- the statistic --------------------------------------------------------


@pytest.mark.mathematical
def test_the_correlation_is_invariant_to_affine_rescaling() -> None:
    # The property that makes it the right statistic here: the two surfaces
    # are log-likelihoods in the same units, separated by a shift and a
    # scale, and that separation is not what is being measured.
    values = np.array([1.0, 2.0, 3.5, 4.0, 5.0])
    assert_allclose(pearson_correlation(values, 7.0 * values - 3.0), 1.0, atol=1e-12)
    assert_allclose(pearson_correlation(values, -2.0 * values), -1.0, atol=1e-12)


@pytest.mark.oracle
def test_the_correlation_matches_its_closed_form_on_a_worked_case() -> None:
    # Centred: a = (-2, -1, 0, 1, 2), b = (-1, -2, 1, 0, 2). a.b = 2+2+0+0+4 = 8,
    # |a| = |b| = sqrt(10), so r = 8/10 = 0.8.
    first = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    second = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    assert_allclose(pearson_correlation(first, second), 0.8, atol=1e-12)


@pytest.mark.structural
def test_the_correlation_is_continuous_where_a_rank_statistic_is_not() -> None:
    # The reason this statistic replaced Spearman's rho, pinned rather than
    # argued. Near-tied values are exactly what the fitted surface produces --
    # several optima agree to within the optimizer's convergence -- and a rank
    # statistic reorders them on any perturbation. This one moves by the size
    # of the perturbation, which is what lets a caption quote it and CI
    # rebuild the same document.
    first = np.arange(50.0)
    second = np.repeat(np.arange(25.0), 2)
    baseline = pearson_correlation(first, second)
    jittered = second + 1e-9 * np.random.default_rng(0).standard_normal(50)
    assert abs(pearson_correlation(first, jittered) - baseline) < 1e-9


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (np.zeros(3), np.zeros(4), "equal length"),
        (np.zeros((2, 2)), np.zeros((2, 2)), "1-D"),
        (np.zeros(1), np.zeros(1), "at least 2 entries"),
        (np.ones(4), np.arange(4.0), "constant sample"),
    ],
)
def test_the_correlation_rejects_a_sample_it_cannot_use(
    first: np.ndarray, second: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pearson_correlation(first, second)


# --- the comparison -------------------------------------------------------


@pytest.mark.simulated_truth
def test_the_two_surfaces_agree_on_the_generating_topology(
    five_taxon: Surfaces,
) -> None:
    # The claim that licenses training on the cheap reward: it picks the same
    # answer. Realized at 5 taxa -- both rank the generating topology first of
    # 15; at 6 taxa, first of 105.
    known, fitted, truth, _, _, _ = five_taxon
    assert int(np.argmax(known)) == truth
    assert int(np.argmax(fitted)) == truth


@pytest.mark.simulated_truth
def test_the_two_surfaces_are_correlated_but_not_identical(
    five_taxon: Surfaces,
) -> None:
    # Both halves matter. High correlation is what makes the substitution
    # sound; falling short of 1 is why it is measured rather than assumed, and
    # why a policy trained on one still has to be judged on the other.
    # Realized: 0.8607 at 5 taxa, 0.9045 at 6.
    known, fitted, _, _, _, _ = five_taxon
    correlation = pearson_correlation(known, fitted)
    assert 0.9 < correlation < 1.0


@pytest.mark.simulated_truth
def test_the_answer_survives_the_choice_of_fixed_branch_length(
    five_taxon: Surfaces,
) -> None:
    # The cheap surface has exactly one free parameter, so the obvious
    # objection is that the agreement was tuned. Across a 50-fold range the
    # two surfaces pick the same best topology every time, at both sizes.
    _, _, _, default, correlations, agreements = five_taxon
    assert set(correlations) == set(BRANCH_LENGTHS)
    assert all(agreements.values())
    assert min(correlations.values()) > 0.5
    assert min(BRANCH_LENGTHS) < default < max(BRANCH_LENGTHS)


@pytest.mark.structural
def test_the_default_branch_length_is_the_generating_mean() -> None:
    params = load_simulation_params(FIVE_TAXA)
    lengths = [0.11, 0.26, 0.07, 0.19, 0.12, 0.08, 0.31]
    assert_allclose(mean_branch_length(params), float(np.mean(lengths)), atol=1e-12)


@pytest.mark.structural
def test_the_caption_reports_the_correlation_it_measured(five_taxon: Surfaces) -> None:
    # `qa/CLAUDE.md`: a caption states what actually ran. The correlation is
    # the figure's whole evidence, so a caption that omitted or rounded away
    # from it would be reporting a different result than the one drawn.
    known, fitted, truth, default, correlations, agreements = five_taxon
    params = load_simulation_params(FIVE_TAXA)
    figure, caption = build_figure(
        known, fitted, truth, default, correlations, agreements, params
    )
    try:
        assert f"{pearson_correlation(known, fitted):.4f}" in caption
        assert f"{min(correlations.values()):.4f}" in caption
        assert f"{len(agreements)} of {len(agreements)}" in caption
        assert str(params.seed) in caption
        assert f"{known.size}" in caption
    finally:
        figure.clear()


@pytest.mark.structural
def test_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    # At 5 taxa, so the whole pipeline -- both surfaces, the sweep, the
    # caption, the render -- is exercised per PR rather than only behind the
    # release gate the 6-taxon document figure sits behind.
    written = rl_reward_surface.main(
        ["--params", str(FIVE_TAXA), "--output-dir", str(tmp_path)]
    )

    assert written.figure_path.is_file()
    assert written.caption_path.is_file()
    assert written.caption == written.caption_path.read_text()
    assert "correlation" in written.caption


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_comparison_holds_at_six_taxa() -> None:
    # The size the technical document reports. Release-gated because the
    # fitted surface is one optimization per topology and there are 105.
    known, fitted, truth, _, _, agreements = reward_surfaces(
        load_simulation_params(SIX_TAXA)
    )
    assert known.size == 105
    assert int(np.argmax(known)) == truth == int(np.argmax(fitted))
    assert pearson_correlation(known, fitted) > 0.9
    assert all(agreements.values())
