"""Regression test for phylo.qa.jc_transition.

The figure's claim is that the simulator's substitution frequencies match the
closed-form Jukes-Cantor transition probabilities. This pins that claim
numerically -- the frequencies the script computes before handing them to
matplotlib, checked against the analytic form at the fixture's declared
tolerance -- rather than that the figure renders (qa/CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from phylo.qa.jc_transition import build_figure, empirical_transitions, main
from phylo.sim.jc import jc_transition_probabilities
from phylo.sim.params import load_simulation_params

from tests._fixtures import FIXTURES_DIR, FOUR_TAXA, load_fixture


@pytest.mark.oracle
@pytest.mark.simulated_truth
def test_empirical_transitions_match_the_analytic_form() -> None:
    # The figure's whole assertion. Checked against sim.jc's closed form --
    # the independent oracle, never against the likelihood code.
    params = load_fixture(FOUR_TAXA)

    for branch_length, p_stay, p_change in empirical_transitions(params):
        expected = jc_transition_probabilities(branch_length, k=params.k)
        assert abs(p_stay - expected[0, 0]) <= params.tolerance
        assert abs(p_change - expected[0, 1]) <= params.tolerance


@pytest.mark.structural
def test_empirical_transitions_cover_every_branch() -> None:
    params = load_fixture(FOUR_TAXA)
    points = empirical_transitions(params)

    # One point per non-root node: the 4-taxon fixture is a trifurcating
    # root over A, B and an ancestor of C and D, so five branches.
    assert len(points) == 5
    assert all(t > 0.0 for t, _, _ in points)


@pytest.mark.mathematical
def test_stay_and_change_probabilities_are_consistent() -> None:
    # p_change is one specific off-diagonal target, so the row must close:
    # p_stay + (k - 1) * p_change == 1.
    params = load_fixture(FOUR_TAXA)

    for _, p_stay, p_change in empirical_transitions(params):
        assert p_stay + (params.k - 1) * p_change == pytest.approx(1.0)


@pytest.mark.structural
def test_empirical_transitions_are_reproducible_from_the_seed() -> None:
    params = load_fixture(FOUR_TAXA)
    assert empirical_transitions(params) == empirical_transitions(params)


@pytest.mark.structural
def test_caption_carries_its_generating_parameters() -> None:
    # A caption without the parameters that produced it is not QA-usable
    # (sim/CLAUDE.md's ground-truth-retention rule).
    params = load_fixture(FOUR_TAXA)
    fig, caption = build_figure(params)
    try:
        assert str(params.seed) in caption
        assert str(params.n_sites) in caption
        assert f"k = {params.k}" in caption
        assert "Jukes-Cantor" in caption
    finally:
        fig.clf()


@pytest.mark.structural
def test_caption_is_plain_text_not_latex() -> None:
    # main.tex pulls the caption in verbatim, so an unescaped special
    # character breaks the document build (qa/CLAUDE.md).
    fig, caption = build_figure(load_fixture(FOUR_TAXA))
    try:
        assert not set(caption) & set("_%\\&#")
    finally:
        fig.clf()


@pytest.mark.structural
def test_main_writes_a_figure_and_its_caption(tmp_path: Path) -> None:
    written = main(
        [
            "--params",
            str(FIXTURES_DIR / FOUR_TAXA),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert written.figure_path.is_file()
    assert written.caption_path.is_file()
    assert written.caption_path.read_text() == written.caption


@pytest.mark.edge_case
def test_a_branch_without_a_length_is_refused() -> None:
    from dataclasses import replace

    from phylo.sim.tree import Node

    params = load_simulation_params(FIXTURES_DIR / FOUR_TAXA)
    broken = replace(
        params,
        tau=Node(
            name="root",
            branch_length=None,
            children=(Node(name="A", branch_length=None),),
        ),
    )

    with pytest.raises(ValueError, match="has no branch_length"):
        empirical_transitions(broken)
