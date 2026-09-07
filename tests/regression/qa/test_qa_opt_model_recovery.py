"""Regression test for phylo.qa.opt_model_recovery.

Pins the numbers the script computes before matplotlib sees them
(qa/CLAUDE.md), and in particular that the caption's coverage counts are the
ones the fits produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.qa.opt_model_recovery import (
    SITES,
    TRUE_EXCHANGEABILITIES,
    TRUE_PI,
    build_figure,
    fit_model,
    main,
    truth_vector,
)
from phylo.sim.gtr import gtr_rate_matrix

from tests._fixtures import EIGHT_TAXA, fixture_path, load_fixture

FIXTURE = EIGHT_TAXA


@pytest.mark.structural
def test_the_truth_vector_pins_the_last_exchangeability() -> None:
    vector = truth_vector(TRUE_EXCHANGEABILITIES, TRUE_PI)
    # Five free exchangeabilities, then four frequencies.
    assert vector.size == 9
    assert_allclose(
        vector[:5], TRUE_EXCHANGEABILITIES[:-1] / TRUE_EXCHANGEABILITIES[-1]
    )
    assert_allclose(vector[5:], TRUE_PI)


@pytest.mark.simulated_truth
def test_fitting_gtr_data_recovers_the_generating_model() -> None:
    params = load_fixture(FIXTURE)
    fitted, spread = fit_model(
        params, gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI), TRUE_PI
    )
    truth = truth_vector(TRUE_EXCHANGEABILITIES, TRUE_PI)

    assert fitted.shape == spread.shape == truth.shape
    assert bool((spread > 0.0).all())
    # Stated in standard errors so the assertion transfers if SITES changes.
    assert float((np.abs(fitted - truth) / spread).max()) < 4.0


@pytest.mark.simulated_truth
def test_fitting_jc_data_does_not_invent_structure() -> None:
    params = load_fixture(FIXTURE)
    uniform = np.full(params.k, 1.0 / params.k)
    fitted, spread = fit_model(params, None, uniform)
    truth = truth_vector(np.ones(TRUE_EXCHANGEABILITIES.size), uniform)

    assert float((np.abs(fitted - truth) / spread).max()) < 4.0


@pytest.mark.structural
def test_the_caption_reports_the_coverage_it_measured() -> None:
    params = load_fixture(FIXTURE)
    general = fit_model(
        params, gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI), TRUE_PI
    )
    uniform = np.full(params.k, 1.0 / params.k)
    jukes_cantor = fit_model(params, None, uniform)

    _, caption = build_figure(params, general, jukes_cantor)

    assert str(params.seed) in caption
    assert str(SITES) in caption
    assert "of 9 intervals covering" in caption
    assert "draw and not a rate" in caption
    # qa/CLAUDE.md: captions are plain text pulled into LaTeX verbatim.
    assert not set(caption) & set("_%\\&#")


@pytest.mark.structural
def test_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    written = main(
        ["--params", str(fixture_path(FIXTURE)), "--output-dir", str(tmp_path)]
    )
    assert written.figure_path.is_file()
    assert written.caption_path.read_text() == written.caption
