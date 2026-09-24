"""Regression test for snakes_and_ladders.qa.opt_model_recovery.

Pins the numbers the script computes before matplotlib sees them
(qa/CLAUDE.md), and in particular that the caption's coverage counts are the
ones the fits produced.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.qa.opt_model_recovery import (
    SITES,
    TRUE_EXCHANGEABILITIES,
    TRUE_PI,
    build_figure,
    fit_model,
    truth_vector,
)
from snakes_and_ladders.sim.gtr import gtr_rate_matrix

from tests._fixtures import EIGHT_TAXA, load_fixture

FIXTURE = EIGHT_TAXA


@pytest.mark.smoke
def test_the_truth_vector_pins_the_last_exchangeability() -> None:
    vector = truth_vector(TRUE_EXCHANGEABILITIES, TRUE_PI)
    # Five free exchangeabilities, then four frequencies.
    assert vector.size == 9
    assert_allclose(
        vector[:5], TRUE_EXCHANGEABILITIES[:-1] / TRUE_EXCHANGEABILITIES[-1]
    )
    assert_allclose(vector[5:], TRUE_PI)


@pytest.mark.end2end
def test_the_fits_recover_the_generating_model_and_the_caption_reports_them() -> None:
    # One GTR fit and one JC fit, judged against the model that generated each
    # dataset and then handed to the caption (issue #982 folded the two
    # recovery tests into this one, which ran the same two fits).
    params = load_fixture(FIXTURE)
    general = fit_model(
        params, gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI), TRUE_PI
    )
    uniform = np.full(params.k, 1.0 / params.k)
    jukes_cantor = fit_model(params, None, uniform)

    fitted, spread = general
    truth = truth_vector(TRUE_EXCHANGEABILITIES, TRUE_PI)
    assert fitted.shape == spread.shape == truth.shape
    assert bool((spread > 0.0).all())
    # Stated in standard errors so the assertion transfers if SITES changes.
    assert float((np.abs(fitted - truth) / spread).max()) < 4.0
    # JC data does not invent structure.
    fitted, spread = jukes_cantor
    truth = truth_vector(np.ones(TRUE_EXCHANGEABILITIES.size), uniform)
    assert float((np.abs(fitted - truth) / spread).max()) < 4.0

    _, caption = build_figure(params, general, jukes_cantor)

    assert str(params.seed) in caption
    assert str(SITES) in caption
    assert "of 9 intervals covering" in caption
    assert "draw and not a rate" in caption
    # qa/CLAUDE.md: captions are plain text pulled into LaTeX verbatim.
    assert not set(caption) & set("_%\\&#")
