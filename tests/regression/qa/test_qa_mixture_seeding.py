"""Regression tests for snakes_and_ladders.qa.mixture_seeding.

The seeding ratios are pinned to the exact dynamic programme they divide by
and to the published bound; the densities to the family's own normalization;
the caption to the numbers it was handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.mixture import MixtureFit, seeding_guarantee
from snakes_and_ladders.qa.mixture_seeding import (
    EM_ITERATIONS,
    MEANS,
    SEED,
    SEEDINGS,
    SeedingRatios,
    build_figure,
    density,
    fitted,
    fixture,
    main,
    seeding_ratios,
)
from snakes_and_ladders.sim.mixture import simulate_mixture


@pytest.mark.oracle
def test_no_seeding_beats_the_exact_optimal_cost_and_kmeanspp_meets_its_bound() -> None:
    observations = simulate_mixture(fixture()).observations
    ratios = seeding_ratios(observations, np.random.default_rng(SEED))

    assert ratios.kmeans_plus_plus.shape == ratios.uniform.shape == (SEEDINGS,)
    assert ratios.kmeans_plus_plus.min() >= 1.0
    assert ratios.uniform.min() >= 1.0
    assert ratios.kmeans_plus_plus.mean() <= seeding_guarantee(len(MEANS))


@pytest.mark.mathematical
def test_the_generating_density_integrates_to_one() -> None:
    truth = fixture()
    grid = np.linspace(-15.0, 15.0, 30001)
    values = density(grid, truth.weights, truth.components)

    assert np.isclose(np.trapezoid(values, grid), 1.0, atol=1e-9)


@pytest.mark.simulated_truth
def test_the_fit_lands_within_the_generating_density_scale() -> None:
    # A density fitted to 500 draws of this mixture cannot be held to the
    # truth to a tolerance nothing derives; what is held is that every fitted
    # component sits inside the generating range with a scale of the same
    # order, which a collapsed or runaway component would break.
    observations = simulate_mixture(fixture()).observations
    fit = fitted(observations, np.random.default_rng(SEED))

    assert fit.iterations <= EM_ITERATIONS
    assert (fit.components.mean.numpy() > min(MEANS) - 1.0).all()
    assert (fit.components.mean.numpy() < max(MEANS) + 1.0).all()
    assert (fit.components.scale.numpy() > 0.2).all()
    assert (fit.components.scale.numpy() < 3.0).all()


@pytest.mark.structural
def test_the_caption_reports_the_numbers_it_was_handed(tmp_path: Path) -> None:
    truth = fixture()
    observations = simulate_mixture(truth).observations
    fit = MixtureFit(
        weights=torch.as_tensor(truth.weights, dtype=torch.float64),
        components=truth.components,
        log_likelihood=-1000.25,
        iterations=EM_ITERATIONS,
    )
    ratios = SeedingRatios(
        kmeans_plus_plus=np.array([1.0, 3.0]),
        uniform=np.array([2.0, 6.0]),
        optimal=1.0,
    )
    _, caption = build_figure(observations, fit, ratios)

    assert f"at the {EM_ITERATIONS}-iteration cap" in caption
    assert "log-likelihood -1000.2" in caption
    assert "k-means++ averages 2.00 (worst 3.00)" in caption
    assert "uniform seeding 4.00 (worst 6.00)" in caption
    assert f"seed {SEED}" in caption

    qa_figure = main(["--output-dir", str(tmp_path)])
    assert qa_figure.figure_path.is_file()
