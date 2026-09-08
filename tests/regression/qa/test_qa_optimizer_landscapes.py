"""Regression tests for snakes_and_ladders.qa.optimizer_landscapes.

The endpoints are pinned to the closed-form minimizers, the drawn surface to
the objective it is a grid of, and the caption to the counts it was handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.testfunctions import HIMMELBLAU_MINIMA, Himmelblau
from snakes_and_ladders.qa.optimizer_landscapes import (
    AT_MINIMUM,
    DOMAINS,
    GRID,
    RESTARTS,
    build_figure,
    endpoints,
    known_minimizers,
    main,
    objectives,
    reached,
    surface,
)


@pytest.mark.oracle
def test_rosenbrock_is_reached_from_every_restart_and_himmelblau_lands_on_a_minimum() -> (
    None
):
    results = endpoints(np.random.default_rng(20260908))

    assert reached("Rosenbrock", results["Rosenbrock"]) == RESTARTS["Rosenbrock"]
    assert reached("Himmelblau", results["Himmelblau"]) == RESTARTS["Himmelblau"]
    for fit in results["Himmelblau"].all_fits:
        _, distance = Himmelblau.nearest_minimum(fit.theta)
        assert distance < AT_MINIMUM


@pytest.mark.oracle
def test_every_rastrigin_endpoint_is_a_stationary_point_of_the_closed_form() -> None:
    # A fit that reports convergence sits where the closed-form gradient
    # vanishes; whether that is the origin is what the caption counts.
    results = endpoints(np.random.default_rng(20260908))
    rastrigin = objectives()["Rastrigin"]

    for fit in results["Rastrigin"].all_fits:
        gradient = rastrigin.gradient(fit.theta)  # type: ignore[attr-defined]
        assert float(gradient.abs().max()) < 1e-5
    assert 0 <= reached("Rastrigin", results["Rastrigin"]) <= RESTARTS["Rastrigin"]


@pytest.mark.oracle
def test_the_surface_is_the_objective_on_its_grid() -> None:
    for name, objective in objectives().items():
        xs, ys, values = surface(objective, name)
        x_min, x_max, y_min, y_max = DOMAINS[name]

        assert values.shape == (GRID, GRID)
        assert (xs[0], xs[-1], ys[0], ys[-1]) == (x_min, x_max, y_min, y_max)
        corner = float(objective(torch.tensor([x_max, y_min], dtype=torch.float64)))
        assert values[0, -1] == corner
        assert values.min() >= 0.0


@pytest.mark.oracle
def test_known_minimizers_are_the_published_ones() -> None:
    assert np.array_equal(known_minimizers("Himmelblau"), np.array(HIMMELBLAU_MINIMA))
    assert np.array_equal(known_minimizers("Rosenbrock"), np.array([[1.0, 1.0]]))
    assert np.array_equal(known_minimizers("Rastrigin"), np.zeros((1, 2)))


@pytest.mark.structural
def test_the_caption_counts_the_endpoints_it_was_handed(tmp_path: Path) -> None:
    results = endpoints(np.random.default_rng(20260908))
    _, caption = build_figure(results)
    for name in RESTARTS:
        assert f"{reached(name, results[name])} of {RESTARTS[name]}" in caption

    qa_figure = main(["--output-dir", str(tmp_path)])
    assert qa_figure.figure_path.is_file()
    assert "seed 20260908" in qa_figure.caption
