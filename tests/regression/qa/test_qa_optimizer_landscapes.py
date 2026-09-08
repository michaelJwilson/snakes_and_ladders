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
    build_figure,
    endpoints,
    known_minimizers,
    main,
    objectives,
    reached,
    surface,
)
from snakes_and_ladders.sim.fixtures import fixture

FIXTURE = fixture("test_functions", "ci")
SUITE = FIXTURE.params
DECLARED = SUITE.named()
RESTARTS = {name: function.restarts for name, function in DECLARED.items()}


@pytest.mark.oracle
def test_rosenbrock_is_reached_from_every_restart_and_himmelblau_lands_on_a_minimum() -> (
    None
):
    results = endpoints(SUITE, np.random.default_rng(SUITE.seed))

    assert (
        reached(DECLARED["Rosenbrock"], results["Rosenbrock"], SUITE.at_minimum)
        == RESTARTS["Rosenbrock"]
    )
    assert (
        reached(DECLARED["Himmelblau"], results["Himmelblau"], SUITE.at_minimum)
        == RESTARTS["Himmelblau"]
    )
    for fit in results["Himmelblau"].all_fits:
        _, distance = Himmelblau.nearest_minimum(fit.theta)
        assert distance < SUITE.at_minimum


@pytest.mark.oracle
def test_every_rastrigin_endpoint_is_a_stationary_point_of_the_closed_form() -> None:
    # A fit that reports convergence sits where the closed-form gradient
    # vanishes; whether that is the origin is what the caption counts.
    results = endpoints(SUITE, np.random.default_rng(SUITE.seed))
    rastrigin = objectives(SUITE)["Rastrigin"]

    for fit in results["Rastrigin"].all_fits:
        gradient = rastrigin.gradient(fit.theta)  # type: ignore[attr-defined]
        assert float(gradient.abs().max()) < 1e-5
    assert (
        0
        <= reached(DECLARED["Rastrigin"], results["Rastrigin"], SUITE.at_minimum)
        <= RESTARTS["Rastrigin"]
    )


@pytest.mark.oracle
def test_the_surface_is_the_objective_on_its_grid() -> None:
    for name, objective in objectives(SUITE).items():
        xs, ys, values = surface(objective, DECLARED[name], SUITE.grid)
        x_min, x_max, y_min, y_max = DECLARED[name].domain

        assert values.shape == (SUITE.grid, SUITE.grid)
        assert (xs[0], xs[-1], ys[0], ys[-1]) == (x_min, x_max, y_min, y_max)
        corner = float(objective(torch.tensor([x_max, y_min], dtype=torch.float64)))
        assert values[0, -1] == corner
        assert values.min() >= 0.0


@pytest.mark.oracle
def test_known_minimizers_are_the_published_ones() -> None:
    # The fixture states the published values; this is what holds the file to
    # the module's own copy of Himmelblau's four and to the two exact ones.
    assert np.array_equal(
        known_minimizers(DECLARED["Himmelblau"]), np.array(HIMMELBLAU_MINIMA)
    )
    assert np.array_equal(
        known_minimizers(DECLARED["Rosenbrock"]), np.array([[1.0, 1.0]])
    )
    assert np.array_equal(known_minimizers(DECLARED["Rastrigin"]), np.zeros((1, 2)))


@pytest.mark.structural
def test_the_caption_counts_the_endpoints_it_was_handed(tmp_path: Path) -> None:
    results = endpoints(SUITE, np.random.default_rng(SUITE.seed))
    _, caption = build_figure(SUITE, results)
    for name, function in DECLARED.items():
        hits = reached(function, results[name], SUITE.at_minimum)
        assert f"{hits} of {function.restarts}" in caption

    qa_figure = main(["--params", str(FIXTURE.path), "--output-dir", str(tmp_path)])
    assert qa_figure.figure_path.is_file()
    assert f"seed {SUITE.seed}" in qa_figure.caption
