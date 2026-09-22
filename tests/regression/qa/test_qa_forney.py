"""Regression tests for snakes_and_ladders.qa.forney.

The renderer draws a window of the declared instance, so what is pinned is
the window: it stays inside the declared caps at every tier, the edges it
draws join two drawn sites, and the caption it ships passes the check a
document's ``\\input`` relies on (issue #891).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from snakes_and_ladders.qa.figure import check_latex_safe
from snakes_and_ladders.qa.forney import (
    LATTICE_CAP,
    POSITION_CAP,
    build_figure,
    forney_window,
)
from snakes_and_ladders.sim.fixtures import fixture

PROBLEM = "spatio_sequential_counts"


@pytest.mark.smoke
@pytest.mark.parametrize("tier", ["ci", "release"])
def test_the_figure_builds_inside_its_caps_with_a_latex_safe_caption(
    tier: str,
) -> None:
    params = fixture(PROBLEM, tier).params.model
    window = forney_window(params, emission_stride=2, classes_shown=2)

    assert window.rows <= LATTICE_CAP
    assert window.columns <= LATTICE_CAP
    assert window.positions == min(params.n_positions, POSITION_CAP)
    drawn = set(window.sites.tolist())
    assert all(int(a) in drawn and int(b) in drawn for a, b in window.edges)
    assert set(window.subsample.tolist()) <= drawn
    assert window.highlighted in set(window.subsample.tolist())
    assert np.array_equal(np.unique(window.labels), np.arange(params.n_classes))

    figure, caption = build_figure(params)
    try:
        check_latex_safe(caption)
        assert f"M = {params.n_classes}" in caption
        assert len(figure.axes) == 1
    finally:
        plt.close(figure)


@pytest.mark.smoke
def test_a_window_asking_for_more_chains_than_classes_is_refused() -> None:
    params = fixture(PROBLEM, "ci").params.model

    with pytest.raises(ValueError, match="classes_shown"):
        forney_window(params, emission_stride=2, classes_shown=params.n_classes + 1)
    with pytest.raises(ValueError, match="emission_stride"):
        forney_window(params, emission_stride=0, classes_shown=1)
