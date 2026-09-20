"""The backend-agreement figure, and the oracle it rests on.

What this pins is the claim the figure makes, not that it rendered: every
backend agrees with brute-force marginalization, and the deviation is
reported relative because the quantity is a sum over sites.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.qa import backend_agreement
from snakes_and_ladders.qa.backend_agreement import (
    BACKENDS,
    SITE_COUNTS,
    agreement,
    build_figure,
)
from snakes_and_ladders.sim.params import SimulationParams

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "tree_jc/stress.yaml"

# The backends compute in float64 and sum over sites, so the deviation is
# bounded by accumulated rounding rather than by machine epsilon. Measured
# worst case across every backend and site count: 4.0e-14.
_AGREEMENT = 1e-12


@pytest.fixture(scope="module")
def measured() -> dict[str, list[tuple[int, float]]]:
    return agreement(load_params(FIXTURE, SimulationParams))


@pytest.mark.oracle
def test_every_backend_agrees_with_brute_force(
    measured: dict[str, list[tuple[int, float]]],
) -> None:
    # The span is part of the claim. The deviation is reported relative, so
    # one bound holding across a tenfold range of site counts is what says it
    # does not grow with the sum.
    assert SITE_COUNTS[-1] // SITE_COUNTS[0] >= 10
    assert set(measured) == set(BACKENDS)
    for name, points in measured.items():
        assert [size for size, _ in points] == list(SITE_COUNTS)
        worst = max(value for _, value in points)
        assert worst < _AGREEMENT, f"{name} deviates by {worst:.2e}"


@pytest.mark.smoke
def test_the_caption_reports_the_worst_deviation_it_measured(
    measured: dict[str, list[tuple[int, float]]],
) -> None:
    params = load_params(FIXTURE, SimulationParams)
    figure, caption = build_figure(measured, params)
    try:
        worst = max(value for points in measured.values() for _, value in points)
        assert f"{worst:.2e}" in caption
        assert str(params.seed) in caption
        assert "brute-force" in caption.lower()
    finally:
        figure.clear()


@pytest.mark.smoke
def test_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    written = backend_agreement.main(
        ["--params", str(FIXTURE), "--output-dir", str(tmp_path)]
    )
    assert written.figure_path.is_file()
    assert written.caption_path.is_file()
    assert written.caption == written.caption_path.read_text()


@pytest.mark.smoke
def test_float64_epsilon_is_below_the_measured_deviation(
    measured: dict[str, list[tuple[int, float]]],
) -> None:
    # The figure draws epsilon as a reference line. If the deviation were at
    # epsilon the line would say nothing; it is above it, which is what a sum
    # over hundreds of sites should do, and the line is what shows that.
    worst = max(value for points in measured.values() for _, value in points)
    assert worst > float(np.finfo(np.float64).eps)
    assert_allclose(float(np.finfo(np.float64).eps), 2.220446049250313e-16)
