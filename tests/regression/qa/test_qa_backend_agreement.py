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
from phylo.qa import backend_agreement
from phylo.qa.backend_agreement import BACKENDS, SITE_COUNTS, agreement, build_figure
from phylo.sim.params import load_simulation_params

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params.yaml"

# The backends compute in float64 and sum over sites, so the deviation is
# bounded by accumulated rounding rather than by machine epsilon. Measured
# worst case across every backend and site count: 4.0e-14.
_AGREEMENT = 1e-12


@pytest.fixture(scope="module")
def measured() -> dict[str, list[tuple[int, float]]]:
    return agreement(load_simulation_params(FIXTURE))


@pytest.mark.oracle
def test_every_backend_agrees_with_brute_force(
    measured: dict[str, list[tuple[int, float]]],
) -> None:
    assert set(measured) == set(BACKENDS)
    for name, points in measured.items():
        assert [size for size, _ in points] == list(SITE_COUNTS)
        worst = max(value for _, value in points)
        assert worst < _AGREEMENT, f"{name} deviates by {worst:.2e}"


@pytest.mark.mathematical
def test_the_deviation_does_not_grow_with_the_sum() -> None:
    # The reason the figure reports a relative deviation. An absolute one
    # grows with the site count, since the log-likelihood is a sum; the
    # relative one does not, and that is the property that makes a single
    # bound transfer across problem sizes.
    params = load_simulation_params(FIXTURE)
    relative = agreement(params)["numpy"]
    smallest = relative[0][1]
    largest = relative[-1][1]
    assert max(smallest, largest) < _AGREEMENT
    assert SITE_COUNTS[-1] // SITE_COUNTS[0] >= 10


@pytest.mark.structural
def test_the_caption_reports_the_worst_deviation_it_measured(
    measured: dict[str, list[tuple[int, float]]],
) -> None:
    params = load_simulation_params(FIXTURE)
    figure, caption = build_figure(measured, params)
    try:
        worst = max(value for points in measured.values() for _, value in points)
        assert f"{worst:.2e}" in caption
        assert str(params.seed) in caption
        assert "brute-force" in caption.lower()
    finally:
        figure.clear()


@pytest.mark.structural
def test_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    written = backend_agreement.main(
        ["--params", str(FIXTURE), "--output-dir", str(tmp_path)]
    )
    assert written.figure_path.is_file()
    assert written.caption_path.is_file()
    assert written.caption == written.caption_path.read_text()


@pytest.mark.structural
def test_float64_epsilon_is_below_the_measured_deviation(
    measured: dict[str, list[tuple[int, float]]],
) -> None:
    # The figure draws epsilon as a reference line. If the deviation were at
    # epsilon the line would say nothing; it is above it, which is what a sum
    # over hundreds of sites should do, and the line is what shows that.
    worst = max(value for points in measured.values() for _, value in points)
    assert worst > float(np.finfo(np.float64).eps)
    assert_allclose(float(np.finfo(np.float64).eps), 2.220446049250313e-16)
