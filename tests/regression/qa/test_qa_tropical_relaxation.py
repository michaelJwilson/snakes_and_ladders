"""The number `fig:tropical-relaxation` plots, pinned (issue #729, step 4).

The renderer had no test (78 statements). With no oracle of its own
(`sandbox.tropical` and `search.infer` are refereed in their modules), pinned:
`STATUS.md`'s leakage certified under `1e-11` by
`sum_Q 2 exp(-g_Q / tau) R_Q`, and the float64 residue at the inverted
temperature, `3.8e-16` relative at worst over 5, 6 and 7 taxa; both on the
5-taxon instance.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.qa.tropical_relaxation import (
    TEMPERATURES,
    TOLERANCE,
    Surfaces,
    Sweep,
    measure,
)
from snakes_and_ladders.sim.fixtures import fixture

from tests.regression.sandbox.test_sandbox_tropical import CORNER_RELATIVE_TOLERANCE

FIXTURE = fixture("tree_search", "ci")


def _rounding(sweep: Sweep) -> float:
    """The band the gap at and below the derived temperature is pinned in.

    Whole ulps of ``|D|``: two (7.28e-12) on MKL AVX-512, one (3.64e-12) under
    ``MKL_CBWR=AVX2``/``COMPATIBLE``, matching a runner bitwise (#895); so the
    corner tolerance of `sandbox/test_sandbox_tropical.py`, not bits.
    """
    return CORNER_RELATIVE_TOLERANCE * abs(sweep.discrete)


@pytest.fixture(scope="module")
def measured() -> tuple[Sweep, Surfaces]:
    """One sweep (4.3 s), taken once and only where selected, not at collection."""
    return measure(FIXTURE.params)


@pytest.mark.smoke
@pytest.mark.snapshot
def test_panel_a_plots_the_temperature_and_the_residual_status_records(
    measured: tuple[Sweep, Surfaces],
) -> None:
    """The two numbers the caption reads off panel (a), to be conserved.

    On `tree_search/ci.yaml`, all 15 topologies: temperature
    0.0193186491251672 and score 25,135.90, pinned to 1e-12 on three MKL
    paths; gap 7.2759576141834e-12 (2.89e-16 relative, inside `STATUS.md`'s
    3.8e-16 and the 1e-11), one ulp on a runner (3.637978807091713e-12), so
    pinned at :func:`_rounding`.
    """
    sweep, surfaces = measured

    assert TOLERANCE == 1e-11
    assert len(surfaces.quartet) == 15

    assert sweep.derived == pytest.approx(0.0193186491251672, rel=1e-12)
    assert sweep.at_derived == pytest.approx(7.2759576141834e-12, abs=_rounding(sweep))
    assert sweep.discrete == pytest.approx(-25135.9048514472, rel=1e-12)

    relative = sweep.at_derived / abs(sweep.discrete)
    assert relative == pytest.approx(2.89464718186e-16, abs=CORNER_RELATIVE_TOLERANCE)
    assert sweep.at_derived <= TOLERANCE


@pytest.mark.smoke
@pytest.mark.snapshot
def test_panel_a_sweeps_the_temperatures_it_declares_and_falls_with_them(
    measured: tuple[Sweep, Surfaces],
) -> None:
    """The sweep's shape, which is what the reader takes off the log axes.

    25 temperatures, 1 to 1e-3: gap 111.83 to 7.28e-12 (runner 3.64e-12), under the bound.
    """
    sweep, _ = measured

    assert TEMPERATURES.shape == (25,)
    assert (TEMPERATURES[0], TEMPERATURES[-1]) == (1.0, 1e-3)

    assert sweep.measured.max() == pytest.approx(111.828161016223, rel=1e-12)
    assert sweep.measured.min() == pytest.approx(
        7.2759576141834e-12, abs=_rounding(sweep)
    )
    above = sweep.derived <= TEMPERATURES
    assert bool(np.all(sweep.bound[above] >= sweep.measured[above]))


@pytest.mark.smoke
@pytest.mark.snapshot
def test_panel_b_plots_two_surfaces_that_share_their_maximizer(
    measured: tuple[Sweep, Surfaces],
) -> None:
    """What licenses relaxing one surface in place of the other.

    Both maximize at topology 13 of 15; values correlate at 0.985, not 1.
    """
    _, surfaces = measured

    assert surfaces.argmax_agrees
    assert int(surfaces.quartet.argmax()) == int(surfaces.likelihood.argmax()) == 13
    assert surfaces.quartet.shape == surfaces.likelihood.shape == (15,)

    correlation = float(np.corrcoef(surfaces.quartet, surfaces.likelihood)[0, 1])
    assert correlation == pytest.approx(0.98492, abs=5e-5)
