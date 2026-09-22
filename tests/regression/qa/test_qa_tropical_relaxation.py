"""The number `fig:tropical-relaxation` plots, pinned (issue #729, step 4).

The renderer had no test at all: 78 statements, and nothing in the tier
imported it.
A renderer has no oracle --- what it draws is what `sandbox.tropical` and
`search.infer` computed, and those are refereed in their own modules --- so
what is held here is that the figure keeps plotting the numbers `STATUS.md`
records for the relaxation, at the fixture `qa.manifest` renders it from.

`STATUS.md` records the softmin's leakage certified under `1e-11` by
`sum_Q 2 exp(-g_Q / tau) R_Q`, and what is left at the temperature that
inverts for as float64 rounding of a sum over quartets rather than the
relaxation --- `3.8e-16` relative at worst over the 5-, 6- and 7-taxon
fixtures. Both are what panel (a) draws, and both are pinned below on the
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

    That gap is float64 rounding of a sum over quartets, a whole number of
    ulps of ``|D|`` whose count is the host's: two (7.28e-12) where MKL's
    vectorized ``exp`` and ``log`` take their AVX-512 path, as on the
    reference host, and one (3.64e-12) under ``MKL_CBWR=AVX2`` and
    ``MKL_CBWR=COMPATIBLE``, the second of which reproduces a GitHub runner's
    value bitwise (#895). So it is held to the corner tolerance
    `sandbox/test_sandbox_tropical.py` declares for the same comparison on
    the same fixture, and not to its bits.
    """
    return CORNER_RELATIVE_TOLERANCE * abs(sweep.discrete)


@pytest.fixture(scope="module")
def measured() -> tuple[Sweep, Surfaces]:
    """One sweep, taken once, and only where a test of it is selected.

    A module-level call would be paid at *collection*, so the 4.3 s would
    land on every run that collects this directory, the early gate included.
    """
    return measure(FIXTURE.params)


@pytest.mark.smoke
@pytest.mark.snapshot
def test_panel_a_plots_the_temperature_and_the_residual_status_records(
    measured: tuple[Sweep, Surfaces],
) -> None:
    """The two numbers the caption reads off panel (a), to be conserved.

    On `tree_search/ci.yaml`, over the corners of all 15 unrooted topologies:
    the largest temperature the certificate inverts to `1e-11` for is
    **0.0193186491251672**, and the largest measured gap there is
    **7.2759576141834e-12**, which is **2.89e-16** of the discrete score's
    magnitude **25,135.90**, on the reference host. The relative figure is
    inside the `3.8e-16` `STATUS.md` reports over the three fixtures, and the
    absolute one inside the `1e-11` the certificate was inverted for. The
    temperature and the score are pinned to 1e-12 relative, and hold there on
    all three MKL code paths measured; the gap is two ulps of the score, one
    on a GitHub runner (3.637978807091713e-12, 1.45e-16 relative), so it is
    pinned at :func:`_rounding`, the declared corner tolerance, rather than to
    its bits.
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

    25 temperatures from 1 to 1e-3, the measured gap falling **111.83 to
    7.28e-12** across them (3.64e-12 on a GitHub runner, :func:`_rounding`)
    and the bound lying above it at every temperature down to the derived
    one. Below that the gap stops improving, which is the
    claim the panel exists to make and the reason the axis is drawn that far.
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

    The quartet score and the fitted log-likelihood maximize at the same
    topology, index **13** of the 15 enumerated, which is the caption's
    "share their maximizer". They are different surfaces: their values
    correlate at **0.985** and are not each other.
    """
    _, surfaces = measured

    assert surfaces.argmax_agrees
    assert int(surfaces.quartet.argmax()) == int(surfaces.likelihood.argmax()) == 13
    assert surfaces.quartet.shape == surfaces.likelihood.shape == (15,)

    correlation = float(np.corrcoef(surfaces.quartet, surfaces.likelihood)[0, 1])
    assert correlation == pytest.approx(0.98492, abs=5e-5)
