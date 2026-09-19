"""The number `fig:turbo-waterfall` plots, pinned (issue #729, step 4).

The renderer had no test at all. A renderer has no oracle --- the waterfall
is what `likelihood.turbo.measure_error_rates` computed, and that is held to
enumeration in its own module --- so what is pinned here is that the same
call keeps returning the same rates for the same seed, beside the one
statement the figure makes that a closed form decides: where each coded
point sits against the uncoded antipodal rate `Q(sqrt(2 E_b / N_0))` drawn
over it.

**The instance is the enumerable one, not the rendered one.** `qa.manifest`
renders the figure from `turbo/stress.yaml`, 6 points x 50 frames at
`K = 256`, which is 15.4 s of decoding and outside `DEV.md`'s 10 s per-PR
cap; `measure` is the same call on either, so the pin is taken on
`turbo/ci.yaml` --- 4 points x 100 frames at `K = 12`, 1.6 s --- and runs
per pull request rather than at the release gate.

**Pull request #767 changes this figure's cost, not its number.** It flips
the BCJR default to the Rust kernel, whose pass is bitwise on the (7, 5)
register; the rates below are what the Python default computes on this base
and are what the compiled default must reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.turbo import ErrorRates, uncoded_bit_error_rate
from snakes_and_ladders.qa.turbo_waterfall import DRAWN_ITERATIONS, measure
from snakes_and_ladders.sim.fixtures import fixture

FIXTURE = fixture("turbo", "ci")
PARAMS = FIXTURE.params


@pytest.fixture(scope="module")
def rates() -> list[ErrorRates]:
    """One ensemble, decoded once, and only where a test of it is selected.

    A module-level call would be paid at *collection*, so the 1.6 s would
    land on every run that collects this directory, the early gate included.
    """
    return measure(PARAMS)


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_waterfall_plots_the_bit_error_rates_this_seed_produces(
    rates: list[ErrorRates],
) -> None:
    """Panel (a)'s deepest curve, to be conserved.

    At `K = 12`, 100 frames and 1,200 message bits per point under seed 233,
    the bit error rate at 8 iterations is **0.100833, 0.072500, 0.026667 and
    0.006667** at 0, 1, 2 and 3 dB. Pinned by equality on the error counts
    the rates are ratios of --- 121, 87, 32 and 8 of 1,200 --- since a rate
    over a fixed denominator is an integer and admits no tolerance.
    """
    assert tuple(PARAMS.eb_n0_db) == (0.0, 1.0, 2.0, 3.0)
    assert [rate.bits for rate in rates] == [1200] * 4

    deepest = np.array([rate.bit_error_rate[-1] for rate in rates])
    errors = np.rint(deepest * 1200).astype(int)

    assert errors.tolist() == [121, 87, 32, 8]
    np.testing.assert_allclose(
        deepest, np.array([121, 87, 32, 8]) / 1200.0, rtol=0.0, atol=0.0
    )


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_frame_error_rates_of_panel_b_and_the_iteration_it_draws(
    rates: list[ErrorRates],
) -> None:
    """Panel (b), and the four curves panel (a) separates.

    The frame error rate at 8 iterations is **0.39, 0.23, 0.10 and 0.02**
    over the four points, of 100 frames each, so each is an integer count of
    erring frames and is pinned by equality. Panel (a) draws iterations
    1, 2, 4 and 8, and the deepest beats the first at every point but 2 dB,
    where `STATUS.md` records the two tying.
    """
    assert DRAWN_ITERATIONS == (1, 2, 4, 8)

    frames = np.array([rate.frame_error_rate[-1] for rate in rates])
    assert np.rint(frames * 100).astype(int).tolist() == [39, 23, 10, 2]

    first = np.array([rate.bit_error_rate[0] for rate in rates])
    deepest = np.array([rate.bit_error_rate[-1] for rate in rates])
    assert bool(np.all(deepest <= first))
    assert float(deepest.sum()) < float(first.sum())


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_coded_curve_sits_where_the_uncoded_closed_form_puts_it(
    rates: list[ErrorRates],
) -> None:
    """The reference curve the figure is read against.

    `Q(sqrt(2 E_b / N_0))` is **0.0786, 0.0563, 0.0375 and 0.0229** at the
    four points, and the coded rate at 8 iterations is below it at 2 and
    3 dB alone: **0.0267 against 0.0375** and **0.0067 against 0.0229**. At
    0 and 1 dB the code buys nothing --- 0.1008 against 0.0786 and 0.0725
    against 0.0563 --- which is the second negative result `STATUS.md`
    records at this interleaver length, pinned rather than smoothed over.
    The `K = 256` instance the figure is rendered from turns between 0.4 and
    1.2 dB instead, which is why the waterfall is drawn there.
    """
    uncoded = uncoded_bit_error_rate(np.array(PARAMS.eb_n0_db))
    deepest = np.array([rate.bit_error_rate[-1] for rate in rates])

    np.testing.assert_allclose(
        uncoded, [0.078650, 0.056282, 0.037506, 0.022878], rtol=1e-4
    )
    assert bool(np.all(deepest[2:] < uncoded[2:]))
    assert bool(np.all(deepest[:2] > uncoded[:2]))
