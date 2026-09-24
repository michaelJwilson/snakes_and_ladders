"""The number `fig:turbo-waterfall` plots, pinned (issue #729, step 4).

The renderer had no test. With no oracle of its own
(`likelihood.turbo.measure_error_rates` is held to enumeration in its module),
pinned: the same call returns the same rates for the same seed, and each coded
point's place against `Q(sqrt(2 E_b / N_0))`. On `turbo/ci.yaml` (4 points x
100 frames, `K = 12`, 1.6 s), not the rendered `turbo/stress.yaml` (15.4 s,
over the 10 s cap). PR #767 flips BCJR to the Rust kernel, bitwise on the
(7, 5) register: the cost changes, these rates do not.
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
    """One ensemble (1.6 s), decoded once and only where selected, not at collection."""
    return measure(PARAMS)


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_waterfall_plots_the_bit_error_rates_this_seed_produces(
    rates: list[ErrorRates],
) -> None:
    """Panel (a)'s deepest curve, to be conserved.

    Seed 233, 8 iterations: 0.100833, 0.072500, 0.026667, 0.006667 at 0-3 dB,
    pinned by equality as 121, 87, 32 and 8 errors of 1,200 bits.
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

    Frame error 0.39, 0.23, 0.10, 0.02 of 100 frames; iteration 8 beats 1 except at 2 dB.
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

    `Q(sqrt(2 E_b / N_0))` = 0.0786, 0.0563, 0.0375, 0.0229; coded below it only
    at 2 and 3 dB (0.0267, 0.0067), above at 0 and 1 dB (0.1008, 0.0725), as
    `STATUS.md` records at `K = 12`; `K = 256` turns between 0.4 and 1.2 dB.
    """
    uncoded = uncoded_bit_error_rate(np.array(PARAMS.eb_n0_db))
    deepest = np.array([rate.bit_error_rate[-1] for rate in rates])

    np.testing.assert_allclose(
        uncoded, [0.078650, 0.056282, 0.037506, 0.022878], rtol=1e-4
    )
    assert bool(np.all(deepest[2:] < uncoded[2:]))
    assert bool(np.all(deepest[:2] > uncoded[:2]))
