"""The three statistics, each against a source that is not another program.

All exist because the repository carries no `scipy`, so each needs pinning
against something outside this codebase rather than against a second
implementation of the same series. The chi-square tail is checked at published
critical values; the autocorrelation time at the closed form for an AR(1)
process, whose value is known exactly from its parameter; the sign test at
binomial tail sums small enough to write out.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sal.sample.statistics import (
    chi_square_p_value,
    integrated_autocorrelation_time,
    sign_test_p_value,
)

from tests._rows import every_row

# Published chi-square critical values: the statistic at which the upper tail
# equals the stated significance. Any table gives these; they are not derived
# from the function under test.
CRITICAL_VALUES = [
    (1, 3.841, 0.05),
    (1, 6.635, 0.01),
    (2, 5.991, 0.05),
    (5, 11.070, 0.05),
    (10, 23.209, 0.01),
    (15, 24.996, 0.05),
    (15, 30.578, 0.01),
    (30, 50.892, 0.01),
    (63, 82.529, 0.05),
]


@pytest.mark.oracle
def test_the_chi_square_tail_matches_published_critical_values() -> None:
    # Two cells each `sqrt(statistic / 2)` from an expectation of 1: the p-value
    # must be the tabulated significance, to 5e-4 (three published decimals).
    def check(degrees_of_freedom: int, statistic: float, tail: float) -> None:
        deviation = float(np.sqrt(statistic / 2.0))
        expected = np.array([1.0, 1.0])
        observed = np.array([1.0 + deviation, 1.0 - deviation])

        realized = chi_square_p_value(
            observed, expected, degrees_of_freedom=degrees_of_freedom
        )

        assert realized == pytest.approx(tail, abs=5e-4)

    every_row(CRITICAL_VALUES, check)


@pytest.mark.smoke
def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="disagree"):
        chi_square_p_value(np.ones(3), np.ones(4))


@pytest.mark.smoke
def test_a_zero_expected_count_is_refused() -> None:
    # The statistic is infinite for any observation against a category the
    # model cannot produce, so the caller has chosen the wrong categories.
    with pytest.raises(ValueError, match="must be positive"):
        chi_square_p_value(np.ones(2), np.array([1.0, 0.0]))


@pytest.mark.analytic
def test_independent_draws_have_the_autocorrelation_time_of_independence() -> None:
    # 1 is the value for a series with no correlation, in the convention
    # `tau = 1 + 2 sum_t rho(t)` that `n / tau` reads as effective draws.
    rng = np.random.default_rng(0)

    realized = integrated_autocorrelation_time(rng.normal(size=200_000))

    assert realized == pytest.approx(1.0, abs=0.04)


@pytest.mark.oracle
def test_an_ar1_process_matches_its_closed_form() -> None:
    # For `x_t = rho x_{t-1} + noise`, `rho(t) = rho**t` exactly, so
    # `tau = 1 + 2 sum_{t>=1} rho**t = (1 + rho) / (1 - rho)`. An estimator
    # that failed to truncate its sum would drift above this with the series
    # length rather than converge to it.
    correlation = 0.8
    rng = np.random.default_rng(0)
    series = np.zeros(400_000)
    for step in range(1, series.shape[0]):
        series[step] = correlation * series[step - 1] + rng.normal()

    realized = integrated_autocorrelation_time(series)

    assert realized == pytest.approx(
        (1.0 + correlation) / (1.0 - correlation), rel=0.02
    )


#: Sigmas of Sokal's estimator error the AR(1) referee admits.
AR1_SIGMAS = 4.0
AR1_LENGTH = 100_000


@pytest.mark.oracle
@pytest.mark.parametrize("coefficient", [0.0, 0.5, 0.9])
def test_ar1_autocorrelation_time_is_held_to_sokals_error(coefficient: float) -> None:
    # Closed form `tau = (1 + phi) / (1 - phi)` (issue #1319). The tolerance
    # is derived, not fitted: Sokal (1997, sec. 3) gives the windowed
    # estimator's variance `var(tau_S) ~ 2 (2M + 1) / n tau_S**2`, with
    # `tau_S = tau / 2` and the window `M = c tau_S`, c = 5; the truncation
    # bias `2 phi**(M + 1) / (1 - phi)` is added, under 0.06 at phi = 0.9.
    exact = (1.0 + coefficient) / (1.0 - coefficient)
    window = math.ceil(5.0 * exact / 2.0)
    relative = AR1_SIGMAS * math.sqrt(2.0 * (2 * window + 1) / AR1_LENGTH)
    bias = 2.0 * coefficient ** (window + 1) / (1.0 - coefficient)
    rng = np.random.default_rng([1319, round(10 * coefficient)])
    noise = rng.normal(size=AR1_LENGTH)
    series = np.empty(AR1_LENGTH)
    series[0] = noise[0] / math.sqrt(1.0 - coefficient**2)
    for step in range(1, AR1_LENGTH):
        series[step] = coefficient * series[step - 1] + noise[step]

    realized = integrated_autocorrelation_time(series)

    assert abs(realized - exact) <= relative * exact + bias, (coefficient, realized)


@pytest.mark.smoke
def test_a_series_that_never_moved_reports_the_floor() -> None:
    # A constant chain has no correlation to measure, and is a sampler that
    # never moved rather than a fast one; the caller's own test sees it.
    assert integrated_autocorrelation_time(np.ones(1_000)) == 1.0


@pytest.mark.smoke
def test_a_series_too_short_to_have_an_autocorrelation_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two sweeps"):
        integrated_autocorrelation_time(np.array([1.0]))


# --- the sign test ---------------------------------------------------------

# Two-sided exact p-values from the binomial(n, 1/2) tails, summed by hand:
# 8 of 16 is the median, 10 of 10 is 2 / 2^10, 1 of 5 is 2 (1 + 5) / 32, and
# 6 of 8 is 2 (1 + 8 + 28) / 256.
SIGN_TEST_CASES = [
    (16, 8, 1.0),
    (10, 10, 2.0 / 1024.0),
    (5, 1, 12.0 / 32.0),
    (8, 6, 74.0 / 256.0),
]


@pytest.mark.oracle
def test_the_sign_test_matches_the_binomial_tail_sums() -> None:
    def check(n: int, positive: int, expected: float) -> None:
        differences = np.array([1.0] * positive + [-1.0] * (n - positive))
        assert sign_test_p_value(differences) == expected
        # Symmetric: the same count of the other sign gives the same p-value.
        assert sign_test_p_value(-differences) == expected

    every_row(SIGN_TEST_CASES, check)


@pytest.mark.smoke
def test_ties_carry_no_sign_and_all_ties_is_nothing_to_test() -> None:
    assert sign_test_p_value(np.zeros(6)) == 1.0
    assert sign_test_p_value(np.array([0.0, 0.0, 2.0, 0.5])) == 0.5
