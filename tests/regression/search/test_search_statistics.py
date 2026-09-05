"""The two statistics, each against a source that is not another program.

Both exist because the repository carries no `scipy`, so both need pinning
against something outside this codebase rather than against a second
implementation of the same series. The chi-square tail is checked at published
critical values; the autocorrelation time at the closed form for an AR(1)
process, whose value is known exactly from its parameter.
"""

from __future__ import annotations

import numpy as np
import pytest
from phylo.search.statistics import (
    chi_square_p_value,
    integrated_autocorrelation_time,
)

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
@pytest.mark.parametrize(("degrees_of_freedom", "statistic", "tail"), CRITICAL_VALUES)
def test_the_chi_square_tail_matches_published_critical_values(
    degrees_of_freedom: int, statistic: float, tail: float
) -> None:
    # Two categories whose squared deviation over expectation is exactly the
    # critical statistic, so the p-value must come back as the significance
    # that value was tabulated at. The tolerance is 5e-4 because the published
    # values are quoted to three decimals.
    # Two cells, each deviating by `sqrt(statistic / 2)` from an expectation
    # of 1, so the statistic is exactly the tabulated value.
    deviation = float(np.sqrt(statistic / 2.0))
    expected = np.array([1.0, 1.0])
    observed = np.array([1.0 + deviation, 1.0 - deviation])

    realized = chi_square_p_value(
        observed, expected, degrees_of_freedom=degrees_of_freedom
    )

    assert realized == pytest.approx(tail, abs=5e-4)


@pytest.mark.edge_case
def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="disagree"):
        chi_square_p_value(np.ones(3), np.ones(4))


@pytest.mark.edge_case
def test_a_zero_expected_count_is_refused() -> None:
    # The statistic is infinite for any observation against a category the
    # model cannot produce, so the caller has chosen the wrong categories.
    with pytest.raises(ValueError, match="must be positive"):
        chi_square_p_value(np.ones(2), np.array([1.0, 0.0]))


@pytest.mark.mathematical
def test_independent_draws_have_the_autocorrelation_time_of_independence() -> None:
    # 0.5 is the value for a series with no correlation, in the convention
    # `tau = 0.5 + sum_t rho(t)`.
    rng = np.random.default_rng(0)

    realized = integrated_autocorrelation_time(rng.normal(size=200_000))

    assert realized == pytest.approx(0.5, abs=0.02)


@pytest.mark.oracle
def test_an_ar1_process_matches_its_closed_form() -> None:
    # For `x_t = rho x_{t-1} + noise`, `rho(t) = rho**t` exactly, so
    # `tau = 0.5 + sum_{t>=1} rho**t = 0.5 + rho / (1 - rho)`. An estimator
    # that failed to truncate its sum would drift above this with the series
    # length rather than converge to it.
    correlation = 0.8
    rng = np.random.default_rng(0)
    series = np.zeros(400_000)
    for step in range(1, series.shape[0]):
        series[step] = correlation * series[step - 1] + rng.normal()

    realized = integrated_autocorrelation_time(series)

    assert realized == pytest.approx(0.5 + correlation / (1.0 - correlation), rel=0.02)


@pytest.mark.edge_case
def test_a_series_that_never_moved_reports_the_floor() -> None:
    # A constant chain has no correlation to measure, and is a sampler that
    # never moved rather than a fast one; the caller's own test sees it.
    assert integrated_autocorrelation_time(np.ones(1_000)) == 0.5


@pytest.mark.edge_case
def test_a_series_too_short_to_have_an_autocorrelation_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two sweeps"):
        integrated_autocorrelation_time(np.array([1.0]))
