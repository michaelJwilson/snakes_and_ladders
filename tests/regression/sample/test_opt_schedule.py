"""What a temperature schedule can get wrong, and the library it mirrors.

The properties a schedule is actually got wrong on are not distributional:
an endpoint off by one ulp, a final temperature reached one step early or
late, a temperature that touches zero. None of them would be localized by a
goodness-of-fit test on the chain that consumed the schedule, so each is
pinned here on the schedule alone. The schedules also claim to mirror
``torch.optim.lr_scheduler``, and that claim is checked against the library
rather than asserted (issue #267).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.sample.schedule import (
    ConstantTempSchedule,
    CosineTempSchedule,
    ExponentialTempSchedule,
    LinearTempSchedule,
    TempSchedule,
    adapt_ladder,
    adapt_ladder_by_round_trips,
    temperatures,
)

from tests._rows import every_row, every_value

CURVES = [LinearTempSchedule, ExponentialTempSchedule, CosineTempSchedule]
ENDPOINTS = [(4.0, 0.05, 50), (0.1, 3.0, 7), (2.5, 2.5, 12), (1.0, 1e-3, 2)]


@pytest.mark.oracle
@pytest.mark.parametrize("curve", CURVES)
def test_both_endpoints_are_reached_exactly_at_the_declared_steps(
    curve: type[LinearTempSchedule],
) -> None:
    # `==`, not a tolerance. A schedule that ends at `end * (1 - 1e-16)` is
    # not at `end`, and a consumer comparing "did we reach the target
    # temperature" by equality would say no forever.
    def check(start: float, end: float, n_steps: int) -> None:
        schedule = curve(start, end, n_steps)

        assert schedule(0) == start
        assert schedule(n_steps - 1) == end
        assert len(temperatures(schedule)) == n_steps

    every_row(ENDPOINTS, check)


@pytest.mark.analytic
@pytest.mark.parametrize("curve", CURVES)
def test_the_curve_is_strictly_monotone_in_the_declared_direction(
    curve: type[LinearTempSchedule],
) -> None:
    def check(start: float, end: float) -> None:
        values = np.array(temperatures(curve(start, end, 40)))
        steps = np.diff(values)

        assert bool((steps < 0.0).all()) if end < start else bool((steps > 0.0).all())

    every_row([(4.0, 0.05), (0.1, 3.0)], check)


@pytest.mark.smoke
@pytest.mark.parametrize("curve", CURVES)
def test_a_step_outside_the_schedule_is_refused_not_clamped(
    curve: type[LinearTempSchedule],
) -> None:
    # The off-by-one no downstream test localizes: a consumer that runs one
    # iteration longer than it declared would, under clamping, sit at the
    # final temperature for one extra step and look exactly like one that
    # did not.
    schedule = curve(2.0, 0.5, 10)

    with pytest.raises(ValueError, match="outside a schedule of 10 steps"):
        schedule(10)
    with pytest.raises(ValueError, match="outside a schedule"):
        schedule(-1)


@pytest.mark.infra
def test_a_constant_schedule_is_constant_and_the_default_is_temperature_one() -> None:
    schedule = ConstantTempSchedule(1.0, 25)

    assert temperatures(schedule) == [1.0] * 25
    assert isinstance(schedule, TempSchedule)


@pytest.mark.parametrize(
    "schedule",
    [
        ConstantTempSchedule(1.0, 3),
        LinearTempSchedule(1.0, 0.5, 3),
        ExponentialTempSchedule(1.0, 0.5, 3),
        CosineTempSchedule(1.0, 0.5, 3),
    ],
)
@pytest.mark.infra
def test_every_schedule_satisfies_the_protocol(schedule: object) -> None:
    assert isinstance(schedule, TempSchedule)


@pytest.mark.smoke
def test_a_non_positive_temperature_is_refused() -> None:
    # At zero every acceptance ratio is 0 or 1 and the chain is a descent;
    # NaN compares false to everything and would pass a `<= 0` check.
    def check(bad: float) -> None:
        with pytest.raises(ValueError, match="positive temperature"):
            ConstantTempSchedule(bad, 5)
        with pytest.raises(ValueError, match="positive temperature"):
            LinearTempSchedule(1.0, bad, 5)
        with pytest.raises(ValueError, match="positive temperature"):
            ExponentialTempSchedule(bad, 1.0, 5)

    every_value([0.0, -1.0, math.nan], check)


@pytest.mark.smoke
def test_an_empty_schedule_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        ConstantTempSchedule(1.0, 0)
    with pytest.raises(ValueError, match="at least one step"):
        CosineTempSchedule(1.0, 0.5, 0)


@pytest.mark.infra
def test_a_one_step_schedule_must_have_one_temperature() -> None:
    # `t = 0 / 0` otherwise; and a schedule that starts at 2 and ends at 1 in
    # a single step has no step at which either could be true.
    with pytest.raises(ValueError, match="one-step schedule has one temperature"):
        LinearTempSchedule(2.0, 1.0, 1)

    assert LinearTempSchedule(2.0, 2.0, 1)(0) == 2.0


@pytest.mark.infra
def test_the_exponential_schedule_has_a_constant_ratio() -> None:
    # The characterization independent of the formula: geometric means the
    # ratio of successive temperatures never changes, and its value is the
    # `gamma` torch's ExponentialLR would need to land at `end`.
    values = np.array(temperatures(ExponentialTempSchedule(4.0, 0.05, 50)))
    ratios = values[1:] / values[:-1]

    assert_allclose(ratios, (0.05 / 4.0) ** (1.0 / 49.0), rtol=1e-12)


@pytest.mark.analytic
def test_the_linear_schedule_has_a_constant_difference() -> None:
    values = np.array(temperatures(LinearTempSchedule(4.0, 0.05, 50)))

    assert_allclose(np.diff(values), (0.05 - 4.0) / 49.0, rtol=1e-12)


def _torch_schedule(
    scheduler: type[torch.optim.lr_scheduler.LRScheduler],
    start: float,
    n_steps: int,
    **kwargs: float | int,
) -> list[float]:
    """The learning rates torch's scheduler produces from a base rate ``start``."""
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=start)
    schedule = scheduler(optimizer, **kwargs)  # type: ignore[arg-type]
    rates = [float(schedule.get_last_lr()[0])]
    for _ in range(n_steps - 1):
        optimizer.step()
        schedule.step()
        rates.append(float(schedule.get_last_lr()[0]))
    return rates


@pytest.mark.infra
def test_linear_mirrors_torch_linear_lr() -> None:
    # `LinearLR` scales a base rate from `start_factor` to `end_factor` over
    # `total_iters` steps; with the base rate as `start` and the factors
    # chosen to land on `end`, it is this schedule.
    start, end, n_steps = 4.0, 0.05, 50
    expected = _torch_schedule(
        torch.optim.lr_scheduler.LinearLR,
        start,
        n_steps,
        start_factor=1.0,
        end_factor=end / start,
        total_iters=n_steps - 1,
    )

    assert_allclose(
        temperatures(LinearTempSchedule(start, end, n_steps)), expected, rtol=1e-12
    )


@pytest.mark.infra
def test_exponential_mirrors_torch_exponential_lr() -> None:
    start, end, n_steps = 4.0, 0.05, 50
    expected = _torch_schedule(
        torch.optim.lr_scheduler.ExponentialLR,
        start,
        n_steps,
        gamma=(end / start) ** (1.0 / (n_steps - 1)),
    )

    assert_allclose(
        temperatures(ExponentialTempSchedule(start, end, n_steps)), expected, rtol=1e-12
    )


@pytest.mark.infra
def test_cosine_mirrors_torch_cosine_annealing_lr() -> None:
    # torch computes the cosine schedule recursively, accumulating rounding
    # over the run, so the agreement is to 1e-10 rather than 1e-12 -- and
    # the two agree *exactly* at both ends, which the recursion does not
    # guarantee and this schedule does.
    start, end, n_steps = 4.0, 0.05, 50
    expected = _torch_schedule(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        start,
        n_steps,
        T_max=n_steps - 1,
        eta_min=end,
    )
    realized = temperatures(CosineTempSchedule(start, end, n_steps))

    assert_allclose(realized, expected, rtol=1e-10)
    assert realized[0] == start
    assert realized[-1] == end


# --- a ladder from what it measures (#333) ----------------------------------


def _ratio_acceptance(ladder: tuple[float, ...]) -> list[float]:
    """A synthetic exchange acceptance, ``min(T) / max(T)`` per pair: closed form."""
    return [min(a, b) / max(a, b) for a, b in itertools.pairwise(ladder)]


BAND = (0.4, 0.7)


@pytest.mark.oracle
def test_a_gap_below_the_band_is_bisected_until_the_ladder_is_geometric() -> None:
    # From two endpoints a factor of 16 apart, two rounds of geometric
    # bisection reach ratios of 2, acceptance 0.5, and the third measurement
    # finds every pair inside the band; the ladder is then the geometric one
    # in closed form, and nothing was inserted past what the band asked for.
    result = adapt_ladder(_ratio_acceptance, (8.0, 0.5), BAND, 10, 16)

    assert result.within_band
    assert result.rounds == 3
    assert result.replicas_measured == 2 + 3 + 5
    assert_allclose(result.temperatures, [8.0 / 2**k for k in range(5)], rtol=1e-12)
    assert_allclose(result.acceptance, [0.5] * 4)


@pytest.mark.oracle
@pytest.mark.analytic
def test_a_temperature_between_two_gaps_above_the_band_is_removed() -> None:
    # A ladder twice as dense as the band needs loses every other interior
    # temperature, never two adjacent ones in one round, until the pairs
    # exchange inside the band.
    dense = tuple(8.0 / math.sqrt(2) ** k for k in range(9))  # ratios of sqrt(2)

    result = adapt_ladder(_ratio_acceptance, dense, BAND, 10, 16)

    assert result.within_band
    assert_allclose(result.temperatures, [8.0 / 2**k for k in range(5)], rtol=1e-12)


@pytest.mark.oracle
@pytest.mark.analytic
def test_a_gap_above_the_band_beside_one_inside_it_moves_their_shared_temperature() -> (
    None
):
    # Neither insertion nor removal applies: the high pair has no gap to
    # bisect and its shared temperature is needed by the pair inside the
    # band. Moving it halfway toward the far end widens the one and narrows
    # the other, and both land inside; the ladder keeps its three rungs.
    result = adapt_ladder(_ratio_acceptance, (4.0, 3.5, 1.5), BAND, 10, 16)

    assert result.within_band
    assert len(result.temperatures) == 3
    assert result.temperatures[0] == 4.0
    assert result.temperatures[2] == 1.5
    assert result.temperatures[1] == pytest.approx(math.sqrt(3.5 * 1.5))


@pytest.mark.smoke
def test_two_endpoints_above_the_band_are_reported_not_changed() -> None:
    # The endpoints are the caller's, so a pair of them that exchanges above
    # the band has nothing the warm-up may do; it says so in one round.
    result = adapt_ladder(_ratio_acceptance, (1.0, 0.9), BAND, 10, 16)

    assert not result.within_band
    assert result.rounds == 1
    assert result.temperatures == (1.0, 0.9)


@pytest.mark.smoke
def test_the_replica_budget_caps_insertion_and_is_reported() -> None:
    result = adapt_ladder(_ratio_acceptance, (8.0, 0.5), BAND, 10, 3)

    assert not result.within_band
    assert result.temperatures == (8.0, 2.0, 0.5)
    assert result.rounds == 2


@pytest.mark.smoke
def test_a_ladder_or_band_the_warm_up_cannot_use_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two temperatures"):
        adapt_ladder(_ratio_acceptance, (1.0,), BAND, 1, 4)
    with pytest.raises(ValueError, match="strictly monotone"):
        adapt_ladder(_ratio_acceptance, (1.0, 3.0, 2.0), BAND, 1, 4)
    with pytest.raises(ValueError, match="positive temperature"):
        adapt_ladder(_ratio_acceptance, (1.0, 0.0), BAND, 1, 4)
    with pytest.raises(ValueError, match="0 < low < high < 1"):
        adapt_ladder(_ratio_acceptance, (2.0, 1.0), (0.7, 0.4), 1, 4)
    with pytest.raises(ValueError, match="max_rounds"):
        adapt_ladder(_ratio_acceptance, (2.0, 1.0), BAND, 0, 4)
    with pytest.raises(ValueError, match="already has"):
        adapt_ladder(_ratio_acceptance, (2.0, 1.0, 0.5), BAND, 1, 2)
    with pytest.raises(ValueError, match="one per neighbouring pair"):
        adapt_ladder(lambda _ladder: [0.5], (2.0, 1.0, 0.5), BAND, 1, 4)


# --- the other criterion: a ladder placed by its round trips (#756) ---------
#
# Per-pair acceptance against a whole-ladder statistic; both synthetic
# measurements below are closed forms.


def _linear_up_fraction(ladder: tuple[float, ...]) -> list[float]:
    """The up-fraction falling linearly in temperature: 1 at the first rung, 0 at the last.

    Mass ``sqrt(-df dT)`` is proportional to width, so the optimum is uniform in ``T``.
    """
    values = np.array(ladder)
    return list((values - values[-1]) / (values[0] - values[-1]))


def _bottleneck_up_fraction(ladder: tuple[float, ...]) -> list[float]:
    """An up-fraction that falls over a width of 0.03 about ``T = 1``: a barrier there."""
    values = np.array(ladder)
    fraction = 1.0 / (1.0 + np.exp((values - 1.0) / 0.03))
    return list((fraction - fraction.min()) / (fraction.max() - fraction.min()))


@pytest.mark.analytic
@pytest.mark.critical
def test_a_linear_up_fraction_places_the_ladder_uniform_in_temperature() -> None:
    # The closed-form fixed point: from a ladder whose steps are a factor of
    # two, one placement returns the uniform one, and the second measurement
    # moves nothing, so the warm-up converges on round 2 having measured ten
    # replicas.
    result = adapt_ladder_by_round_trips(
        _linear_up_fraction, (8.0, 4.0, 2.0, 1.0, 0.5), 1e-9, 5
    )

    assert result.converged
    assert result.rounds == 2
    assert result.replicas_measured == 10
    assert_allclose(
        result.temperatures, [8.0 - 1.875 * k for k in range(5)], rtol=1e-12
    )
    assert result.up_fraction == (1.0, 0.75, 0.5, 0.25, 0.0)


@pytest.mark.analytic
def test_the_rungs_concentrate_where_the_up_fraction_falls() -> None:
    # Katzgraber's claim, on a barrier at T = 1 that a geometric ladder puts
    # one rung inside: the placement moves five of nine rungs into the window
    # where the walkers are held up, and leaves both endpoints where the
    # caller put them, bitwise.
    start = tuple(0.5 * 1.3**k for k in range(9))
    inside = [0.9 <= value <= 1.1 for value in start]
    assert sum(inside) == 1

    result = adapt_ladder_by_round_trips(_bottleneck_up_fraction, start, 1e-3, 6)

    assert sum(0.9 <= value <= 1.1 for value in result.temperatures) == 5
    assert result.temperatures[0] == start[0]
    assert result.temperatures[-1] == start[-1]
    assert not result.converged


@pytest.mark.smoke
def test_a_ladder_or_a_measurement_the_placement_cannot_use_is_refused() -> None:
    # A ladder of two is two endpoints and nothing to place; a flat
    # up-fraction is a run in which no walker circulated, which is a
    # measurement of nothing rather than a ladder that is already right.
    with pytest.raises(ValueError, match="at least three temperatures"):
        adapt_ladder_by_round_trips(_linear_up_fraction, (2.0, 1.0), 1e-3, 4)
    with pytest.raises(ValueError, match="strictly monotone"):
        adapt_ladder_by_round_trips(_linear_up_fraction, (1.0, 3.0, 2.0), 1e-3, 4)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        adapt_ladder_by_round_trips(_linear_up_fraction, (4.0, 2.0, 1.0), 0.0, 4)
    with pytest.raises(ValueError, match="max_rounds"):
        adapt_ladder_by_round_trips(_linear_up_fraction, (4.0, 2.0, 1.0), 1e-3, 0)
    with pytest.raises(ValueError, match="one per rung"):
        adapt_ladder_by_round_trips(lambda _l: [1.0, 0.0], (4.0, 2.0, 1.0), 1e-3, 1)
    with pytest.raises(ValueError, match="must be finite"):
        adapt_ladder_by_round_trips(
            lambda _l: [1.0, float("nan"), 0.0], (4.0, 2.0, 1.0), 1e-3, 1
        )
    with pytest.raises(ValueError, match="flat over the whole ladder"):
        adapt_ladder_by_round_trips(
            lambda _l: [1.0, 1.0, 1.0], (4.0, 2.0, 1.0), 1e-3, 1
        )
