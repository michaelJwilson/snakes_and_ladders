"""A schedule as parameters builds the schedule it names, at the length asked (issue #1038).

Referees are properties of the curve whatever the implementation: the first
step is ``t_start`` and every step from the ramp's last is ``t_end``, both
exactly; the temperatures never turn back; the length is the one asked. With
no hold the build is the shape's own schedule, equal as a value, so a consumer
that took the default is handed the floats it had.
"""

from __future__ import annotations

import itertools
import math

import pytest
from sal.sample.schedule import (
    CosineTempSchedule,
    ExponentialTempSchedule,
    HeldTempSchedule,
    LinearTempSchedule,
    ScheduleParams,
    ScheduleShape,
    temperatures,
)

#: Lengths, holds and endpoint pairs, cooling and heating.
LENGTHS = (1, 2, 7, 1000)
HOLDS = (0.0, 0.25, 0.9)
ENDPOINTS = ((2.0, 0.05), (0.5, 1.5), (1.0, 1.0))


def _cases() -> list[tuple[ScheduleShape, float, float, float, int]]:
    """Every combination a build accepts: a one-step ramp needs equal endpoints."""
    cases = []
    for shape, (start, end), hold, length in itertools.product(
        ScheduleShape, ENDPOINTS, HOLDS, LENGTHS
    ):
        ramp = length - math.floor(hold * length)
        if ramp == 1 and start != end:
            continue
        cases.append((shape, start, end, hold, length))
    return cases


@pytest.mark.analytic
@pytest.mark.parametrize(("shape", "start", "end", "hold", "length"), _cases())
def test_endpoints_are_exact_the_curve_is_monotone_and_the_length_is_asked(
    shape: ScheduleShape, start: float, end: float, hold: float, length: int
) -> None:
    values = temperatures(ScheduleParams(shape, start, end, hold).build(length))

    assert len(values) == length
    assert values[0] == start
    held = math.floor(hold * length)
    assert values[length - held - 1 :] == [end] * (held + 1)
    # Monotone in the direction the endpoints set, to the rounding of one step.
    sign = 1.0 if end >= start else -1.0
    assert all(
        sign * (later - earlier) >= -1e-15 * max(start, end)
        for earlier, later in itertools.pairwise(values)
    )


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("shape", "kind"),
    [
        (ScheduleShape.EXPONENTIAL, ExponentialTempSchedule),
        (ScheduleShape.LINEAR, LinearTempSchedule),
        (ScheduleShape.COSINE, CosineTempSchedule),
    ],
)
def test_no_hold_builds_the_shapes_own_schedule(
    shape: ScheduleShape, kind: type[ExponentialTempSchedule]
) -> None:
    built = ScheduleParams(shape, 2.0, 0.05).build(1000)

    assert built == kind(2.0, 0.05, 1000)


@pytest.mark.analytic
def test_a_hold_repeats_the_ramps_last_temperature_after_it() -> None:
    built = ScheduleParams(ScheduleShape.LINEAR, 2.0, 0.05, 0.3).build(10)

    assert built == HeldTempSchedule(LinearTempSchedule(2.0, 0.05, 7), 3)
    assert temperatures(built)[:7] == temperatures(LinearTempSchedule(2.0, 0.05, 7))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "arguments",
    [
        (ScheduleShape.LINEAR, 0.0, 0.05, 0.0),
        (ScheduleShape.LINEAR, 2.0, -1.0, 0.0),
        (ScheduleShape.LINEAR, 2.0, 0.05, 1.0),
        (ScheduleShape.LINEAR, 2.0, 0.05, -0.1),
    ],
)
def test_a_temperature_or_hold_out_of_range_is_refused(
    arguments: tuple[ScheduleShape, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="temperature|fraction"):
        ScheduleParams(*arguments)


@pytest.mark.smoke
def test_a_one_step_ramp_between_two_temperatures_is_refused() -> None:
    with pytest.raises(ValueError, match="one-step"):
        ScheduleParams(ScheduleShape.EXPONENTIAL, 2.0, 0.05, 0.5).build(2)


@pytest.mark.smoke
def test_a_held_schedule_refuses_a_step_past_its_end() -> None:
    built = ScheduleParams(ScheduleShape.COSINE, 2.0, 0.05, 0.5).build(10)

    with pytest.raises(ValueError, match="outside"):
        built(10)
