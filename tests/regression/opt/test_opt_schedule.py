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

import math

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.opt.schedule import (
    Constant,
    Cosine,
    Exponential,
    Linear,
    Schedule,
    temperatures,
)

CURVES = [Linear, Exponential, Cosine]
ENDPOINTS = [(4.0, 0.05, 50), (0.1, 3.0, 7), (2.5, 2.5, 12), (1.0, 1e-3, 2)]


@pytest.mark.parametrize("curve", CURVES)
@pytest.mark.parametrize(("start", "end", "n_steps"), ENDPOINTS)
def test_both_endpoints_are_reached_exactly_at_the_declared_steps(
    curve: type[Linear], start: float, end: float, n_steps: int
) -> None:
    # `==`, not a tolerance. A schedule that ends at `end * (1 - 1e-16)` is
    # not at `end`, and a consumer comparing "did we reach the target
    # temperature" by equality would say no forever.
    schedule = curve(start, end, n_steps)

    assert schedule(0) == start
    assert schedule(n_steps - 1) == end
    assert len(temperatures(schedule)) == n_steps


@pytest.mark.parametrize("curve", CURVES)
@pytest.mark.parametrize(("start", "end"), [(4.0, 0.05), (0.1, 3.0)])
def test_the_curve_is_strictly_monotone_in_the_declared_direction(
    curve: type[Linear], start: float, end: float
) -> None:
    values = np.array(temperatures(curve(start, end, 40)))
    steps = np.diff(values)

    assert bool((steps < 0.0).all()) if end < start else bool((steps > 0.0).all())


@pytest.mark.parametrize("curve", CURVES)
def test_a_step_outside_the_schedule_is_refused_not_clamped(
    curve: type[Linear],
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


def test_a_constant_schedule_is_constant_and_the_default_is_temperature_one() -> None:
    schedule = Constant(1.0, 25)

    assert temperatures(schedule) == [1.0] * 25
    assert isinstance(schedule, Schedule)


@pytest.mark.parametrize(
    "schedule",
    [
        Constant(1.0, 3),
        Linear(1.0, 0.5, 3),
        Exponential(1.0, 0.5, 3),
        Cosine(1.0, 0.5, 3),
    ],
)
def test_every_schedule_satisfies_the_protocol(schedule: object) -> None:
    assert isinstance(schedule, Schedule)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_a_non_positive_temperature_is_refused(bad: float) -> None:
    # At zero every acceptance ratio is 0 or 1 and the chain is a descent;
    # NaN compares false to everything and would pass a `<= 0` check.
    with pytest.raises(ValueError, match="positive temperature"):
        Constant(bad, 5)
    with pytest.raises(ValueError, match="positive temperature"):
        Linear(1.0, bad, 5)
    with pytest.raises(ValueError, match="positive temperature"):
        Exponential(bad, 1.0, 5)


def test_an_empty_schedule_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        Constant(1.0, 0)
    with pytest.raises(ValueError, match="at least one step"):
        Cosine(1.0, 0.5, 0)


def test_a_one_step_schedule_must_have_one_temperature() -> None:
    # `t = 0 / 0` otherwise; and a schedule that starts at 2 and ends at 1 in
    # a single step has no step at which either could be true.
    with pytest.raises(ValueError, match="one-step schedule has one temperature"):
        Linear(2.0, 1.0, 1)

    assert Linear(2.0, 2.0, 1)(0) == 2.0


def test_the_exponential_schedule_has_a_constant_ratio() -> None:
    # The characterization independent of the formula: geometric means the
    # ratio of successive temperatures never changes, and its value is the
    # `gamma` torch's ExponentialLR would need to land at `end`.
    values = np.array(temperatures(Exponential(4.0, 0.05, 50)))
    ratios = values[1:] / values[:-1]

    assert_allclose(ratios, (0.05 / 4.0) ** (1.0 / 49.0), rtol=1e-12)


def test_the_linear_schedule_has_a_constant_difference() -> None:
    values = np.array(temperatures(Linear(4.0, 0.05, 50)))

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

    assert_allclose(temperatures(Linear(start, end, n_steps)), expected, rtol=1e-12)


def test_exponential_mirrors_torch_exponential_lr() -> None:
    start, end, n_steps = 4.0, 0.05, 50
    expected = _torch_schedule(
        torch.optim.lr_scheduler.ExponentialLR,
        start,
        n_steps,
        gamma=(end / start) ** (1.0 / (n_steps - 1)),
    )

    assert_allclose(
        temperatures(Exponential(start, end, n_steps)), expected, rtol=1e-12
    )


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
    realized = temperatures(Cosine(start, end, n_steps))

    assert_allclose(realized, expected, rtol=1e-10)
    assert realized[0] == start
    assert realized[-1] == end
