"""Every annealing ramp makes some ``g(T)`` linear in some ``s(k)``, with exact endpoints (issue #1333).

Tolerances, declared before the run:

* endpoints: bitwise, every shape;
* ``g(T_k) - ((1 - s_k) g(start) + s_k g(end))``: ``1e-15`` times
  ``max(|g(start)|, |g(end)|)`` for every closed-form shape;
* Cauchy ``start / (1 + c k)``, ``c`` from the endpoints, against
  ``INVERSE_LINEAR``: bitwise on every step but the last, which the ramp pins
  to ``end``;
* ``LOGARITHMIC``'s ``c / log(n - 1 + d)`` against ``end``: ``1e-12``
  relative.

The ``THERMODYNAMIC`` ramp and the ``HUANG`` schedule are declined and their
tests moved with them (issue #1352):
``tests/regression/sandbox/test_adaptive_schedules.py``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove, Recolour
from sal.sample.schedule import (
    CosineTempSchedule,
    ExponentialTempSchedule,
    LinearTempSchedule,
    LogarithmicTempSchedule,
    Ramp,
    ScheduleParams,
    ScheduleShape,
    ramp,
    temperatures,
)
from sal.sample.tune import Criterion, tune_schedule
from sal.sim.graph import BoundaryCondition, lattice_graph

#: Cooling endpoint pairs and lengths every closed-form shape is read at.
ENDPOINTS = ((2.0, 0.05), (10.0, 9.0), (0.8, 0.3))
LENGTHS = (2, 3, 17, 1000)
LINEARITY = 1e-15
LOG_ENDPOINT = 1e-12


def _g_and_s(shape: ScheduleShape, n_steps: int, built: object) -> tuple:  # type: ignore[type-arg]
    """``g`` and ``s_k`` as #1333's table states them, for the closed forms."""
    t = np.arange(n_steps) / (n_steps - 1)
    k = np.arange(n_steps)
    match shape:
        case ScheduleShape.LINEAR:
            return (lambda x: x), t
        case ScheduleShape.EXPONENTIAL:
            return math.log, t
        case ScheduleShape.COSINE:
            return (lambda x: x), 0.5 * (1.0 - np.cos(np.pi * t))
        case ScheduleShape.INVERSE_LINEAR:
            return (lambda x: 1.0 / x), t
        case ScheduleShape.POWER:
            return math.log, np.log1p(k) / math.log(n_steps)
        case ScheduleShape.LOGARITHMIC:
            assert isinstance(built, LogarithmicTempSchedule)
            x = built.log_offset
            return (lambda v: 1.0 / v), np.log1p(k * math.exp(-x)) / math.log1p(
                (n_steps - 1) * math.exp(-x)
            )
    raise AssertionError(shape)


CLOSED = list(ScheduleShape)


@pytest.mark.analytic
@pytest.mark.parametrize("shape", CLOSED)
@pytest.mark.parametrize(("start", "end"), ENDPOINTS)
@pytest.mark.parametrize("n_steps", LENGTHS)
def test_g_is_linear_in_s_the_endpoints_exact_and_the_ramp_strictly_monotone(
    shape: ScheduleShape, start: float, end: float, n_steps: int
) -> None:
    built = ramp(shape, start, end, n_steps)
    values = temperatures(built)
    g, s = _g_and_s(shape, n_steps, built)

    assert values[0] == start
    assert values[-1] == end
    assert all(later < earlier for earlier, later in itertools.pairwise(values))
    scale = max(abs(g(start)), abs(g(end)))
    for value, fraction in zip(values, s, strict=True):
        line = (1.0 - fraction) * g(start) + fraction * g(end)
        assert abs(g(value) - line) <= LINEARITY * scale + 4 * math.ulp(scale)


@pytest.mark.analytic
@pytest.mark.parametrize(("start", "end"), ENDPOINTS)
@pytest.mark.parametrize("n_steps", LENGTHS)
def test_cauchy_with_c_from_the_endpoints_is_inverse_linear_bitwise(
    start: float, end: float, n_steps: int
) -> None:
    rate = (start / end - 1.0) / (n_steps - 1)
    cauchy = [start / (1.0 + rate * k) for k in range(n_steps)]
    values = temperatures(ramp.inverse_linear(start, end, n_steps))

    assert values[:-1] == cauchy[:-1]
    assert values[-1] == end
    assert abs(cauchy[-1] - end) <= LINEARITY * end + math.ulp(end)


@pytest.mark.analytic
@pytest.mark.parametrize(("start", "end"), [*ENDPOINTS, (2.0, 2.0 - 1e-13)])
@pytest.mark.parametrize("n_steps", LENGTHS)
def test_the_logarithmic_offset_meets_its_endpoint(
    start: float, end: float, n_steps: int
) -> None:
    built = ramp.logarithmic(start, end, n_steps)
    assert isinstance(built, LogarithmicTempSchedule)
    x = built.log_offset
    # Geman & Geman's c / log(n - 1 + d), c = start log d, in x = log d.
    reached = start * x / (x + math.log1p((n_steps - 1) * math.exp(-x)))

    assert abs(reached - end) <= LOG_ENDPOINT * end


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("shape", "kind"),
    [
        (ScheduleShape.EXPONENTIAL, ExponentialTempSchedule),
        (ScheduleShape.LINEAR, LinearTempSchedule),
        (ScheduleShape.COSINE, CosineTempSchedule),
    ],
)
def test_the_existing_shapes_are_bitwise_unchanged(
    shape: ScheduleShape, kind: type[LinearTempSchedule]
) -> None:
    by_hand = kind(2.0, 0.05, 1000)
    for built in (
        ramp(shape, 2.0, 0.05, 1000),
        getattr(ramp, shape.value)(2.0, 0.05, 1000),
    ):
        assert built == by_hand
        assert temperatures(built) == temperatures(by_hand)


@pytest.mark.smoke
@pytest.mark.parametrize("shape", list(ScheduleShape))
def test_every_shape_has_an_explicit_call_a_match_case_and_a_tune_candidate(
    shape: ScheduleShape,
) -> None:
    explicit = getattr(Ramp, shape.value)
    assert temperatures(ramp(shape, 2.0, 0.9, 9)) == temperatures(explicit(2.0, 0.9, 9))
    params = ScheduleParams(shape, 2.0, 0.9)
    tuned = tune_schedule(
        lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8),
        np.zeros((9, 2)),
        move=PottsMove.SINGLE_SITE,
        recolour=Recolour.UNIFORM,
        budget=Budget(Cost.SITE_VISITS, 9 * 4),
        criterion=Criterion.LOWEST_ENERGY,
        rng=np.random.default_rng(1333),
        grid=(params,),
    )
    assert tuned.params == params


@pytest.mark.smoke
def test_the_generic_call_refuses_by_name() -> None:
    with pytest.raises(ValueError, match="no ramp shape 'quadratic'"):
        ramp("quadratic", 2.0, 0.1, 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cools"):
        ramp(ScheduleShape.LOGARITHMIC, 0.1, 2.0, 5)
