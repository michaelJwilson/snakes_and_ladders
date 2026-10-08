"""Every annealing ramp makes some ``g(T)`` linear in some ``s(k)``, with exact endpoints (issue #1333).

Tolerances, declared before the run:

* endpoints: bitwise, every shape;
* ``g(T_k) - ((1 - s_k) g(start) + s_k g(end))``: ``1e-15`` times
  ``max(|g(start)|, |g(end)|)`` for every closed-form shape;
* Cauchy ``start / (1 + c k)``, ``c`` from the endpoints, against
  ``INVERSE_LINEAR``: bitwise on every step but the last, which the ramp pins
  to ``end``;
* ``LOGARITHMIC``'s ``c / log(n - 1 + d)`` against ``end``: ``1e-12``
  relative;
* ``THERMODYNAMIC`` on the ``8 x 8`` Ising torus, ``sigma_E`` from Kaufman's
  exact energy by a central difference at ``h = 1e-5``: equal thermodynamic
  length per step to ``1e-10`` of the total; the difference's error, read
  against ``h = 2e-5``, below ``1e-6`` relative;
* ``HUANG``: the recurrence bitwise on a stated energy sequence.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Stop
from sal.sample.loop import Moved, anneal_adaptive
from sal.sample.potts_mcmc import PottsMove, Recolour
from sal.sample.schedule import (
    CosineTempSchedule,
    ExponentialTempSchedule,
    LinearTempSchedule,
    LogarithmicTempSchedule,
    Ramp,
    ScheduleParams,
    ScheduleShape,
    ThermodynamicPilot,
    adaptive,
    ramp,
    temperatures,
)
from sal.sample.tune import Criterion, tune_schedule
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import kaufman_energy

#: Cooling endpoint pairs and lengths every closed-form shape is read at.
ENDPOINTS = ((2.0, 0.05), (10.0, 9.0), (0.8, 0.3))
LENGTHS = (2, 3, 17, 1000)
LINEARITY = 1e-15
LOG_ENDPOINT = 1e-12
LENGTH_TOLERANCE = 1e-10
#: The 8 x 8 torus, its sites, and the central difference's step in ``T``.
SIDE = 8
N_SITES = SIDE * SIDE
STEP = 1e-5
DIFFERENCE_TOLERANCE = 1e-6


def _ising_sigma(temperature: float, step: float = STEP) -> float:
    """``sigma_E = T sqrt(N de/dT)`` from Kaufman's exact energy per site, ``J = 1``."""

    def energy(t: float) -> float:
        # kaufman_energy takes beta J and returns beta e; e = T * that.
        return t * kaufman_energy(1.0 / t, SIDE)

    slope = (energy(temperature + step) - energy(temperature - step)) / (2.0 * step)
    return temperature * math.sqrt(N_SITES * slope)


#: ``sigma_E`` on 401 temperatures from 0.8 to 4.0, across T_c = 1.13.
ISING_GRID = tuple(0.8 + 3.2 * k / 400 for k in range(401))
ISING_PILOT = ThermodynamicPilot(ISING_GRID, tuple(_ising_sigma(t) for t in ISING_GRID))


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


CLOSED = [shape for shape in ScheduleShape if shape is not ScheduleShape.THERMODYNAMIC]


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


@pytest.mark.oracle
def test_thermodynamic_places_equal_length_on_the_exact_ising_sigma() -> None:
    start, end, n_steps = 3.5, 0.9, 50
    built = ramp.thermodynamic(start, end, n_steps, pilot=ISING_PILOT)
    values = temperatures(built)
    lengths = np.array([ISING_PILOT.length(value) for value in values])
    total = lengths[0] - lengths[-1]
    line = lengths[0] - total * np.arange(n_steps) / (n_steps - 1)

    assert values[0] == start
    assert values[-1] == end
    assert all(later < earlier for earlier, later in itertools.pairwise(values))
    assert np.max(np.abs(lengths - line)) <= LENGTH_TOLERANCE * total
    # The ramp slows where sigma_E / T^2 peaks: the smallest step sits near T_c.
    gaps = -np.diff(values)
    assert 1.0 < values[int(np.argmin(gaps))] < 1.4
    # The central difference's error, read against twice the step.
    coarse = np.array([_ising_sigma(t, 2.0 * STEP) for t in ISING_GRID[::40]])
    fine = np.array(ISING_PILOT.sigma[::40])
    assert np.max(np.abs(coarse / fine - 1.0)) <= DIFFERENCE_TOLERANCE


@pytest.mark.oracle
def test_huang_reproduces_its_recurrence_bitwise() -> None:
    schedule = adaptive.huang(2.0, 0.1, rate=0.7, budget=100)
    blocks = [[0.0, 1.0, 3.0], [2.0, 2.5], [1.0, 4.0, 4.0, 9.0]]
    temperature, expected = 2.0, []
    for energies in blocks:
        sigma = float(np.std(energies))
        expected.append(max(temperature * math.exp(-0.7 * temperature / sigma), 0.1))
        temperature = expected[-1]
    got, temperature = [], 2.0
    for energies in blocks:
        reached = schedule.next(temperature, energies)
        assert reached is not None
        got.append(reached)
        temperature = reached

    assert got == expected
    assert schedule.next(0.1, [1.0, 2.0]) is None


def _walk(budget: int) -> tuple:  # type: ignore[type-arg]
    """``anneal_adaptive`` on a step whose energies alternate 0 and 1: sigma_E = 0.5."""

    def step(
        state: int, _energy: float, _carried: None, _temperature: float, _rng: None
    ) -> Moved[int, None]:
        return Moved(state + 1, float((state + 1) % 2), None, 1)

    schedule = adaptive.huang(2.0, 0.1, rate=0.7, budget=budget)
    return schedule, anneal_adaptive(
        step, schedule, Moved(0, 0.0, None, 0), None, lambda s: s, block=4
    )


@pytest.mark.oracle
def test_anneal_adaptive_stops_converged_at_end_or_on_budget() -> None:
    schedule, walked = _walk(100)
    expected, temperature = [2.0], 2.0
    while temperature > 0.1:
        temperature = max(temperature * math.exp(-0.7 * temperature / 0.5), 0.1)
        expected.append(temperature)

    assert walked.temperatures == tuple(expected)
    assert walked.termination.reason is Stop.CONVERGED
    assert walked.termination.iterations == len(expected)
    assert walked.spent == 4 * len(expected)

    _, capped = _walk(2)
    assert capped.temperatures == tuple(expected[:2])
    assert capped.termination.reason is Stop.BUDGET


@pytest.mark.smoke
@pytest.mark.parametrize("shape", list(ScheduleShape))
def test_every_shape_has_an_explicit_call_a_match_case_and_a_tune_candidate(
    shape: ScheduleShape,
) -> None:
    pilot = {"pilot": ISING_PILOT} if shape is ScheduleShape.THERMODYNAMIC else {}
    explicit = getattr(Ramp, shape.value)
    assert temperatures(ramp(shape, 2.0, 0.9, 9, **pilot)) == temperatures(
        explicit(2.0, 0.9, 9, **pilot)
    )
    params = ScheduleParams(shape, 2.0, 0.9, pilot=pilot.get("pilot"))
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
    with pytest.raises(ValueError, match="takes no pilot"):
        ramp(ScheduleShape.LINEAR, 2.0, 0.1, 5, pilot=ISING_PILOT)
    with pytest.raises(ValueError, match="needs pilot"):
        ramp(ScheduleShape.THERMODYNAMIC, 2.0, 0.9, 5)
    with pytest.raises(ValueError, match="cools"):
        ramp(ScheduleShape.LOGARITHMIC, 0.1, 2.0, 5)
