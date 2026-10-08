"""The declined ``THERMODYNAMIC`` ramp and ``HUANG`` schedule of #1333, from their sandbox home (issue #1352).

Moved from ``tests/regression/sample/test_schedule_ramps.py`` unchanged but
for the import. Tolerances, declared before the run:

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
from sal.opt.termination import Stop
from sal.sample.loop import Moved
from sal.sample.schedule import temperatures
from sal.sandbox.adaptive_schedules import (
    ThermodynamicPilot,
    adaptive,
    anneal_adaptive,
    thermodynamic,
)
from sal.sim.potts import kaufman_energy

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


@pytest.mark.oracle
def test_thermodynamic_places_equal_length_on_the_exact_ising_sigma() -> None:
    start, end, n_steps = 3.5, 0.9, 50
    built = thermodynamic(start, end, n_steps, pilot=ISING_PILOT)
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
def test_the_thermodynamic_ramp_refuses_a_temperature_off_its_pilot() -> None:
    with pytest.raises(ValueError, match="outside the pilot's grid"):
        thermodynamic(5.0, 0.9, 5, pilot=ISING_PILOT)
