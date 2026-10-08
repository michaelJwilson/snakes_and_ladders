"""A start temperature in units of ``J * d`` carries from one cell to another (issue #1390, question 1).

`sal.qa.temperature_scale` tunes ``T0`` per `potts_reference` variant and
finds ``T0 / (J d)`` the scale whose stress-tuned constant transfers
(``docs/experiments/032-a-start-temperature-in-the-instances-units.md``).
Here the stress constant, ``c = 1/3``, is applied without a pilot to the
``ci`` cell and to its square-lattice twin, whose degree is 4 rather than 6.
"""

from __future__ import annotations

import dataclasses
import statistics

import numpy as np
import pytest
from sal.fixtures import Scale
from sal.qa import temperature_scale
from sal.sim.fixtures import fixture

#: ``T0 / (J d)`` tuned on the stress cell, read off the study's table.
STRESS_CONSTANT = 1.0 / 3.0
#: How far above the ladder's best held-out mean the transferred start may land.
SLACK = 1.0


@pytest.mark.experiment
@pytest.mark.parametrize("graph", ["triangular", "square"])
def test_a_coupling_relative_start_lands_near_the_ladders_best(graph: str) -> None:
    # The held-out gap to the TRW-S bound at T0 = c * J * d, against every
    # ladder start at the same visits and seeds: within SLACK nats of the best
    # ladder start, and below the coldest, which a start in absolute units
    # tuned for a larger scale would sit nearer.
    params = fixture("potts_reference", Scale.CI).params
    cell = temperature_scale.cell(dataclasses.replace(params, graph=graph), graph)
    t0 = STRESS_CONSTANT * cell.scales["coupling"]

    moved = statistics.fmean(cell.gaps(temperature_scale.relative(t0)))
    ladder = [
        statistics.fmean(cell.gaps(schedule)) for schedule in temperature_scale.LADDER
    ]

    assert moved >= -1e-9
    assert moved <= min(ladder) + SLACK
    assert moved < ladder[0]
    assert np.argmin(ladder) > 0
