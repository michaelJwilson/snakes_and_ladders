"""A discrete instance descent does not solve, whose optimum is still enumerated (issue #406).

Since #198 no problem failed a baseline (the seven-taxon tree: 0.476 by
descent, 1.0 by restarts). Asserted of the fixture: single-site descent reaches
the ground state from well under 0.4 of starts, failures are genuine local
minima, and the ground state is enumerated, 7.0 below the planted state.
Random restarts reach it on every seed: a single descent has 0.92 of
headroom, restarts none (#198's distinction). The numbers are the baseline
record's, recomputed at release (`infra/baselines.py`) and here for one seed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.fixtures import Scale
from sal.search.alpha_expansion import Labelling
from sal.search.icm import iterated_conditional_modes
from sal.sim.canonical import FrustratedLatticeParams, PlantedSpinGlass
from sal.sim.fixtures import Baseline, baseline, fixture
from sal.sim.potts import energy

PROBLEM, TIER = "planted_glass", Scale.CI
#: Zero field: the difficulty is in the couplings, not in a site preference.
FIELD = np.zeros(2)
#: `STATUS.md`'s bar: a baseline solves an instance when it reaches the
#: optimum from 0.4 or more of starts.
SOLVED = 0.4


@pytest.fixture(scope="module")
def record() -> Baseline:
    return baseline(PROBLEM, TIER)


@pytest.fixture(scope="module")
def glass() -> PlantedSpinGlass:
    """The declared instance, built from the fixture's own seed."""
    params: FrustratedLatticeParams = fixture(PROBLEM, TIER).params
    (frustration,) = params.glass_frustrations
    return params.glass(frustration, np.random.default_rng(params.seed))


def _descents(instance: PlantedSpinGlass, seed: int, restarts: int) -> list[Labelling]:
    generator = np.random.default_rng(seed)
    return [
        iterated_conditional_modes(instance.graph, FIELD, 2, generator)
        for _ in range(restarts)
    ]


@pytest.mark.oracle
def test_single_site_descent_does_not_solve_the_declared_glass(
    record: Baseline, glass: PlantedSpinGlass
) -> None:
    # The claim the fixture exists to support, at one seed of the sixteen the
    # record averages. The bound is the recorded rate plus its interval rather
    # than the rate itself, because one seed of fifty restarts is a noisier
    # estimate than sixteen and pinning the mean would be pinning the noise.
    best = record.value("enumerated_ground_energy")
    rate = float(
        np.mean(
            [abs(found - best) < 1e-9 for _, found in _descents(glass, 20260908, 50)]
        )
    )

    assert rate < SOLVED, f"descent solved the fixture at {rate}"
    assert abs(rate - record.value("descent_rate")) < 4 * record.value(
        "descent_rate_half_width"
    ), rate


@pytest.mark.oracle
def test_the_ground_state_is_enumerated_and_the_planted_state_is_not_it(
    record: Baseline, glass: PlantedSpinGlass
) -> None:
    # At frustration 0.30 the planted state is 7.0 above the ground state, so
    # the oracle is `enumeration` (`planted_spin_glass`'s warning, realized).
    best = record.value("enumerated_ground_energy")
    planted = record.value("planted_energy")

    assert float(glass.planted_energy) == planted
    assert planted > best
    assert glass.unsatisfied > 0, "an unfrustrated glass is a ferromagnet"


@pytest.mark.analytic
def test_every_descent_failure_stops_at_a_genuine_local_minimum(
    record: Baseline, glass: PlantedSpinGlass
) -> None:
    # The difference between a hard instance and a truncated run. If a descent
    # stopped with an improving flip still available, the failure would
    # measure the sweep budget rather than the environment.
    best = record.value("enumerated_ground_energy")
    failures = [
        labelling
        for labelling, found in _descents(glass, 20260908, 20)
        if abs(found - best) >= 1e-9
    ]

    assert failures, "expected descent to fall short on this fixture"
    for labelling in failures:
        here = energy(glass.graph, FIELD, labelling)
        for node in range(glass.graph.n_nodes):
            flipped = labelling.copy()
            flipped[node] = 1 - flipped[node]
            assert energy(glass.graph, FIELD, flipped) >= here - 1e-12, node


@pytest.mark.oracle
def test_random_restart_descent_still_reaches_the_ground_state(
    record: Baseline, glass: PlantedSpinGlass
) -> None:
    # The baseline to beat is restart descent: 1.000 over 50 restarts on every
    # seed measured; here 8 of 50 within 1e-9 of `enumerated_ground_energy`.
    best = record.value("enumerated_ground_energy")
    reached = [abs(found - best) < 1e-9 for _, found in _descents(glass, 20260908, 50)]

    assert any(reached)
    assert record.value("restart_success") == 1.0
