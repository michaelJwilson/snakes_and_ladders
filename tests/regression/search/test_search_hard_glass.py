"""A discrete instance descent does not solve, whose optimum is still enumerated (issue #406).

`STATUS.md` has recorded since #198 that the repository has no problem a
baseline fails on: hill climbing reaches the enumerated maximum from every
start of the six-taxon tree, and from 0.476 of the seven-taxon one while
random restarts reach it always. Without such an instance every Stage 2 claim
about a learned searcher is a claim about a solved problem.

What is asserted here is a property of the *fixture*, as
`test_search_hard_fixture.py` asserts one of the tree: that single-site
descent reaches the ground state from well under the 0.4 of starts `STATUS.md`
calls solved, that the failures are genuine local minima rather than truncated
runs, and that the ground state is an enumerated fact rather than the planted
bound --- which on this instance it is not, the planted state scoring 7.0
above it.

The counterweight is asserted too, because the fixture would otherwise
overstate itself: random-restart descent reaches the ground state on every
seed. A single descent has 0.92 of headroom here; restarts have none. That is
the same distinction #198 drew on the tree, made explicit rather than left to
a later reader.

The numbers come from the fixture's baseline record, which
`infra/baselines.py` recomputes at the release gate; the recomputation here is
one seed of it, which is what makes the record a check rather than a memo.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.search.alpha_expansion import (
    energy,
    iterated_conditional_modes,
)
from snakes_and_ladders.sim.canonical import FrustratedLatticeParams, PlantedSpinGlass
from snakes_and_ladders.sim.fixtures import Baseline, baseline, fixture

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


def _descents(
    instance: PlantedSpinGlass, seed: int, restarts: int
) -> list[tuple[np.ndarray, float]]:
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
    # Why the fixture states `enumeration` as its oracle and not the planted
    # energy `sim.canonical` carries. At frustration 0.30 the planted state is
    # 7.0 above the ground state, so a search scored against it would be
    # scored against the wrong number --- the case `planted_spin_glass`'s
    # docstring warns of, realized.
    best = record.value("enumerated_ground_energy")
    planted = record.value("planted_energy")

    assert float(glass.planted_energy) == planted
    assert planted > best
    assert glass.unsatisfied > 0, "an unfrustrated glass is a ferromagnet"


@pytest.mark.mathematical
def test_every_descent_failure_stops_at_a_genuine_local_minimum(
    record: Baseline, glass: PlantedSpinGlass
) -> None:
    # The difference between a hard instance and a truncated run. If a descent
    # stopped with an improving flip still available, the failure would
    # measure the sweep budget rather than the landscape.
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


@pytest.mark.structural
def test_random_restart_descent_still_reaches_the_ground_state(
    record: Baseline, glass: PlantedSpinGlass
) -> None:
    # The counterweight, and the reason this fixture does not by itself settle
    # Milestone 2.1: the baseline a learned searcher has to beat is
    # random-restart descent, and over the declared 50 restarts that baseline
    # is 1.000 on every seed measured. The headroom is against a single run.
    best = record.value("enumerated_ground_energy")
    reached = [abs(found - best) < 1e-9 for _, found in _descents(glass, 20260908, 50)]

    assert any(reached)
    assert record.value("restart_success") == 1.0
