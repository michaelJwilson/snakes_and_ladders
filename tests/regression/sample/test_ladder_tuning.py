"""``temperatures="auto"`` with a ``LadderTuning`` on the Potts tempering entry points (issue #1337).

The fixture is a 4x4 open lattice at ``q = 3``, coupling ``1.0``, with a
seeded per-site field. The referee is the band itself: a tuned ladder's
last measured exchange acceptances lie in ``band``, or ``within_band`` is
``False`` and the tuning's ``Termination`` says it stopped on its budget.
A given ladder is held bitwise to the run before the option existed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Stop
from sal.sample.potts_mcmc import cluster_tempering, parallel_tempering
from sal.sample.tune import LadderTuning, TunedLadder
from sal.search.ground_state import Problem, run_tempering
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

SEED = 1337
FIELD = np.random.default_rng(SEED).normal(0.0, 0.3, (16, 3))
BAND = (0.2, 0.3)
START = (0.3, 0.6, 1.2, 2.4)


def _graph() -> PottsGraph:
    return lattice_graph((4, 4), BoundaryCondition.OPEN, 1.0)


def _tuning(start: tuple[float, ...] = START) -> LadderTuning:
    return LadderTuning(
        Budget(Cost.SWEEPS, 12 * 100 * 8), 12, start, BAND, n_sweeps=100
    )


def _check_band(tuned: TunedLadder | None) -> None:
    assert tuned is not None
    acceptance = tuned.adapted.acceptance
    inside = all(BAND[0] <= value <= BAND[1] for value in acceptance)
    assert tuned.within_band == inside
    assert tuned.termination.converged == inside
    if not inside:
        assert tuned.termination.reason is Stop.BUDGET
    assert tuned.spent > 0
    assert len(tuned.adapted.temperatures) <= 12
    assert tuned.adapted.temperatures[0] == START[0]
    assert tuned.adapted.temperatures[-1] == START[-1]


@pytest.mark.analytic
def test_parallel_tempering_auto_ladder_lies_in_the_band_or_says_not() -> None:
    run = parallel_tempering(
        _graph(),
        FIELD,
        "auto",
        np.random.default_rng(0),
        50,
        ladder_tuning=_tuning(),
    )
    _check_band(run.tuned_ladder)
    assert run.tuned_ladder is not None
    assert run.temperatures == run.tuned_ladder.adapted.temperatures


@pytest.mark.analytic
def test_cluster_tempering_auto_ladder_lies_in_the_band_or_says_not() -> None:
    run = cluster_tempering(
        _graph(),
        FIELD,
        "auto",
        np.random.default_rng(0),
        50,
        ladder_tuning=_tuning(),
    )
    _check_band(run.tuned_ladder)


@pytest.mark.analytic
def test_run_tempering_auto_charges_the_pilot_and_reports_the_ladder() -> None:
    problem = Problem(_graph(), FIELD, 3)
    budget = Budget(Cost.SITE_VISITS, 200_000)
    run = run_tempering(
        problem,
        budget,
        np.random.default_rng(0),
        temperatures="auto",
        ladder_tuning=_tuning(),
    )
    assert run.tuned_ladder is not None
    assert run.spent >= run.tuned_ladder.spent > 0


@pytest.mark.patch
@pytest.mark.analytic
@pytest.mark.parametrize("entry", ["parallel", "cluster"])
def test_auto_without_tuning_is_refused_before_any_work(entry: str) -> None:
    call = parallel_tempering if entry == "parallel" else cluster_tempering
    with pytest.raises(ValueError, match="LadderTuning"):
        call(_graph(), FIELD, "auto", np.random.default_rng(0), 10)


@pytest.mark.patch
@pytest.mark.analytic
def test_a_given_ladder_with_a_tuning_is_refused() -> None:
    with pytest.raises(ValueError, match="LadderTuning"):
        parallel_tempering(
            _graph(),
            FIELD,
            START,
            np.random.default_rng(0),
            10,
            ladder_tuning=_tuning(),
        )


@pytest.mark.patch
@pytest.mark.oracle
def test_a_given_ladder_is_unchanged_bitwise() -> None:
    first = parallel_tempering(_graph(), FIELD, START, np.random.default_rng(4), 20)
    second = parallel_tempering(
        _graph(), FIELD, START, np.random.default_rng(4), 20, ladder_tuning=None
    )
    assert first.tuned_ladder is None
    assert np.array_equal(first.states, second.states)
    assert first.energy == second.energy
