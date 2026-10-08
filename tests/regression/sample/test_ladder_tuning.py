"""``temperatures="auto"`` with a ``LadderTuning`` on the Potts tempering entry points (issue #1337).

The fixture is a 4x4 open lattice at ``q = 3``, coupling ``1.0``, with a
seeded per-site field. The referee is the band itself: a tuned ladder's
last measured exchange acceptances lie in ``band``, or ``within_band`` is
``False`` and the tuning's ``Termination`` says it stopped on its budget.
A given ladder is held bitwise to the run before the option existed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Stop
from sal.sample.potts_mcmc import chains, cluster_tempering, parallel_tempering
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


# --- one ladder order and a start per rung (#1343) ---------------------------

#: Both Potts temperings, as a test calls them: a ladder, a seed, a start.
SAMPLERS = (parallel_tempering, cluster_tempering)


@pytest.mark.analytic
@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
def test_both_temperings_refuse_a_ladder_hottest_first(sampler: Any) -> None:
    # The one order is coldest first; the reverse is refused, naming it,
    # rather than run as reversed draws.
    with pytest.raises(ValueError, match="increasing, coldest first"):
        sampler(_graph(), FIELD, START[::-1], np.random.default_rng(0), 2)


@pytest.mark.analytic
@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
def test_one_start_is_every_rungs_start_bitwise(sampler: Any) -> None:
    # `(n_nodes,)` is the `(n_rungs, n_nodes)` start of that row repeated:
    # the same chains, bitwise.
    row = np.random.default_rng(1).integers(0, 3, size=16)
    runs = [
        sampler(_graph(), FIELD, START, np.random.default_rng(2), 5, start=start)
        for start in (row, np.tile(row, (len(START), 1)))
    ]
    assert np.array_equal(runs[0].best, runs[1].best)
    assert runs[0].energy == runs[1].energy
    assert np.array_equal(runs[0].swap_acceptance, runs[1].swap_acceptance)


@pytest.mark.analytic
@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
def test_a_start_of_neither_shape_is_refused(sampler: Any) -> None:
    # A one-dimensional start of the wrong length is `check_labelling`'s refusal.
    for shape in ((len(START) - 1, 16), (15,), (len(START), 16, 1)):
        with pytest.raises(ValueError, match=r"start (is one labelling|must hold one)"):
            sampler(
                _graph(),
                FIELD,
                START,
                np.random.default_rng(0),
                2,
                start=np.zeros(shape, dtype=np.int64),
            )


@pytest.mark.analytic
def test_a_per_rung_start_maps_onto_the_tuned_rungs_by_nearest_temperature() -> None:
    # Starting rungs (0.5, 2.0); tuned rungs 0.5, 1.0, 1.25, 1.5, 2.0. 1.0 is
    # nearer 0.5, 1.5 nearer 2.0, and 1.25 is equidistant and goes to the
    # colder; the endpoints take their own rows.
    tuning = LadderTuning(Budget(Cost.SWEEPS, 100), 5, (0.5, 2.0), n_sweeps=20)
    rows = np.array([[0] * 4, [1] * 4])
    states = chains._rung_starts(rows, (0.5, 1.0, 1.25, 1.5, 2.0), tuning, [], 4, 2)
    assert states[:, 0].tolist() == [0, 0, 0, 1, 1]
    assert states.dtype == np.int64 and states.flags.c_contiguous


@pytest.mark.analytic
def test_auto_takes_a_start_per_starting_rung() -> None:
    # Under "auto" the per-rung start is one row per rung of the starting
    # ladder, whatever the tuned ladder grows to; one row per tuned rung is
    # not that shape unless the two lengths agree.
    rows = np.tile(np.arange(16) % 3, (len(START), 1))
    run = cluster_tempering(
        _graph(),
        FIELD,
        "auto",
        np.random.default_rng(0),
        5,
        start=rows,
        ladder_tuning=_tuning(),
    )
    assert run.tuned_ladder is not None
    assert run.temperatures[0] == START[0]
    with pytest.raises(ValueError, match="one per rung of the ladder given"):
        cluster_tempering(
            _graph(),
            FIELD,
            "auto",
            np.random.default_rng(0),
            5,
            start=rows[:-1],
            ladder_tuning=_tuning(),
        )
