"""``temperatures="auto"`` with a ``LadderTuning`` on the Potts tempering entry points (issue #1337).

The fixture is a 4x4 open lattice at ``q = 3``, coupling ``1.0``, with a
seeded per-site field. The referee is the band itself: a tuned ladder's
last measured exchange acceptances lie in ``band``, or ``within_band`` is
``False`` and the tuning's ``Termination`` says it stopped on its budget.
A given ladder is held bitwise to the run before the option existed.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Stop
from sal.sample.loop import swap_log_ratio
from sal.sample.potts_mcmc import chains, cluster_tempering
from sal.sample.schedule import adapt_ladder
from sal.sample.tune import LadderTuning, TunedLadder
from sal.sandbox.potts_tempering import parallel_tempering, run_tempering
from sal.search.ground_state import Problem
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
    assert states.dtype == np.int64
    assert states.flags.c_contiguous


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


@pytest.mark.analytic
def test_the_exchange_ratio_does_not_read_the_ladder_order() -> None:
    # Reversing the ladder relabels each pair (i, i + 1) as (i + 1, i); the
    # Metropolis exchange ratio is the same number under that relabelling, so
    # the order changes which pair proposes first, not the law (#1343).
    rng = np.random.default_rng(0)
    for _ in range(100):
        beta_i, beta_j, energy_i, energy_j = rng.normal(size=4)
        assert swap_log_ratio(beta_i, beta_j, energy_i, energy_j) == swap_log_ratio(
            beta_j, beta_i, energy_j, energy_i
        )


# --- the pilot's limits and its cost (#1402) ---------------------------------


@pytest.mark.bug
@pytest.mark.analytic
def test_adapt_ladder_misses_a_band_one_geometric_ladder_reaches() -> None:
    # Noiseless exchange acceptance exp(-|log(T_j / T_i)|): the geometric
    # four-rung ladder over (0.05, 2.0) has every pair at 40^(-1/3) = 0.292,
    # inside (0.2, 0.3). From the two endpoints a pair moves only beside one
    # in the band, so the ladder bisects to 3 and 5 rungs and removes back,
    # and ends out of band on its budget. Fails when the revision reaches it.
    def measure(rungs: tuple[float, ...]) -> list[float]:
        seen.append(len(rungs))
        return [math.exp(-abs(math.log(b / a))) for a, b in itertools.pairwise(rungs)]

    seen: list[int] = []
    geometric = tuple(np.geomspace(0.05, 2.0, 4))
    assert all(BAND[0] <= value <= BAND[1] for value in measure(geometric))
    seen.clear()
    adapted = adapt_ladder(measure, (0.05, 2.0), BAND, 50, 200)
    assert not adapted.within_band
    assert adapted.rounds == 50
    assert set(seen) == {2, 3, 5}


@pytest.mark.analytic
def test_cluster_tempering_reports_the_pilot_apart_from_the_run() -> None:
    # The pilot's visits are tuned_ladder.spent and the run's are spent, as
    # anneal_potts reports its schedule pilot's. Each step charges one sweep,
    # n_nodes + 2 n_edges, per replica and per Houdayer pair; the pilot runs
    # one Houdayer pair on every ladder it measures.
    graph = _graph()
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    tuning = _tuning()
    run = cluster_tempering(
        graph, FIELD, "auto", np.random.default_rng(0), 50, ladder_tuning=tuning
    )
    assert run.tuned_ladder is not None
    adapted = run.tuned_ladder.adapted
    assert run.spent == per_sweep * 50 * (len(run.temperatures) + 1)
    assert run.tuned_ladder.spent == per_sweep * tuning.n_sweeps * (
        adapted.replicas_measured + adapted.rounds
    )
