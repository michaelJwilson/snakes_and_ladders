"""Racing, common random numbers and the polished rank in ``tune_schedule``, held to hand-run pilots (issue #1337).

The fixture is #1317's: a 3x3 open lattice at ``q = 3`` with a seeded
per-site field and a four-schedule grid. Every comparison is bitwise; no
tolerance is declared because none is used. Racing's regret against the
full-budget brute force is measured, not bounded.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove, Recolour, anneal_potts
from sal.sample.schedule import Polish, ScheduleParams, ScheduleShape
from sal.sample.tune import Criterion, ScheduleTuning, tune_schedule
from sal.search.icm import iterated_conditional_modes
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy

SEED = 1317
FIELD = np.random.default_rng(SEED).normal(0.0, 0.5, (9, 3))
GRID = (
    ScheduleParams(ScheduleShape.EXPONENTIAL, 2.0, 0.05),
    ScheduleParams(ScheduleShape.EXPONENTIAL, 0.8, 0.3),
    ScheduleParams(ScheduleShape.LINEAR, 2.0, 0.05),
    ScheduleParams(ScheduleShape.LINEAR, 0.8, 0.3, 0.25),
)
#: 4 candidates x 9 sites x 20 sweeps: racing runs 5, 10 then 20 sweeps.
BUDGET = Budget(Cost.SITE_VISITS, 4 * 9 * 20)
MOVE = PottsMove.SINGLE_SITE
RECOLOUR = Recolour.UNIFORM


def _graph() -> PottsGraph:
    return lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8)


def _run(
    params: ScheduleParams, sweeps: int, rng: np.random.Generator
) -> tuple[float, int]:
    run = anneal_potts(
        _graph(), FIELD, params.build(sweeps), rng, move=MOVE, recolour=RECOLOUR
    )
    return run.energy, run.spent


def _anneal(params: ScheduleParams, sweeps: int, rng: np.random.Generator) -> float:
    return _run(params, sweeps, rng)[0]


def _tune(seed: int, **options: object) -> object:
    return tune_schedule(
        _graph(),
        FIELD,
        move=MOVE,
        recolour=RECOLOUR,
        budget=BUDGET,
        criterion=Criterion.LOWEST_ENERGY,
        rng=np.random.default_rng(seed),
        grid=GRID,
        **options,  # type: ignore[arg-type]
    )


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_racing_is_successive_halving_run_by_hand_bitwise(seed: int) -> None:
    """Each round's pilots on the streams ``rng`` spawns in order, the better half kept, ties to grid index."""
    rng = np.random.default_rng(seed)
    alive, spent_rounds, spent = list(range(len(GRID))), [], 0
    for sweeps in (5, 10, 20):
        children = rng.spawn(len(alive))
        runs = {
            k: _run(GRID[k], sweeps, c) for k, c in zip(alive, children, strict=True)
        }
        scores = {k: run[0] for k, run in runs.items()}
        spent += sum(run[1] for run in runs.values())
        spent_rounds.append((len(alive), sweeps))
        ranked = sorted(alive, key=lambda k: (scores[k], k))
        winner = ranked[0]
        alive = sorted(ranked[: max(1, len(alive) // 2)])
        if len(ranked) == 1:
            break
    tuned = _tune(seed, racing=True)
    assert tuned.params == GRID[winner]  # type: ignore[attr-defined]
    assert tuned.rounds == tuple(spent_rounds)  # type: ignore[attr-defined]
    assert tuned.spent == spent  # type: ignore[attr-defined]


@pytest.mark.oracle
def test_racing_returns_the_brute_force_argmin_where_the_winner_leads_every_round() -> (
    None
):
    """On paired streams, a grid whose brute-force winner leads at 5, 10 and 20 sweeps is raced to it."""
    full = _tune(7, common=True)
    raced = _tune(7, racing=True, common=True)
    shared = np.random.default_rng(7)
    leads = []
    for sweeps in (5, 10, 20):
        (stream,) = shared.spawn(1)
        scores = [_anneal(p, sweeps, copy.deepcopy(stream)) for p in GRID]
        leads.append(int(np.argmin(scores)))
    best = min(range(len(GRID)), key=lambda k: (full.candidates[k].lowest_energy, k))  # type: ignore[attr-defined]
    if all(lead == best for lead in leads):
        assert raced.params == full.params  # type: ignore[attr-defined]
    regret = (
        full.candidates[GRID.index(raced.params)].lowest_energy  # type: ignore[attr-defined]
        - full.candidates[best].lowest_energy  # type: ignore[attr-defined]
    )
    assert regret >= 0.0


@pytest.mark.analytic
def test_common_numbers_give_every_candidate_identical_starts() -> None:
    """Under ``common=True`` a one-step pilot at one temperature is the same draw for every candidate."""
    flat = tuple(ScheduleParams(ScheduleShape.LINEAR, 1.0, 1.0) for _ in range(4))
    tuned = tune_schedule(
        _graph(),
        FIELD,
        move=MOVE,
        recolour=RECOLOUR,
        budget=Budget(Cost.SITE_VISITS, 4 * 9 * 2),
        criterion=Criterion.LOWEST_ENERGY,
        rng=np.random.default_rng(3),
        grid=flat,
        common=True,
    )
    energies = {c.lowest_energy for c in tuned.candidates}
    assert len(energies) == 1
    independent = tune_schedule(
        _graph(),
        FIELD,
        move=MOVE,
        recolour=RECOLOUR,
        budget=Budget(Cost.SITE_VISITS, 4 * 9 * 2),
        criterion=Criterion.LOWEST_ENERGY,
        rng=np.random.default_rng(3),
        grid=flat,
    )
    assert len({c.lowest_energy for c in independent.candidates}) > 1


@pytest.mark.analytic
def test_common_numbers_parallel_equal_serial_bitwise() -> None:
    serial = _tune(5, racing=True, common=True)
    threaded = _tune(5, racing=True, common=True, workers=2, pool="threads")
    assert serial == threaded


def _icm(graph: PottsGraph) -> object:
    def polish(labelling: np.ndarray) -> np.ndarray:
        return iterated_conditional_modes(
            graph, FIELD, np.random.default_rng(0), start=labelling
        ).labelling

    return polish


@pytest.mark.analytic
def test_polished_gap_equals_the_polish_applied_by_hand() -> None:
    graph = _graph()
    polish = _icm(graph)
    tuned = tune_schedule(
        graph,
        FIELD,
        move=MOVE,
        recolour=RECOLOUR,
        budget=BUDGET,
        criterion=Criterion.POLISHED_GAP,
        rng=np.random.default_rng(11),
        grid=GRID,
        polish=polish,  # type: ignore[arg-type]
    )
    children = np.random.default_rng(11).spawn(len(GRID))
    by_hand = []
    for params, child in zip(GRID, children, strict=True):
        run = anneal_potts(
            graph, FIELD, params.build(20), child, move=MOVE, recolour=RECOLOUR
        )
        by_hand.append(energy(graph, FIELD, polish(run.best)))  # type: ignore[operator]
    assert [c.polished_energy for c in tuned.candidates] == by_hand
    assert tuned.params == GRID[min(range(len(GRID)), key=lambda k: (by_hand[k], k))]


@pytest.mark.patch
@pytest.mark.analytic
def test_polished_gap_without_polish_is_refused_before_any_work() -> None:
    with pytest.raises(ValueError, match="polish"):
        ScheduleTuning(BUDGET, Criterion.POLISHED_GAP, 10, GRID)
    with pytest.raises(ValueError, match="polish"):
        ScheduleTuning(BUDGET, Criterion.LOWEST_ENERGY, 10, GRID, polish=lambda x: x)


@pytest.mark.patch
@pytest.mark.oracle
def test_every_option_off_is_the_run_before_it_bitwise() -> None:
    assert _tune(9) == _tune(9, racing=False, common=False, polish=None)


@pytest.mark.oracle
@pytest.mark.parametrize("member", list(Polish))
def test_a_polish_member_pilot_is_the_polished_anneal_bitwise(member: Polish) -> None:
    """Each pilot's lowest, polished energy and spend are ``anneal_potts(..., polish=member)``'s, on the same spawned stream (#1390)."""
    graph = _graph()
    tuned = tune_schedule(
        graph,
        FIELD,
        move=MOVE,
        recolour=RECOLOUR,
        budget=BUDGET,
        criterion=Criterion.POLISHED_GAP,
        rng=np.random.default_rng(13),
        grid=GRID,
        polish=member,
    )
    children = np.random.default_rng(13).spawn(len(GRID))
    runs = [
        anneal_potts(
            graph,
            FIELD,
            params.build(20),
            child,
            move=MOVE,
            recolour=RECOLOUR,
            polish=member,
        )
        for params, child in zip(GRID, children, strict=True)
    ]
    assert [c.polished_energy for c in tuned.candidates] == [r.energy for r in runs]
    assert [c.lowest_energy for c in tuned.candidates] == [
        r.stages[0].energy for r in runs
    ]
    assert [c.spent for c in tuned.candidates] == [r.spent for r in runs]
    assert tuned.spent == sum(r.spent for r in runs)
    assert tuned.spent > sum(r.stages[0].spent for r in runs)
    scores = [r.energy for r in runs]
    assert tuned.params == GRID[min(range(len(GRID)), key=lambda k: (scores[k], k))]


@pytest.mark.oracle
def test_a_polish_member_leaves_the_schedule_stage_unchanged_bitwise() -> None:
    """The schedule's lowest energy under a ``Polish`` is the unpolished pilot's: the polish draws after it (#1390)."""
    tuned = {
        criterion: tune_schedule(
            _graph(),
            FIELD,
            move=MOVE,
            recolour=RECOLOUR,
            budget=BUDGET,
            criterion=criterion,
            rng=np.random.default_rng(17),
            grid=GRID,
            polish=polish,
        )
        for criterion, polish in (
            (Criterion.LOWEST_ENERGY, None),
            (Criterion.POLISHED_GAP, Polish.ICM_MERGE),
        )
    }
    plain, polished = tuned.values()
    assert [c.lowest_energy for c in plain.candidates] == [
        c.lowest_energy for c in polished.candidates
    ]


@pytest.mark.patch
@pytest.mark.analytic
@pytest.mark.parametrize("member", list(Polish))
def test_a_polish_member_without_polished_gap_is_refused(member: Polish) -> None:
    with pytest.raises(ValueError, match="polish"):
        ScheduleTuning(BUDGET, Criterion.LOWEST_ENERGY, 10, GRID, polish=member)
