"""A tuned annealing schedule on every annealed Potts entry point, held to brute force (issue #1317, stage 2).

The fixture is a 3x3 open lattice at ``q = 3`` with a per-site field from a
declared seed. :func:`~sal.sample.tune.tune_schedule` is refereed by the
brute-force loop it replaces: every candidate of a four-schedule grid
annealed by :func:`~sal.sample.potts_mcmc.anneal_potts` on the generator
:func:`sal.parallel.map_tasks` spawns for it, and the argmin taken by hand.
Every comparison is bitwise; no tolerance is declared because none is used.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample import annealed
from sal.sample.potts_keyed import SwendsenWangMove, cluster_moves
from sal.sample.potts_mcmc import (
    MoveKind,
    PottsMove,
    Recolour,
    anneal_potts,
    swendsen_wang_heat_bath_sweep,
)
from sal.sample.schedule import (
    InverseTemperatures,
    ScheduleParams,
    ScheduleShape,
    temperatures,
)
from sal.sample.tune import (
    AUTO,
    Criterion,
    ScheduleTuning,
    tune_schedule,
)
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

SHAPE = (3, 3)
N_STATES = 3
COUPLING = 0.8
SEED = 1317
FIELD = np.random.default_rng(SEED).normal(0.0, 0.5, (9, N_STATES))
#: Four candidates, two shapes by two starts; small so the brute force is cheap.
GRID = (
    ScheduleParams(ScheduleShape.EXPONENTIAL, 2.0, 0.05),
    ScheduleParams(ScheduleShape.EXPONENTIAL, 0.8, 0.3),
    ScheduleParams(ScheduleShape.LINEAR, 2.0, 0.05),
    ScheduleParams(ScheduleShape.LINEAR, 0.8, 0.3, 0.25),
)
#: 4 candidates x 9 sites x 20 sweeps.
BUDGET = Budget(Cost.SITE_VISITS, 4 * 9 * 20)
N_STEPS = 30


def _graph() -> PottsGraph:
    return lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)


def _tuning() -> ScheduleTuning:
    return ScheduleTuning(BUDGET, Criterion.LOWEST_ENERGY, N_STEPS, GRID)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("move", "recolour"),
    [
        (PottsMove.SINGLE_SITE, Recolour.UNIFORM),
        ((PottsMove.WOLFF, PottsMove.SINGLE_SITE), Recolour.HEAT_BATH),
    ],
    ids=["single-site", "wolff-heat-bath+gibbs"],
)
def test_tune_schedule_returns_the_brute_force_argmin_bitwise(
    move: PottsMove | tuple[PottsMove, ...], recolour: Recolour
) -> None:
    tuned = tune_schedule(
        _graph(),
        FIELD,
        move=move,
        recolour=recolour,
        budget=BUDGET,
        criterion=Criterion.LOWEST_ENERGY,
        rng=np.random.default_rng(SEED),
        grid=GRID,
    )
    # The referee: each candidate annealed by hand on the child map_tasks spawns.
    children = np.random.default_rng(SEED).spawn(len(GRID))
    energies = [
        anneal_potts(
            _graph(), FIELD, params.build(20), child, move=move, recolour=recolour
        ).energy
        for params, child in zip(GRID, children, strict=True)
    ]
    assert tuned.sweeps == 20
    assert [c.lowest_energy for c in tuned.candidates] == energies
    assert tuned.params == GRID[int(np.argmin(energies))]


@pytest.mark.oracle
def test_tune_schedule_on_threads_is_its_serial_run_bitwise() -> None:
    def run(workers: int, pool: str) -> object:
        return tune_schedule(
            _graph(),
            FIELD,
            move=PottsMove.SINGLE_SITE,
            recolour=Recolour.UNIFORM,
            budget=BUDGET,
            criterion=Criterion.LOWEST_ENERGY,
            rng=np.random.default_rng(SEED),
            grid=GRID,
            workers=workers,
            pool=pool,  # type: ignore[arg-type]
        )

    assert run(1, "serial") == run(4, "threads")


@pytest.mark.oracle
def test_auto_is_the_tuned_schedule_run_explicitly_bitwise() -> None:
    graph = _graph()
    auto = anneal_potts(
        graph, FIELD, AUTO, np.random.default_rng(SEED), tuning=_tuning()
    )
    again = anneal_potts(
        graph, FIELD, AUTO, np.random.default_rng(SEED), tuning=_tuning()
    )
    assert auto.tuned is not None
    np.testing.assert_array_equal(auto.best, again.best)
    assert auto.energy == again.energy
    # The referee: the chosen schedule, given, on a generator whose children
    # the pilots took.
    rng = np.random.default_rng(SEED)
    rng.spawn(len(GRID))
    given = anneal_potts(graph, FIELD, auto.tuned.params.build(N_STEPS), rng)
    np.testing.assert_array_equal(auto.best, given.best)
    np.testing.assert_array_equal(auto.final, given.final)
    assert auto.energy == given.energy


@pytest.mark.oracle
def test_auto_ais_is_the_tuned_ladder_from_zero_bitwise() -> None:
    graph = _graph()
    auto = annealed.annealed_importance_sampling(
        graph, FIELD, AUTO, np.random.default_rng(SEED), 4, tuning=_tuning()
    )
    rng = np.random.default_rng(SEED)
    tuned = tune_schedule(
        graph,
        FIELD,
        move=PottsMove.SINGLE_SITE,
        recolour=Recolour.UNIFORM,
        budget=BUDGET,
        criterion=Criterion.LOWEST_ENERGY,
        rng=rng,
        grid=GRID,
    )
    ladder = InverseTemperatures(
        (0.0, *(1.0 / t for t in temperatures(tuned.params.build(N_STEPS))))
    )
    given = annealed.annealed_importance_sampling(graph, FIELD, ladder, rng, 4)
    assert auto.log_partition == given.log_partition
    np.testing.assert_array_equal(auto.log_weights, given.log_weights)


ENTRY = [
    lambda rng, s, t: anneal_potts(_graph(), FIELD, s, rng, tuning=t),
    lambda rng, s, t: annealed.annealed_importance_sampling(
        _graph(), FIELD, s, rng, 4, tuning=t
    ),
    lambda rng, s, t: annealed.population_annealing(
        _graph(), FIELD, s, rng, 4, tuning=t
    ),
    lambda rng, s, t: annealed.simulated_tempering(
        _graph(), FIELD, s, np.zeros(2), rng, 4, tuning=t
    ),
]


@pytest.mark.analytic
@pytest.mark.parametrize("entry", ENTRY, ids=["anneal", "ais", "pa", "st"])
def test_auto_without_a_tuning_is_refused_before_any_draw(entry: object) -> None:
    rng = np.random.default_rng(SEED)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="needs a ScheduleTuning"):
        entry(rng, AUTO, None)  # type: ignore[operator]
    assert rng.bit_generator.state == before


@pytest.mark.analytic
@pytest.mark.parametrize("entry", ENTRY, ids=["anneal", "ais", "pa", "st"])
def test_a_tuning_beside_a_given_schedule_is_refused(entry: object) -> None:
    given = GRID[0].build(N_STEPS)
    with pytest.raises(ValueError, match="a tuning chooses"):
        entry(np.random.default_rng(SEED), given, _tuning())  # type: ignore[operator]


@pytest.mark.analytic
def test_schedule_tuning_refuses_another_unit_and_criterion() -> None:
    with pytest.raises(ValueError, match="site visits"):
        ScheduleTuning(Budget(Cost.SWEEPS, 10), Criterion.LOWEST_ENERGY, 5)
    with pytest.raises(ValueError, match="LOWEST_ENERGY"):
        ScheduleTuning(BUDGET, Criterion.ESJD, 5)


@pytest.mark.oracle
def test_cluster_moves_default_is_the_three_moves_as_before() -> None:
    moves = cluster_moves(_graph(), FIELD)
    assert list(moves) == [MoveKind.WOLFF, MoveKind.SWENDSEN_WANG, MoveKind.NIEDERMAYER]


@pytest.mark.oracle
def test_cluster_moves_heat_bath_swendsen_wang_is_the_heat_bath_sweep_bitwise() -> None:
    graph = _graph()
    moves = cluster_moves(
        graph, FIELD, move=PottsMove.SWENDSEN_WANG, recolour=Recolour.HEAT_BATH
    )
    assert list(moves) == [MoveKind.SWENDSEN_WANG]
    sw = moves[MoveKind.SWENDSEN_WANG]
    assert isinstance(sw, SwendsenWangMove)
    state = np.random.default_rng(SEED).integers(0, N_STATES, 9)
    proposed, _ = sw.propose(
        state, temperature=0.7, site=0, label=0, rng=np.random.default_rng(SEED)
    )
    # The referee: the sweep called directly on the same generator.
    expected = state.astype(np.int64).copy()
    swendsen_wang_heat_bath_sweep(
        expected, graph, FIELD, np.random.default_rng(SEED), 1.0 / 0.7
    )
    np.testing.assert_array_equal(proposed, expected)


@pytest.mark.analytic
def test_cluster_moves_refuses_a_keyed_wolff_under_a_heat_bath() -> None:
    with pytest.raises(ValueError, match="names the label"):
        cluster_moves(
            _graph(), FIELD, move=PottsMove.WOLFF, recolour=Recolour.HEAT_BATH
        )
