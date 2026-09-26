"""The annealed entries take a schedule, a step count and a start, and their defaults are today's runs bitwise (issue #1038).

The referee for the defaults is the call they replace, spelled out: the
exponential schedule from ``ANNEAL_START`` to ``ANNEAL_END`` at the budget's
step count, passed to ``anneal_potts`` from a uniform draw. The seeds are the
notebook's reported ones, 0 to 4, on its instance at a shortened budget. A warm
chain's descent is ICM's own run on the same generator, and it is charged
the sweeps it ran.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove, anneal_potts
from sal.sample.schedule import (
    ExponentialTempSchedule,
    ScheduleParams,
    ScheduleShape,
)
from sal.search.ground_state import (
    ANNEAL_END,
    ANNEAL_OPTIONS,
    ANNEAL_SCHEDULE,
    ANNEAL_START,
    METHODS,
    Rung,
    chain,
    descend,
    part,
    run_annealed,
    run_icm,
)
from sal.search.potts_starts import (
    LabellingEnergy,
    ScheduleStart,
    SolverStart,
    spatio_rung,
)
from sal.sim.fixtures import fixture

#: The notebook's reported seeds.
SEEDS = (0, 1, 2, 3, 4)
#: Sweeps' worth of site visits: the notebook's 1,000 cut to keep each run
#: under a tenth of a second.
SWEEPS = 40
#: The annealed entries and their move sets.
ENTRIES = {
    "anneal": PottsMove.SINGLE_SITE,
    "swendsen-wang": PottsMove.SWENDSEN_WANG,
    "wolff": PottsMove.WOLFF,
}


@functools.cache
def _rung() -> Rung:
    return spatio_rung(fixture("spatio_only", "release").params, "release-q10")


def _budget(rung: Rung) -> Budget:
    return Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize("name", list(ENTRIES))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_default_schedule_is_the_run_it_replaces_bitwise(
    name: str, seed: int
) -> None:
    rung = _rung()
    budget = _budget(rung)
    move = ENTRIES[name]

    entry = METHODS[name](rung, budget, np.random.default_rng([seed, 0]))
    replaced = anneal_potts(
        rung.graph,
        rung.field,
        ExponentialTempSchedule(ANNEAL_START, ANNEAL_END, SWEEPS),
        np.random.default_rng([seed, 0]),
        move=move,
        cluster_backend=Backend.RUST
        if move is PottsMove.SWENDSEN_WANG
        else Backend.PYTHON,
    )

    assert np.array_equal(entry.labelling, replaced.labelling)
    assert entry.energy == replaced.energy
    assert entry.spent == replaced.site_visits


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_a_start_given_is_the_start_drawn_bitwise(move: PottsMove) -> None:
    rung = _rung()
    schedule = ANNEAL_SCHEDULE.build(SWEEPS)
    drawn = anneal_potts(
        rung.graph, rung.field, schedule, np.random.default_rng(7), move=move
    )
    rng = np.random.default_rng(7)
    # The draw `anneal_potts` makes, made here and handed in.
    initial = rng.integers(0, rung.n_states, size=rung.n_nodes)
    given = anneal_potts(
        rung.graph, rung.field, schedule, rng, move=move, start=initial
    )

    assert np.array_equal(drawn.labelling, given.labelling)
    assert drawn.energy == given.energy
    assert drawn.site_visits == given.site_visits


@pytest.mark.smoke
def test_a_start_of_the_wrong_shape_or_range_is_refused() -> None:
    rung = _rung()
    schedule = ANNEAL_SCHEDULE.build(2)
    for initial in (
        np.zeros(rung.n_nodes - 1, dtype=np.int64),
        np.full(rung.n_nodes, rung.n_states, dtype=np.int64),
    ):
        with pytest.raises(ValueError, match="one integer state"):
            anneal_potts(
                rung.graph,
                rung.field,
                schedule,
                np.random.default_rng(0),
                start=initial,
            )


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize("seed", SEEDS)
def test_the_warm_chains_descent_is_icms_run_on_the_same_generator(seed: int) -> None:
    rung = _rung()
    budget = _budget(rung)

    labelling, sweeps = descend(
        rung, np.random.default_rng([seed, 0]), budget.size // rung.visits_per_sweep
    )
    icm = run_icm(rung, budget, np.random.default_rng([seed, 0]))

    assert np.array_equal(labelling, icm.labelling)
    assert 1 <= sweeps < SWEEPS


@pytest.mark.smoke
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_a_warm_chain_is_charged_its_descent_and_its_anneal(move: PottsMove) -> None:
    rung = _rung()
    budget = _budget(rung)
    schedule = ScheduleParams(ScheduleShape.EXPONENTIAL, 0.5, ANNEAL_END)

    _, sweeps = descend(
        rung, np.random.default_rng([0, 0]), budget.size // rung.visits_per_sweep
    )
    warm = chain(
        "descent",
        part(run_annealed, takes=ANNEAL_OPTIONS, move=move, schedule=schedule),
    )(rung, budget, np.random.default_rng([0, 0]))

    descent = sweeps * rung.visits_per_sweep
    remaining = Budget(budget.unit, budget.size - descent)
    rng = np.random.default_rng([0, 0])
    labelling, _ = descend(rung, rng, budget.size // rung.visits_per_sweep)
    anneal = run_annealed(
        rung, remaining, rng, move, schedule=schedule, start=labelling
    )
    assert warm.spent == descent + anneal.spent
    assert warm.energy == anneal.energy
    assert np.array_equal(warm.labelling, anneal.labelling)
    assert warm.spent <= budget.size


@pytest.mark.smoke
def test_a_fixed_step_count_replaces_the_budgets() -> None:
    rung = _rung()

    run = run_annealed(
        rung,
        _budget(rung),
        np.random.default_rng(0),
        PottsMove.WOLFF,
        steps=3 * SWEEPS,
    )

    assert len(run.trace) == 3 * SWEEPS


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize("name", ["swendsen-wang", "wolff"])
def test_a_schedule_start_on_the_default_is_the_solver_start_bitwise(
    name: str,
) -> None:
    rung = _rung()
    objective = LabellingEnergy(rung)
    budget = _budget(rung)

    solver = SolverStart(name, budget, np.random.default_rng([3, 0]))
    scheduled = ScheduleStart(
        ENTRIES[name],
        budget,
        ANNEAL_SCHEDULE,
        None,
        False,
        np.random.default_rng([3, 0]),
    )

    assert np.array_equal(
        solver.starts(objective)[0].numpy(), scheduled.starts(objective)[0].numpy()
    )
