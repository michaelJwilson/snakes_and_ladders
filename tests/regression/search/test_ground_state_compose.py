"""Chains of ground-state solvers named ``a>b>...``, against the runs they are made of (issue #1077).

A chain spelled in `ground_state`'s grammar is its parts run by hand: the
first on the budget, each next from the one before's labelling on what was
left, one generator throughout. So each chain is pinned bitwise to that
hand-run loop, and the hybrid arm the grammar can spell is pinned to the
arm. The arms themselves are pinned to the functions they replaced by
`test_ground_state_start.py`'s pre-#1052 hashes.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.schedule import ScheduleParams
from sal.search.ground_state import (
    ANNEAL_SCHEDULE,
    ARMS,
    EXPANSION_RESERVE_CYCLES,
    EXPANSION_SW_SCHEDULE,
    SWENDSEN_WANG_SCHEDULE,
    WARM_SCHEDULE,
    MethodRun,
    Rung,
    chain,
    ground_state,
    part,
    run_alpha_expansion,
    run_descent,
    run_field_argmax,
    run_swendsen_wang,
)
from sal.search.potts_starts import spatio_rung
from sal.sim.fixtures import fixture
from sal.sim.potts import energy

SEEDS = (0, 1, 2, 3, 4)


@functools.cache
def _rung() -> Rung:
    return spatio_rung(fixture("spatio_only", "ci").params, "ci")


def _budget(rung: Rung) -> Budget:
    return Budget(Cost.SITE_VISITS, 50 * rung.visits_per_sweep)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", SEEDS)
def test_a_spelled_chain_is_its_parts_run_by_hand(seed: int) -> None:
    # The field's argmax, ICM to its first clean sweep, then the expansion.
    rung = _rung()
    budget = _budget(rung)
    chain = ground_state(
        rung.graph,
        rung.field,
        "field_argmax>descent>alpha-expansion",
        budget,
        np.random.default_rng(seed),
    )
    rng = np.random.default_rng(seed)
    argmax = run_field_argmax(rung, budget, rng)
    left = budget.size - argmax.spent
    descent = run_descent(rung, Budget(budget.unit, left), rng, start=argmax.labelling)
    left -= descent.spent
    expansion = run_alpha_expansion(
        rung, Budget(budget.unit, left), rng, start=descent.labelling
    )

    assert np.array_equal(chain.labelling, expansion.labelling)
    assert chain.energy == expansion.energy
    assert chain.spent == argmax.spent + descent.spent + expansion.spent
    assert chain.energy == energy(rung.graph, rung.field, chain.labelling)
    assert chain.energy <= descent.energy <= argmax.energy


@pytest.mark.oracle
@pytest.mark.parametrize("seed", SEEDS)
def test_the_spelled_expansion_then_anneal_is_the_arm(seed: int) -> None:
    # `alpha-expansion>swendsen-wang` on the arm's schedule is the arm: the
    # grammar and the table build one chain.
    rung = _rung()
    budget = _budget(rung)
    spelled = ground_state(
        rung.graph,
        rung.field,
        "alpha-expansion>swendsen-wang",
        budget,
        np.random.default_rng(seed),
        schedule=EXPANSION_SW_SCHEDULE,
    )
    arm = ARMS["expansion>swendsen-wang"](rung, budget, np.random.default_rng(seed))

    assert np.array_equal(spelled.labelling, arm.labelling)
    assert (spelled.energy, spelled.spent) == (arm.energy, arm.spent)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("method", "keywords", "match"),
    [
        ("field_argmax>nothing", {}, "no ground-state part"),
        ("descent>field_argmax", {}, "field_argmax"),
        ("icm>alpha-expansion", {}, "step 1 is left 0"),
        ("field_argmax>descent", {"steps": 3}, "runs no anneal"),
        ("field_argmax>alpha-expansion", {"min_sites": 2}, "no single-site descent"),
    ],
)
def test_a_chain_that_cannot_run_is_refused_by_name(
    method: str, keywords: dict[str, int], match: str
) -> None:
    # An unknown part; a part that refuses a start placed after another;
    # ICM, which charges its whole budget, before another part; an option
    # no part takes.
    rung = _rung()
    with pytest.raises(ValueError, match=match):
        ground_state(
            rung.graph,
            rung.field,
            method,
            _budget(rung),
            np.random.default_rng(0),
            **keywords,  # type: ignore[arg-type]
        )


@pytest.mark.oracle
def test_a_keyword_at_call_time_replaces_the_one_a_part_bound() -> None:
    # The factory binds a schedule into the arm; `ground_state(schedule=)`
    # replaces it, so the arm on another schedule is the chain built on it.
    rung = _rung()
    budget = _budget(rung)
    bound = chain("alpha-expansion", part("swendsen-wang", schedule=ANNEAL_SCHEDULE))
    replaced = ground_state(
        rung.graph,
        rung.field,
        "expansion>swendsen-wang",
        budget,
        np.random.default_rng(7),
        schedule=ANNEAL_SCHEDULE,
    )
    built = bound(rung, budget, np.random.default_rng(7))

    assert np.array_equal(replaced.labelling, built.labelling)
    assert (replaced.energy, replaced.spent) == (built.energy, built.spent)


def _text(schedule: ScheduleParams) -> str:
    """A schedule as :func:`~sal.search.ground_state.rendered` reads it."""
    return (
        f"shape={schedule.shape.value},t_start={schedule.t_start!r},"
        f"t_end={schedule.t_end!r},hold={schedule.hold!r}"
    )


def _same(first: MethodRun, second: MethodRun) -> None:
    assert np.array_equal(first.labelling, second.labelling)
    assert (first.energy, first.spent) == (second.energy, second.spent)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("arm", "text"),
    [
        (
            "swendsen-wang>expansion",
            f"swendsen-wang({_text(SWENDSEN_WANG_SCHEDULE)},"
            f"reserve_cycles={EXPANSION_RESERVE_CYCLES})>alpha-expansion",
        ),
        (
            "expansion>swendsen-wang",
            f"alpha-expansion>swendsen-wang({_text(EXPANSION_SW_SCHEDULE)})",
        ),
        ("warm-anneal", f"descent>swendsen-wang({_text(WARM_SCHEDULE)})"),
    ],
)
def test_an_arm_written_as_text_is_the_arm(arm: str, text: str) -> None:
    # The table's chains are names for text the factory reads on the fly:
    # each spelled with its schedule and reserve is the entry, bitwise.
    rung = _rung()
    budget = _budget(rung)
    for seed in SEEDS:
        rendered = ground_state(
            rung.graph, rung.field, text, budget, np.random.default_rng(seed)
        )
        _same(rendered, ARMS[arm](rung, budget, np.random.default_rng(seed)))


@pytest.mark.oracle
def test_a_part_with_arguments_and_a_chain_in_no_table_run_on_the_fly() -> None:
    # One part with a step count is the method called with it; a chain no
    # table names, built in code or read from text, is charged within the
    # budget and returns a labelling whose energy it reports.
    rung = _rung()
    budget = _budget(rung)
    one = ground_state(
        rung.graph,
        rung.field,
        "swendsen-wang(steps=7)",
        budget,
        np.random.default_rng(2),
    )
    _same(one, run_swendsen_wang(rung, budget, np.random.default_rng(2), steps=7))

    text = "field_argmax>descent>wolff(t_start=2.0,steps=30)>alpha-beta-swap"
    built = chain(
        "field_argmax",
        "descent",
        part(
            "wolff",
            schedule=ScheduleParams(ANNEAL_SCHEDULE.shape, 2.0, ANNEAL_SCHEDULE.t_end),
            steps=30,
        ),
        "alpha-beta-swap",
    )
    read = ground_state(rung.graph, rung.field, text, budget, np.random.default_rng(3))
    handed = ground_state(
        rung.graph, rung.field, built, budget, np.random.default_rng(3)
    )

    _same(read, handed)
    assert read.spent <= budget.size
    assert read.energy == energy(rung.graph, rung.field, read.labelling)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("alpha-expansion(steps=3)", "runs no anneal"),
        ("icm(t_start=1.0)>alpha-expansion", "runs no anneal"),
        ("alpha-expansion(min_sites=2)", "no single-site descent"),
        ("warm-anneal(steps=3)>alpha-expansion", "an arm is a chain already"),
        ("swendsen-wang(bogus=1)", "unknown arguments"),
        ("swendsen-wang(steps)", "key=value"),
        ("swendsen-wang(reserve_cycles=1,reserve_sweeps=1)>icm", "one reserve"),
        ("nothing(steps=3)", "no ground-state part"),
    ],
)
def test_an_argument_a_part_cannot_take_is_refused(text: str, match: str) -> None:
    rung = _rung()
    with pytest.raises(ValueError, match=match):
        ground_state(
            rung.graph, rung.field, text, _budget(rung), np.random.default_rng(0)
        )
