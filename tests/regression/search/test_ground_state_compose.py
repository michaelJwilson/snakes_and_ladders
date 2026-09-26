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
import pickle

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
    METHODS,
    SWENDSEN_WANG_SCHEDULE,
    WARM_SCHEDULE,
    MethodRun,
    Rung,
    SolverChain,
    SolverRealizations,
    SolverStage,
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


@pytest.mark.oracle
def test_a_chain_read_from_text_takes_its_arguments_by_update() -> None:
    # The object route: parse the stages, update one, hand the instance to
    # `ground_state`. With the arm's schedule and reserve it is the arm; its
    # text reads back to an equal chain, and a pickled copy runs the same.
    rung = _rung()
    budget = _budget(rung)
    solver = SolverChain.parse("swendsen-wang>alpha-expansion")
    solver.stages[0].update(
        schedule=SWENDSEN_WANG_SCHEDULE, reserve_cycles=EXPANSION_RESERVE_CYCLES
    )
    copy = pickle.loads(pickle.dumps(solver))

    assert SolverChain.parse(str(solver)) == solver
    for seed in SEEDS:
        run = ground_state(
            rung.graph, rung.field, solver, budget, np.random.default_rng(seed)
        )
        _same(
            run,
            ARMS["swendsen-wang>expansion"](rung, budget, np.random.default_rng(seed)),
        )
        _same(run, copy(rung, budget, np.random.default_rng(seed)))

    solver.stages[0].update(t_start=1.0)
    assert str(solver).startswith("swendsen-wang(shape=linear,t_start=1.0,")


@pytest.mark.smoke
def test_an_update_a_stage_cannot_take_is_refused_and_leaves_it_unchanged() -> None:
    solver = SolverChain.parse("alpha-expansion>swendsen-wang(steps=4)")
    with pytest.raises(ValueError, match="runs no anneal"):
        solver.stages[0].update(t_start=1.0)
    with pytest.raises(ValueError, match="unknown arguments"):
        solver.stages[1].update(bogus=1)

    assert str(solver) == "alpha-expansion>swendsen-wang(steps=4)"


@pytest.mark.oracle
@pytest.mark.parametrize("seed", SEEDS)
def test_repeated_realizations_are_the_best_of_their_branches_by_hand(
    seed: int,
) -> None:
    # `(descent>alpha-expansion)**3`: three realizations, each on a third of
    # the budget and its own spawned generator, from the same (drawn) start;
    # the lowest energy is kept and every branch's spend is charged.
    rung = _rung()
    budget = _budget(rung)
    run = ground_state(
        rung.graph,
        rung.field,
        "(descent>alpha-expansion)**3",
        budget,
        np.random.default_rng(seed),
    )
    share = Budget(budget.unit, budget.size // 3)
    branches = [
        chain("descent", "alpha-expansion")(rung, share, stream)
        for stream in np.random.default_rng(seed).spawn(3)
    ]
    best = min(branches, key=lambda branch: branch.energy)

    assert np.array_equal(run.labelling, best.labelling)
    assert run.energy == best.energy
    assert run.spent == sum(branch.spent for branch in branches) <= budget.size


@pytest.mark.oracle
def test_alternatives_start_from_the_same_labelling() -> None:
    # `icm|alpha-expansion` from a given start: each realization starts from
    # it, so each is its solver called from that start on its share.
    rung = _rung()
    budget = _budget(rung)
    start = rung.field.argmax(axis=1).astype(np.int64)
    run = ground_state(
        rung.graph,
        rung.field,
        "icm|alpha-expansion",
        budget,
        np.random.default_rng(4),
        start=start,
    )
    share = Budget(budget.unit, budget.size // 2)
    first, second = np.random.default_rng(4).spawn(2)
    icm = METHODS.__getitem__("icm")(rung, share, first, start=start)
    cut = METHODS.__getitem__("alpha-expansion")(rung, share, second, start=start)

    best = min((icm, cut), key=lambda branch: branch.energy)

    assert np.array_equal(run.labelling, best.labelling)
    assert run.energy == best.energy
    assert run.spent == icm.spent + cut.spent


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("descent>alpha-expansion**2", "descent>(alpha-expansion**2)"),
        ("(descent>alpha-expansion)**3", "(descent>alpha-expansion)**3"),
        ("icm|anneal(steps=5)>alpha-expansion", "icm|anneal(steps=5)>alpha-expansion"),
        ("swendsen-wang**1>alpha-expansion", "swendsen-wang>alpha-expansion"),
        (
            "field_argmax>(descent>alpha-expansion|swendsen-wang**2)>alpha-beta-swap",
            "field_argmax>(descent>alpha-expansion|swendsen-wang**2)>alpha-beta-swap",
        ),
    ],
)
def test_repetition_binds_tighter_than_a_chain_and_a_chain_than_alternatives(
    text: str, shown: str
) -> None:
    solver = SolverChain.parse(text)

    assert str(solver) == shown
    assert SolverChain.parse(str(solver)) == solver


@pytest.mark.smoke
def test_repeated_copies_update_on_their_own() -> None:
    solver = SolverChain.parse("swendsen-wang**2")
    realizations = solver.stages[0]
    assert isinstance(realizations, SolverRealizations)
    first = realizations.branches[0]
    assert isinstance(first, SolverStage)
    first.update(steps=3)

    assert str(solver) == "swendsen-wang(steps=3)|swendsen-wang"


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("swendsen-wang**", "a count after"),
        ("swendsen-wang**0", "at least one"),
        ("(descent>alpha-expansion", "closing parenthesis"),
        ("descent>>alpha-expansion", "a stage name"),
        ("descent alpha-expansion", "the end"),
    ],
)
def test_text_that_does_not_read_is_refused_where_it_stops(
    text: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        SolverChain.parse(text)


@pytest.mark.smoke
def test_an_update_on_realizations_reaches_every_branch_or_none() -> None:
    solver = SolverChain.parse("swendsen-wang**2>alpha-expansion")
    solver.stages[0].update(steps=3)

    assert str(solver) == "(swendsen-wang(steps=3)**2)>alpha-expansion"
    with pytest.raises(ValueError, match="runs no anneal"):
        SolverChain.parse("(swendsen-wang|alpha-expansion)>icm").stages[0].update(
            steps=3
        )
