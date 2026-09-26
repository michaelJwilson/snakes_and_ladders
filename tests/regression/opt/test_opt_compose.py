"""The chain of stages, checked on stages whose runs are arithmetic (issue #1077).

`opt.compose` is model-agnostic, so it is tested on stages that know no
model: each returns the start it was given plus one, spends a declared share
of its budget and records what it saw. What is asserted is the accounting
--- the budget each step is handed, the spend summed, the start handed on,
the options routed --- and the refusals. The Potts chains are pinned against
the hand-written arms they replaced in `test_ground_state_start.py`.
"""

from __future__ import annotations

import functools
import operator
import pickle
from dataclasses import dataclass, replace

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.compose import Step, Then, then


@dataclass(frozen=True)
class Run:
    """An arithmetic run: its answer, spend, time, and what the stage was handed."""

    x: int
    spent: int
    seconds: float
    seen: tuple[object, ...]


def _stage(
    problem: int,
    budget: Budget,
    rng: np.random.Generator,
    *,
    start: int | None = None,
    share: int = 2,
    scale: int = 1,
) -> Run:
    """``start + scale`` (``problem`` when none), spending ``budget // share``."""
    begun = problem if start is None else start
    draw = int(rng.integers(0, 1_000))
    return Run(begun + scale, budget.size // share, 0.0, (budget.size, begun, draw))


def _held(problem: int) -> int:
    """What the first step holds back: ``problem`` units."""
    return problem


def _chain(*steps: Step) -> Then:
    return Then(steps, operator.attrgetter("x"))


@pytest.mark.analytic
def test_each_step_gets_what_the_steps_before_it_left_less_its_reserve() -> None:
    # 100 units, a reserve of 10 on the first: it sees 90 and spends 45; the
    # second sees 55 and spends 27; the chain is charged 72.
    chain = _chain(Step(_stage, reserve=_held), Step(_stage))
    run = chain(10, Budget(Cost.SITE_VISITS, 100), np.random.default_rng(0))

    assert run.spent == 45 + 27
    assert run.seen[0] == 55
    assert run.x == 12


@pytest.mark.analytic
def test_the_answer_is_handed_on_and_the_generator_is_shared_in_order() -> None:
    # The second step starts from the first's answer, and its draw is the
    # generator's second: the loop a hand-written arm would run.
    chain = _chain(Step(_stage), Step(_stage))
    run = chain(3, Budget(Cost.SITE_VISITS, 8), np.random.default_rng(1), start=5)
    rng = np.random.default_rng(1)
    rng.integers(0, 1_000)

    assert run.seen[1] == 6
    assert run.seen[2] == int(rng.integers(0, 1_000))
    assert run.x == 7


@pytest.mark.analytic
def test_an_option_reaches_the_steps_that_take_it_and_no_other() -> None:
    chain = _chain(Step(_stage), Step(_stage, takes=frozenset({"scale"})))
    run = chain(0, Budget(Cost.SITE_VISITS, 8), np.random.default_rng(0), scale=10)

    assert run.x == 1 + 10
    assert chain.takes == frozenset({"scale"})
    with pytest.raises(ValueError, match="no step takes"):
        chain(0, Budget(Cost.SITE_VISITS, 8), np.random.default_rng(0), share=4)


@pytest.mark.analytic
def test_a_step_left_no_budget_and_an_empty_chain_are_refused() -> None:
    spend_all = functools.partial(_stage, share=1)
    chain = then([spend_all, _stage], operator.attrgetter("x"))

    with pytest.raises(ValueError, match="step 1 is left 0"):
        chain(0, Budget(Cost.SITE_VISITS, 8), np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one step"):
        then([], operator.attrgetter("x"))


@pytest.mark.analytic
def test_a_chain_of_module_level_parts_crosses_a_process_boundary() -> None:
    # `opt.budget.compare` runs its cells on a process pool.
    chain = _chain(
        Step(_stage, reserve=_held), Step(functools.partial(_stage, scale=3))
    )
    copy = pickle.loads(pickle.dumps(chain))
    budget = Budget(Cost.SITE_VISITS, 40)

    copied = copy(2, budget, np.random.default_rng(4))
    original = chain(2, budget, np.random.default_rng(4))

    assert replace(copied, seconds=0.0) == replace(original, seconds=0.0)
