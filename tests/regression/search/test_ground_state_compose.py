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
from sal.search.ground_state import (
    ARMS,
    EXPANSION_SW_SCHEDULE,
    Rung,
    ground_state,
    run_alpha_expansion,
    run_descent,
    run_field_argmax,
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
