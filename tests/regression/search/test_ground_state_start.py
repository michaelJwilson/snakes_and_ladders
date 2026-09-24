"""One entry, `search.ground_state.ground_state`, reaches every solver and arm from a start (issue #1052).

Referees: with no start, schedule or step count every `METHODS` entry and
every `ARMS` entry is the run the code before #1052 made, seed for seed, pinned
as a digest recorded before the change; a start equal to the labelling a
solver would draw or build gives its run bitwise, and a descent ends at or
below its start; a solver with no single starting labelling refuses one by
name; and at `spatio_only/ci`, nine sites, no solver or arm lands below the
enumerated minimum, the cut-based ones reach it at two states, and the
expansion's factor-2 bracket holds from every start at three.
"""

from __future__ import annotations

import functools
import hashlib
import itertools

import numpy as np
import pytest
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.sample.schedule import ScheduleParams, ScheduleShape
from snakes_and_ladders.search.ground_state import (
    ANNEALED,
    ARMS,
    METHODS,
    Rung,
    expansion_bracket,
    ground_state,
    rung_field,
)
from snakes_and_ladders.search.potts_starts import spatio_rung
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.potts import SpatioOnlyParams, energy

#: The notebook's reported seeds.
SEEDS = (0, 1, 2, 3, 4)
#: Sweeps' worth of site visits per tier: at `ci` every entry settles, and at
#: `release` 40 is the smallest count giving the expansion its three cycles.
SWEEPS = {"ci": 50, "release": 40}
#: Two exact routes sum the same weights in a different order.
EXACT = 1e-9

#: The solvers that refuse a start, and the ones whose own start is the
#: field's argmax rather than a uniform draw.
REFUSED = ("greedy", "tempering", "max-product", "bifurcation")
ARGMAX_START = ("alpha-expansion", "alpha-beta-swap", "expansion>swendsen-wang")
#: The solvers that end on a descent from the start they are handed.
DESCENTS = ("icm", "alpha-expansion", "alpha-beta-swap", "expansion>swendsen-wang")

#: Each cell's run on `SEEDS` before #1052, as sha256 over the labelling's
#: bytes, the energy's repr and the spend, first 16 hex digits. The `METHODS`
#: cells are `ground_state` at `origin/main` b6bc5d63; the `ARMS` cells are
#: the calls #1038 and #1041 made there: `run_annealed` on the tuned schedule,
#: on Wolff's matched steps, `warm_anneal` from 1.0, and the two hybrids of
#: `search.cluster_moves`.
BEFORE = {
    "ci/greedy": "dd8ab86c204a8c39",
    "ci/icm": "eba7c65d69aeb916",
    "ci/icm-random": "f61e867f262cd619",
    "ci/anneal": "b599a8a6d3e3ebbc",
    "ci/swendsen-wang": "baf7e04c393e8a9e",
    "ci/wolff": "6ae699293ac05e3a",
    "ci/tempering": "efb5e19b6fdbcea1",
    "ci/alpha-expansion": "43fe1263a3fb0417",
    "ci/alpha-beta-swap": "212124e7656e961a",
    "ci/max-product": "2fd81f289869933b",
    "ci/bifurcation": "3c108e6281f26ea7",
    "release/greedy": "d32a79fb13f61671",
    "release/icm": "0b81269cf9db1847",
    "release/icm-random": "354122e191105cb6",
    "release/anneal": "6582386d83d0024b",
    "release/swendsen-wang": "922f0038e98fe6f4",
    "release/wolff": "8628d1e1aca72bda",
    "release/tempering": "62689b14a52e1547",
    "release/alpha-expansion": "a5415d349d97d72f",
    "release/alpha-beta-swap": "7d8f0bca122dcb29",
    "release/max-product": "a6c10c7ef9f08414",
    "release/bifurcation": "43cc16107de6c84e",
    "ci/tuned-swendsen-wang": "2b9e8db7dac75d76",
    "ci/matched-wolff": "b334d6cd3b03f823",
    "ci/warm-anneal": "3ada7396e5abf18c",
    "ci/swendsen-wang>expansion": "e4b07c088fe1f16c",
    "ci/expansion>swendsen-wang": "cedea74f5b4f41a8",
    "release/tuned-swendsen-wang": "2b1a8d8cd28788e3",
    "release/warm-anneal": "0b81269cf9db1847",
    "release/expansion>swendsen-wang": "9ccfe74792ba2496",
}
#: Cells near or over the per-PR cap of 10 s: 41,250 Wolff steps a seed,
#: 12.6 s for five, and max-product's 40 flooding iterations at 5,041 sites,
#: 9 s for five.
SLOW = {"ci/matched-wolff", "release/max-product"}

CI: SpatioOnlyParams = fixture("spatio_only", "ci").params


@functools.cache
def _rung(tier: str) -> Rung:
    return spatio_rung(fixture("spatio_only", tier).params, tier)


def _budget(rung: Rung, sweeps: int) -> Budget:
    return Budget(Cost.SITE_VISITS, sweeps * rung.visits_per_sweep)


def _cells() -> list[object]:
    return [
        pytest.param(cell, marks=pytest.mark.release) if cell in SLOW else cell
        for cell in BEFORE
    ]


@pytest.mark.analytic
@pytest.mark.snapshot
@pytest.mark.parametrize("cell", _cells())
def test_no_start_is_the_run_before_the_entry_took_one(cell: str) -> None:
    tier, method = cell.split("/", 1)
    rung = _rung(tier)
    budget = _budget(rung, SWEEPS[tier])
    digest = hashlib.sha256()
    for seed in SEEDS:
        run = ground_state(
            rung.graph, rung.field, method, budget, np.random.default_rng(seed)
        )
        digest.update(np.asarray(run.labelling, dtype=np.int64).tobytes())
        digest.update(repr(float(run.energy)).encode())
        digest.update(str(int(run.spent)).encode())
    assert digest.hexdigest()[:16] == BEFORE[cell]


def _takes_start() -> list[str]:
    return [name for name in (*METHODS, *ARMS) if name not in REFUSED]


@pytest.mark.analytic
@pytest.mark.parametrize("method", _takes_start())
@pytest.mark.parametrize("seed", [0, 1])
def test_the_start_a_solver_would_make_is_its_run_bitwise(
    method: str, seed: int
) -> None:
    rung = _rung("ci")
    budget = _budget(rung, SWEEPS["ci"])
    own = ground_state(
        rung.graph, rung.field, method, budget, np.random.default_rng(seed)
    )
    rng = np.random.default_rng(seed)
    # The labelling the solver makes for itself, made here and handed in: the
    # field's argmax for the cut-based ones, a uniform draw from the same
    # generator for the rest.
    start = (
        rung.field.argmax(axis=1).astype(np.int64)
        if method in ARGMAX_START
        else rng.integers(0, rung.n_states, size=rung.n_nodes)
    )
    given = ground_state(rung.graph, rung.field, method, budget, rng, start=start)

    np.testing.assert_array_equal(given.labelling, own.labelling)
    assert given.energy == own.energy
    assert given.spent == own.spent


@pytest.mark.analytic
@pytest.mark.parametrize("method", DESCENTS)
def test_a_descent_ends_at_or_below_its_start(method: str) -> None:
    rung = _rung("ci")
    budget = _budget(rung, SWEEPS["ci"])
    draws = np.random.default_rng(1052)
    for _ in range(8):
        start = draws.integers(0, rung.n_states, size=rung.n_nodes)
        run = ground_state(
            rung.graph,
            rung.field,
            method,
            budget,
            np.random.default_rng(0),
            start=start,
        )
        assert run.energy <= energy(rung.graph, rung.field, start) + EXACT


@pytest.mark.smoke
@pytest.mark.parametrize("method", REFUSED)
def test_a_solver_with_no_single_start_refuses_one_by_name(method: str) -> None:
    rung = _rung("ci")
    with pytest.raises(ValueError, match=f"'{method}' takes no start"):
        ground_state(
            rung.graph,
            rung.field,
            method,
            _budget(rung, 5),
            np.random.default_rng(0),
            start=np.zeros(rung.n_nodes, dtype=np.int64),
        )


@pytest.mark.smoke
def test_a_bad_start_schedule_or_name_is_refused() -> None:
    rung = _rung("ci")
    budget = _budget(rung, 5)
    rng = np.random.default_rng(0)
    for start in (
        np.zeros(rung.n_nodes - 1, dtype=np.int64),
        np.full(rung.n_nodes, rung.n_states, dtype=np.int64),
        np.zeros(rung.n_nodes),
    ):
        with pytest.raises(ValueError, match="one integer state"):
            ground_state(rung.graph, rung.field, "icm", budget, rng, start=start)
    schedule = ScheduleParams(ScheduleShape.LINEAR, 1.0, 0.1)
    with pytest.raises(ValueError, match="'icm' runs no anneal"):
        ground_state(rung.graph, rung.field, "icm", budget, rng, schedule=schedule)
    with pytest.raises(ValueError, match="'alpha-expansion' runs no anneal"):
        ground_state(rung.graph, rung.field, "alpha-expansion", budget, rng, steps=3)
    with pytest.raises(ValueError, match="the arms"):
        ground_state(rung.graph, rung.field, "gradient", budget, rng)


@pytest.mark.smoke
@pytest.mark.parametrize("method", sorted(ANNEALED))
def test_a_schedule_and_steps_reach_every_annealed_name(method: str) -> None:
    # A schedule and a step count of the caller's reach the anneal, and the
    # run on them is a function of the seed.
    rung = _rung("ci")
    budget = _budget(rung, SWEEPS["ci"])
    schedule = ScheduleParams(ScheduleShape.LINEAR, 0.5, 0.1)

    def run(seed: int) -> tuple[np.ndarray, float]:
        result = ground_state(
            rung.graph,
            rung.field,
            method,
            budget,
            np.random.default_rng(seed),
            schedule=schedule,
            steps=3,
        )
        return result.labelling, result.energy

    first, second = run(0), run(0)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


def _enumerated_minimum(rung: Rung) -> float:
    """The exact minimum by exhaustive enumeration, which nine sites affords."""
    return min(
        energy(rung.graph, rung.field, np.asarray(labelling))
        for labelling in itertools.product(range(rung.n_states), repeat=rung.n_nodes)
    )


def _ci_rung(n_states: int) -> Rung:
    field, alpha = rung_field(CI, n_states)
    return Rung(
        name=f"ci-q{n_states}",
        graph=CI.graph,
        field=field,
        alpha=alpha,
        sizes=CI.sizes,
        n_states=n_states,
        optimum=None,
    )


def _from_seeded_start(rung: Rung, method: str, seed: int) -> float:
    start = (
        None
        if method in REFUSED
        else np.random.default_rng([1052, seed]).integers(
            0, rung.n_states, size=rung.n_nodes
        )
    )
    return ground_state(
        rung.graph,
        rung.field,
        method,
        _budget(rung, 200),
        np.random.default_rng(seed),
        start=start,
    ).energy


#: The solvers that end on an expansion's local minimum, or below one.
EXPANSIONS = ("alpha-expansion", "swendsen-wang>expansion", "expansion>swendsen-wang")


@pytest.mark.oracle
@pytest.mark.parametrize(
    "method",
    [
        # Six runs of 41,250 Wolff steps are 16 s, over the per-PR cap; the
        # start test above runs the arm per PR.
        pytest.param(name, marks=pytest.mark.release)
        if name == "matched-wolff"
        else name
        for name in (*METHODS, *ARMS)
    ],
)
def test_no_solver_lands_below_the_enumerated_minimum(method: str) -> None:
    # At two states one expansion move is the whole binary problem and a
    # graph cut solves it, so every cut-based solver reaches the minimum from
    # any start; at three the expansion's factor-2 bracket holds from any
    # start, since the bound is on every expansion local minimum.
    for n_states in (2, 3):
        rung = _ci_rung(n_states)
        exact = _enumerated_minimum(rung)
        for seed in (0, 1, 2):
            found = _from_seeded_start(rung, method, seed)
            assert found >= exact - EXACT
            if n_states == 2 and method in (*EXPANSIONS, "alpha-beta-swap"):
                assert found == pytest.approx(exact, abs=EXACT)
            if n_states == 3 and method in EXPANSIONS:
                assert expansion_bracket(rung, found).lower <= exact + EXACT
