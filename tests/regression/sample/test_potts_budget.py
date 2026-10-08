"""A Wolff run spends its site-visit budget (issue #1344).

A Wolff step flips one cluster, charged by its size, so no step count spends
a budget in site visits. Given a :class:`~sal.opt.budget.Budget`,
``anneal_potts`` and ``sample_potts`` take steps until ``spent`` reaches it,
the schedule read at the spent fraction. The referees:

* ``analytic``: the run overspends by less than one step, and one Wolff step
  costs at most one lattice-spanning cluster; the schedule's last
  temperature runs;
* ``oracle``: on #1322's 2x3 ``q = 3`` fixture with a forbidden label, a
  budgeted chain fits :func:`tests._chains.enumerated_law` by chi-square;
* bitwise: a move set of fixed cost per step given ``n_steps`` steps' cost is
  the step-count run, and the step-count Wolff run is main's, pinned.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.loop import Moved, anneal_spent
from sal.sample.potts_mcmc import (
    PottsMove,
    Recolour,
    anneal_potts,
    move_set,
    sample_potts,
)
from sal.sample.potts_mcmc.chains import step_visits
from sal.sample.schedule import ramp
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

from tests._chains import cell_counts, enumerated_law
from tests.regression.sample.test_potts_recolour import (
    FIELD,
    RECORDED,
    SIGNIFICANCE,
    THINNING,
    _graph,
    _pooled_p_value,
)

SEED = 1344
N_STATES = 3
#: A lattice near the q = 3 transition, ``K_c = ln(1 + sqrt 3) = 1.005``,
#: where a Wolff cluster's size spans the lattice's range.
LATTICE = (12, 12)
COUPLING = 1.0
N_STEPS = 40


def _lattice() -> PottsGraph:
    return lattice_graph(LATTICE, BoundaryCondition.PERIODIC, COUPLING)


def _field() -> np.ndarray:
    return np.zeros(N_STATES)


def _per_sweep(graph: PottsGraph) -> int:
    return step_visits(PottsMove.SINGLE_SITE, graph)


@pytest.mark.analytic
@pytest.mark.parametrize("sweeps", [5, 50])
def test_a_wolff_anneal_spends_its_budget_within_one_cluster(sweeps: int) -> None:
    """``budget <= spent < budget + one step whose cluster spans``, whatever the clusters built."""
    graph = _lattice()
    budget = sweeps * _per_sweep(graph)
    schedule = ramp.linear(3.0, 0.5, N_STEPS)
    run = anneal_potts(
        graph,
        _field(),
        schedule,
        np.random.default_rng(SEED),
        move=PottsMove.WOLFF,
        recolour=Recolour.UNIFORM,
        budget=Budget(Cost.SITE_VISITS, budget),
    )
    # The costliest step: every move of the set, its cluster spanning the lattice.
    largest = sum(
        step_visits(each, graph, graph.n_nodes)
        for each in move_set(PottsMove.WOLFF, Recolour.UNIFORM)
    )
    assert budget <= run.spent < budget + largest
    assert run.n_sweeps == run.termination.iterations
    assert run.n_sweeps != N_STEPS


@pytest.mark.analytic
def test_the_spent_fraction_reaches_the_last_temperature() -> None:
    """Steps of random cost at most ``budget / n`` run every index, the last included, in order."""
    schedule = ramp.linear(4.0, 1.0, N_STEPS)
    budget = 10_000
    costs = np.random.default_rng(SEED).integers(1, budget // N_STEPS + 1, 10_000)
    seen: list[float] = []

    def step(
        state: int, energy: float, carried: None, temperature: float, _rng: None
    ) -> Moved[int, None]:
        seen.append(temperature)
        return Moved(state, energy, carried, int(costs[len(seen) - 1]))

    walked = anneal_spent(
        step, schedule, Moved(0, 0.0, None, 0), None, int, budget=budget
    )
    expected = [schedule(index) for index in range(N_STEPS)]
    assert sorted(set(seen), reverse=True) == expected
    assert seen[0] == schedule(0)
    assert seen[-1] == schedule(N_STEPS - 1)
    assert budget <= walked.spent < budget + budget // N_STEPS
    assert walked.termination.iterations == len(seen)


@pytest.mark.analytic
@pytest.mark.parametrize(
    "move", [PottsMove.SINGLE_SITE, PottsMove.SWENDSEN_WANG, (PottsMove.SWENDSEN_WANG,)]
)
def test_a_fixed_cost_anneal_under_its_budget_is_the_step_run_bitwise(
    move: PottsMove | tuple[PottsMove, ...],
) -> None:
    """Budget ``n * c`` reads step ``k`` at index ``k``: the step-count run, state for state."""
    graph = _lattice()
    schedule = ramp.linear(3.0, 0.5, N_STEPS)
    stepped = anneal_potts(
        graph, _field(), schedule, np.random.default_rng(SEED), move=move
    )
    budget = Budget(Cost.SITE_VISITS, stepped.spent)
    spent = anneal_potts(
        graph, _field(), schedule, np.random.default_rng(SEED), move=move, budget=budget
    )
    assert spent.spent == stepped.spent
    assert spent.n_sweeps == stepped.n_sweeps == N_STEPS
    assert spent.energy == stepped.energy
    np.testing.assert_array_equal(spent.best, stepped.best)
    np.testing.assert_array_equal(spent.final, stepped.final)


#: Main's step-count Wolff run (57a52f90), uniform recolouring alone and the
#: default composition with a Gibbs sweep: ``(energy, spent, final[:8])``.
WOLFF_PINS = {
    "uniform": (-280.0, 32430, (1, 1, 1, 1, 1, 1, 1, 1)),
    "default": (-272.0, 31320, (2, 2, 2, 2, 2, 2, 2, 2)),
}


@pytest.mark.smoke
@pytest.mark.snapshot
@pytest.mark.parametrize("name", WOLFF_PINS)
def test_the_step_count_wolff_anneal_is_mains_bitwise(name: str) -> None:
    """No budget: one step per schedule entry, the run main returned."""
    recolour = Recolour.UNIFORM if name == "uniform" else Recolour.PER_MOVE
    run = anneal_potts(
        _lattice(),
        _field(),
        ramp.linear(3.0, 0.5, N_STEPS),
        np.random.default_rng(SEED),
        move=PottsMove.WOLFF,
        recolour=recolour,
        # Main's run is the oracle's stream; the Rust step is the same law
        # on another (#1362).
        cluster_backend=Backend.PYTHON,
    )
    energy, spent, head = WOLFF_PINS[name]
    assert (run.energy, run.spent, tuple(int(v) for v in run.final[:8])) == (
        energy,
        spent,
        head,
    )


@pytest.mark.analytic
@pytest.mark.parametrize("move", [PottsMove.SINGLE_SITE, PottsMove.SWENDSEN_WANG])
def test_a_fixed_cost_chain_under_its_budget_is_the_step_chain_bitwise(
    move: PottsMove,
) -> None:
    """``sample_potts`` given ``n`` steps' site visits records the ``n``-step chain."""
    graph = _graph()
    n_sweeps, burn_in, thin = 200, 20, 3
    per_step = sum(
        step_visits(each, graph)
        for each in (
            (move,)
            if move is PottsMove.SINGLE_SITE
            else (PottsMove.SWENDSEN_WANG_HEAT_BATH, PottsMove.SINGLE_SITE)
        )
    )
    stepped = sample_potts(
        graph, FIELD, move, np.random.default_rng(SEED), n_sweeps, burn_in, thin
    )
    spent = sample_potts(
        graph,
        FIELD,
        move,
        np.random.default_rng(SEED),
        Budget(Cost.SITE_VISITS, n_sweeps * thin * per_step),
        burn_in,
        thin,
    )
    np.testing.assert_array_equal(spent.states, stepped.states)
    assert spent.acceptance == stepped.acceptance
    assert spent.termination == stepped.termination


@pytest.mark.analytic
def test_a_budget_in_another_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="site_visits|SITE_VISITS|site visits"):
        anneal_potts(
            _graph(),
            FIELD,
            ramp.linear(2.0, 1.0, 4),
            np.random.default_rng(SEED),
            budget=Budget(Cost.SWEEPS, 10),
        )


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("moves", "recolour"),
    [((PottsMove.WOLFF,), Recolour.UNIFORM), (PottsMove.WOLFF, Recolour.PER_MOVE)],
)
def test_a_budgeted_wolff_chain_keeps_the_boltzmann_law(
    moves: PottsMove | tuple[PottsMove, ...], recolour: Recolour
) -> None:
    """#1322's chi-square on the 2x3 fixture, the chain sized by site visits."""
    graph = _graph()
    index, probability = enumerated_law(graph, FIELD)
    cost = step_visits(PottsMove.WOLFF, graph, 2)
    chain = sample_potts(
        graph,
        FIELD,
        moves,
        np.random.default_rng(SEED),
        Budget(Cost.SITE_VISITS, RECORDED * THINNING * cost),
        burn_in=RECORDED // 10,
        thin=THINNING,
        recolour=recolour,
    )
    counts = cell_counts(index, chain.states)
    support = probability > 0
    assert len(chain.states) > 0
    assert counts[~support].sum() == 0
    assert _pooled_p_value(probability[support], counts[support]) > SIGNIFICANCE
