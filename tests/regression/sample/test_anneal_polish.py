"""Annealing ends with a polisher run to its own convergence (issue #1363).

The schedule runs its full length; the polisher then runs from the final
state until its own criterion holds. The referees:

* ``analytic``: the returned labelling is an ICM fixed point, one more index
  sweep changes nothing; ``spent`` is the schedule's plus the polish's;
  ``polish=None`` is the unpolished run, bitwise; a forbidden label stays
  unused;
* ``oracle``: on an enumerable instance the converged energy is at least the
  exact minimum found by enumeration;
* HMC: the polished point's gradient meets L-BFGS's own tolerance.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.objective import Objective, value_and_gradient
from sal.opt.starts import PolishedPoint, polish_by_fit
from sal.opt.termination import Stop, Termination
from sal.sample.hmc import anneal
from sal.sample.potts_mcmc import PottsMove, anneal_potts
from sal.sample.potts_mcmc.chains import step_visits
from sal.sample.schedule import ExponentialTempSchedule, Polish
from sal.search.icm import iterated_conditional_modes
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy

from tests._posteriors import GAUSSIAN

N_STATES = 3
SCHEDULE = ExponentialTempSchedule(2.0, 0.3, 20)


def _graph() -> PottsGraph:
    return lattice_graph((6, 6), BoundaryCondition.PERIODIC, 1.0)


@pytest.mark.analytic
@pytest.mark.parametrize("seed", range(4))
def test_the_polished_labelling_is_an_icm_fixed_point(seed: int) -> None:
    graph, field = _graph(), np.array([0.0, 0.3, -0.2])
    run = anneal_potts(
        graph, field, SCHEDULE, np.random.default_rng(seed), polish=Polish.ICM
    )
    again = iterated_conditional_modes(
        graph, field, np.random.default_rng(0), start=run.best, max_iterations=1
    )

    assert np.array_equal(again.labelling, run.best)
    assert run.termination.reason is Stop.CONVERGED
    assert run.polished_by == "icm"
    assert run.energy == pytest.approx(energy(graph, field, run.best), abs=1e-9)


@pytest.mark.analytic
def test_spent_is_the_schedule_plus_the_polish_and_none_is_bitwise() -> None:
    graph, field = _graph(), np.zeros(N_STATES)
    plain = anneal_potts(graph, field, SCHEDULE, np.random.default_rng(3))
    unpolished = anneal_potts(
        graph, field, SCHEDULE, np.random.default_rng(3), polish=None
    )
    polished = anneal_potts(
        graph, field, SCHEDULE, np.random.default_rng(3), polish=Polish.ICM
    )
    sweep = step_visits(PottsMove.SINGLE_SITE, graph)

    assert np.array_equal(plain.best, unpolished.best)
    assert np.array_equal(plain.final, unpolished.final)
    assert (plain.energy, plain.spent, plain.termination) == (
        unpolished.energy,
        unpolished.spent,
        unpolished.termination,
    )
    assert (plain.polish_spent, plain.polished_by) == (0, None)
    assert polished.spent - polished.polish_spent == plain.spent
    assert polished.polish_spent > 0
    assert polished.polish_spent % sweep == 0
    assert polished.energy <= plain.energy + 1e-9
    assert polished.n_sweeps == plain.n_sweeps


@pytest.mark.analytic
def test_the_polish_follows_a_budget_and_honours_a_forbidden_label() -> None:
    graph = _graph()
    field = np.array([0.0, -np.inf, 0.2])
    budget = Budget(Cost.SITE_VISITS, 15 * step_visits(PottsMove.SINGLE_SITE, graph))
    run = anneal_potts(
        graph,
        field,
        SCHEDULE,
        np.random.default_rng(7),
        start=np.zeros(graph.n_nodes, dtype=np.int64),
        budget=budget,
        polish=Polish.ICM,
    )

    assert not bool((run.best == 1).any())
    assert run.termination.reason is Stop.CONVERGED
    assert run.spent - run.polish_spent >= budget.size


@pytest.mark.analytic
def test_a_floor_without_a_polish_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a polish"):
        anneal_potts(
            _graph(),
            np.zeros(N_STATES),
            SCHEDULE,
            np.random.default_rng(0),
            min_sites=2,
        )


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(6))
def test_the_converged_energy_is_at_least_the_enumerated_minimum(seed: int) -> None:
    # 2 x 4 open lattice, q = 3, couplings of either sign: 6,561 labellings.
    rng = np.random.default_rng(1363 + seed)
    base = lattice_graph((2, 4), BoundaryCondition.OPEN, 1.0)
    graph = PottsGraph(
        base.n_nodes, base.edges, tuple(rng.normal(0.0, 1.0, len(base.edges)))
    )
    field = rng.normal(0.0, 0.5, (graph.n_nodes, N_STATES))
    exact = min(
        energy(graph, field, np.array(labels))
        for labels in itertools.product(range(N_STATES), repeat=graph.n_nodes)
    )
    run = anneal_potts(
        graph, field, SCHEDULE, np.random.default_rng(seed), polish=Polish.ICM
    )

    assert run.energy >= exact - 1e-9
    assert run.termination.reason is Stop.CONVERGED


@pytest.mark.analytic
def test_the_hmc_polish_meets_its_gradient_tolerance() -> None:
    start = torch.tensor([4.0, 3.0], dtype=torch.float64)
    run = anneal(
        GAUSSIAN,
        ExponentialTempSchedule(4.0, 0.5, 40),
        rng=torch.Generator().manual_seed(2),
        step_size=0.2,
        n_steps=10,
        start=start,
        polish=polish_by_fit,
        polish_budget=Budget(Cost.ITERATIONS, 100),
    )
    plain = anneal(
        GAUSSIAN,
        ExponentialTempSchedule(4.0, 0.5, 40),
        rng=torch.Generator().manual_seed(2),
        step_size=0.2,
        n_steps=10,
        start=start,
    )
    value, gradient = value_and_gradient(GAUSSIAN, run.best)

    assert run.termination.reason is Stop.CONVERGED
    assert run.polished_by == "polish_by_fit"
    assert float(gradient.abs().max()) / max(1.0, abs(float(value))) <= 1e-8
    assert run.spent - run.polish_spent == plain.spent
    assert run.value <= plain.value


@pytest.mark.analytic
def test_an_hmc_polish_at_its_cap_ends_on_the_budget() -> None:
    # A polisher that spends its cap without meeting its criterion: the run
    # ends on the budget and names it.
    def capped(
        objective: Objective, theta: torch.Tensor, budget: Budget
    ) -> PolishedPoint:
        return PolishedPoint(
            value=float(objective(theta)),
            termination=Termination.after(budget.size, converged=False),
            theta=theta,
        )

    run = anneal(
        GAUSSIAN,
        ExponentialTempSchedule(4.0, 0.5, 5),
        rng=torch.Generator().manual_seed(2),
        step_size=0.2,
        n_steps=10,
        polish=capped,
        polish_budget=Budget(Cost.ITERATIONS, 3),
    )

    assert run.termination.reason is Stop.BUDGET
    assert run.polished_by == "capped"
    assert run.polish_spent >= 3
