"""Annealing ends with a polisher run to its own convergence (issues #1363, #1373, #1374).

The schedule runs its full length; the polisher then runs from the best
state alone until its own criterion holds, and ``Polish.ICM_MERGE`` then
merges whole labels. The referees:

* ``analytic``: the returned labelling is an ICM fixed point, one more index
  sweep changes nothing; ``spent`` is the schedule's plus the polish's;
  ``polish=None`` is the unpolished run, bitwise; a forbidden label stays
  unused;
* ``oracle``: on an enumerable instance the converged energy is at least the
  exact minimum found by enumeration;
* HMC: the polished point's gradient meets L-BFGS's own tolerance;
* ``oracle``: the polish is ICM from the unpolished run's best, bitwise;
  every merge ``delta(u, v)`` is the :func:`energies` difference to 1e-12
  relative; the merge is a brute-force greedy over all pairs, bitwise;
* ``analytic``: the stages' ``spent`` sums to ``spent``, their energy never
  rises and the last is the top level; no admissible merge lowers the merged
  energy; one label has no merge.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.objective import Objective, value_and_gradient
from sal.opt.starts import PolishedPoint, polish_by_fit
from sal.opt.termination import Stop, Termination
from sal.sample.hmc import anneal
from sal.sample.potts_mcmc import PottsMove, anneal_potts
from sal.sample.potts_mcmc.chains import merge_labels, step_visits
from sal.sample.schedule import ExponentialTempSchedule, Polish
from sal.search.icm import iterated_conditional_modes
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energies, energy

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


@pytest.mark.oracle
@pytest.mark.parametrize("loop_backend", [Backend.PYTHON, Backend.RUST])
def test_the_polish_descends_the_best_alone(loop_backend: Backend) -> None:
    """``best`` is ICM run from the unpolished run's ``best`` to its fixed point, bitwise (#1374)."""
    graph, field = _graph(), np.array([0.0, 0.3, -0.2])
    plain, polished = (
        anneal_potts(
            graph,
            field,
            SCHEDULE,
            np.random.default_rng(11),
            loop_backend=loop_backend,
            polish=polish,
        )
        for polish in (None, Polish.ICM)
    )
    expected = iterated_conditional_modes(
        graph,
        field,
        np.random.default_rng(0),
        start=plain.best,
        max_iterations=graph.n_nodes,
    )

    np.testing.assert_array_equal(polished.best, expected.labelling)
    assert polished.energy == expected.energy
    assert polished.energy <= plain.energy
    sweep = step_visits(PottsMove.SINGLE_SITE, graph)
    assert polished.polish_spent == expected.sweeps * sweep
    assert polished.spent == plain.spent + polished.polish_spent


@pytest.mark.analytic
@pytest.mark.parametrize("loop_backend", [Backend.PYTHON, Backend.RUST])
@pytest.mark.parametrize("polish", [None, Polish.ICM, Polish.ICM_MERGE])
def test_the_stages_sum_to_the_run_and_the_last_is_the_result(
    loop_backend: Backend, polish: Polish | None
) -> None:
    """``init``, ``polish``, ``merge`` as run: costs sum to ``spent``, energy never rises (#1373)."""
    graph, field = _graph(), np.array([0.0, 0.3, -0.2])
    plain = anneal_potts(
        graph, field, SCHEDULE, np.random.default_rng(5), loop_backend=loop_backend
    )
    run = anneal_potts(
        graph,
        field,
        SCHEDULE,
        np.random.default_rng(5),
        loop_backend=loop_backend,
        polish=polish,
    )
    names = {None: ["init"], Polish.ICM: ["init", "polish"]}.get(
        polish, ["init", "polish", "merge"]
    )

    assert [stage.name for stage in run.stages] == names
    np.testing.assert_array_equal(run.stages[0].best, plain.best)
    assert run.stages[0].energy == plain.energy
    assert sum(stage.spent for stage in run.stages) == run.spent
    assert sum(stage.spent for stage in run.stages[1:]) == run.polish_spent
    levels = [stage.energy for stage in run.stages]
    assert levels == sorted(levels, reverse=True)
    last = run.stages[-1]
    np.testing.assert_array_equal(last.best, run.best)
    assert (last.energy, last.termination) == (run.energy, run.termination)
    assert all(stage.seconds >= 0.0 for stage in run.stages)


def _merge_instance(seed: int) -> tuple[PottsGraph, np.ndarray]:
    """A 2 x 3 open lattice, q = 3, random positive couplings and fields, label 2 forbidden at site 0."""
    rng = np.random.default_rng(1373 + seed)
    base = lattice_graph((2, 3), BoundaryCondition.OPEN, 1.0)
    graph = PottsGraph(
        base.n_nodes, base.edges, tuple(rng.uniform(0.2, 1.5, len(base.edges)))
    )
    field = rng.normal(0.0, 0.5, (graph.n_nodes, N_STATES))
    field[0, 2] = -np.inf
    return graph, field


def _admissible(field: np.ndarray, labels: np.ndarray) -> list[tuple[int, int]]:
    """The pairs ``(u, v)`` of labels in use, ``v`` allowed at every site of ``u``."""
    used = sorted({int(label) for label in labels})
    return [
        (u, v)
        for u in used
        for v in used
        if u != v and bool(np.isfinite(field[labels == u, v]).all())
    ]


def _brute_greedy(
    graph: PottsGraph, field: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    """The greedy merge by enumeration: each round scores every admissible merge by :func:`energies`."""
    labels = labels.copy()
    levels = [float(energies(graph, field, labels[None])[0])]
    while True:
        chosen, lowest = None, 0.0
        for u, v in _admissible(field, labels):
            merged = np.where(labels == u, v, labels)
            delta = float(energies(graph, field, merged[None])[0]) - levels[-1]
            if delta < lowest:
                chosen, lowest = (u, v), delta
        if chosen is None:
            return labels, levels
        labels = np.where(labels == chosen[0], chosen[1], labels)
        levels.append(float(energies(graph, field, labels[None])[0]))


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(4))
def test_each_merge_delta_is_the_energy_difference(seed: int) -> None:
    """``delta(u, v) = -(U[u, v] - U[u, u]) - (B[u, v] + B[v, u]) / 2`` is ``energies(merged) - energies(labels)`` (#1373)."""
    graph, field = _merge_instance(seed)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    for labels in itertools.product(range(N_STATES), repeat=graph.n_nodes):
        labelling = np.array(labels)
        if not np.isfinite(field[np.arange(graph.n_nodes), labelling]).all():
            continue
        before = float(energies(graph, field, labelling[None])[0])
        for u, v in _admissible(field, labelling):
            sites = labelling == u
            unary = field[sites, v].sum() - field[sites, u].sum()
            bonds = 0.0
            for node in range(graph.n_nodes):
                for position in range(offsets[node], offsets[node + 1]):
                    pair = (labelling[node], labelling[neighbours[position]])
                    if pair in ((u, v), (v, u)):
                        bonds += couplings[position]
            delta = -unary - bonds / 2.0
            merged = np.where(sites, v, labelling)
            after = float(energies(graph, field, merged[None])[0])
            assert delta == pytest.approx(after - before, rel=1e-12, abs=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(4))
def test_the_merge_is_the_brute_force_greedy(seed: int) -> None:
    """On 2 x 3, q = 3, :func:`merge_labels` from every allowed labelling is the enumerated greedy, bitwise (#1373)."""
    graph, field = _merge_instance(seed)
    for labels in itertools.product(range(N_STATES), repeat=graph.n_nodes):
        labelling = np.array(labels, dtype=np.int64)
        if not np.isfinite(field[np.arange(graph.n_nodes), labelling]).all():
            continue
        merged, rounds = merge_labels(graph, field, labelling)
        expected, levels = _brute_greedy(graph, field, labelling)

        np.testing.assert_array_equal(merged, expected)
        assert rounds == len(levels)
        assert levels == sorted(levels, reverse=True)
        # The fixed point: no admissible merge lowers the energy.
        final = float(energies(graph, field, merged[None])[0])
        for u, v in _admissible(field, merged):
            after = np.where(merged == u, v, merged)
            assert float(energies(graph, field, after[None])[0]) >= final


@pytest.mark.analytic
def test_one_label_has_no_merge() -> None:
    graph, field = _merge_instance(0)
    labelling = np.zeros(graph.n_nodes, dtype=np.int64)
    merged, rounds = merge_labels(graph, field, labelling)

    np.testing.assert_array_equal(merged, labelling)
    assert rounds == 1


@pytest.mark.analytic
def test_the_hmc_stages_are_the_schedule_then_the_polish() -> None:
    """HMC ``anneal``: ``init`` and ``polish``, costs summing to ``spent``, the last the result (#1373)."""
    run = anneal(
        GAUSSIAN,
        ExponentialTempSchedule(4.0, 0.5, 40),
        rng=torch.Generator().manual_seed(2),
        polish=polish_by_fit,
        step_size=0.2,
        n_steps=10,
        polish_budget=Budget(Cost.ITERATIONS, 100),
    )
    plain = anneal(
        GAUSSIAN,
        ExponentialTempSchedule(4.0, 0.5, 40),
        rng=torch.Generator().manual_seed(2),
        step_size=0.2,
        n_steps=10,
    )

    assert [stage.name for stage in plain.stages] == ["init"]
    assert [stage.name for stage in run.stages] == ["init", "polish"]
    assert torch.equal(run.stages[0].best, plain.best)
    assert sum(stage.spent for stage in run.stages) == run.spent
    assert run.stages[1].energy <= run.stages[0].energy
    assert torch.equal(run.stages[-1].best, run.best)
    assert (run.stages[-1].energy, run.stages[-1].termination) == (
        run.value,
        run.termination,
    )
