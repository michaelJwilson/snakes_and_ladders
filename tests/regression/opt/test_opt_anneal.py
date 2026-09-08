"""The annealing driver and the exchange step, against the loops they replaced.

Issue #386 merged eight entry points onto `opt.anneal.drive` and
`opt.anneal.exchange`. The ticket's acceptance criterion is draw-for-draw
equality, so every reference below is the replaced loop copied verbatim from
the commit before the seam and every assertion is exact: the same states in
the same order, not the same distribution. A distributional check would pass
on a seam that consumed its uniforms in a different order, and the order is
the thing eight callers had committed chains under.

The seam's own two behaviours are pinned first, because they are what the
adapters rest on: the strict comparison that fixes which of two equal-valued
states is kept, and the `draw_first` flag that decides whether an exchange
uniform is consumed when the ratio already settles the outcome.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.relaxed import (
    MINIMUM_TEMPERATURE,
)
from snakes_and_ladders.learn.relaxed import (
    anneal as relaxed_anneal,
)
from snakes_and_ladders.opt.anneal import (
    Extremum,
    drive,
    exchange,
    swap_log_ratio,
)
from snakes_and_ladders.opt.hmc import (
    anneal as hmc_anneal,
)
from snakes_and_ladders.opt.hmc import (
    parallel_tempering as hmc_parallel_tempering,
)
from snakes_and_ladders.opt.hmc import (
    sample,
)
from snakes_and_ladders.opt.schedule import Constant, Exponential, Linear
from snakes_and_ladders.search.gibbs import anneal_factor_graph
from snakes_and_ladders.search.potts_mcmc import anneal_potts, parallel_tempering
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

from tests._objective_checks import AnalyticGaussian

SHAPE = (3, 3)
COUPLING = 0.7
FIELD = np.array([0.3, -0.7, 0.15])
LADDER = (0.6, 1.0, 1.8)


@pytest.mark.structural
def test_the_driver_keeps_the_first_state_to_reach_the_best_value() -> None:
    # The tie rule, which no adapter states and all four depend on: a strict
    # comparison keeps the earliest maximizer, a non-strict one the latest.
    # Both drivers ran strict, so a run whose value plateaus reports the step
    # it first plateaued at.
    values = [0.0, 1.0, 1.0, 0.5]
    seen: list[int] = []

    def transition(state: int, _: float, __: float) -> tuple[int, float, bool]:
        seen.append(state)
        return state + 1, values[state + 1], True

    run = drive(0, values[0], transition, Constant(1.0, 3), keep=Extremum.MAXIMUM)
    assert run.best == 1
    assert run.best_value == 1.0
    assert run.final == 3
    assert run.accepted == 3
    assert np.array_equal(run.trajectory, np.array(values))
    assert seen == [0, 1, 2]


@pytest.mark.structural
def test_the_driver_snapshots_a_mutable_state_when_told_to() -> None:
    # A heat-bath sweep writes through its state array. Without `copy` the
    # best labelling aliases the final one and the driver reports the last
    # sweep as the best, which is a silent failure with a plausible answer.
    def transition(
        state: np.ndarray, _: float, __: float
    ) -> tuple[np.ndarray, float, bool]:
        state += 1
        return state, -float(state[0]), False

    start = np.zeros(2)
    run = drive(
        start,
        0.0,
        transition,
        Constant(1.0, 3),
        keep=Extremum.MINIMUM,
        copy=lambda state: state.copy(),
    )
    assert run.best_value == -3.0
    assert np.array_equal(run.best, np.array([3.0, 3.0]))
    assert run.best is not run.final


@pytest.mark.structural
def test_draw_first_decides_whether_a_settled_pair_consumes_a_uniform() -> None:
    # Two replicas whose energies make the ratio non-negative, so the
    # exchange is certain. Lazily the uniform is never drawn; eagerly it is,
    # and every later draw in the caller's stream shifts by one. That is the
    # difference between `opt.hmc.parallel_tempering` and the three discrete
    # entry points, and the reason the flag is not a preference.
    drawn: list[int] = []

    def draw() -> float:
        drawn.append(1)
        return 0.5

    betas = [1.0 / temperature for temperature in (1.0, 2.0)]
    energies = [5.0, 0.0]  # the colder replica already holds the higher
    # energy, so `(beta_i - beta_j)(E_i - E_j)` is positive and the
    # exchange is certain before any uniform is looked at.
    assert swap_log_ratio(betas[0], betas[1], *energies) >= 0.0

    lazy = exchange(energies, betas, draw, lambda _, __: None)
    assert drawn == []
    eager = exchange(energies, betas, draw, lambda _, __: None, draw_first=True)
    assert drawn == [1]
    assert np.array_equal(lazy, eager)
    assert np.array_equal(lazy, np.array([1.0]))


@pytest.mark.structural
def test_an_accepted_exchange_is_applied_before_the_next_pair_is_scored() -> None:
    # A configuration can cross more than one rung in a pass, which is what
    # all four entry points did. Scoring every pair against the energies the
    # pass started with would be a shuffle rather than a ladder, and would
    # look identical in the acceptance rates.
    order: list[tuple[int, int]] = []
    betas = [3.0, 2.0, 1.0]
    energies = [9.0, 5.0, 0.0]

    accepted = exchange(
        energies,
        betas,
        lambda: 0.0,
        lambda first, second: order.append((first, second)),
    )
    assert np.array_equal(accepted, np.array([1.0, 1.0]))
    assert order == [(0, 1), (1, 2)]


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_anneal_potts_is_the_loop_it_replaced_draw_for_draw(seed: int) -> None:
    # The reference is `anneal_potts`'s body before #386, run on a generator
    # seeded identically. Equality of the labelling *and* the final state
    # *and* the energy: the driver consumes the sweep's uniforms in the same
    # order or the two chains part on the first step.
    from snakes_and_ladders.search.backend import Backend
    from snakes_and_ladders.search.potts_mcmc import _adjacency, _sweep_at, energies

    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    schedule = Exponential(start=2.0, end=0.1, n_steps=60)

    rng = np.random.default_rng(seed)
    state = rng.integers(0, int(FIELD.shape[0]), size=graph.n_nodes)
    neighbours = _adjacency(graph)
    best_state = state.copy()
    best_energy = float(energies(graph, FIELD, state[None])[0])
    sweep = _sweep_at(graph, FIELD, neighbours, Backend.PYTHON)
    for step in range(schedule.n_steps):
        sweep(state, rng, 1.0 / schedule(step))
        energy = float(energies(graph, FIELD, state[None])[0])
        if energy < best_energy:
            best_state, best_energy = state.copy(), energy

    run = anneal_potts(graph, FIELD, schedule, np.random.default_rng(seed))
    assert np.array_equal(run.labelling, best_state)
    assert run.energy == best_energy
    assert np.array_equal(run.final, state)
    assert run.n_sweeps == schedule.n_steps


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_anneal_factor_graph_is_the_loop_it_replaced_draw_for_draw(seed: int) -> None:
    # The general annealer, against its own pre-seam body. The trajectory is
    # asserted element by element, not just its endpoints: it is what the
    # notebooks plot, so a driver that recorded the value before the step
    # rather than after would move a committed figure.
    from snakes_and_ladders.search.gibbs import _Indexed, gibbs_sweep

    graph = from_potts(lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING), FIELD)
    schedule = Linear(start=2.0, end=0.2, n_steps=50)

    rng = np.random.default_rng(seed)
    indexed = _Indexed(graph)
    state = indexed.start(rng, None)
    best_state = state.copy()
    best = indexed.log_density(state)
    trajectory = [best]
    for step in range(schedule.n_steps):
        gibbs_sweep(indexed, state, rng, beta=1.0 / schedule(step))
        value = indexed.log_density(state)
        trajectory.append(value)
        if value > best:
            best, best_state = value, state.copy()

    run = anneal_factor_graph(graph, schedule, np.random.default_rng(seed))
    assert np.array_equal(run.state, best_state)
    assert run.log_density == best
    assert np.array_equal(run.trajectory, np.array(trajectory))


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1])
def test_potts_parallel_tempering_is_the_loop_it_replaced_draw_for_draw(
    seed: int,
) -> None:
    # The exchange pass, on the ladder the committed chi-square tests use.
    # Every recorded sweep is compared, so a swap applied to the wrong
    # replica or a uniform drawn out of turn shows up rather than averaging
    # out of the acceptance rate.
    from snakes_and_ladders.search.backend import Backend
    from snakes_and_ladders.search.potts_mcmc import _adjacency, _sweep_at, energies

    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    n_sweeps, burn_in, thin = 40, 5, 2

    rng = np.random.default_rng(seed)
    n_replicas = len(LADDER)
    betas = [1.0 / temperature for temperature in LADDER]
    children = rng.spawn(n_replicas)
    states = np.stack(
        [child.integers(0, FIELD.shape[0], size=graph.n_nodes) for child in children]
    )
    neighbours = _adjacency(graph)
    recorded = np.empty((n_sweeps, n_replicas, graph.n_nodes), dtype=np.int64)
    proposed = np.zeros(n_replicas - 1)
    accepted = np.zeros(n_replicas - 1)
    current = energies(graph, FIELD, states)
    best_index = int(np.argmin(current))
    best, best_energy = states[best_index].copy(), float(current[best_index])
    sweep = _sweep_at(graph, FIELD, neighbours, Backend.PYTHON)
    for step in range(-burn_in * thin, n_sweeps * thin):
        for replica in range(n_replicas):
            sweep(states[replica], children[replica], betas[replica])
        current = energies(graph, FIELD, states)
        for pair in range(n_replicas - 1):
            log_ratio = (betas[pair] - betas[pair + 1]) * (
                current[pair] - current[pair + 1]
            )
            proposed[pair] += 1
            if log_ratio >= 0.0 or rng.random() < np.exp(log_ratio):
                accepted[pair] += 1
                states[[pair, pair + 1]] = states[[pair + 1, pair]]
                current[[pair, pair + 1]] = current[[pair + 1, pair]]
        lowest = int(np.argmin(current))
        if current[lowest] < best_energy:
            best, best_energy = states[lowest].copy(), float(current[lowest])
        if step >= 0 and (step + 1) % thin == 0:
            recorded[step // thin] = states

    run = parallel_tempering(
        graph,
        FIELD,
        LADDER,
        np.random.default_rng(seed),
        n_sweeps,
        burn_in=burn_in,
        thin=thin,
    )
    assert np.array_equal(run.states, recorded)
    assert np.array_equal(run.swap_acceptance, accepted / proposed)
    assert np.array_equal(run.best, best)
    assert run.best_energy == best_energy


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_constant_schedule_reproduces_hmc_sample_draw_for_draw(seed: int) -> None:
    # The oracle `STATUS.md`'s audit table names for the continuous annealer.
    # `sample` and `anneal` run the same transition; at a constant
    # temperature the only difference is the bookkeeping, so the final
    # position must be bitwise the sampler's last draw.
    objective = AnalyticGaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    n_draws, step_size = 30, 0.4

    drawn = sample(
        objective,
        torch.Generator().manual_seed(seed),
        n_draws,
        step_size=step_size,
        theta0=torch.zeros(2, dtype=torch.float64),
    )
    annealed = hmc_anneal(
        objective,
        Constant(1.0, n_draws),
        torch.Generator().manual_seed(seed),
        step_size=step_size,
        theta0=torch.zeros(2, dtype=torch.float64),
    )
    assert torch.equal(annealed.final, drawn.theta[-1])
    assert annealed.acceptance_rate == drawn.acceptance_rate


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1])
def test_hmc_parallel_tempering_is_the_loop_it_replaced_draw_for_draw(
    seed: int,
) -> None:
    # The eager draw is the whole content of this one: the pre-seam loop drew
    # its exchange uniform before testing the ratio. Run against a lazy
    # `exchange` every position after the first certain exchange differs, so
    # this pins the flag as much as the loop.
    from snakes_and_ladders.opt.hmc import _start, _transition, leapfrog

    objective = AnalyticGaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    temperatures = (1.0, 1.6, 2.4)
    n_rounds, step_size, n_steps = 12, 0.35, 8

    parent = torch.Generator().manual_seed(seed)
    n_replicas = len(temperatures)
    children = [
        torch.Generator().manual_seed(int(child))
        for child in torch.randint(0, 2**31 - 1, (n_replicas,), generator=parent)
    ]
    start = _start(objective, None)
    positions = [start.clone() for _ in range(n_replicas)]
    values = [float(objective(start))] * n_replicas
    best, best_value = start.clone(), values[0]
    accepted = torch.zeros(n_replicas, dtype=torch.float64)
    swapped = torch.zeros(n_replicas - 1, dtype=torch.float64)
    recorded = torch.empty((n_rounds, n_replicas, start.shape[0]), dtype=torch.float64)
    for round_index in range(n_rounds):
        for replica in range(n_replicas):
            positions[replica], _, was_accepted, _ = _transition(
                objective,
                positions[replica],
                temperatures[replica],
                children[replica],
                step_size,
                n_steps,
                leapfrog,
            )
            accepted[replica] += was_accepted
            values[replica] = float(objective(positions[replica]))
        for pair in range(n_replicas - 1):
            log_ratio = (1.0 / temperatures[pair] - 1.0 / temperatures[pair + 1]) * (
                values[pair] - values[pair + 1]
            )
            uniform = float(torch.rand(1, generator=parent))
            if log_ratio >= 0.0 or uniform < math.exp(log_ratio):
                swapped[pair] += 1
                positions[pair], positions[pair + 1] = (
                    positions[pair + 1],
                    positions[pair],
                )
                values[pair], values[pair + 1] = values[pair + 1], values[pair]
        lowest = min(range(n_replicas), key=values.__getitem__)
        if values[lowest] < best_value:
            best, best_value = positions[lowest].clone(), values[lowest]
        recorded[round_index] = torch.stack(positions)

    run = hmc_parallel_tempering(
        objective,
        temperatures,
        torch.Generator().manual_seed(seed),
        n_rounds,
        step_size=step_size,
        n_steps=n_steps,
    )
    assert torch.equal(run.positions, recorded)
    assert torch.equal(run.swap_acceptance, swapped / n_rounds)
    assert torch.equal(run.acceptance_rate, accepted / n_rounds)
    assert torch.equal(run.theta, best)
    assert run.value == best_value


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("start", "end", "steps"),
    [(2.0, 0.1, 50), (1.0, MINIMUM_TEMPERATURE, 7), (0.5, 0.5, 3), (3.0, 0.2, 2)],
)
def test_the_relaxation_schedule_is_the_exponential_schedule(
    start: float, end: float, steps: int
) -> None:
    # `learn.relaxed.anneal` was `opt.schedule.Exponential` written in a
    # different algebra: `start * (end / start) ** f` against
    # `start ** (1 - f) * end ** f`. The bound is what the two roundings
    # differ by, and it is asserted rather than described.
    schedule = Exponential(start=start, end=end, n_steps=steps)
    realized = [relaxed_anneal(start, end, steps, step) for step in range(steps)]
    reference = [
        start * (end / start) ** (min(step, steps - 1) / (steps - 1))
        for step in range(steps)
    ]
    assert realized == [schedule(step) for step in range(steps)]
    assert realized[0] == start
    assert realized[-1] == end
    deviation = max(
        abs(a - b) / b for a, b in zip(realized, reference, strict=True) if b
    )
    assert deviation < 1e-15, deviation


@pytest.mark.edge_case
def test_the_relaxation_schedule_clamps_a_step_past_the_end() -> None:
    # `Exponential` refuses a step outside its range and the optimizer's last
    # iterate asks for one, so the clamp stays at this call site rather than
    # loosening the schedule for every other consumer.
    assert relaxed_anneal(2.0, 0.1, 5, 9) == relaxed_anneal(2.0, 0.1, 5, 4)
    with pytest.raises(ValueError, match="outside a schedule"):
        Exponential(start=2.0, end=0.1, n_steps=5)(9)
