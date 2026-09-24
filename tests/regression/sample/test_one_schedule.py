"""Two spellings of a ladder are one schedule, and a run cannot tell them apart (issue #827).

Four annealers took a :class:`~snakes_and_ladders.sample.schedule.TempSchedule`
and six temperings took a bare sequence of temperatures, so a caller who had
built a schedule could not hand it to a tempering. `LadderTempSchedule` is the
sequence as a schedule and `ladder` reads either spelling into the tuple the
run consumes. What is pinned here is that the two spellings are the **same
floats** --- every chain, swap and estimate below is bitwise what it was ---
on the smallest instance each function runs on, so the pin is cheap and the
referee is equality rather than a tolerance.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.testfunctions import Rosenbrock
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.annealed import (
    annealed_importance_sampling,
    population_annealing,
    simulated_tempering,
)
from snakes_and_ladders.sample.potts_mcmc import adapt_ladder_potts, parallel_tempering
from snakes_and_ladders.sample.schedule import (
    ExponentialTempSchedule,
    LadderTempSchedule,
    adapt_ladder,
    adapt_ladder_by_round_trips,
    ladder,
    temperatures,
)
from snakes_and_ladders.sample.tempered import (
    adapt_ladder_round_trips,
    tempered_factor_graph,
    tempered_potts_pair,
    tempered_topologies,
)
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.simulator import simulate_tree
from snakes_and_ladders.sim.topology import enumerate_topologies

from tests._fixtures import FOUR_TAXA, load_fixture

LADDER = (1.0, 1.5, 2.25)
FIELD = np.array([0.3, -0.1])
SWEEPS = 6
BAND = (0.2, 0.6)


def _graph() -> PottsGraph:
    return lattice_graph((2, 2), BoundaryCondition.OPEN, 0.5)


def _acceptance(candidate: tuple[float, ...]) -> list[float]:
    """Exchange acceptance as the ratio of each neighbouring pair: a measurement, not a model."""
    return [
        min(cold, hot) / max(cold, hot) for cold, hot in itertools.pairwise(candidate)
    ]


def _up_fraction(candidate: tuple[float, ...]) -> list[float]:
    """The up-fraction falling linearly from 1 at rung 0 to 0 at the last."""
    return [1.0 - rung / (len(candidate) - 1) for rung in range(len(candidate))]


@pytest.mark.analytic
def test_the_ladder_schedule_is_the_sequence_step_by_step() -> None:
    schedule = LadderTempSchedule(LADDER)

    assert schedule.n_steps == 3
    assert temperatures(schedule) == list(LADDER)
    assert ladder(schedule) == LADDER
    assert ladder(list(LADDER)) == LADDER
    geometric = ExponentialTempSchedule(4.0, 1.0, 5)
    assert ladder(geometric) == tuple(temperatures(geometric))


@pytest.mark.smoke
def test_the_ladder_schedule_refuses_what_every_schedule_refuses() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        LadderTempSchedule(())
    with pytest.raises(ValueError, match="positive temperature"):
        LadderTempSchedule((1.0, 0.0))
    with pytest.raises(ValueError, match="outside a schedule"):
        LadderTempSchedule(LADDER)(3)


@pytest.mark.analytic
def test_potts_tempering_reads_both_spellings_to_the_same_chains() -> None:
    graph = _graph()
    runs = [
        parallel_tempering(graph, FIELD, spelling, np.random.default_rng(0), SWEEPS)
        for spelling in (LADDER, LadderTempSchedule(LADDER))
    ]

    assert runs[0].temperatures == runs[1].temperatures == LADDER
    np.testing.assert_array_equal(runs[0].states, runs[1].states)
    np.testing.assert_array_equal(runs[0].swap_acceptance, runs[1].swap_acceptance)


@pytest.mark.analytic
def test_the_tempered_ensembles_read_both_spellings_to_the_same_draws() -> None:
    graph = _graph()
    factor = [
        tempered_factor_graph(
            from_potts(graph, FIELD), spelling, np.random.default_rng(1), SWEEPS
        )
        for spelling in (LADDER, LadderTempSchedule(LADDER))
    ]
    pair = [
        tempered_potts_pair(graph, FIELD, spelling, np.random.default_rng(2), SWEEPS)
        for spelling in (LADDER, LadderTempSchedule(LADDER))
    ]
    params = load_fixture(FOUR_TAXA)
    dataset = simulate_tree(params, np.random.default_rng(1), n_sites=30)
    alignment = dict(dataset.alignment)
    start = next(iter(enumerate_topologies(sorted(alignment))))
    topologies = [
        tempered_topologies(
            alignment, params.k, spelling, np.random.default_rng(3), 4, start=start
        )
        for spelling in (LADDER, LadderTempSchedule(LADDER))
    ]

    for ensembles in (factor, pair, topologies):
        first, second = ensembles
        assert first.temperatures == second.temperatures == LADDER
        np.testing.assert_array_equal(first.log_densities, second.log_densities)


@pytest.mark.analytic
def test_hamiltonian_tempering_reads_both_spellings_to_the_same_positions() -> None:
    objective = Rosenbrock(dimension=2)
    runs = [
        hmc.parallel_tempering(
            objective,
            spelling,
            generator=torch.Generator().manual_seed(4),
            n_rounds=5,
            step_size=0.05,
            n_steps=3,
        )
        for spelling in (LADDER, LadderTempSchedule(LADDER))
    ]

    assert torch.equal(runs[0].positions, runs[1].positions)
    assert torch.equal(runs[0].swap_acceptance, runs[1].swap_acceptance)


@pytest.mark.analytic
def test_the_warm_ups_read_both_spellings_and_the_zero_anchor_stays_refused() -> None:
    """The six ladders #834 left on a bare sequence, read from a schedule (issue #861).

    Four place alike; the two anchored at ``beta = 0`` are refused: no schedule spells it.
    """
    graph = _graph()
    warm_ups = (
        [
            adapt_ladder(_acceptance, spelling, BAND, 2, 6)
            for spelling in (LADDER, LadderTempSchedule(LADDER))
        ],
        [
            adapt_ladder_by_round_trips(_up_fraction, spelling, 1e-3, 2)
            for spelling in (LADDER, LadderTempSchedule(LADDER))
        ],
        [
            adapt_ladder_potts(
                graph, FIELD, spelling, np.random.default_rng(6), SWEEPS, BAND, 2, 6
            )
            for spelling in (LADDER, LadderTempSchedule(LADDER))
        ],
        [
            adapt_ladder_round_trips(
                graph, FIELD, spelling, np.random.default_rng(7), SWEEPS, 0.02, 2
            )
            for spelling in (LADDER, LadderTempSchedule(LADDER))
        ],
    )

    for sequence, schedule in warm_ups:
        assert sequence == schedule

    for estimator in (annealed_importance_sampling, population_annealing):
        with pytest.raises(ValueError, match="must start at beta = 0"):
            estimator(
                graph, FIELD, LadderTempSchedule(LADDER), np.random.default_rng(8), 2
            )


@pytest.mark.analytic
def test_simulated_tempering_reads_a_schedule_as_temperatures_inverted() -> None:
    # Coldest rung last: the ladder is strictly increasing in beta, so the
    # schedule that spells it runs hot to cold.
    graph = _graph()
    hot_to_cold = tuple(reversed(LADDER))
    betas = tuple(1.0 / temperature for temperature in hot_to_cold)
    first = simulated_tempering(
        graph, FIELD, betas, np.ones(3), np.random.default_rng(5), SWEEPS
    )
    second = simulated_tempering(
        graph,
        FIELD,
        LadderTempSchedule(hot_to_cold),
        np.ones(3),
        np.random.default_rng(5),
        SWEEPS,
    )

    assert first.betas == second.betas == betas
    np.testing.assert_array_equal(first.rungs, second.rungs)
