"""Correctness by distribution, not by inspection of cluster sizes.

At an enumerable lattice size the exact Boltzmann distribution is available,
so each move set is tested by whether the chain's realized visit frequencies
are drawn from it --- a chi-square goodness-of-fit at a declared significance
and chain length. A sampler with a broken accept step produces plausible
configurations, runs to completion, and fails this; it passes any test that
only checks the chain moved.

Two things this file also pins, because both were wrong while it was written
and neither would have been caught by a test that only ran the sampler:

The **field accept step**. `test_dropping_the_field_accept_step_is_caught`
replaces it with an unconditional recolouring and asserts this same test
rejects --- so the tests above are known to have the power they claim rather
than assumed to.

The **thinning**. A chi-square assumes independent draws, and successive
sweeps are not independent. Run on every sweep it rejects a *correct*
sampler: measured here, single-site at `thin = 1` returned p = 0.038 and
Swendsen-Wang p = 0.0024 on chains that are right. The thinning below is
therefore part of the test, not a speed knob, and Wolff needs more of it
because one Wolff sweep flips one cluster while the other two touch every
site.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.opt.schedule import Constant, Exponential
from snakes_and_ladders.search import potts_mcmc
from snakes_and_ladders.search.alpha_expansion import energy, iterated_conditional_modes
from snakes_and_ladders.search.potts_mcmc import (
    PottsChain,
    PottsMove,
    TemperedChains,
    anneal_potts,
    energies,
    parallel_tempering,
    sample_potts,
    tempered,
)
from snakes_and_ladders.search.statistics import (
    chi_square_p_value,
    integrated_autocorrelation_time,
)
from snakes_and_ladders.sim.canonical import (
    frustrated_triangular_lattice,
    minimum_frustrated_edges,
    planted_spin_glass,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

# Declared significance. The worst p-value over 36 runs -- six seeds across
# all three move sets, with and without a field -- was 0.0145, so 0.001 does
# not reject a correct sampler at this chain length while the ablation below
# fails it at p = 0.0.
SIGNIFICANCE = 0.001
SWEEPS = 10_000
SEED = 4242

# 2x2, two states: 16 configurations, so every cell of the chi-square has an
# expected count in the hundreds at this chain length.
SHAPE = (2, 2)
COUPLING = 0.8
NO_FIELD = np.zeros(2)
WITH_FIELD = np.array([0.6, -0.4])

# One Wolff sweep is one cluster flip; one sweep of either other move set
# touches every site. Equal `thin` would compare a decorrelated chain against
# a correlated one and reject Wolff for being thinned less.
THINNING = {
    PottsMove.SINGLE_SITE: 5,
    PottsMove.SWENDSEN_WANG: 5,
    PottsMove.WOLFF: 25,
}


def _exact_distribution(
    graph: PottsGraph, field: np.ndarray
) -> tuple[dict[tuple[int, ...], int], np.ndarray]:
    """Every configuration and its exact Boltzmann probability."""
    n_states = int(field.shape[0])
    configurations = np.array(
        list(itertools.product(range(n_states), repeat=graph.n_nodes)),
        dtype=np.int64,
    )
    weights = log_weights(graph, field, configurations)
    weights = weights - weights.max()
    probability = np.exp(weights)
    probability /= probability.sum()
    index = {tuple(row): position for position, row in enumerate(configurations)}
    return index, probability


def _goodness_of_fit(move: PottsMove, field: np.ndarray, seed: int = SEED) -> float:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, probability = _exact_distribution(graph, field)

    chain = sample_potts(
        graph,
        field,
        move,
        seed,
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING[move],
    )

    observed = np.zeros(len(probability))
    for row in chain.states:
        observed[index[tuple(row)]] += 1
    return chi_square_p_value(observed, probability * SWEEPS)


@pytest.mark.parametrize("move", list(PottsMove))
def test_the_chain_is_drawn_from_the_exact_boltzmann_distribution(
    move: PottsMove,
) -> None:
    assert _goodness_of_fit(move, NO_FIELD) > SIGNIFICANCE


@pytest.mark.parametrize("move", list(PottsMove))
def test_the_chain_is_still_exact_in_an_external_field(move: PottsMove) -> None:
    # The case the ticket exists for. Wolff's cluster construction alone does
    # not preserve detailed balance in a field, so this is where the accept
    # step is doing work and the test above is not.
    assert _goodness_of_fit(move, WITH_FIELD) > SIGNIFICANCE


@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_dropping_the_field_accept_step_is_caught(
    move: PottsMove, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Evidence that the two tests above have the power they claim. Without
    # this, "the sampler passes a chi-square" and "the chi-square could not
    # tell" are indistinguishable.
    def unconditional(
        state: np.ndarray,
        members: np.ndarray,
        field: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        state[members] = int(rng.integers(field.shape[0]))

    monkeypatch.setattr(potts_mcmc, "_recolour", unconditional)

    assert _goodness_of_fit(move, WITH_FIELD) < SIGNIFICANCE


@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_a_cluster_move_refuses_a_negative_coupling(move: PottsMove) -> None:
    # `1 - exp(-J)` is above 1 for J < 0, so it is not a probability, and an
    # antiferromagnet has no like-spin regions to flip. The algorithm does not
    # apply rather than applying badly.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, -0.5)

    with pytest.raises(ValueError, match="needs every coupling >= 0"):
        sample_potts(graph, NO_FIELD, move, SEED, 10)


def test_single_site_still_runs_on_a_negative_coupling() -> None:
    # The refusal is a property of the cluster construction, not of the model.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, -0.5)

    chain = sample_potts(graph, NO_FIELD, PottsMove.SINGLE_SITE, SEED, 10)

    assert chain.states.shape == (10, graph.n_nodes)


# The exact q-state Potts transition on a square lattice, where the
# correlation length diverges and single-site updates slow critically. Pinned
# as a closed form so the comparison happens where the physics says it is
# interesting, not at a coupling that flatters cluster updates.
TRANSITION = math.log(1.0 + math.sqrt(3.0))


def _autocorrelation_in_site_updates(move: PottsMove, extent: int) -> float:
    """Energy autocorrelation time, normalized to the work a sweep costs.

    A Wolff sweep flips one cluster; the other two touch every site. Reporting
    all three in sweeps would make Wolff look free, so each is scaled by the
    sites its sweep actually touched.
    """
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, TRANSITION)
    field = np.zeros(3)
    n_sweeps = 12_000 if move is PottsMove.WOLFF else 1_500

    chain: PottsChain = sample_potts(
        graph, field, move, 7, n_sweeps, burn_in=n_sweeps // 5
    )
    tau = integrated_autocorrelation_time(energies(graph, field, chain.states))
    return tau * chain.mean_cluster_size / graph.n_nodes


def test_cluster_updates_decorrelate_faster_at_the_transition() -> None:
    # The reason for having them, as a number rather than an assertion. At
    # 12x12 the measured times are roughly 6.7, 4.2 and 2.3 site-updates for
    # single-site, Swendsen-Wang and Wolff. Asserted as an ordering rather
    # than as values: the gap widens with lattice extent, so pinning a ratio
    # here would pin a finite-size effect.
    single = _autocorrelation_in_site_updates(PottsMove.SINGLE_SITE, 12)
    swendsen_wang = _autocorrelation_in_site_updates(PottsMove.SWENDSEN_WANG, 12)
    wolff = _autocorrelation_in_site_updates(PottsMove.WOLFF, 12)

    assert swendsen_wang < single
    assert wolff < single


def test_a_wolff_cluster_is_smaller_than_the_lattice_but_larger_than_a_site() -> None:
    # What makes the normalization above necessary, pinned so a change that
    # made every cluster a single site -- which would silently turn Wolff into
    # an expensive single-site sampler -- is visible.
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, TRANSITION)

    chain = sample_potts(graph, np.zeros(3), PottsMove.WOLFF, 7, 2_000, burn_in=200)

    assert 1.0 < chain.mean_cluster_size < graph.n_nodes


# --- temperature ------------------------------------------------------------

TEMPERATURES = [2.0, 0.5]


def _tempered_exact_distribution(
    graph: PottsGraph, field: np.ndarray, temperature: float
) -> tuple[dict[tuple[int, ...], int], np.ndarray]:
    """``exp(-E / T)`` from the *unscaled* model: the oracle a tempered chain is held to.

    Computed from `log_weights` of the model as declared, divided by the
    temperature, so it shares nothing with `tempered` -- a chain that ran the
    scaled model wrongly would be caught here and not by a test that scaled
    the oracle the same way.
    """
    n_states = int(field.shape[0])
    configurations = np.array(
        list(itertools.product(range(n_states), repeat=graph.n_nodes)),
        dtype=np.int64,
    )
    log_probability = log_weights(graph, field, configurations) / temperature
    probability = np.exp(log_probability - log_probability.max())
    probability /= probability.sum()
    index = {tuple(row): position for position, row in enumerate(configurations)}
    return index, probability


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_tempering_is_model_scaling_exactly(temperature: float) -> None:
    # The consistency check the model itself provides: the coupling absorbs
    # beta, so the energy of the scaled model is the energy over T, and the
    # deviation is 0.0 rather than a tolerance -- a division on each term.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    scaled_graph, scaled_field = tempered(graph, WITH_FIELD, temperature)

    scaled = energies(scaled_graph, scaled_field, configurations)
    expected = energies(graph, WITH_FIELD, configurations) / temperature

    assert np.abs(scaled - expected).max() == 0.0


@pytest.mark.parametrize("move", list(PottsMove))
@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_a_tempered_chain_is_drawn_from_the_tempered_boltzmann_distribution(
    move: PottsMove, temperature: float
) -> None:
    # Hot and cold, in a field, for every move set: the bond probabilities
    # and the field accept step are tempered by the same division as the heat
    # bath, and this is what says so. Realized p-values over two seeds range
    # 0.016 to 0.89 against the 0.001 significance the untempered tests use.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, probability = _tempered_exact_distribution(graph, WITH_FIELD, temperature)

    chain = sample_potts(
        graph,
        WITH_FIELD,
        move,
        SEED,
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING[move],
        temperature=temperature,
    )
    observed = np.zeros(len(probability))
    for row in chain.states:
        observed[index[tuple(row)]] += 1

    assert chi_square_p_value(observed, probability * SWEEPS) > SIGNIFICANCE


def test_a_non_positive_temperature_is_refused() -> None:
    # At zero the heat bath is an argmin and the chain is a descent that
    # samples nothing; a negative temperature inverts the model.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="temperature must be positive"):
        sample_potts(graph, NO_FIELD, PottsMove.SINGLE_SITE, SEED, 10, temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be positive"):
        tempered(graph, NO_FIELD, -1.0)


def test_annealing_reaches_the_closed_form_ground_energy_where_descent_does_not() -> (
    None
):
    # The optimizer built from the sampler, against the one frustrated
    # instance with a ground-state energy known at every size: the periodic
    # triangular antiferromagnet, where at least one edge in three is
    # unsatisfied (`sim.canonical.minimum_frustrated_edges`). At 9x9 over 20
    # seeds and 200 sweeps: annealing 20/20, single-site descent (ICM) 2/20,
    # and the same 200 sweeps at a *constant* temperature of 1 -- the control
    # that separates the schedule from the wandering -- 7/20.
    #
    # What this does not say: that annealing beats descent at equal budget.
    # ICM converges in 2.6 sweeps here, so 200 sweeps buy 78 restarts, and
    # the best of 78 also reaches the ground state 20/20. The instance is too
    # easy for restarts to lose on; the comparison at equal evaluations on
    # instances where they might is #267's second pull request.
    graph = frustrated_triangular_lattice((9, 9), BoundaryCondition.PERIODIC, -1.0)
    field = np.zeros(2)
    ground = float(minimum_frustrated_edges(graph))  # |J| = 1
    schedule = Exponential(2.0, 0.05, 200)

    annealed = [anneal_potts(graph, field, schedule, seed) for seed in range(20)]
    constant = [
        anneal_potts(graph, field, Constant(1.0, 200), seed) for seed in range(20)
    ]
    descended = [
        iterated_conditional_modes(graph, field, 2, seed)[1] for seed in range(20)
    ]

    for result in annealed:
        # The reported energy is the energy of the reported labelling, in the
        # convention the exact solvers use, and never below the closed form.
        assert result.energy == pytest.approx(energy(graph, field, result.labelling))
        assert result.energy >= ground - 1e-12
        assert result.n_sweeps == 200
    annealed_hits = sum(abs(result.energy - ground) < 1e-12 for result in annealed)
    constant_hits = sum(abs(result.energy - ground) < 1e-12 for result in constant)
    descent_hits = sum(abs(value - ground) < 1e-12 for value in descended)

    assert annealed_hits >= 18
    assert annealed_hits > constant_hits
    assert descent_hits < annealed_hits


# --- parallel tempering -----------------------------------------------------

LADDER = (4.0, 2.0, 1.0)


def _replica_p_values(
    run: TemperedChains, graph: PottsGraph, field: np.ndarray
) -> list[float]:
    """Each replica's chi-square against its own tempered target, from the unscaled model."""
    p_values = []
    for replica, temperature in enumerate(run.temperatures):
        index, probability = _tempered_exact_distribution(graph, field, temperature)
        observed = np.zeros(len(probability))
        for row in run.states[:, replica]:
            observed[index[tuple(row)]] += 1
        p_values.append(chi_square_p_value(observed, probability * run.states.shape[0]))
    return p_values


def test_every_replica_is_drawn_from_its_own_tempered_distribution() -> None:
    # The oracle the plan named: the joint target is a product of tempered
    # marginals, so with exchanges *on* each replica must still pass the
    # chi-square against exp(-E / T_r) enumerated from the unscaled model. A
    # wrong exchange ratio contaminates the cold replica with hot
    # configurations and this is what says so. Realized p-values over two
    # seeds: 0.024 to 0.70 at the 0.001 significance; exchange acceptance
    # 0.78 and 0.57 for the two pairs, so the exchanges are actually
    # happening and the test has power.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    run = parallel_tempering(
        graph, WITH_FIELD, LADDER, SEED, SWEEPS, burn_in=SWEEPS // 10, thin=5
    )

    assert run.states.shape == (SWEEPS, len(LADDER), graph.n_nodes)
    assert bool((run.swap_acceptance > 0.3).all()), run.swap_acceptance
    assert min(_replica_p_values(run, graph, WITH_FIELD)) > SIGNIFICANCE


def test_omitting_the_exchange_term_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # The negative case, paired with the positive one above so the two are
    # distinguishable. An exchange that ignores (beta_i - beta_j)(E_i - E_j)
    # still runs and still mixes -- every exchange is accepted -- and every
    # replica's marginal is the wrong distribution: realized p = 0.0 at all
    # three temperatures.
    def always_exchange(*_: float) -> float:
        return 0.0

    monkeypatch.setattr(potts_mcmc, "_swap_log_ratio", always_exchange)
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    run = parallel_tempering(
        graph, WITH_FIELD, LADDER, SEED, SWEEPS, burn_in=SWEEPS // 10, thin=5
    )

    assert bool((run.swap_acceptance == 1.0).all())
    assert max(_replica_p_values(run, graph, WITH_FIELD)) < SIGNIFICANCE


def test_replicas_draw_from_separate_streams_and_one_seed_reproduces_them() -> None:
    # Two replicas at the *same* temperature with no field would be identical
    # chains if they shared a stream, and the whole point would be lost while
    # every diagnostic looked healthy. Spawned children differ; the parent
    # seed still reproduces the run bitwise.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    first = parallel_tempering(graph, NO_FIELD, (1.0, 1.0), 3, 200)
    second = parallel_tempering(graph, NO_FIELD, (1.0, 1.0), 3, 200)

    assert not np.array_equal(first.states[:, 0], first.states[:, 1])
    assert np.array_equal(first.states, second.states)


def test_the_best_configuration_is_the_lowest_energy_any_replica_visited() -> None:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    run = parallel_tempering(graph, WITH_FIELD, LADDER, SEED, 300)

    visited = energies(graph, WITH_FIELD, run.states.reshape(-1, graph.n_nodes))
    assert run.best_energy == pytest.approx(
        energies(graph, WITH_FIELD, run.best[None])[0]
    )
    assert run.best_energy <= visited.min() + 1e-12


def test_a_ladder_of_one_or_a_cold_temperature_is_refused() -> None:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="at least two temperatures"):
        parallel_tempering(graph, NO_FIELD, (1.0,), SEED, 10)
    with pytest.raises(ValueError, match="must be positive"):
        parallel_tempering(graph, NO_FIELD, (1.0, 0.0), SEED, 10)


def test_tempering_and_annealing_beat_restarts_at_equal_budget_on_the_glass() -> None:
    # The comparison the ticket asked for, on the instance where restarts can
    # lose: the planted Viana-Bray spin glass, 60 sites at mean degree 4 and
    # frustration 0.2, whose planted energy upper-bounds the ground state and
    # whose ground state enumeration cannot reach. Budget: 400 heat-bath
    # sweeps per method -- annealing spends them on one chain, tempering on
    # four replicas of 100, and single-site descent on 100 restarts of at
    # most 4 sweeps (it converges in 2 to 4). Realized over 12 instances,
    # against the best energy any method found: annealing and tempering
    # 12/12, restarts 5/12 with a mean gap of 0.75; at frustration 0.35 it
    # is 9/12, 9/12 against 5/12, and at 100 sites and degree 6, 7/12 and
    # 8/12 against 2/12. The plan predicted tempering would be hard to
    # justify at these sizes; it is not, and the prediction is retracted.
    #
    # Asserted at the margin the measurement supports, not the measurement:
    # restarts below both, and the two tempered methods at or below the
    # planted energy on every instance (restarts miss it on 1 of 12).
    budget, n_instances = 400, 12
    ladder = (2.0, 1.2, 0.7, 0.4)
    best: dict[str, list[float]] = {"restarts": [], "anneal": [], "tempering": []}
    planted = []
    for seed in range(n_instances):
        instance = planted_spin_glass(60, 4.0, 0.2, np.random.default_rng(1000 + seed))
        graph, field = instance.graph, np.zeros(2)
        planted.append(instance.planted_energy)
        best["restarts"].append(
            min(
                iterated_conditional_modes(
                    graph, field, 2, 5000 * seed + r, max_sweeps=4
                )[1]
                for r in range(budget // 4)
            )
        )
        best["anneal"].append(
            anneal_potts(graph, field, Exponential(2.0, 0.05, budget), seed).energy
        )
        best["tempering"].append(
            parallel_tempering(
                graph, field, ladder, seed, budget // len(ladder)
            ).best_energy
        )

    known = np.min(np.array(list(best.values())), axis=0)
    hits = {
        name: int((np.array(values) <= known + 1e-9).sum())
        for name, values in best.items()
    }

    for name in ("anneal", "tempering"):
        assert bool((np.array(best[name]) <= np.array(planted) + 1e-9).all()), name
    assert hits["restarts"] < hits["anneal"], hits
    assert hits["restarts"] < hits["tempering"], hits
    assert hits["anneal"] >= 10, hits
    assert hits["tempering"] >= 10, hits
