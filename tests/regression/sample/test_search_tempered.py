"""The tempered ensemble's marginal weight, held to enumeration over 20 seeds.

The replica at temperature one of a replica-exchange run over topologies and
over labellings estimates the enumerated weight -- the flat-prior weight over
fitted likelihoods of the four-taxon fixture's three topologies, the
Boltzmann weight of a 2 x 2 Potts lattice's labellings, the path posterior
of the ambiguous chain's decodings -- within a Monte Carlo tolerance stated
here and measured over 20 seeds, with the exchange acceptance and
autocorrelation time that license calling it a posterior weight reported
beside it (issue #331).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from sal.backend import Backend
from sal.likelihood.hmm_paths import emission_log_density
from sal.likelihood.potts import log_weights
from sal.sample.potts_mcmc import (
    PottsMove,
    adapt_ladder_potts,
    houdayer_move,
    parallel_tempering,
    sweep_for,
)
from sal.sample.statistics import chi_square_p_value, sign_test_p_value
from sal.sample.tempered import (
    TemperedEnsemble,
    adapt_ladder_round_trips,
    round_trip_time,
    round_trips,
    tempered_factor_graph,
    tempered_potts_pair,
    tempered_topologies,
    up_fraction,
)
from sal.search.support import (
    SupportKind,
    TemperedSupport,
    enumerated_labelling_support,
    enumerated_support,
    tempered_labelling_support,
    tempered_topology_support,
)
from sal.sim.canonical import (
    AMBIGUOUS_OBSERVATIONS,
    ambiguous_hmm,
    frustrated_triangular_lattice,
)
from sal.sim.factor_graph import FactorGraph, from_hmm, from_potts
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, site_field
from sal.sim.simulator import simulate_tree
from sal.sim.topology import enumerate_topologies, leaf_bipartitions

from tests._fixtures import FOUR_TAXA, load_fixture

#: The ladder every ensemble here runs on, and its length. Geometric with
#: ratio 2 and three rungs: enough for the exchange acceptance to sit inside
#: (0.5, 0.95) on every instance below, and no rung redundant.
LADDER = (1.0, 2.0, 4.0)
N_SWEEPS = 1000
BURN_IN = 100
SEEDS = 20

#: Over 20 seeds the worst single-seed weight deviation was 0.040
#: (topologies, 30 sites), 0.029 (lattice), 0.024 (chain); means within 0.003.
#: Asserted at 1.5x the first and 3x the second.
SEED_TOLERANCE = 0.06
MEAN_TOLERANCE = 0.01

#: What "mixed" means here: no adjacent pair exchanges less than half the
#: time or more than 95 percent of it, and no indicator series is correlated
#: past one recorded sweep. Realized: acceptance 0.56 to 0.93 across the
#: instances, autocorrelation times 0.50 to 0.76 recorded sweeps.
ACCEPTANCE = (0.5, 0.95)
AUTOCORRELATION = 1.0


def _four_taxa() -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(FOUR_TAXA)
    dataset = simulate_tree(params, np.random.default_rng(1), n_sites=30)
    return dict(dataset.alignment), params.k


def _potts_lattice() -> FactorGraph:
    return from_potts(
        lattice_graph((2, 2), BoundaryCondition.OPEN, 0.8), np.array([0.6, -0.4])
    )


def _ambiguous_chain() -> FactorGraph:
    params = ambiguous_hmm()
    return from_hmm(
        np.log(params.initial),
        np.log(params.transition),
        emission_log_density(params, AMBIGUOUS_OBSERVATIONS),
    )


def _mixed(support: TemperedSupport) -> None:
    assert all(ACCEPTANCE[0] <= a <= ACCEPTANCE[1] for a in support.swap_acceptance), (
        support.swap_acceptance
    )
    assert support.autocorrelation_time <= AUTOCORRELATION, support.autocorrelation_time
    assert support.n_recorded == N_SWEEPS
    assert support.kind is SupportKind.TEMPERED


def _held_to(
    estimates: np.ndarray, exact: np.ndarray, supports: list[TemperedSupport]
) -> None:
    assert estimates.shape == (SEEDS, len(exact))
    assert np.abs(estimates - exact).max() < SEED_TOLERANCE, np.abs(estimates - exact)
    assert np.abs(estimates.mean(axis=0) - exact).max() < MEAN_TOLERANCE
    for support in supports:
        _mixed(support)


@pytest.mark.oracle
def test_the_tempered_weight_of_the_four_taxon_topologies_is_the_enumerated_one() -> (
    None
):
    # 30 sites, so the three weights are 0.66, 0.17, 0.17 rather than a
    # point mass -- the two wrong topologies tie because each fits best with
    # its internal branch at zero, the star. Three fits in all, cached.
    alignment, k = _four_taxa()
    topologies = list(enumerate_topologies(sorted(alignment)))
    exact = np.array([enumerated_support(t, alignment, k).weight for t in topologies])
    assert exact.max() < 0.8

    cache: dict[frozenset[frozenset[str]], float] = {}
    estimates = np.zeros((SEEDS, 3))
    supports = []
    for seed in range(SEEDS):
        ensemble = tempered_topologies(
            alignment,
            k,
            LADDER,
            np.random.default_rng(seed),
            N_SWEEPS,
            BURN_IN,
            start=topologies[0],
            scores=cache,
        )
        for column, topology in enumerate(topologies):
            support = tempered_topology_support(topology, alignment, k, ensemble)
            estimates[seed, column] = support.weight
            supports.append(support)
            assert support.log_score == cache[leaf_bipartitions(topology)]
            assert support.n_candidates == 3

    assert len(cache) == 3
    _held_to(estimates, exact, supports)
    best = int(exact.argmax())
    assert all(s.margin > 0 for s in supports[best::3])


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("build", "labellings"),
    [
        (_potts_lattice, ([0, 0, 0, 0], [1, 0, 0, 0])),
        (_ambiguous_chain, ([0, 0, 0, 0, 0], [0, 1, 0, 1, 0])),
    ],
    ids=["potts", "hmm"],
)
def test_the_tempered_weight_of_a_labelling_is_the_enumerated_one(
    build: Callable[[], FactorGraph], labellings: tuple[list[int], ...]
) -> None:
    # The ground state and a one-flip excitation of the lattice (weights
    # 0.68 and 0.05); the Viterbi and posterior-decoded paths of the chain
    # (0.14 and 0.08). One ensemble per seed serves every labelling.
    graph = build()
    states = [np.array(labelling) for labelling in labellings]
    exact = np.array([enumerated_labelling_support(graph, s).weight for s in states])

    estimates = np.zeros((SEEDS, len(states)))
    supports = []
    for seed in range(SEEDS):
        ensemble = tempered_factor_graph(
            graph, LADDER, np.random.default_rng(seed), N_SWEEPS, BURN_IN
        )
        for column, state in enumerate(states):
            support = tempered_labelling_support(graph, state, ensemble)
            estimates[seed, column] = support.weight
            supports.append(support)
            assert support.log_score == graph.log_density(
                {v.name: int(x) for v, x in zip(graph.variables, state, strict=True)}
            )
            assert support.n_candidates <= int(
                np.prod([v.cardinality for v in graph.variables])
            )

    _held_to(estimates, exact, supports)


@pytest.mark.smoke
def test_a_seed_reproduces_the_run_and_every_recorded_density_is_its_structure_s_score() -> (
    None
):
    # One parent generator: one seed reproduces the run; per-replica scores are
    # keyed so `tempered_topology_support` reads them without a fit.
    alignment, k = _four_taxa()
    start = next(enumerate_topologies(sorted(alignment)))
    cache: dict[frozenset[frozenset[str]], float] = {}
    first = tempered_topologies(
        alignment, k, LADDER, np.random.default_rng(3), 50, start=start, scores=cache
    )
    again = tempered_topologies(
        alignment, k, LADDER, np.random.default_rng(3), 50, start=start, scores=cache
    )

    assert first.keys == again.keys
    assert np.array_equal(first.log_densities, again.log_densities)
    assert first.log_densities.shape == (50, 3)
    assert first.swap_acceptance.shape == (2,)
    assert first.temperatures == LADDER
    assert set(first.scores) <= set(cache)
    for replica, keys in enumerate(first.keys):
        for sweep, key in enumerate(keys):
            assert isinstance(key, frozenset)
            assert (
                first.log_densities[sweep, replica] == first.scores[key] == cache[key]
            )


@pytest.mark.smoke
def test_an_unusable_ladder_and_a_ladder_without_temperature_one_are_refused() -> None:
    graph = _potts_lattice()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least two temperatures"):
        tempered_factor_graph(graph, (1.0,), rng, 10)
    with pytest.raises(ValueError, match="positive temperature"):
        tempered_factor_graph(graph, (1.0, 0.0), rng, 10)
    with pytest.raises(ValueError, match="n_sweeps"):
        tempered_factor_graph(graph, (1.0, 2.0), rng, 0)
    alignment, k = _four_taxa()
    with pytest.raises(ValueError, match="at least two temperatures"):
        tempered_topologies(
            alignment, k, (1.0,), rng, 10, start=next(enumerate_topologies("ABCD"))
        )

    hot = tempered_factor_graph(graph, (2.0, 4.0), rng, 10)
    assert isinstance(hot, TemperedEnsemble)
    with pytest.raises(ValueError, match="no replica at temperature 1.0"):
        tempered_labelling_support(graph, np.zeros(4, dtype=np.int64), hot)
    with pytest.raises(ValueError, match="inside its cardinality"):
        tempered_labelling_support(graph, np.array([0, 0, 0, 2]), hot)


# --- the pair ensemble and its round trips ----------------------------------
#
# Issue #756. The ladder carries the pair; exchanges use its summed energy.
# Read as round trips: a per-pair acceptance can look healthy with no crossing.

#: The ladder the round-trip readings run on: ten rungs, ratio 1.32, from 0.2
#: to 2.43. Denser than `LADDER` above because the pair's energy is the sum of
#: two, so its variance is twice a single replica's and the acceptance falls
#: with it; realized 0.18 to 0.82 across the rungs of both instances.
PAIR_LADDER = tuple(round(0.2 * 1.32**rung, 4) for rung in range(10))

#: The 2x2 open antiferromagnet the pair ensemble's marginal is enumerated
#: against: 16 configurations, and a negative coupling, which is the case
#: `sample_potts` refuses both Fortuin-Kasteleyn cluster moves on.
PAIR_COUPLING = -0.5
PAIR_SWEEPS = 4_000
PAIR_THIN = 5
PAIR_SIGNIFICANCE = 0.001

#: Eight paired seeds: an exact sign test gives p = 0.0703 at seven of eight
#: and 0.0078 at eight, so a direction is refutable.
ROUND_TRIP_SEEDS = 8
ROUND_TRIP_SWEEPS = 2_000

#: Where the paired sign test is read. A failure to reject is what is wanted,
#: so the number is stated rather than the usual 0.05 being assumed.
SEPARATION = 0.05


@pytest.mark.analytic
@pytest.mark.critical
def test_a_round_trip_is_the_cold_rung_reached_through_the_hot_one() -> None:
    """The definition, on traces whose answer is counted by hand.

    Rattling at the cold end scores nothing; each top-and-back trip scores one.
    """
    trace = np.array(
        [
            # rattles at the cold end; reaches the top once and returns twice
            # over; ends at the top having not come back.
            [0, 0, 0],
            [1, 1, 2],
            [0, 2, 1],
            [1, 1, 2],
            [0, 0, 1],
            [1, 2, 2],
            [0, 0, 2],
        ]
    )

    assert round_trips(trace).tolist() == [0, 2, 0]


@pytest.mark.oracle
@pytest.mark.release
def test_each_replica_of_the_cold_rungs_pair_is_the_enumerated_boltzmann_law() -> None:
    """The ensemble's marginal at temperature one, against enumeration.

    The tempered pair path; p per replica: 0.4404 / 0.1214 and 0.8203 / 0.3536.
    """
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, PAIR_COUPLING)
    field = np.array([0.6, -0.4])
    configurations = [
        tuple(values) for values in itertools.product(range(2), repeat=graph.n_nodes)
    ]
    weights = log_weights(graph, field, np.array(configurations, dtype=np.int64))
    exact = np.exp(weights - weights.max())
    exact /= exact.sum()
    index = {values: position for position, values in enumerate(configurations)}

    ensemble = tempered_potts_pair(
        graph,
        field,
        (1.0, 2.0, 4.0),
        np.random.default_rng(1),
        PAIR_SWEEPS,
        burn_in=PAIR_SWEEPS // 10,
        thin=PAIR_THIN,
    )

    cold = ensemble.replica_at(1.0)
    for half in (slice(0, graph.n_nodes), slice(graph.n_nodes, None)):
        observed = np.zeros(len(configurations))
        for key in ensemble.keys[cold]:
            # The pair's key is both replicas' labels concatenated, so a half
            # of it is one replica's own configuration.
            pair = cast(tuple[int, ...], key)
            observed[index[pair[half]]] += 1
        p_value = chi_square_p_value(observed, exact * observed.sum())
        assert p_value > PAIR_SIGNIFICANCE, p_value


@pytest.mark.analytic
@pytest.mark.release
def test_the_round_trip_does_not_separate_with_houdayers_move() -> None:
    """The measurement issue #756's plan asks for, and it does not separate.

    12x12 triangular antiferromagnet, sign test over paired seeds: 1,115.5
    sweeps without the move against 1,090.7 with, five of eight shorter,
    p = 0.6875, at 3.96x the wall (17.0 s against 67.3 s);
    `docs/experiments/022-cluster-moves-for-frustrated-lattices.md`.
    """
    graph = frustrated_triangular_lattice((12, 12), BoundaryCondition.PERIODIC, -1.0)
    times = {}
    for houdayer in (False, True):
        readings = []
        for seed in range(ROUND_TRIP_SEEDS):
            ensemble = tempered_potts_pair(
                graph,
                np.zeros(2),
                PAIR_LADDER,
                np.random.default_rng(seed),
                ROUND_TRIP_SWEEPS,
                burn_in=ROUND_TRIP_SWEEPS // 10,
                houdayer=houdayer,
            )
            assert round_trips(ensemble.walkers).sum() > 0
            readings.append(ensemble.round_trip_time)
        times[houdayer] = np.array(readings)

    difference = times[True] - times[False]

    assert sign_test_p_value(difference) > SEPARATION, difference


@pytest.mark.analytic
@pytest.mark.critical
def test_the_overlap_defect_percolates_on_the_frustrated_lattice() -> None:
    """Why the round trip does not move: the cluster is most of the defect.

    Defect 65-72 of 144 sites, largest component 43-59: a near-global swap.
    """
    graph = frustrated_triangular_lattice((12, 12), BoundaryCondition.PERIODIC, -1.0)
    rows = site_field(np.zeros(2), graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    sweep = sweep_for(
        PottsMove.SINGLE_SITE, graph, rows, offsets, neighbours, couplings, Backend.RUST
    )

    for temperature in (0.2, 0.46, 1.06):
        rng = np.random.default_rng(7)
        pair = [
            np.ascontiguousarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
            for _ in range(2)
        ]
        defects, clusters = [], []
        for step in range(300):
            for replica in pair:
                sweep(replica, rng, 1.0 / temperature)
            defect = int((pair[0] != pair[1]).sum())
            size = houdayer_move(pair[0], pair[1], offsets, neighbours, rng)
            if step >= 100:
                defects.append(defect)
                clusters.append(size)

        assert 60 < float(np.mean(defects)) < 80, (temperature, np.mean(defects))
        assert float(np.mean(clusters)) > 0.6 * float(np.mean(defects)), (
            temperature,
            np.mean(clusters),
        )


@pytest.mark.smoke
def test_the_pair_ensemble_refuses_houdayers_move_above_two_states() -> None:
    """The restriction stated where the ensemble is configured.

    Houdayer's overlap is Ising (#756): three states are refused at configuration.
    """
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, PAIR_COUPLING)

    with pytest.raises(ValueError, match="Ising overlap"):
        tempered_potts_pair(
            graph, np.zeros(3), (1.0, 2.0), np.random.default_rng(0), 10
        )


# --- placing the ladder by its round trips (#756) ---------------------------
#
# `adapt_ladder_round_trips` places rungs by the up-fraction, flattening the
# diffusivity (Katzgraber et al. 2006); read as round-trip time, paired by seed.

#: Warm-up for both criteria. 1,000 sweeps: at 400 over four rounds the placed
#: ladder ranged 1,469 to 5,357 sweeps per trip against geometric's 1,690
#: (`docs/experiments/023-placing-the-tempering-ladder.md`).
WARM_UP_SWEEPS = 1_000
WARM_UP_ROUNDS = 3
WARM_UP_SEED = 11

#: The acceptance criterion's band and budget, above the starting length so
#: insertion can act; its length is the round-trip criterion's too.
WARM_UP_BAND = (0.2, 0.6)
WARM_UP_REPLICAS = 16

#: The readings: recorded sweeps per replica, and the paired seeds.
PLACEMENT_SWEEPS = 2_000


def _geometric(cold: float, hot: float, n_rungs: int) -> tuple[float, ...]:
    return tuple(
        cold * (hot / cold) ** (rung / (n_rungs - 1)) for rung in range(n_rungs)
    )


def _placed_ladders(
    graph: PottsGraph, cold: float, hot: float, n_start: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The two ladders, placed from the same warm-up budget and the same seed."""
    accepted = adapt_ladder_potts(
        graph,
        np.zeros(2),
        _geometric(cold, hot, n_start),
        np.random.default_rng(WARM_UP_SEED),
        WARM_UP_SWEEPS,
        WARM_UP_BAND,
        WARM_UP_ROUNDS,
        WARM_UP_REPLICAS,
    )
    feedback = adapt_ladder_round_trips(
        graph,
        np.zeros(2),
        _geometric(cold, hot, len(accepted.temperatures)),
        np.random.default_rng(WARM_UP_SEED),
        WARM_UP_SWEEPS,
        0.02,
        WARM_UP_ROUNDS,
    )
    assert len(feedback.temperatures) == len(accepted.temperatures)
    return accepted.temperatures, feedback.temperatures


def _round_trip_times(graph: PottsGraph, ladder: tuple[float, ...]) -> np.ndarray:
    """Round-trip time per seed on one ladder, the seeds paired across ladders."""
    return np.array(
        [
            round_trip_time(
                parallel_tempering(
                    graph,
                    np.zeros(2),
                    ladder,
                    np.random.default_rng(seed),
                    PLACEMENT_SWEEPS,
                    burn_in=PLACEMENT_SWEEPS // 10,
                ).walkers
            )
            for seed in range(ROUND_TRIP_SEEDS)
        ]
    )


@pytest.mark.analytic
@pytest.mark.critical
def test_the_up_fraction_labels_a_walker_by_the_end_it_last_touched() -> None:
    """The measurement the placement reads, on the trace whose answer is counted by hand.

    Rung 0: 8 up; rung 1: 4 up, 3 down; top: 6, none up. So ``f`` is 1 and 0 at the ends.
    """
    trace = np.array(
        [
            [0, 0, 0],
            [1, 1, 2],
            [0, 2, 1],
            [1, 1, 2],
            [0, 0, 1],
            [1, 2, 2],
            [0, 0, 2],
        ]
    )

    assert up_fraction(trace).tolist() == [1.0, 4.0 / 7.0, 0.0]


@pytest.mark.smoke
@pytest.mark.critical
def test_the_tempering_trace_holds_one_walker_per_rung() -> None:
    """A swap moves configurations between rungs; it does not create them.

    Every recorded sweep is a permutation of rungs, or trip times are wrong.
    """
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.4)

    run = parallel_tempering(
        graph, np.zeros(2), (0.5, 1.0, 2.0), np.random.default_rng(0), 50
    )

    assert run.walkers.shape == (50, 3)
    for row in run.walkers.tolist():
        assert sorted(row) == [0, 1, 2]
    assert round_trip_time(run.walkers) == pytest.approx(
        150.0 / float(round_trips(run.walkers).sum())
    )


@pytest.mark.analytic
@pytest.mark.release
def test_the_round_trip_placed_ladder_does_not_separate_at_the_transition() -> None:
    """The measurement issue #756's plan asks for, on the 16x16 square at ``J_c``.

    ``J_c = ln(1 + sqrt(2))``, 12 rungs from 0.6 to 2 each: 1,053.7 against
    1,194.4 sweeps per trip, six of eight shorter, p = 0.2891; again 1,053.1
    against 1,092.8, p = 1.0 (load 3.86, 12.2 s). No separation;
    `docs/experiments/023-...` reports both beating geometric.
    """
    graph = lattice_graph((16, 16), BoundaryCondition.OPEN, critical_coupling(2))
    accepted, feedback = _placed_ladders(graph, 0.6, 2.0, 10)

    difference = _round_trip_times(graph, feedback) - _round_trip_times(graph, accepted)

    assert sign_test_p_value(difference) > SEPARATION, difference


@pytest.mark.analytic
@pytest.mark.release
def test_the_round_trip_placed_ladder_does_not_separate_on_the_frustrated_lattice() -> (
    None
):
    """The same comparison on the 12x12 triangular antiferromagnet of step 5.

    9 rungs, 0.2 to 2.4333: 366.6 against 360.1, five of eight, p = 0.7266 (again 353.4).
    """
    graph = frustrated_triangular_lattice((12, 12), BoundaryCondition.PERIODIC, -1.0)
    accepted, feedback = _placed_ladders(graph, 0.2, 2.4333, 10)

    difference = _round_trip_times(graph, feedback) - _round_trip_times(graph, accepted)

    assert sign_test_p_value(difference) > SEPARATION, difference
