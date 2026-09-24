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
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.hmm_paths import emission_log_density
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.sample.potts_mcmc import (
    PottsMove,
    adapt_ladder_potts,
    houdayer_move,
    parallel_tempering,
    sweep_for,
)
from snakes_and_ladders.sample.statistics import chi_square_p_value, sign_test_p_value
from snakes_and_ladders.sample.tempered import (
    TemperedEnsemble,
    adapt_ladder_round_trips,
    round_trip_time,
    round_trips,
    tempered_factor_graph,
    tempered_potts_pair,
    tempered_topologies,
    up_fraction,
)
from snakes_and_ladders.search.support import (
    SupportKind,
    TemperedSupport,
    enumerated_labelling_support,
    enumerated_support,
    tempered_labelling_support,
    tempered_topology_support,
)
from snakes_and_ladders.sim.canonical import (
    AMBIGUOUS_OBSERVATIONS,
    ambiguous_hmm,
    frustrated_triangular_lattice,
)
from snakes_and_ladders.sim.factor_graph import FactorGraph, from_hmm, from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling, site_field
from snakes_and_ladders.sim.simulator import simulate_tree
from snakes_and_ladders.sim.topology import enumerate_topologies, leaf_bipartitions

from tests._fixtures import FOUR_TAXA, load_fixture

#: The ladder every ensemble here runs on, and its length. Geometric with
#: ratio 2 and three rungs: enough for the exchange acceptance to sit inside
#: (0.5, 0.95) on every instance below, and no rung redundant.
LADDER = (1.0, 2.0, 4.0)
N_SWEEPS = 1000
BURN_IN = 100
SEEDS = 20

#: The Monte Carlo tolerance, measured: over 20 seeds the largest deviation
#: of one seed's weight from the enumerated one was 0.040 on the topologies
#: (30 sites), 0.029 on the lattice and 0.024 on the chain, and the mean
#: over seeds was within 0.003 on every instance. Asserted at 1.5 times the
#: largest single-seed deviation and 3 times the largest mean deviation.
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
    # One parent generator spawns the replicas and draws the exchanges, so
    # one seed reproduces the run; and the densities recorded per replica
    # are the scores the ensemble reports per structure, by key, which is
    # what lets `tempered_topology_support` read a visited topology's score
    # without a fit.
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
# Issue #756. Houdayer's move acts on two replicas at one temperature, so the
# state the ladder carries is the pair and the exchange ratio takes the pair's
# summed energy. What that buys is read as a *round trip* --- a walker from the
# cold rung to the hot one and back --- because an exchange acceptance is a
# per-pair number a ladder can look healthy in while nothing crosses it.

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

#: Seeds behind the round-trip comparison, and the recorded sweeps each runs.
#: Eight is the smallest paired sample whose exact sign test can reject at
#: `SEPARATION`: seven of eight one way is `p = 0.0703` and eight is
#: `p = 0.0078`, so a real direction is refutable here and a null result is
#: not merely a shortage of seeds.
ROUND_TRIP_SEEDS = 8
ROUND_TRIP_SWEEPS = 2_000

#: Where the paired sign test is read. A failure to reject is what is wanted,
#: so the number is stated rather than the usual 0.05 being assumed.
SEPARATION = 0.05


@pytest.mark.analytic
@pytest.mark.critical
def test_a_round_trip_is_the_cold_rung_reached_through_the_hot_one() -> None:
    """The definition, on traces whose answer is counted by hand.

    A walker rattling at the cold end scores nothing however often it returns
    to rung 0, and one that reaches the top and comes back scores one each
    time --- which is the whole reason the statistic is preferred to an
    exchange acceptance.
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

    The pair sampler is pinned on one temperature in
    `tests/regression/search/test_potts_mcmc.py`; this pins the *tempered*
    path --- the exchange on the pair's summed energy, the move applied at
    every rung --- by reading the cold rung's two replicas against
    `log_weights` over all 16 configurations. Realized p per replica at the
    declared seed and the next: 0.4404 / 0.1214 and 0.8203 / 0.3536.
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

    Round-trip time in recorded sweeps on the 12x12 periodic triangular
    antiferromagnet, `ROUND_TRIP_SEEDS` seeds paired by seed and read by
    `sample.statistics.sign_test_p_value`: 1,115.5 sweeps without the move
    against 1,090.7 with it, five seeds of eight shorter with the move and
    two longer, `p = 0.6875`. So the direction is not established in either
    sense, at 3.96x the wall --- 17.0 s against 67.3 s.

    Asserted as a failure to separate rather than as a win or a loss, which is
    what eight paired seeds can carry; the mechanism is
    `test_the_overlap_defect_percolates_on_the_frustrated_lattice` and the
    numbers are in
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

    Houdayer's move swaps one connected component of the region where the two
    replicas disagree. On this instance that region is 65 to 72 sites of 144
    and its largest component 43 to 59 of that --- over half the defect at
    every temperature read --- so the move is a near-global exchange of the
    two replicas, and an exchange of two replicas that differ everywhere
    carries the pair nowhere new.
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

    Houdayer's overlap `q_i = s_i s'_i` is the Ising one, and issue #756
    validates the move at two states alone; a three-state model is refused
    here rather than at the first move.
    """
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, PAIR_COUPLING)

    with pytest.raises(ValueError, match="Ising overlap"):
        tempered_potts_pair(
            graph, np.zeros(3), (1.0, 2.0), np.random.default_rng(0), 10
        )


# --- placing the ladder by its round trips (#756) ---------------------------
#
# The other criterion. `adapt_ladder_potts` places a ladder by its exchange
# acceptance, which is a per-pair number; `adapt_ladder_round_trips` places
# one of the same length by the fraction of walkers moving up at each rung,
# and flattens the local diffusivity (Katzgraber et al. 2006). What that buys
# is read as a round-trip time, paired by seed.

#: The warm-up both criteria are given: sweeps per replica per measurement,
#: measurements, and the seed that fixes the ladder each returns. 1,000 sweeps
#: rather than 400 because the placement is read from an up-fraction and a
#: measurement too short to resolve it places rungs on its noise --- at 400
#: sweeps over four rounds the placed ladder ranged from 1,469 to 5,357
#: sweeps per round trip against a geometric ladder's 1,690 on the 16x16
#: (`docs/experiments/023-placing-the-tempering-ladder.md`).
WARM_UP_SWEEPS = 1_000
WARM_UP_ROUNDS = 3
WARM_UP_SEED = 11

#: The acceptance criterion's band and budget. The budget is above the
#: starting length, because insertion is the acceptance criterion's whole
#: remedy for a gap and capping it at the starting length leaves it with
#: nothing to do; the length it settles on is then the length the round-trip
#: criterion is given, so the comparison is at equal rungs and equal cost.
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

    The same three walkers `test_a_round_trip_is_the_cold_rung_reached_through_the_hot_one`
    counts trips in. Rung 0 sees 8 labelled visits and every one is a walker
    on its way up; rung 1 sees 4 up and 3 down; the top rung sees 6 and none
    of them up. So ``f`` is 1 and 0 at the ends by construction, which is what
    makes the placement's cumulative read a fraction of a whole.
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

    `parallel_tempering` records the trace `round_trips` and `up_fraction`
    read, so every recorded sweep must be a permutation of the rungs --- a
    trace in which two walkers sit at one rung would still produce a
    round-trip time, and a wrong one.
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

    256 sites at ``J_c = ln(1 + sqrt(2))``, where the correlation length is
    the lattice and the walkers are held up. The acceptance warm-up settles on
    12 rungs from 0.6 to 2; the round-trip warm-up places 12 of its own, and
    over `ROUND_TRIP_SEEDS` paired seeds the times are 1,053.7 recorded sweeps
    per trip against 1,194.4, six seeds of eight shorter, `p = 0.2891`. A
    second reading at another warm-up seed gave 1,053.1 against 1,092.8 and
    `p = 1.0`, at a 1-minute load of 3.86 and 12.2 s of wall for the eight
    pairs.

    Asserted as a failure to separate, which is what eight paired seeds carry;
    both ladders beat the geometric one they were placed from, which
    `docs/experiments/023-placing-the-tempering-ladder.md` reports and this
    does not assert.
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

    The acceptance warm-up settles on 9 rungs from 0.2 to 2.4333 and the
    round-trip warm-up places 9. Round-trip time 366.6 recorded sweeps
    against 360.1, five seeds of eight shorter, `p = 0.7266`; a second reading
    gave 353.4 against 360.1 at the same `p`. So neither criterion is
    established over the other here, at 7.0 s of wall for the eight pairs.
    """
    graph = frustrated_triangular_lattice((12, 12), BoundaryCondition.PERIODIC, -1.0)
    accepted, feedback = _placed_ladders(graph, 0.2, 2.4333, 10)

    difference = _round_trip_times(graph, feedback) - _round_trip_times(graph, accepted)

    assert sign_test_p_value(difference) > SEPARATION, difference
