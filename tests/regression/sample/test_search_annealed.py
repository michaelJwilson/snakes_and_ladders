"""``log Z`` from an annealed run, against the transfer matrix and against enumeration.

Issue #756. Three estimators and one claim each, every one refereed outside
the sampler: importance sampling and population annealing recover
`likelihood.potts.strip_log_partition` at three widths and
`opt.potts.log_partition_graph` on the enumerable lattices, each within three
times *its own* standard error over two seeds; the simulated-tempering walker
visits the enumerated Boltzmann law at every rung by chi-square, and its rungs
in the proportion its weights predict.

**A ``log Z`` without its error is not an estimate**, so every deviation below
is read in standard errors rather than in nats, and the two ablations are read
the same way: dropping the importance weight misses by 7.9 errors on a coarse
ladder, and dropping the resampling turns population annealing back into the
importance sampler to 1.8e-15.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood.potts import log_weights, strip_log_partition
from snakes_and_ladders.opt.potts import graph_statistics, log_partition_graph
from snakes_and_ladders.sample.annealed import (
    LogPartition,
    Resampling,
    annealed_importance_sampling,
    geometric_betas,
    population_annealing,
    rung_weights,
    simulated_tempering,
)
from snakes_and_ladders.sample.potts_mcmc import PottsMove
from snakes_and_ladders.sample.statistics import chi_square_p_value
from snakes_and_ladders.sample.tempered import round_trips
from snakes_and_ladders.sim.canonical import frustrated_triangular_lattice
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

#: The ladder both estimators run on, and the population. 200 rungs at 128
#: replicas holds the weight distribution narrow enough to read the answer:
#: realized ESS 59 to 105 of 128 on the strips below, and no deviation past
#: 1.63 standard errors over three widths, two methods and two seeds.
RUNGS = 200
POPULATION = 128
BETA_MIN = 1e-3

#: Where every estimate is read: three times its own standard error. A
#: two-sided normal interval at that width covers 0.997, so one seed outside
#: it is a defect rather than a draw.
ERRORS = 3.0

#: Seeds every claim here is made over. Two, because an estimator that is
#: right at one seed and biased is right at one seed.
SEEDS = (0, 1)

#: The open strip the transfer matrix reaches: 8 columns of 4, 6 and 8 sites,
#: at two states. The widths are the oracle's own limit rather than the
#: sampler's --- a column carries ``2 ** M`` states -- so this is the largest
#: instance in this file with an exact answer.
STRIP_LENGTH = 8
STRIP_WIDTHS = (4, 6, 8)
STRIP_COUPLING = 0.5
STRIP_FIELD = np.array([0.3, -0.2])

#: The 3x3 lattices `test_potts_mcmc.py` enumerates: the open square, and the
#: periodic triangular antiferromagnet whose couplings are all negative.
WIDER_COUPLING = 0.4
WIDER_FIELD = np.array([0.6, -0.4])

#: The ladder simulated tempering runs over, and the recorded sweeps. Four
#: rungs from beta 0.25 to 2, the coldest the model as declared at beta 1 and
#: the hottest twice that; the walker crosses it in both directions, realized
#: 520 to 525 round trips over 6,000 recorded sweeps.
TEMPERING_BETAS = (0.25, 0.5, 1.0, 2.0)
TEMPERING_SWEEPS = 6_000
TEMPERING_THIN = 5

#: Thinning for the rung occupation, where the chi-square is over 4 cells
#: rather than 16 and the rung series is the thing correlated: at 15 the
#: integrated autocorrelation time of the rung is 0.52 to 0.59 recorded
#: sweeps.
OCCUPATION_THIN = 15
TEMPERING_SIGNIFICANCE = 0.001


def _strip(width: int) -> tuple[PottsGraph, float]:
    """The strip and its exact ``log Z``, from the transfer matrix."""
    shape = (STRIP_LENGTH, width)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, STRIP_COUPLING)
    exact = strip_log_partition(
        shape, BoundaryCondition.OPEN, STRIP_COUPLING, STRIP_FIELD
    )
    return graph, exact


def _enumerated_log_z(graph: PottsGraph, coupling: float, field: np.ndarray) -> float:
    """``log Z`` by enumeration, through the statistics `opt.potts` scores it from."""
    agreements, counts = graph_statistics(
        int(field.shape[0]), list(graph.edges), graph.n_nodes
    )
    return float(
        log_partition_graph(
            torch.tensor(coupling, dtype=torch.float64),
            torch.tensor(field, dtype=torch.float64),
            agreements,
            counts,
        )
    )


def _enumerable() -> dict[str, tuple[PottsGraph, float, np.ndarray]]:
    """The two enumerable 3x3 instances, with the coupling the oracle needs."""
    return {
        "3x3-open": (
            lattice_graph((3, 3), BoundaryCondition.OPEN, WIDER_COUPLING),
            WIDER_COUPLING,
            WIDER_FIELD,
        ),
        "frustrated-triangular": (
            frustrated_triangular_lattice((3, 3), BoundaryCondition.PERIODIC, -1.0),
            -1.0,
            np.zeros(2),
        ),
    }


def _within(estimate: LogPartition, exact: float) -> float:
    """The deviation in standard errors, which is the only unit read here."""
    return abs(estimate.log_z - exact) / estimate.stderr


@pytest.mark.oracle
def test_the_importance_sampled_log_partition_is_the_transfer_matrix() -> None:
    """Neal's estimator against the exact strip, at three widths.

    Realized deviations, in standard errors: 0.61 and 0.52 at width 4, 0.98
    and 0.41 at width 6, 1.44 and 0.64 at width 8, over the two seeds; the
    weights carried an ESS of 59 to 90 of 128.
    """
    betas = geometric_betas(1.0, RUNGS, beta_min=BETA_MIN)
    for width in STRIP_WIDTHS:
        graph, exact = _strip(width)
        for seed in SEEDS:
            estimate = annealed_importance_sampling(
                graph,
                STRIP_FIELD,
                betas,
                np.random.default_rng(seed),
                POPULATION,
            )

            assert _within(estimate, exact) < ERRORS, (width, seed, estimate)
            assert 1.0 < estimate.ess <= POPULATION
            assert estimate.family_entropy == pytest.approx(math.log(POPULATION))


@pytest.mark.oracle
def test_the_population_annealed_log_partition_is_the_transfer_matrix() -> None:
    """The resampled population against the same exact strip.

    Realized deviations, in standard errors: 1.01 and 0.20 at width 4, 0.36
    and 1.27 at width 6, 1.63 and 0.68 at width 8. The population's family
    entropy is 3.45 to 3.88 nats of the 4.85 it starts with, which is the
    diagnostic resampling costs: the copies share ancestors, and a run whose
    entropy has collapsed is one estimate rather than 128.
    """
    betas = geometric_betas(1.0, RUNGS, beta_min=BETA_MIN)
    for width in STRIP_WIDTHS:
        graph, exact = _strip(width)
        for seed in SEEDS:
            estimate = population_annealing(
                graph,
                STRIP_FIELD,
                betas,
                np.random.default_rng(seed),
                POPULATION,
            )

            assert _within(estimate, exact) < ERRORS, (width, seed, estimate)
            assert 3.0 < estimate.family_entropy < math.log(POPULATION)


@pytest.mark.oracle
def test_both_estimators_recover_the_enumerated_log_partition() -> None:
    """The same claim where the answer is a sum over every configuration.

    The 3x3 open square and the 3x3 periodic triangular antiferromagnet, 512
    configurations each. Realized deviations in standard errors: 0.35 and
    0.15 for the importance sampler, 0.04 and 0.45 for the population, on the
    open square; 0.15 and 0.14, 0.90 and 1.35 on the frustrated one, where
    every coupling is negative and both Fortuin-Kasteleyn cluster moves are
    refused.
    """
    betas = geometric_betas(1.0, RUNGS // 2, beta_min=BETA_MIN)
    for name, (graph, coupling, field) in _enumerable().items():
        exact = _enumerated_log_z(graph, coupling, field)
        for estimator in (annealed_importance_sampling, population_annealing):
            for seed in SEEDS:
                estimate = estimator(
                    graph, field, betas, np.random.default_rng(seed), POPULATION
                )

                assert _within(estimate, exact) < ERRORS, (name, estimator, estimate)


@pytest.mark.oracle
@pytest.mark.critical
def test_the_zero_rung_is_n_log_q_bitwise() -> None:
    """``==``, not a tolerance: the uniform law's normalizer is exact.

    A one-rung ladder draws from ``beta = 0`` and stops, so both estimators
    must return ``n log q`` to the last bit and an error of exactly zero.
    An estimator that returned ``n log q * (1 - 1e-16)`` here has a bias
    nothing downstream would localize, the zero rung being what every other
    rung is measured from.
    """
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, WIDER_COUPLING)
    exact = graph.n_nodes * math.log(2)

    for estimator in (annealed_importance_sampling, population_annealing):
        estimate = estimator(
            graph, WIDER_FIELD, (0.0,), np.random.default_rng(0), POPULATION
        )

        assert estimate.log_z == exact
        assert estimate.stderr == 0.0
        assert estimate.ess == POPULATION
        assert estimate.rung_log_z.tolist() == [exact]


@pytest.mark.analytic
def test_dropping_the_importance_weight_misses_log_z_by_more_than_three_errors() -> (
    None
):
    """The ablation: a plain annealing run reports ``mean(log w)``, not ``log mean(w)``.

    Six rungs on the 3x3 open square, where the ladder is coarse enough for
    the weights to spread. The importance sampler lands at 11.7015 +- 0.1458
    against the enumerated 11.5199, 1.25 errors; the average of the *logs* ---
    Jensen's lower bound, which is what an annealing run that keeps no weight
    can report --- lands at 10.3647, **7.9 errors low** and low every time,
    the gap being a variance and so one-signed.
    """
    graph, coupling, field = _enumerable()["3x3-open"]
    exact = _enumerated_log_z(graph, coupling, field)
    coarse = geometric_betas(1.0, 6, beta_min=1e-2)

    estimate = annealed_importance_sampling(
        graph, field, coarse, np.random.default_rng(0), 256
    )
    plain = graph.n_nodes * math.log(2) + float(np.mean(estimate.log_weights))

    assert _within(estimate, exact) < ERRORS, estimate
    assert (exact - plain) / estimate.stderr > ERRORS, (plain, exact, estimate.stderr)


@pytest.mark.analytic
@pytest.mark.critical
def test_population_annealing_without_resampling_is_the_importance_sampler() -> None:
    """The second ablation, and the identity that makes the two one estimator.

    With the resampling off, the per-rung normalizers telescope into
    ``logsumexp(w) - log N``: the same trajectories from the same seed, so
    the two estimates agree to 1.8e-15 of a ``log Z`` near 11.44, and the
    population's normalized log weights are the importance sampler's own
    shifted by their normalizer.
    """
    graph, _, field = _enumerable()["3x3-open"]
    betas = geometric_betas(1.0, RUNGS // 2, beta_min=BETA_MIN)

    neal = annealed_importance_sampling(
        graph, field, betas, np.random.default_rng(3), 64
    )
    unresampled = population_annealing(
        graph, field, betas, np.random.default_rng(3), 64, resample=Resampling.NONE
    )

    assert unresampled.log_z == pytest.approx(neal.log_z, abs=1e-12)
    assert unresampled.family_entropy == pytest.approx(math.log(64))
    shift = float(np.log(np.exp(neal.log_weights - neal.log_weights.max()).sum()))
    normalized = neal.log_weights - neal.log_weights.max() - shift
    assert np.abs(unresampled.log_weights - normalized).max() < 1e-12


@pytest.mark.analytic
def test_systematic_resampling_keeps_the_families_multinomial_loses() -> None:
    """Why the default is systematic, in the diagnostic that separates them.

    Both are unbiased and both land inside three standard errors of the
    enumerated answer --- realized 0.04 to 1.05 over four seeds --- so the
    estimate does not choose between them. The genealogy does: over 100 rungs
    on the 3x3 open square, systematic resampling ends at a family entropy of
    4.01 to 4.17 nats of a possible 4.85, and multinomial at 0.19 to 0.80 ---
    two to three effective families of 128. Systematic gives a member whose
    weight is ``1 / N`` exactly one copy where multinomial draws its count,
    and that variance compounds once per rung.
    """
    graph, coupling, field = _enumerable()["3x3-open"]
    exact = _enumerated_log_z(graph, coupling, field)
    betas = geometric_betas(1.0, RUNGS // 2, beta_min=BETA_MIN)
    entropy: dict[Resampling, list[float]] = {}
    for scheme in (Resampling.SYSTEMATIC, Resampling.MULTINOMIAL):
        entropy[scheme] = []
        for seed in range(4):
            estimate = population_annealing(
                graph,
                field,
                betas,
                np.random.default_rng(seed),
                POPULATION,
                resample=scheme,
            )
            assert _within(estimate, exact) < ERRORS, (scheme, seed, estimate)
            entropy[scheme].append(estimate.family_entropy)

    assert min(entropy[Resampling.SYSTEMATIC]) > 4.0
    assert max(entropy[Resampling.MULTINOMIAL]) < 1.0


def _tempering_fixture() -> tuple[PottsGraph, np.ndarray, np.ndarray, list[np.ndarray]]:
    """The 2x2 lattice, its 16 configurations, and the exact law at every rung."""
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.8)
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    laws = []
    for beta in TEMPERING_BETAS:
        weights = log_weights(
            lattice_graph((2, 2), BoundaryCondition.OPEN, 0.8 * beta),
            WIDER_FIELD * beta,
            configurations,
        )
        law = np.exp(weights - weights.max())
        laws.append(law / law.sum())
    return graph, WIDER_FIELD, configurations, laws


@pytest.mark.oracle
@pytest.mark.critical
def test_the_simulated_tempering_walker_is_the_enumerated_law_at_every_rung() -> None:
    """Marinari and Parisi's walker, rung by rung, against enumeration.

    The rung is a sampled variable, so the claim is conditional: the
    configurations recorded *at* rung ``k`` are drawn from the Boltzmann law
    at ``beta_k``, all four laws from one chain. Realized chi-square p over
    the 16 configurations: 0.2635, 0.1572, 0.0941 and 0.6561 at the first
    seed, 0.6100, 0.7028, 0.4190 and 0.0138 at the second. The walker
    completed 520 and 525 round trips over the ladder, so no rung is a law it
    visited once.
    """
    graph, field, configurations, laws = _tempering_fixture()
    index = {
        tuple(row): position for position, row in enumerate(configurations.tolist())
    }
    exact = np.array(
        [
            _enumerated_log_z(
                lattice_graph((2, 2), BoundaryCondition.OPEN, 0.8 * beta),
                0.8 * beta,
                field * beta,
            )
            for beta in TEMPERING_BETAS
        ]
    )

    for seed in SEEDS:
        run = simulated_tempering(
            graph,
            field,
            TEMPERING_BETAS,
            -exact,
            np.random.default_rng(seed),
            TEMPERING_SWEEPS,
            burn_in=TEMPERING_SWEEPS // 12,
            thin=TEMPERING_THIN,
        )

        assert int(round_trips(run.walkers).sum()) > 100
        for rung in range(len(TEMPERING_BETAS)):
            recorded = run.states[run.rungs == rung]
            observed = np.zeros(len(configurations))
            for row in recorded.tolist():
                observed[index[tuple(row)]] += 1.0
            p_value = chi_square_p_value(observed, laws[rung] * observed.sum())
            assert p_value > TEMPERING_SIGNIFICANCE, (seed, rung, p_value)


@pytest.mark.oracle
def test_the_rung_occupation_is_the_one_the_weights_predict() -> None:
    """The rung marginal is ``Z_k exp(g_k)``, and a pilot estimate is what sets it.

    Two readings of the same identity. Given the exact ``g_k = -log Z_k`` the
    occupation is uniform: realized p 0.1907 and 0.7083. Given a pilot
    `annealed_importance_sampling` run's ``g_k``, whose error is 0.0009 to
    0.0736 nats, the prediction tilts to 0.241, 0.242, 0.259, 0.257 and the
    occupation follows it: p 0.2765 and 0.3747. A uniform asserted against
    the pilot's own run would fail at p = 0.0003, which is the estimate's
    error and not the sampler's defect.
    """
    graph, field, _, _ = _tempering_fixture()
    exact = np.array(
        [
            _enumerated_log_z(
                lattice_graph((2, 2), BoundaryCondition.OPEN, 0.8 * beta),
                0.8 * beta,
                field * beta,
            )
            for beta in TEMPERING_BETAS
        ]
    )
    pilot = np.array(
        [
            rung_weights(
                annealed_importance_sampling(
                    graph,
                    field,
                    geometric_betas(beta, 60, beta_min=1e-2),
                    np.random.default_rng(5),
                    POPULATION,
                )
            )[-1]
            for beta in TEMPERING_BETAS
        ]
    )

    for weights in (-exact, pilot):
        predicted = np.exp(exact + weights - (exact + weights).max())
        predicted /= predicted.sum()
        for seed in SEEDS:
            run = simulated_tempering(
                graph,
                field,
                TEMPERING_BETAS,
                weights,
                np.random.default_rng(seed),
                4_000,
                burn_in=500,
                thin=OCCUPATION_THIN,
            )
            counts = run.occupation * run.rungs.shape[0]
            p_value = chi_square_p_value(counts, predicted * counts.sum())

            assert p_value > TEMPERING_SIGNIFICANCE, (weights, seed, p_value)


@pytest.mark.smoke
def test_a_ladder_or_a_population_the_estimators_cannot_use_is_refused() -> None:
    """Every refusal at the entry point, rather than at the first weight.

    A ladder that does not start at zero has no exact rung to anchor on, one
    that is not increasing is not a ladder, and a population of one carries an
    estimate with no statement about it.
    """
    graph, _, field = _enumerable()["3x3-open"]
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="must start at beta = 0"):
        annealed_importance_sampling(graph, field, (0.5, 1.0), rng, 8)
    with pytest.raises(ValueError, match="strictly increasing"):
        population_annealing(graph, field, (0.0, 1.0, 0.5), rng, 8)
    with pytest.raises(ValueError, match="at least two replicas"):
        annealed_importance_sampling(graph, field, (0.0, 1.0), rng, 1)
    with pytest.raises(ValueError, match="at least 2 rungs"):
        simulated_tempering(graph, field, (1.0,), np.zeros(1), rng, 10)
    with pytest.raises(ValueError, match="one g_k per rung"):
        simulated_tempering(graph, field, (0.5, 1.0), np.zeros(3), rng, 10)
    with pytest.raises(ValueError, match="n_sweeps"):
        simulated_tempering(graph, field, (0.5, 1.0), np.zeros(2), rng, 0)
    with pytest.raises(ValueError, match="at least two rungs"):
        geometric_betas(1.0, 1, beta_min=0.1)
    with pytest.raises(ValueError, match="beta_min must lie"):
        geometric_betas(1.0, 4, beta_min=2.0)


@pytest.mark.smoke
def test_a_cluster_move_on_a_negative_coupling_is_refused_by_every_estimator() -> None:
    """The dispatch's refusal, reached through the three new entry points.

    Wolff's bond probability is not a probability below zero, and an
    estimator that ran it there would return a number rather than fail ---
    which is the reason `potts_mcmc.refuse_negative_coupling` is one
    function and every entry point calls it.
    """
    graph, _, field = _enumerable()["frustrated-triangular"]
    rng = np.random.default_rng(0)

    for call in (annealed_importance_sampling, population_annealing):
        with pytest.raises(ValueError, match="needs every coupling >= 0"):
            call(graph, field, (0.0, 1.0), rng, 4, move=PottsMove.WOLFF)
    with pytest.raises(ValueError, match="needs every coupling >= 0"):
        simulated_tempering(
            graph, field, (0.5, 1.0), np.zeros(2), rng, 10, move=PottsMove.WOLFF
        )
