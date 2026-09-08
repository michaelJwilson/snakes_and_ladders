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

from collections.abc import Callable

import numpy as np
import pytest
from snakes_and_ladders.likelihood.hmm_paths import emission_log_density
from snakes_and_ladders.search.support import (
    SupportKind,
    TemperedSupport,
    enumerated_labelling_support,
    enumerated_support,
    tempered_labelling_support,
    tempered_topology_support,
)
from snakes_and_ladders.search.tempered import (
    TemperedEnsemble,
    tempered_factor_graph,
    tempered_topologies,
)
from snakes_and_ladders.search.topology import enumerate_topologies, leaf_bipartitions
from snakes_and_ladders.sim.canonical import AMBIGUOUS_OBSERVATIONS, ambiguous_hmm
from snakes_and_ladders.sim.factor_graph import FactorGraph, from_hmm, from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.simulate import simulate_alignment

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
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(1), 30
    )
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


@pytest.mark.structural
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


@pytest.mark.edge_case
def test_an_unusable_ladder_and_a_ladder_without_temperature_one_are_refused() -> None:
    graph = _potts_lattice()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least two temperatures"):
        tempered_factor_graph(graph, (1.0,), rng, 10)
    with pytest.raises(ValueError, match="must be positive"):
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
