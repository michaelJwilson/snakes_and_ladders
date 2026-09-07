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
from phylo.likelihood.potts import log_weights
from phylo.search import potts_mcmc
from phylo.search.potts_mcmc import PottsChain, PottsMove, energies, sample_potts
from phylo.search.statistics import (
    chi_square_p_value,
    integrated_autocorrelation_time,
)
from phylo.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

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


@pytest.mark.oracle
@pytest.mark.parametrize("move", list(PottsMove))
def test_the_chain_is_drawn_from_the_exact_boltzmann_distribution(
    move: PottsMove,
) -> None:
    assert _goodness_of_fit(move, NO_FIELD) > SIGNIFICANCE


@pytest.mark.oracle
@pytest.mark.parametrize("move", list(PottsMove))
def test_the_chain_is_still_exact_in_an_external_field(move: PottsMove) -> None:
    # The case the ticket exists for. Wolff's cluster construction alone does
    # not preserve detailed balance in a field, so this is where the accept
    # step is doing work and the test above is not.
    assert _goodness_of_fit(move, WITH_FIELD) > SIGNIFICANCE


@pytest.mark.structural
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


@pytest.mark.edge_case
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_a_cluster_move_refuses_a_negative_coupling(move: PottsMove) -> None:
    # `1 - exp(-J)` is above 1 for J < 0, so it is not a probability, and an
    # antiferromagnet has no like-spin regions to flip. The algorithm does not
    # apply rather than applying badly.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, -0.5)

    with pytest.raises(ValueError, match="needs every coupling >= 0"):
        sample_potts(graph, NO_FIELD, move, SEED, 10)


@pytest.mark.edge_case
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


@pytest.mark.mathematical
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


@pytest.mark.mathematical
def test_a_wolff_cluster_is_smaller_than_the_lattice_but_larger_than_a_site() -> None:
    # What makes the normalization above necessary, pinned so a change that
    # made every cluster a single site -- which would silently turn Wolff into
    # an expensive single-site sampler -- is visible.
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, TRANSITION)

    chain = sample_potts(graph, np.zeros(3), PottsMove.WOLFF, 7, 2_000, burn_in=200)

    assert 1.0 < chain.mean_cluster_size < graph.n_nodes
