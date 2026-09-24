"""Correctness by distribution, not by inspection of cluster sizes.

At an enumerable lattice size the exact Boltzmann distribution is available,
so each move set is tested by whether the chain's realized visit frequencies
are drawn from it --- a chi-square goodness-of-fit at a declared significance
and chain length.

Two things this file also pins, both wrong while it was written:

The **field accept step**. `test_dropping_the_field_accept_step_is_caught`
replaces it with an unconditional recolouring and asserts this same test
rejects, so the tests above are known to have the power they claim.

The **thinning**. A chi-square assumes independent draws and successive
sweeps are not independent. Run on every sweep it rejects a *correct*
sampler: single-site at `thin = 1` returned p = 0.038 and Swendsen-Wang
p = 0.0024 on chains that are right. The thinning below is part of the test,
not a speed knob, and Wolff needs more of it because one Wolff sweep flips
one cluster while the other two touch every site.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.opt.budget import Budget, Outcome, compare, restarts
from snakes_and_ladders.sample import potts_mcmc
from snakes_and_ladders.sample.balanced import (
    BalancingFunction,
    log_balanced_weights,
    log_metropolis_ratio,
    log_normalizer,
    log_ratios,
)
from snakes_and_ladders.sample.potts_mcmc import _GUARD as GUARD
from snakes_and_ladders.sample.potts_mcmc import (
    ClusterCounter,
    PottsChain,
    PottsMove,
    TemperedChains,
    adapt_ladder_potts,
    anneal_potts,
    autodiff_log_ratios,
    energies,
    houdayer_cluster,
    niedermayer_threshold,
    parallel_tempering,
    sample_potts,
    sample_potts_pair,
    taylor_log_ratios,
    tempered,
)
from snakes_and_ladders.sample.schedule import (
    ConstantTempSchedule,
    ExponentialTempSchedule,
)
from snakes_and_ladders.sample.statistics import (
    chi_square_p_value,
    integrated_autocorrelation_time,
)
from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes
from snakes_and_ladders.sim.canonical import (
    PlantedSpinGlass,
    frustrated_triangular_lattice,
    minimum_frustrated_edges,
    planted_spin_glass,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import (
    critical_coupling,
    energy,
    heat_bath_log_weights,
    local_fields,
    owner_rows,
    site_field,
)

from tests._chains import enumerated_law, fit_p_value
from tests._scale import at_scale

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
#: A field per site on the same 2x2 graph (issue #919), one row per site, no
#: two rows equal, drawn once from a declared seed: the form `spatio_only`
#: and the coupled model hand the cluster moves, whose accept step sums the
#: field over the cluster's own sites.
PER_SITE_FIELD = np.random.default_rng(919).normal(0.0, 0.6, (4, 2))
#: Both forms, for the exactness tests and their control.
FIELDS = {"shared": WITH_FIELD, "per-site": PER_SITE_FIELD}

# One Wolff sweep is one cluster flip; one sweep of either other move set
# touches every site. Equal `thin` would compare a decorrelated chain against
# a correlated one and reject Wolff for being thinned less.
THINNING = {
    PottsMove.SINGLE_SITE: 5,
    PottsMove.SWENDSEN_WANG: 5,
    PottsMove.WOLFF: 25,
    PottsMove.LOCALLY_BALANCED: 4,
    PottsMove.GIBBS_WITH_GRADIENTS: 4,
    # One cluster per sweep, as Wolff's is, so the same thinning. Read rather
    # than copied: at 15 the pair of chains on the frustrated lattice returned
    # p = 0.0 with a mean cluster of 8.3 sites in 9, the near-percolating
    # cluster being exactly what correlates successive sweeps there.
    PottsMove.NIEDERMAYER: 25,
}

#: The gradient-informed sweeps run in NumPy, one proposal over the whole
#: neighbourhood at a time, so a chain of `SWEEPS` of them costs seconds where
#: the others cost fractions of one. 4,000 recorded sweeps still puts an
#: expected 250 in each of the sixteen cells, which is what the chi-square
#: needs; the length is the test's budget and not part of its claim.
BALANCED_SWEEPS = 4_000
SWEEPS_BY_MOVE = {
    PottsMove.SINGLE_SITE: SWEEPS,
    PottsMove.SWENDSEN_WANG: SWEEPS,
    PottsMove.WOLFF: SWEEPS,
    PottsMove.NIEDERMAYER: SWEEPS,
    PottsMove.LOCALLY_BALANCED: BALANCED_SWEEPS,
    PottsMove.GIBBS_WITH_GRADIENTS: BALANCED_SWEEPS,
}

#: The gradient-informed move sets, which propose from the whole single-flip
#: neighbourhood rather than visiting each site in turn.
BALANCED = [PottsMove.LOCALLY_BALANCED, PottsMove.GIBBS_WITH_GRADIENTS]

#: Recorded sweeps behind an ablation. An ablated chain is rejected at p = 0.0
#: rather than marginally, so a fifth of `SWEEPS` shows it and keeps the two
#: ablations inside the per-pull-request cap.
ABLATION_SWEEPS = 2_000

#: Moves out of each configuration behind the reversibility check. The flow
#: matrix's Monte Carlo error is `1 / sqrt` of this on a probability of one,
#: which is what the bound there is read against.
KERNEL_TRIALS = 12_500


def _goodness_of_fit(move: PottsMove, field: np.ndarray, seed: int = SEED) -> float:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    return _chi_square_against(graph, field, move, seed)


def _chi_square_against(
    graph: PottsGraph,
    field: np.ndarray,
    move: PottsMove,
    seed: int,
    sweeps: int | None = None,
) -> float:
    """One chain's realized frequencies against the enumerated Boltzmann law."""
    index, probability = enumerated_law(graph, field)
    sweeps = SWEEPS_BY_MOVE[move] if sweeps is None else sweeps

    chain = sample_potts(
        graph,
        field,
        move,
        np.random.default_rng(seed),
        sweeps,
        burn_in=sweeps // 10,
        thin=THINNING[move],
    )

    return fit_p_value(index, probability, chain.states, sweeps)


@pytest.mark.oracle
@pytest.mark.parametrize("move", list(PottsMove))
def test_the_chain_is_drawn_from_the_exact_boltzmann_distribution(
    move: PottsMove,
) -> None:
    assert _goodness_of_fit(move, NO_FIELD) > SIGNIFICANCE


@pytest.mark.oracle
@pytest.mark.parametrize("field", list(FIELDS))
@pytest.mark.parametrize("move", list(PottsMove))
def test_the_chain_is_still_exact_in_an_external_field(
    move: PottsMove, field: str
) -> None:
    # Wolff's cluster construction alone does not preserve detailed balance in
    # a field, so this is where the accept step does work and the test above
    # does not. The per-site field is the one whose accept step sums over the
    # cluster's own sites rather than scaling one shared difference (#919).
    assert _goodness_of_fit(move, FIELDS[field]) > SIGNIFICANCE


@pytest.mark.smoke
@pytest.mark.parametrize("field", list(FIELDS))
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_dropping_the_field_accept_step_is_caught(
    move: PottsMove, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Evidence that the two tests above have the power they claim.
    def unconditional(
        state: np.ndarray,
        members: np.ndarray,
        rows: np.ndarray,
        rng: np.random.Generator,
        proposed: int | None = None,
    ) -> potts_mcmc.Recolour:
        # `rows` is the field widened to one row per site (issue #551), so the
        # colour count is its column count. `proposed` is the colour issue
        # #706's Wolff action names; unused here, since the point of this stub
        # is that the accept step is gone, not which colour it skipped.
        del proposed
        state[members] = int(rng.integers(rows.shape[1]))
        return potts_mcmc.Recolour(proposed=True, accepted=True)

    monkeypatch.setattr(potts_mcmc, "_recolour", unconditional)

    assert _goodness_of_fit(move, FIELDS[field]) < SIGNIFICANCE


@pytest.mark.smoke
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_a_cluster_move_refuses_a_negative_coupling(move: PottsMove) -> None:
    # `1 - exp(-J)` is above 1 for J < 0, so it is not a probability, and an
    # antiferromagnet has no like-spin regions to flip.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, -0.5)

    with pytest.raises(ValueError, match="needs every coupling >= 0"):
        sample_potts(graph, NO_FIELD, move, np.random.default_rng(SEED), 10)


@pytest.mark.oracle
def test_single_site_is_still_exact_on_a_negative_coupling() -> None:
    # The refusal above is a property of the cluster construction, not of the
    # model, and what says so is that the heat bath is still drawing from the
    # right distribution at J = -0.5: the same enumeration of all 16
    # configurations refereeing the ferromagnetic tests, at the coupling the
    # cluster moves decline. Realized p = 0.524 at the declared seed and
    # 0.608 at the next, against the 0.001 significance.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, -0.5)
    index, probability = enumerated_law(graph, NO_FIELD)

    chain = sample_potts(
        graph,
        NO_FIELD,
        PottsMove.SINGLE_SITE,
        np.random.default_rng(SEED),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING[PottsMove.SINGLE_SITE],
    )

    observed = np.zeros(len(probability))
    for row in chain.states:
        observed[index[tuple(row)]] += 1
    assert chi_square_p_value(observed, probability * SWEEPS) > SIGNIFICANCE


# --- the gradient-informed proposals ----------------------------------------
#
# Zanella (2020) and Grathwohl et al. (2021), which on this energy are one
# kernel: the Taylor estimate the second proposes from *is* the difference the
# first proposes from, because the relaxed log weight is affine in each site's
# row. That is the claim of
# `test_the_taylor_estimate_is_the_single_flip_energy_difference`, pinned three
# ways --- against the heat bath's own conditional, against the tape, and
# against the enumerated energy --- and it is why two move sets share one
# sweep rather than one move set carrying two names.


def _wider_lattice() -> PottsGraph:
    """The 3x3 open square: 512 configurations, and an interior site of degree 4.

    A weaker coupling than `COUPLING`, so the enumerated law is spread over
    the 512 cells rather than concentrated on the two aligned configurations:
    a chi-square over cells the model almost never visits measures rounding.
    """
    return lattice_graph((3, 3), BoundaryCondition.OPEN, 0.4)


def _frustrated_lattice() -> PottsGraph:
    """The 3x3 periodic triangular antiferromagnet: every coupling negative.

    The instance both cluster moves refuse
    (`test_a_cluster_move_refuses_a_negative_coupling`). The gradient-informed
    moves are single-flip and run on it, which is the point of testing them
    here: a frustrated law is the one a proposal that follows the energy is
    likeliest to get wrong.
    """
    return frustrated_triangular_lattice((3, 3), BoundaryCondition.PERIODIC, -1.0)


ENUMERABLE = {
    "3x3-open": _wider_lattice,
    "frustrated-triangular": _frustrated_lattice,
}

#: Recorded sweeps for the enumerable lattices above: 512 cells, so 10,000
#: draws put a mean 19.5 in each.
WIDE_SWEEPS = 10_000


# --- Niedermayer's bond rule and Houdayer's pair ----------------------------
#
# Issue #756. Wolff's construction needs every coupling non-negative and is
# refused on the frustrated lattice
# (`test_a_cluster_move_refuses_a_negative_coupling`), so the one cluster move
# this package had did not reach the instance cluster moves exist for.
# Niedermayer (1988) activates a bond on its energy relative to a threshold
# `E_0` and carries the Metropolis ratio that leaves; Houdayer (2001) moves a
# *pair* of replicas at one temperature and is isoenergetic for the pair, so
# its acceptance is 1 by an identity rather than by a construction.
#
# The referees below are the file's own, in order: the enumerated law on the
# 512-configuration lattices, the ablations that show those tests have power,
# and the two exact identities the moves rest on.


#: The mixed instance: the frustrated lattice with half its bonds turned
#: ferromagnetic. At the threshold its cluster percolates --- 8.94 sites of 9
#: --- so a chain of these alone is a global spin reversal and a chi-square
#: rejects it at p = 9e-218 however long it runs. It is refereed by
#: reversibility instead (`test_niedermayers_kernel_is_reversible...`), which
#: is the claim a chain cannot carry here.
def _mixed_lattice() -> PottsGraph:
    """The frustrated triangular lattice with every other coupling ferromagnetic."""
    frustrated = _frustrated_lattice()
    return PottsGraph(
        n_nodes=frustrated.n_nodes,
        edges=frustrated.edges,
        coupling=tuple(
            -1.0 if index % 2 else 0.7 for index in range(len(frustrated.edges))
        ),
        shape=frustrated.shape,
    )


#: Sweeps the pair sampler records. The same 10,000 as a single chain, so a
#: cell of the 512-cell chi-square has the same expected count.
PAIR_SWEEPS = 10_000

#: Thinning for the pair, per within-replica move set. Twice a single chain's
#: for the heat bath: Houdayer's move exchanges labels between the replicas
#: rather than changing them, so a pair decorrelates more slowly than either
#: chain alone at the same sweep count.
PAIR_THINNING = {PottsMove.SINGLE_SITE: 10, PottsMove.NIEDERMAYER: 25}


def _pair_chi_square(
    graph: PottsGraph, field: np.ndarray, move: PottsMove, seed: int
) -> tuple[float, float]:
    """Each replica's realized frequencies against the enumerated Boltzmann law."""
    index, probability = enumerated_law(graph, field)
    chains = sample_potts_pair(
        graph,
        field,
        move,
        np.random.default_rng(seed),
        PAIR_SWEEPS,
        burn_in=PAIR_SWEEPS // 10,
        thin=PAIR_THINNING[move],
    )
    realized = [
        fit_p_value(index, probability, chain.states, PAIR_SWEEPS) for chain in chains
    ]
    return realized[0], realized[1]


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("instance", sorted(ENUMERABLE))
def test_the_niedermayer_chain_is_drawn_from_the_exact_boltzmann_distribution(
    instance: str,
) -> None:
    # The same referee as the heat bath and Wolff, on the instances Wolff
    # refuses: every configuration enumerated by `log_weights`, which shares no
    # bond rule, transposition or accept step with the sampler. `release`
    # because 275,000 cluster growths in Python do not fit the per-test cap;
    # the 2x2 case of
    # `test_the_chain_is_drawn_from_the_exact_boltzmann_distribution` is the
    # per-pull-request sibling and runs the same move set.
    # Realized p at the declared seed and the next: 3x3-open 0.0221 and 0.1116,
    # frustrated-triangular 0.0936 and 0.9945.
    graph = ENUMERABLE[instance]()

    p_value = _chi_square_against(
        graph, WITH_FIELD, PottsMove.NIEDERMAYER, SEED, sweeps=WIDE_SWEEPS
    )

    assert p_value > SIGNIFICANCE, p_value


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("instance", sorted(ENUMERABLE))
@pytest.mark.parametrize("move", [PottsMove.SINGLE_SITE, PottsMove.NIEDERMAYER])
def test_each_replica_of_a_houdayer_pair_is_drawn_from_the_exact_boltzmann_distribution(
    instance: str, move: PottsMove
) -> None:
    # The marginal of *one* replica of the pair is the claim: the joint target
    # is the product of the two Boltzmann laws, Houdayer's move is an
    # involution on it with a symmetric proposal, so the product is invariant
    # and the marginal follows. Both replicas are read, since a move that
    # exchanges between them could leave one right and the other wrong.
    # Realized p per replica at the declared seed and the next --- single-site:
    # 3x3-open 0.5086/0.1672 and 0.9580/0.0171, frustrated-triangular
    # 0.0039/0.9058 and 0.8477/0.9913; Niedermayer: 3x3-open 0.5061/0.1717 and
    # 0.1394/0.7302, frustrated-triangular 0.3116/0.9231 and 0.3202/0.0386.
    graph = ENUMERABLE[instance]()

    first, second = _pair_chi_square(graph, WITH_FIELD, move, SEED)

    assert min(first, second) > SIGNIFICANCE, (first, second)


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize(
    ("name", "coupling"),
    [
        ("ferromagnet", (0.8, 0.8, 0.8, 0.8)),
        ("antiferromagnet", (-1.0, -1.0, -1.0, -1.0)),
        ("mixed", (0.7, -1.0, 0.7, -1.0)),
    ],
)
def test_niedermayers_kernel_is_reversible_against_the_enumerated_law(
    name: str, coupling: tuple[float, ...]
) -> None:
    # The claim a chain carries where the move mixes and cannot carry where it
    # percolates, made directly instead: the kernel's own flow `pi(s) K(s, s')`
    # against its transpose, `K` estimated from 12,500 moves out of each of the
    # 16 configurations of a four-cycle and `pi` enumerated by `log_weights`.
    # The mixed instance is the one this exists for --- its cluster is the
    # whole lattice nine times in ten, so a chain there is a global spin
    # reversal whatever its length --- and the other two are the controls.
    # Realized max asymmetry: ferromagnet 0.00137, antiferromagnet 0.00080,
    # mixed 0.00160, against a Monte Carlo standard error of 1 / sqrt(12,500)
    # = 0.0089 on a probability of one, so the bound is the error and not a
    # tolerance fitted to the result.
    graph = PottsGraph(
        n_nodes=4, edges=((0, 1), (1, 2), (2, 3), (3, 0)), coupling=coupling
    )
    configurations = [
        tuple(values) for values in itertools.product(range(2), repeat=graph.n_nodes)
    ]
    weights = log_weights(graph, WITH_FIELD, np.array(configurations, dtype=np.int64))
    exact = np.exp(weights - weights.max())
    exact /= exact.sum()
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    threshold = niedermayer_threshold(couplings)
    rng = np.random.default_rng(SEED)

    index = {values: position for position, values in enumerate(configurations)}
    kernel = np.zeros((len(configurations), len(configurations)))
    for position, values in enumerate(configurations):
        for _ in range(KERNEL_TRIALS):
            state = np.array(values, dtype=np.int64)
            potts_mcmc.niedermayer_sweep(
                state,
                rows,
                offsets,
                neighbours,
                couplings,
                rng,
                beta=1.0,
                threshold=threshold,
            )
            kernel[position, index[tuple(state.tolist())]] += 1
    flow = exact[:, None] * kernel / KERNEL_TRIALS

    assert np.abs(flow - flow.T).max() < 3.0 / np.sqrt(KERNEL_TRIALS), name


@pytest.mark.analytic
@pytest.mark.critical
def test_a_mixed_couplings_cluster_percolates_and_an_antiferromagnets_nearly_does() -> (
    None
):
    # Why Houdayer's move exists beside Niedermayer's, as a number. A single
    # cluster that is the whole lattice is a global spin reversal, and the
    # threshold's bond probabilities put the mixed instance there: 8.94 sites
    # of 9 against the uniform antiferromagnet's 8.31 and the ferromagnet's
    # 2.25, over 4,000 clusters from the declared seed.
    sizes = {}
    for name, graph in (
        ("mixed", _mixed_lattice()),
        ("frustrated", _frustrated_lattice()),
        ("ferromagnet", _wider_lattice()),
    ):
        sizes[name] = _cluster_counter(graph, PottsMove.NIEDERMAYER).mean_size

    assert_allclose(sizes["mixed"], 8.94, atol=0.005)
    assert_allclose(sizes["frustrated"], 8.31, atol=0.005)
    assert_allclose(sizes["ferromagnet"], 2.25, atol=0.005)


@pytest.mark.analytic
@pytest.mark.critical
def test_the_threshold_at_zero_on_an_antiferromagnet_is_a_single_site_flip() -> None:
    # The other end of Niedermayer's one knob, and the reason it is a knob:
    # at `E_0 = 0` an antiferromagnet's like bonds ask for a margin of `J < 0`
    # and its unlike bonds for one of 0, so no bond forms, the cluster is its
    # seed, and the ratio is the exact single-flip energy difference. Read
    # against `energies` on the flipped configuration rather than against the
    # sweep's own arithmetic.
    graph = _frustrated_lattice()
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rng = np.random.default_rng(SEED)

    moved = 0
    for _ in range(200):
        state = np.ascontiguousarray(
            rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64
        )
        root = int(rng.integers(graph.n_nodes))
        flipped = state.copy()
        flipped[root] = 1 - int(state[root])
        difference = float(
            energies(graph, rows, state[None])[0]
            - energies(graph, rows, flipped[None])[0]
        )
        after = state.copy()
        size = potts_mcmc.niedermayer_sweep(
            after,
            rows,
            offsets,
            neighbours,
            couplings,
            np.random.default_rng(11),
            beta=1.0,
            threshold=0.0,
            root=root,
            partner=1 - int(state[root]),
        )

        assert size == 1
        if difference >= 0.0:
            assert np.array_equal(after, flipped)
        moved += int(not np.array_equal(after, state))
    # The flip is refused sometimes and taken sometimes, so the accept step
    # above was exercised in both directions rather than only one.
    assert 20 < moved < 180, moved


@pytest.mark.smoke
def test_dropping_niedermayers_accept_step_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Evidence that the two tests above have the power they claim, and the
    # ablation issue #756's plan names. At the threshold the boundary terms of
    # the ratio cancel, so what this removes is the field term --- the step
    # that makes a Fortuin-Kasteleyn cluster move exact in a field, and the
    # one `test_dropping_the_field_accept_step_is_caught` removes from Wolff.
    # Realized p = 0.0 against the 0.001 significance, where the unablated
    # chain returns 0.0936.
    def unconditional(delta: float, beta: float, rng: np.random.Generator) -> bool:
        # The accept step gone, both terms with it: the field difference the
        # ferromagnetic case also carries, and the boundary terms only a mixed
        # instance has.
        del delta, beta, rng
        return True

    monkeypatch.setattr(potts_mcmc, "_niedermayer_accept", unconditional)

    p_value = _chi_square_against(
        _frustrated_lattice(),
        WITH_FIELD,
        PottsMove.NIEDERMAYER,
        SEED,
        sweeps=ABLATION_SWEEPS,
    )

    assert p_value < SIGNIFICANCE, p_value


@pytest.mark.smoke
def test_swapping_a_site_rather_than_a_component_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The second ablation: Houdayer's cluster replaced by singletons, so the
    # move swaps one defect site rather than a connected component of the
    # defect region. It is the *component* that makes the move isoenergetic
    # --- a neighbour that disagreed would be in the same component, so a
    # boundary neighbour agrees and its two bond terms are exchanged --- and
    # this is what says so. Realized p = 0.0 on the first replica against the
    # 0.001 significance, where the unablated pair returns 0.0039.
    def singletons(
        first: np.ndarray,
        second: np.ndarray,
        offsets: np.ndarray,
        neighbours: np.ndarray,
    ) -> np.ndarray:
        del first, second, neighbours
        return np.arange(int(offsets.shape[0]) - 1)

    monkeypatch.setattr(potts_mcmc, "houdayer_cluster", singletons)

    first, second = _pair_chi_square(
        _frustrated_lattice(), WITH_FIELD, PottsMove.SINGLE_SITE, SEED
    )

    assert min(first, second) < SIGNIFICANCE, (first, second)


@pytest.mark.analytic
@pytest.mark.critical
def test_houdayers_move_leaves_the_pairs_energy_where_it_found_it() -> None:
    # The identity the acceptance of 1 rests on, checked rather than derived.
    # A bond inside the cluster has its two agreements exchanged between the
    # replicas; a bond leaving it has its far end where the replicas agree, so
    # its two terms are exchanged too; and the field term is exchanged site by
    # site. Worst |dE| over 500 drawn pairs: 3.6e-15 against an energy of
    # order 20, which is the double-precision floor for a sum of that size and
    # not a tolerance chosen to admit anything.
    graph = _frustrated_lattice()
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, _ = graph.compressed_adjacency()
    rng = np.random.default_rng(SEED)

    worst, moved = 0.0, 0
    for _ in range(500):
        first = np.ascontiguousarray(
            rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64
        )
        second = np.ascontiguousarray(
            rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64
        )
        before = float(energies(graph, rows, np.stack([first, second])).sum())
        size = potts_mcmc.houdayer_move(first, second, offsets, neighbours, rng)
        after = float(energies(graph, rows, np.stack([first, second])).sum())
        worst = max(worst, abs(after - before))
        moved += int(size > 0)

    assert moved > 400, moved
    assert worst < 1e-12, worst


@pytest.mark.analytic
@pytest.mark.critical
def test_houdayers_move_alone_never_leaves_the_orbit_of_its_draw() -> None:
    # Why `sample_potts_pair` takes a within-replica move set rather than
    # offering Houdayer's move on its own: the move exchanges labels between
    # the replicas, so the unordered pair `{s_i, s'_i}` at every site is
    # exactly what it cannot change. A chain of these alone is confined to the
    # 2**n_nodes orbit of its initial draw and is not a sampler of anything,
    # which is a property of the move and not a gap in the implementation.
    graph = _frustrated_lattice()
    offsets, neighbours, _ = graph.compressed_adjacency()
    rng = np.random.default_rng(SEED)
    first = np.ascontiguousarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
    second = np.ascontiguousarray(
        rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64
    )
    start = np.sort(np.stack([first, second]), axis=0)

    for _ in range(200):
        potts_mcmc.houdayer_move(first, second, offsets, neighbours, rng)
        assert np.array_equal(np.sort(np.stack([first, second]), axis=0), start)


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("temperature", [0.25, 1.0, 4.0])
def test_niedermayers_rule_is_wolffs_bitwise_on_a_ferromagnet(
    temperature: float,
) -> None:
    # The reduction, at the strongest reading available: on a ferromagnet
    # `niedermayer_threshold` is 0, the unlike bonds get probability 0 and the
    # like ones `1 - exp(-beta J)`, and the boundary terms of the ratio cancel
    # to leave Wolff's field accept step. So the two sweeps draw the same
    # uniforms against the same numbers in the same order, and the claim is
    # equality of the labellings rather than of a distribution over them ---
    # 300 draws at each of three temperatures, `q = 3`, on the 4x4 open
    # square, with the root and the colour named so neither draws them.
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, COUPLING)
    rows = site_field(np.array([0.6, -0.4, 0.1]), graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    assert niedermayer_threshold(couplings) == 0.0
    rng = np.random.default_rng(SEED)

    for _ in range(300):
        state = np.ascontiguousarray(
            rng.integers(0, 3, size=graph.n_nodes), dtype=np.int64
        )
        root, colour = int(rng.integers(graph.n_nodes)), int(rng.integers(3))
        wolff, niedermayer = state.copy(), state.copy()

        theirs = potts_mcmc.wolff_sweep(
            wolff,
            rows,
            offsets,
            neighbours,
            couplings,
            np.random.default_rng(7),
            beta=1.0 / temperature,
            root=root,
            proposed=colour,
        )
        ours = potts_mcmc.niedermayer_sweep(
            niedermayer,
            rows,
            offsets,
            neighbours,
            couplings,
            np.random.default_rng(7),
            beta=1.0 / temperature,
            threshold=0.0,
            root=root,
            partner=colour,
        )

        assert np.array_equal(wolff, niedermayer)
        assert theirs == ours


@pytest.mark.analytic
@pytest.mark.critical
def test_the_threshold_is_where_the_antiferromagnets_bonds_become_a_probability() -> (
    None
):
    # `E_0 = max(0, -min J)` is not a tuning constant: it is the smallest
    # threshold at which every bond probability is defined. Below it the like
    # bonds of an antiferromagnet ask for `1 - exp(beta |J|)`, which is
    # negative, and at it they ask for zero while the unlike ones carry the
    # construction --- which is what an antiferromagnet's satisfied bonds are.
    frustrated = _frustrated_lattice()
    _, _, couplings = frustrated.compressed_adjacency()
    ferromagnet = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    assert niedermayer_threshold(couplings) == 1.0
    assert niedermayer_threshold(ferromagnet.compressed_adjacency().couplings) == 0.0
    # The like bond at the threshold: margin zero, so no bond and no draw.
    assert max(0.0, 1.0 + float(couplings.min())) == 0.0
    # The unlike bond at the threshold: the antiferromagnet's own 1 - exp(-b|J|).
    assert max(0.0, 1.0 + 0.0) == 1.0


def _cluster_counter(
    graph: PottsGraph, move: PottsMove, n_clusters: int = 4_000
) -> ClusterCounter:
    """Sizes and field acceptances over ``n_clusters`` single-cluster steps."""
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rng = np.random.default_rng(SEED)
    state = np.ascontiguousarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
    counter = ClusterCounter()
    for _ in range(n_clusters):
        if move is PottsMove.WOLFF:
            potts_mcmc.wolff_sweep(
                state, rows, offsets, neighbours, couplings, rng, counter, graph
            )
        else:
            potts_mcmc.niedermayer_sweep(
                state,
                rows,
                offsets,
                neighbours,
                couplings,
                rng,
                counter,
                graph,
                1.0,
                niedermayer_threshold(couplings),
            )
    return counter


@pytest.mark.analytic
@pytest.mark.critical
def test_the_accepted_fraction_and_the_cluster_are_what_the_instance_makes_them() -> (
    None
):
    # What the moves are read against, side by side, because the reason for
    # Houdayer's move is a number rather than an argument. On the frustrated
    # lattice Wolff is refused outright; Niedermayer runs there and builds a
    # cluster of 8.31 sites in 9 --- a near-percolating cluster, which is a
    # global spin reversal by another name and is what makes the pair move
    # worth having. On the ferromagnet the two agree on both readings.
    frustrated, ferromagnet = _frustrated_lattice(), _wider_lattice()
    readings = {
        "frustrated": _cluster_counter(frustrated, PottsMove.NIEDERMAYER),
        "ferro-niedermayer": _cluster_counter(ferromagnet, PottsMove.NIEDERMAYER),
        "ferro-wolff": _cluster_counter(ferromagnet, PottsMove.WOLFF),
    }

    with pytest.raises(ValueError, match="needs every coupling >= 0"):
        sample_potts(
            frustrated, WITH_FIELD, PottsMove.WOLFF, np.random.default_rng(SEED), 10
        )
    for name, expected in (
        ("frustrated", (0.4868, 8.31)),
        ("ferro-niedermayer", (0.3157, 2.25)),
        ("ferro-wolff", (0.3477, 2.18)),
    ):
        assert_allclose(
            (readings[name].accept_rate, readings[name].mean_size),
            expected,
            atol=0.005,
        )


@pytest.mark.analytic
@pytest.mark.critical
def test_the_overlap_components_are_the_defect_regions_by_breadth_first_search() -> (
    None
):
    # `houdayer_cluster` reads through `potts_mcmc`'s union-find; this walks
    # the adjacency from each defect site instead, so the components are
    # checked against a second reading of what a component is rather than
    # against themselves.
    graph = _frustrated_lattice()
    offsets, neighbours, _ = graph.compressed_adjacency()
    incident = [
        neighbours[offsets[node] : offsets[node + 1]].tolist()
        for node in range(graph.n_nodes)
    ]
    rng = np.random.default_rng(SEED)

    for _ in range(200):
        first = rng.integers(0, 2, size=graph.n_nodes)
        second = rng.integers(0, 2, size=graph.n_nodes)
        partition = houdayer_cluster(first, second, offsets, neighbours)
        defect = first != second
        for node in np.flatnonzero(defect):
            seen, frontier = {int(node)}, [int(node)]
            while frontier:
                current = frontier.pop()
                for neighbour in incident[current]:
                    if neighbour not in seen and defect[neighbour]:
                        seen.add(int(neighbour))
                        frontier.append(int(neighbour))
            assert set(np.flatnonzero(partition == partition[node]).tolist()) == seen


@pytest.mark.smoke
def test_the_pair_sampler_refuses_houdayers_move_above_two_states() -> None:
    # The restriction stated rather than silently generalized: the overlap
    # `q_i = s_i s'_i` is the Ising one, and issue #756 validates the move at
    # two states alone.
    graph = _wider_lattice()

    with pytest.raises(ValueError, match="Ising overlap"):
        sample_potts_pair(
            graph,
            np.zeros(3),
            PottsMove.SINGLE_SITE,
            np.random.default_rng(SEED),
            10,
        )


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("instance", sorted(ENUMERABLE))
def test_the_locally_balanced_chain_is_drawn_from_the_exact_boltzmann_distribution(
    instance: str,
) -> None:
    # The same referee as the heat bath and Wolff: every configuration
    # enumerated by `log_weights`, which shares no proposal, weight or accept
    # step with the sampler. `release` because the sweep is NumPy over the
    # whole neighbourhood per proposal and 360,000 proposals do not fit the
    # per-test cap; the 2x2 case of
    # `test_the_chain_is_drawn_from_the_exact_boltzmann_distribution` is the
    # per-pull-request sibling.
    # Realized p at the declared seed and the next: 3x3-open 0.0240 and
    # 0.2629, frustrated-triangular 0.0758 and 0.9926.
    graph = ENUMERABLE[instance]()

    p_value = _chi_square_against(
        graph, WITH_FIELD, PottsMove.LOCALLY_BALANCED, SEED, sweeps=WIDE_SWEEPS
    )

    assert p_value > SIGNIFICANCE, p_value


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("instance", sorted(ENUMERABLE))
def test_the_gibbs_with_gradients_chain_is_drawn_from_the_exact_boltzmann_distribution(
    instance: str,
) -> None:
    # As above, for the proposal that weights the neighbourhood by the Taylor
    # estimate rather than by the difference. The two agree to 1e-12 on this
    # energy, so what this adds over the test above is that the *route* --- a
    # gradient rebuilt at each state, forward and reverse --- reaches the same
    # law rather than being assumed to.
    # Realized p at the declared seed and the next: 3x3-open 0.0240 and
    # 0.2629, frustrated-triangular 0.0758 and 0.9926 --- the locally balanced
    # chain's, to four figures, on every instance and seed. The two routes
    # draw the same uniforms against the same weights here, so the chains do
    # not merely share a law; they are the same chain.
    graph = ENUMERABLE[instance]()

    p_value = _chi_square_against(
        graph, WITH_FIELD, PottsMove.GIBBS_WITH_GRADIENTS, SEED, sweeps=WIDE_SWEEPS
    )

    assert p_value > SIGNIFICANCE, p_value


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("instance", sorted(ENUMERABLE))
def test_the_taylor_estimate_is_the_single_flip_energy_difference(
    instance: str,
) -> None:
    # The claim that makes Gibbs-with-gradients and the locally balanced
    # proposal one kernel here, and it is refereed rather than derived: the
    # heat bath's own per-site conditional, the tape's gradient at the one-hot
    # state, and the enumerated energy of each flipped configuration. Nothing
    # in the chain of equalities is the sampler's own arithmetic.
    graph = ENUMERABLE[instance]()
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    owner = owner_rows(offsets)
    bounds = offsets.tolist()
    incident, weights = neighbours.tolist(), couplings.tolist()
    rng = np.random.default_rng(SEED)

    for _ in range(8):
        state = np.asarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
        conditional = np.array(
            [
                heat_bath_log_weights(
                    rows[node],
                    state,
                    incident,
                    weights,
                    bounds[node],
                    bounds[node + 1],
                )
                for node in range(graph.n_nodes)
            ]
        )
        expected = log_ratios(conditional, state)
        estimate = taylor_log_ratios(rows, state, neighbours, couplings, owner)

        assert np.abs(estimate - expected).max() < 1e-12
        assert np.abs(autodiff_log_ratios(graph, rows, state) - expected).max() < 1e-12

        current = float(energies(graph, rows, state[None])[0])
        for node in range(graph.n_nodes):
            for colour in range(2):
                moved = state.copy()
                moved[node] = colour
                difference = current - float(energies(graph, rows, moved[None])[0])
                assert estimate[node, colour] == pytest.approx(difference, abs=1e-12)


@pytest.mark.analytic
@pytest.mark.parametrize("function", list(BalancingFunction))
def test_the_accept_step_is_the_ratio_of_the_two_neighbourhood_normalizers(
    function: BalancingFunction,
) -> None:
    # Zanella's collapse, for *both* balancing functions: a balanced weight
    # satisfies `g(t) = t g(1 / t)`, so the target and the weights cancel and
    # the Metropolis-Hastings ratio is `Z(s) / Z(s')` whichever `g` is used.
    # The sweep computes it the general way --- the way Gibbs-with-gradients
    # needs, since an estimate does not cancel --- and this is what says the
    # general form is the collapsed one where the estimate is exact.
    graph = _frustrated_lattice()
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    owner = owner_rows(offsets)
    rng = np.random.default_rng(SEED)

    for _ in range(4):
        state = np.asarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
        here = log_ratios(
            local_fields(rows, state, neighbours, couplings, owner), state
        )
        forward_weights = log_balanced_weights(here, state, function=function)
        forward_total = log_normalizer(forward_weights)

        for node in range(graph.n_nodes):
            colour = 1 - int(state[node])
            moved = state.copy()
            moved[node] = colour
            there = log_ratios(
                local_fields(rows, moved, neighbours, couplings, owner), moved
            )
            reverse_weights = log_balanced_weights(there, moved, function=function)
            reverse_total = log_normalizer(reverse_weights)

            realized = log_metropolis_ratio(
                float(here[node, colour]),
                float(forward_weights[node, colour]),
                forward_total,
                float(reverse_weights[node, int(state[node])]),
                reverse_total,
            )

            assert realized == pytest.approx(forward_total - reverse_total, abs=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("move", BALANCED)
def test_dropping_the_metropolis_correction_is_caught(
    move: PottsMove, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Evidence that the chi-square tests above have the power they claim. A
    # locally balanced proposal accepted unconditionally is stationary at
    # `pi(s) Z(s)` rather than at `pi(s)`, and the normalizer varies over the
    # 16 configurations of a 2x2 lattice in a field, so the enumeration
    # rejects it.
    def always_accept(*arguments: float) -> float:
        del arguments
        return 0.0

    monkeypatch.setattr(potts_mcmc, "log_metropolis_ratio", always_accept)

    assert _goodness_of_fit(move, WITH_FIELD) < SIGNIFICANCE


# The instance at the transition, declared rather than built here (issue
# #413). `potts_lattice/stress` is the 12x12 open square at the exact
# 3-state transition in zero field, and section 9 of
# `docs/nb/potts_chain.ipynb` reads the same file, so the notebook and this
# module share one instance rather than two literals that agree.
CRITICAL = fixture("potts_lattice", "stress").params

#: A Wolff sweep flips one cluster while the other two move every site, so
#: Wolff runs this many times the declared sweeps and burn-in to give the
#: three chains comparable work. A property of the measurement, not of the
#: instance.
WOLFF_SWEEPS = 8


def _critical_lattice() -> PottsGraph:
    """The declared lattice at the transition."""
    return lattice_graph(CRITICAL.shape, CRITICAL.boundary, CRITICAL.coupling)


def _autocorrelation_in_site_updates(move: PottsMove, graph: PottsGraph) -> float:
    """Energy autocorrelation time, normalized to the work a sweep costs.

    Reporting all three in sweeps would make Wolff look free --- one Wolff
    sweep flips one cluster --- so each is scaled by the sites its sweep
    touched.
    """
    field = CRITICAL.field
    factor = WOLFF_SWEEPS if move is PottsMove.WOLFF else 1

    chain: PottsChain = sample_potts(
        graph,
        field,
        move,
        np.random.default_rng(CRITICAL.seed),
        CRITICAL.n_samples * factor,
        burn_in=CRITICAL.burn_in * factor,
    )
    tau = integrated_autocorrelation_time(energies(graph, field, chain.states))
    return tau * chain.mean_cluster_size / graph.n_nodes


@pytest.mark.analytic
def test_cluster_updates_decorrelate_faster_at_the_transition() -> None:
    # The ordering is the claim; the seeded values are pinned beside it
    # because the sibling test below is worth nothing unless both halves of
    # the comparison are pinned the same way.
    #
    # What is *not* claimed is the ratio: it widens with lattice extent
    # (docs/experiments/001-potts-cluster-autocorrelation.md), so a ratio
    # pinned here would pin a finite-size effect.
    graph = _critical_lattice()

    single = _autocorrelation_in_site_updates(PottsMove.SINGLE_SITE, graph)
    swendsen_wang = _autocorrelation_in_site_updates(PottsMove.SWENDSEN_WANG, graph)
    wolff = _autocorrelation_in_site_updates(PottsMove.WOLFF, graph)

    assert single == pytest.approx(6.70, rel=CRITICAL.tolerance)
    assert swendsen_wang == pytest.approx(4.17, rel=CRITICAL.tolerance)
    assert wolff == pytest.approx(2.34, rel=CRITICAL.tolerance)
    assert swendsen_wang < single
    assert wolff < single


@pytest.mark.analytic
def test_the_cluster_advantage_is_absent_at_the_registry_instance() -> None:
    # The other half of the claim above, and why the comparison runs at the
    # transition. The registry's lattice is 9 sites at J = 0.6, well below
    # J_c = 1.005: the
    # correlation length is shorter than the lattice, there is no critical
    # slowing to remove, and the cluster moves buy nothing for the bonds they
    # cost. Pinned as the measurement `docs/nb/potts_chain.ipynb` reports, so
    # the notebook states nothing the suite does not.
    instance = fixture("potts_lattice", "ci").params
    graph = lattice_graph(instance.shape, instance.boundary, instance.coupling)
    field = instance.field - math.log(float(np.exp(instance.field).sum()))
    sweeps = 4_000

    times: dict[PottsMove, float] = {}
    for move in (PottsMove.SINGLE_SITE, PottsMove.SWENDSEN_WANG, PottsMove.WOLFF):
        chain = sample_potts(
            graph, field, move, np.random.default_rng(7), sweeps, burn_in=sweeps // 5
        )
        tau = integrated_autocorrelation_time(energies(graph, field, chain.states))
        times[move] = tau * chain.mean_cluster_size / graph.n_nodes

    assert times[PottsMove.SINGLE_SITE] == pytest.approx(1.00, rel=0.1)
    assert times[PottsMove.SWENDSEN_WANG] == pytest.approx(1.71, rel=0.1)
    assert times[PottsMove.WOLFF] == pytest.approx(1.59, rel=0.1)
    assert times[PottsMove.SINGLE_SITE] < min(
        times[PottsMove.SWENDSEN_WANG], times[PottsMove.WOLFF]
    )


@pytest.mark.analytic
def test_a_wolff_cluster_is_smaller_than_the_lattice_but_larger_than_a_site() -> None:
    # What makes the normalization above necessary, pinned so a change that
    # made every cluster a single site -- which would silently turn Wolff into
    # an expensive single-site sampler -- is visible.
    # Extent 8 rather than the declared 12: the claim is about the cluster
    # construction, and a smaller lattice makes it in a quarter of the sweeps.
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, critical_coupling(3))

    chain = sample_potts(
        graph,
        np.zeros(3),
        PottsMove.WOLFF,
        np.random.default_rng(7),
        2_000,
        burn_in=200,
    )

    assert 1.0 < chain.mean_cluster_size < graph.n_nodes


# --- temperature ------------------------------------------------------------

TEMPERATURES = [2.0, 0.5]


@pytest.mark.oracle
@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_tempering_is_model_scaling_exactly(temperature: float) -> None:
    # The coupling absorbs beta, so the scaled model's energy is the energy
    # over T and the deviation is 0.0 rather than a tolerance: one division
    # per term.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    scaled_graph, scaled_field = tempered(graph, WITH_FIELD, temperature)

    scaled = energies(scaled_graph, scaled_field, configurations)
    expected = energies(graph, WITH_FIELD, configurations) / temperature

    assert np.abs(scaled - expected).max() == 0.0


@pytest.mark.oracle
@pytest.mark.parametrize("move", list(PottsMove))
@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_a_tempered_chain_is_drawn_from_the_tempered_boltzmann_distribution(
    move: PottsMove, temperature: float
) -> None:
    # Refereed by `tests._chains.enumerated_law`, the enumeration of all 16
    # configurations at the same temperature, which shares no sweep, cluster
    # or accept step with the sampler.
    # Hot and cold, in a field, for every move set: the bond probabilities
    # and the field accept step are tempered by the same division as the heat
    # bath. Realized p-values over two seeds range 0.016 to 0.89 against the
    # 0.001 significance the untempered tests use.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, probability = enumerated_law(graph, WITH_FIELD, temperature)
    sweeps = SWEEPS_BY_MOVE[move]

    chain = sample_potts(
        graph,
        WITH_FIELD,
        move,
        np.random.default_rng(SEED),
        sweeps,
        burn_in=sweeps // 10,
        thin=THINNING[move],
        temperature=temperature,
    )
    observed = np.zeros(len(probability))
    for row in chain.states:
        observed[index[tuple(row)]] += 1

    assert chi_square_p_value(observed, probability * sweeps) > SIGNIFICANCE


@pytest.mark.smoke
def test_a_non_positive_temperature_is_refused() -> None:
    # At zero the heat bath is an argmin and the chain is a descent that
    # samples nothing; a negative temperature inverts the model.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="temperature must be positive"):
        sample_potts(
            graph,
            NO_FIELD,
            PottsMove.SINGLE_SITE,
            np.random.default_rng(SEED),
            10,
            temperature=0.0,
        )
    with pytest.raises(ValueError, match="temperature must be positive"):
        tempered(graph, NO_FIELD, -1.0)


@pytest.mark.oracle
@pytest.mark.release
def test_annealing_reaches_the_closed_form_ground_energy_where_descent_does_not() -> (
    None
):
    # The optimizer built from the sampler, against the frustrated instance
    # with a ground-state energy known at every size: the periodic triangular
    # antiferromagnet, where at least one edge in three is unsatisfied
    # (`sim.canonical.minimum_frustrated_edges`). At 9x9 over 20
    # seeds and 200 sweeps: annealing 20/20, single-site descent (ICM) 2/20,
    # and the same 200 sweeps at a *constant* temperature of 1 -- the control
    # that separates the schedule from the wandering -- 7/20.
    #
    # What this does not say: that annealing beats descent at equal budget.
    # ICM converges in 2.6 sweeps here, so 200 sweeps buy 78 restarts, and
    # the best of 78 also reaches the ground state 20/20. The comparison at
    # equal evaluations, on instances restarts can lose, is #267's second
    # pull request.
    graph = frustrated_triangular_lattice((9, 9), BoundaryCondition.PERIODIC, -1.0)
    field = np.zeros(2)
    ground = float(minimum_frustrated_edges(graph))  # |J| = 1
    schedule = ExponentialTempSchedule(2.0, 0.05, 200)

    annealed = [
        anneal_potts(graph, field, schedule, np.random.default_rng(seed))
        for seed in range(20)
    ]
    constant = [
        anneal_potts(
            graph, field, ConstantTempSchedule(1.0, 200), np.random.default_rng(seed)
        )
        for seed in range(20)
    ]
    descended = [
        iterated_conditional_modes(graph, field, 2, np.random.default_rng(seed)).energy
        for seed in range(20)
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
        index, probability = enumerated_law(graph, field, temperature)
        p_values.append(
            fit_p_value(index, probability, run.states[:, replica], run.states.shape[0])
        )
    return p_values


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
def test_every_replica_is_drawn_from_its_own_tempered_distribution(
    backend: Backend,
) -> None:
    # Refereed by the same enumeration, once per rung of the ladder.
    # The joint target is a product of tempered marginals, so with exchanges
    # *on* each replica must still pass the chi-square against exp(-E / T_r)
    # enumerated from the unscaled model; a wrong exchange ratio contaminates
    # the cold replica with hot configurations. Realized p-values over two
    # seeds: 0.024 to 0.70 at the 0.001 significance; exchange acceptance
    # 0.78 and 0.57 for the two pairs, so the exchanges happen and the test
    # has power.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    run = parallel_tempering(
        graph,
        WITH_FIELD,
        LADDER,
        np.random.default_rng(SEED),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=5,
        backend=backend,
    )

    assert run.states.shape == (SWEEPS, len(LADDER), graph.n_nodes)
    assert bool((run.swap_acceptance > 0.3).all()), run.swap_acceptance
    assert min(_replica_p_values(run, graph, WITH_FIELD)) > SIGNIFICANCE


@pytest.mark.smoke
def test_omitting_the_exchange_term_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # The negative case, paired with the positive one above. An exchange that
    # ignores (beta_i - beta_j)(E_i - E_j) still runs and still mixes -- every
    # exchange is accepted -- and every replica's marginal is the wrong
    # distribution: realized p = 0.0 at all three temperatures.
    def always_exchange(*_: float) -> float:
        return 0.0

    monkeypatch.setattr(potts_mcmc, "swap_log_ratio", always_exchange)
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    run = parallel_tempering(
        graph,
        WITH_FIELD,
        LADDER,
        np.random.default_rng(SEED),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=5,
    )

    assert bool((run.swap_acceptance == 1.0).all())
    assert max(_replica_p_values(run, graph, WITH_FIELD)) < SIGNIFICANCE


@pytest.mark.analytic
def test_replicas_draw_from_separate_streams_and_one_seed_reproduces_them() -> None:
    # Two replicas at the *same* temperature with no field would be identical
    # chains if they shared a stream, while every diagnostic looked healthy.
    # Spawned children differ; the parent seed still reproduces the run
    # bitwise.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    first = parallel_tempering(
        graph, NO_FIELD, (1.0, 1.0), np.random.default_rng(3), 200
    )
    second = parallel_tempering(
        graph, NO_FIELD, (1.0, 1.0), np.random.default_rng(3), 200
    )

    assert not np.array_equal(first.states[:, 0], first.states[:, 1])
    assert np.array_equal(first.states, second.states)


@pytest.mark.oracle
def test_the_best_configuration_is_the_lowest_energy_any_replica_visited() -> None:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    run = parallel_tempering(
        graph, WITH_FIELD, LADDER, np.random.default_rng(SEED), 300
    )

    visited = energies(graph, WITH_FIELD, run.states.reshape(-1, graph.n_nodes))
    assert run.best_energy == pytest.approx(
        energies(graph, WITH_FIELD, run.best[None])[0]
    )
    assert run.best_energy <= visited.min() + 1e-12


@pytest.mark.oracle
@pytest.mark.critical
def test_tempering_reaches_the_ground_energy_annealing_reaches() -> None:
    # The rung below (issue #734), on the rung below's own instance: the
    # periodic triangular antiferromagnet, whose ground energy is |J| * N at
    # every size (`sim.canonical.minimum_frustrated_edges`). Annealing
    # reaches it here from every seed, and the claim is that the coldest
    # replica of a tempered run reaches the same energy at the same budget --
    # 200 sweeps, spent as one annealed chain or as four replicas of 50.
    #
    # The coldest replica rather than `best_energy`: an exchange moves
    # configurations between temperatures, so the run's best can be a state
    # only a hot replica ever held, and what the cold end returns is the
    # claim. Realized over six seeds: every annealed run and every coldest
    # replica at 81.0 exactly, difference 0.0 against the 1e-12 declared.
    # The cold pair exchanges at 0.26 to 0.38 of proposals, asserted so the
    # ladder is known to be four coupled chains rather than four independent
    # ones; the middle pair reaches 0.0 on one seed, which is why the
    # acceptance is read at the cold end rather than over every pair.
    graph = frustrated_triangular_lattice((9, 9), BoundaryCondition.PERIODIC, -1.0)
    field = np.zeros(2)
    ground = float(minimum_frustrated_edges(graph))  # |J| = 1
    ladder = (2.0, 1.0, 0.5, 0.25)
    coldest = int(np.argmin(ladder))

    annealed = [
        anneal_potts(
            graph,
            field,
            ExponentialTempSchedule(2.0, 0.05, 200),
            np.random.default_rng(seed),
        )
        for seed in range(6)
    ]
    tempered_runs = [
        parallel_tempering(graph, field, ladder, np.random.default_rng(seed), 50)
        for seed in range(6)
    ]

    for annealing, run in zip(annealed, tempered_runs, strict=True):
        assert annealing.energy == pytest.approx(ground, abs=1e-12)
        visited = energies(graph, field, run.states[:, coldest])
        assert float(visited.min()) == pytest.approx(annealing.energy, abs=1e-12)
        assert run.best_energy == pytest.approx(ground, abs=1e-12)
        assert run.swap_acceptance[coldest - 1] > 0.2, run.swap_acceptance


@pytest.mark.smoke
def test_a_ladder_of_one_or_a_cold_temperature_is_refused() -> None:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="at least two temperatures"):
        parallel_tempering(graph, NO_FIELD, (1.0,), np.random.default_rng(SEED), 10)
    with pytest.raises(ValueError, match="positive temperature"):
        parallel_tempering(graph, NO_FIELD, (1.0, 0.0), np.random.default_rng(SEED), 10)


@pytest.mark.end2end
def test_tempering_and_annealing_beat_restarts_at_equal_budget_on_the_glass() -> None:
    # The instance where restarts can lose: the planted Viana-Bray spin
    # glass, 60 sites at mean degree 4 and frustration 0.2, whose planted
    # energy upper-bounds a ground state enumeration cannot reach. Budget:
    # 400 heat-bath sweeps per method, held equal by `opt.budget.compare`
    # (issue #281) -- annealing on one chain, tempering on four replicas of
    # 100, single-site descent on 100 restarts of at most 4 sweeps (it
    # converges in 2 to 4). Realized over 12 instances, against the best
    # energy any method found: tempering 12/12, annealing 10/12, restarts
    # 4/12 with a mean gap of 0.75. The plan predicted tempering would be
    # hard to justify at these sizes; it is not, and the prediction is
    # retracted.
    #
    # Asserted at the margin the measurement supports: restarts below both,
    # and the two tempered methods at or below the planted energy on every
    # instance. The referee is that planted energy: `planted_spin_glass`
    # plants a configuration and reports its energy, so a solver at or below
    # it has recovered the truth the instance was generated from.
    budget = Budget(Cost.SWEEPS, 400)
    ladder = (2.0, 1.2, 0.7, 0.4)
    instances = [
        planted_spin_glass(60, 4.0, 0.2, np.random.default_rng(1000 + seed))
        for seed in range(12)
    ]
    planted = np.array([instance.planted_energy for instance in instances])

    def descent(
        instance: PlantedSpinGlass, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        energy = iterated_conditional_modes(
            instance.graph, np.zeros(2), 2, rng, max_sweeps=budget.size
        ).energy
        return Outcome(energy, budget.size)

    def anneal(
        instance: PlantedSpinGlass, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        run = anneal_potts(
            instance.graph,
            np.zeros(2),
            ExponentialTempSchedule(2.0, 0.05, budget.size),
            rng,
        )
        return Outcome(run.energy, budget.size)

    def tempering(
        instance: PlantedSpinGlass, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        per_replica = budget.size // len(ladder)
        run = parallel_tempering(instance.graph, np.zeros(2), ladder, rng, per_replica)
        return Outcome(run.best_energy, per_replica * len(ladder))

    result = compare(
        {"restarts": restarts(descent, 4), "anneal": anneal, "tempering": tempering},
        instances,
        budget,
        seeds=(0,),
        workers=1,
    )
    hits = result.hits()

    for row, name in enumerate(result.methods):
        if name != "restarts":
            assert bool((result.best[row] <= planted + 1e-9).all()), name
    assert hits["restarts"] < hits["anneal"], hits
    assert hits["restarts"] < hits["tempering"], hits
    assert hits["anneal"] >= 10, hits
    assert hits["tempering"] >= 10, hits


@pytest.mark.smoke
def test_the_sweep_has_no_numba_backend() -> None:
    # The descent has one and the sampler does not: a sampler's pin is
    # distributional, and root CLAUDE.md admits one compiled path per
    # measurement. Refused by name rather than falling back silently.
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="no numba backend"):
        anneal_potts(
            graph,
            NO_FIELD,
            ConstantTempSchedule(1.0, 5),
            np.random.default_rng(0),
            backend=Backend.NUMBA,
        )


# --- a ladder from its own exchange acceptance (#333) ------------------------

#: The band the adapted ladder is driven into. Each acceptance is a fraction
#: of 50 exchange proposals, a binomial sd of 0.07 at 0.5, and a band this
#: wide is one that noise cannot keep a settled ladder out of: measured over
#: 20 seeds every warm-up settled inside it, in 4.2 rounds on average.
BAND = (0.25, 0.75)
PROBE_SWEEPS = 50
HAND_LADDER = (2.0, 1.2, 0.7, 0.4)


@pytest.mark.smoke
@at_scale("n_seeds", ci=10, stress=20)
def test_the_adapted_ladder_exchanges_within_the_band_on_the_frustrated_lattice(
    n_seeds: int,
) -> None:
    # The 9x9 periodic triangular antiferromagnet, from the endpoints alone.
    # The warm-up reports every pair inside the band, and a *fresh* run on
    # the ladder it returned -- a different seed, four times the sweeps --
    # exchanges inside the band widened by the measurement's noise. The
    # second is what matters: a warm-up that stopped on a lucky measurement
    # would pass the first alone. Realized over 20 seeds: 5 to 8 rungs,
    # fresh acceptances 0.16 to 0.72.
    # The hand ladder's pairs, for comparison, exchange at 0.28, 0.15 and
    # 0.12 -- two of three below the band.
    graph = frustrated_triangular_lattice((9, 9), BoundaryCondition.PERIODIC, -1.0)
    field = np.zeros(2)

    for seed in range(n_seeds):
        adapted = adapt_ladder_potts(
            graph,
            field,
            (2.0, 0.4),
            np.random.default_rng(seed),
            PROBE_SWEEPS,
            BAND,
            10,
            12,
            backend=Backend.RUST,
        )
        assert adapted.within_band, adapted
        assert adapted.temperatures[0] == 2.0
        assert adapted.temperatures[-1] == 0.4
        assert all(BAND[0] <= value <= BAND[1] for value in adapted.acceptance)

        fresh = parallel_tempering(
            graph,
            field,
            adapted.temperatures,
            np.random.default_rng(1000 + seed),
            4 * PROBE_SWEEPS,
            backend=Backend.RUST,
        )
        assert bool((fresh.swap_acceptance > 0.1).all()), fresh.swap_acceptance
        assert bool((fresh.swap_acceptance < 0.9).all()), fresh.swap_acceptance


@pytest.mark.oracle
@at_scale("n_seeds", ci=10, stress=20)
def test_the_adapted_ladder_reaches_the_ground_state_at_equal_sweeps(
    n_seeds: int,
) -> None:
    # Refereed by `sim.canonical.minimum_frustrated_edges`, the closed-form
    # ground energy |J| * N of the periodic triangular antiferromagnet --- the
    # same referee the annealing and tempering rungs above are judged against
    # --- so a hit is the exact minimum and not a solver's own best.
    # The warm-up is charged: 2400 sweeps per seed, of which the adapted
    # ladder spends its warm-up (895 on average, 500 to 1900) and splits the
    # rest across its rungs, while the hand ladder spends 600 per replica on
    # four. Realized over 20 seeds: adapted 20/20, hand 20/20; at 1600 sweeps
    # 19/20 against 20/20, and the hand ladder hits 18/20 at 100 sweeps, so
    # the instance does not separate them. What the adapted ladder buys is
    # the band, which the hand ladder is outside on two pairs. Asserted at
    # the margin the measurement supports.
    graph = frustrated_triangular_lattice((9, 9), BoundaryCondition.PERIODIC, -1.0)
    field = np.zeros(2)
    ground = float(minimum_frustrated_edges(graph))
    budget = 2400

    adapted_hits = 0
    for seed in range(n_seeds):
        # The warm-up is run here and not read from a helper the smoke test
        # above shares: it is half of what this test judges -- the adapted
        # ladder is charged for the sweeps it spends adapting -- and a
        # cached run is recorded against whichever test asked for it first,
        # so sharing it across the judged boundary moved the four statements
        # of `adapt_ladder_potts` out of the judged set (issue #745, #729's
        # floor was measured before #735 merged).
        rng = np.random.default_rng(seed)
        adapted = adapt_ladder_potts(
            graph,
            field,
            (2.0, 0.4),
            rng,
            PROBE_SWEEPS,
            BAND,
            10,
            12,
            backend=Backend.RUST,
        )
        remaining = budget - adapted.replicas_measured * PROBE_SWEEPS
        assert remaining > 0, adapted
        run = parallel_tempering(
            graph,
            field,
            adapted.temperatures,
            rng,
            remaining // len(adapted.temperatures),
            backend=Backend.RUST,
        )
        adapted_hits += abs(run.best_energy - ground) < 1e-12
    hand_hits = sum(
        abs(
            parallel_tempering(
                graph,
                field,
                HAND_LADDER,
                np.random.default_rng(seed),
                budget // len(HAND_LADDER),
                backend=Backend.RUST,
            ).best_energy
            - ground
        )
        < 1e-12
        for seed in range(n_seeds)
    )

    assert adapted_hits >= n_seeds - 2, (adapted_hits, hand_hits)
    assert hand_hits >= n_seeds - 2, (adapted_hits, hand_hits)


# --- the Rust sweep's field, and where beta is applied (issue #571) -----------


def _swept(
    graph: PottsGraph, rows: np.ndarray, backend: Backend, beta: float
) -> np.ndarray:
    """Five sweeps from one seed, on the backend named.

    The adjacency is passed in the compressed form both backends now read
    (issue #277); the list-of-lists this test used to build for the Python
    closure is the shape that builder replaced.
    """
    offsets, neighbours, couplings = graph.compressed_adjacency()
    sweep = potts_mcmc._sweep_at(rows, offsets, neighbours, couplings, backend)
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    rng = np.random.default_rng(7)
    for _ in range(5):
        sweep(state, rng, beta)
    return state


@pytest.mark.oracle
@pytest.mark.parametrize("beta", [1.0, 0.37, 2.5], ids=lambda value: f"beta{value}")
@pytest.mark.parametrize("field", ["shared", "per_site"])
def test_the_backends_agree_bitwise_at_every_temperature(
    field: str, beta: float
) -> None:
    # Two failures this covers, and the second is why `beta` moved into the
    # kernel. The per-site field the Rust sweep could not express at all
    # (issue #571); and the *order* `beta` is applied in, which the caller used
    # to get wrong by pre-scaling the arguments -- computing `beta * h + sum
    # (beta * J)` where the oracle computes `(h + sum J) * beta`. Those agree
    # in real arithmetic and differ in the last bits, so the old arrangement
    # could only ever have been bitwise at beta = 1.0, which is the only place
    # it was tested.
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.7)
    rows = (
        site_field(np.array([0.3, -0.2, 0.5]), graph.n_nodes)
        if field == "shared"
        else np.random.default_rng(11).normal(size=(graph.n_nodes, 3))
    )

    np.testing.assert_array_equal(
        _swept(graph, rows, Backend.PYTHON, beta),
        _swept(graph, rows, Backend.RUST, beta),
    )


@pytest.mark.smoke
def test_the_kernel_names_the_shape_it_wanted_and_the_shape_it_got() -> None:
    # PyO3 reports a dimensionality mismatch as "'ndarray' object is not an
    # instance of 'ndarray'", which names neither shape (issue #571).
    from snakes_and_ladders import oxisal

    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.5)
    offsets, index, couplings = graph.compressed_adjacency()
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    draws = np.full(graph.n_nodes, 0.5)

    with pytest.raises(ValueError, match="one row per site"):
        oxisal.single_site_sweeps(
            state,
            np.zeros((graph.n_nodes + 1, 3)),
            offsets,
            index,
            couplings,
            draws,
            1,
            1.0,
            GUARD,
            0,
        )
    with pytest.raises(ValueError, match="beta must be finite"):
        oxisal.single_site_sweeps(
            state,
            np.zeros((graph.n_nodes, 3)),
            offsets,
            index,
            couplings,
            draws,
            1,
            float("nan"),
            GUARD,
            0,
        )


@pytest.mark.smoke
def test_a_field_of_the_wrong_dimensionality_names_its_shape() -> None:
    # PyO3 would reject a 1-D field before the kernel body, as "'ndarray'
    # object is not an instance of 'ndarray'" (issue #571). The field is taken
    # as a dynamic array so the refusal names the shape instead.
    from snakes_and_ladders import oxisal

    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.5)
    offsets, index, couplings = graph.compressed_adjacency()

    with pytest.raises(ValueError, match=r"must be 2-D.*got shape \[3\]"):
        oxisal.single_site_sweeps(
            np.zeros(graph.n_nodes, dtype=np.int64),
            np.zeros(3),
            offsets,
            index,
            couplings,
            np.full(graph.n_nodes, 0.5),
            1,
            1.0,
            GUARD,
            0,
        )


@pytest.mark.smoke
@pytest.mark.backend
@pytest.mark.parametrize("move", [PottsMove.WOLFF, PottsMove.NIEDERMAYER])
def test_the_adjacency_converted_once_is_the_per_step_stream_bitwise(
    move: PottsMove,
) -> None:
    # Issue #919: the lists a chain builds once are the ones each step used to
    # build, so 300 steps at three temperatures draw the same stream and leave
    # the same state and the same cluster sizes.
    graph = lattice_graph((6, 6), BoundaryCondition.OPEN, COUPLING)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rows = site_field(np.random.default_rng(1).normal(0.0, 0.6, (36, 2)), graph.n_nodes)
    lists = potts_mcmc.adjacency_lists(offsets, neighbours, couplings)
    step = (
        potts_mcmc.wolff_sweep
        if move is PottsMove.WOLFF
        else potts_mcmc.niedermayer_sweep
    )
    for beta in (0.5, 1.0, 2.0):
        runs = []
        for held in (None, lists):
            rng = np.random.default_rng(919)
            state = rng.integers(0, 2, graph.n_nodes)
            sizes = [
                step(
                    state,
                    rows,
                    offsets,
                    neighbours,
                    couplings,
                    rng,
                    beta=beta,
                    lists=held,
                )
                for _ in range(300)
            ]
            runs.append((state.copy(), sizes))
        assert np.array_equal(runs[0][0], runs[1][0]), beta
        assert runs[0][1] == runs[1][1], beta
