"""Correctness by distribution, not by inspection of cluster sizes.

At an enumerable size each move set's visit frequencies are tested against the
exact Boltzmann law by chi-square at a declared significance and length.
`test_dropping_the_field_accept_step_is_caught` shows the tests have power.
Thinning is part of the test: at `thin = 1` correct chains returned p = 0.038
(single-site) and 0.0024 (Swendsen-Wang); Wolff needs more, flipping one
cluster per sweep.
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
from snakes_and_ladders.sample.potts_mcmc import (
    ClusterCounter,
    PottsChain,
    PottsMove,
    TemperedChains,
    adapt_ladder_potts,
    anneal_potts,
    autodiff_log_ratios,
    chains,
    energies,
    houdayer_cluster,
    niedermayer_threshold,
    parallel_tempering,
    sample_potts,
    sample_potts_pair,
    sweeps,
    taylor_log_ratios,
    tempered,
)
from snakes_and_ladders.sample.potts_mcmc.sweeps import GUARD
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
from tests._rows import every_value
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
    # A pass over every site, as Swendsen-Wang's is (issue #1041).
    PottsMove.GHOST_SPIN: 5,
    PottsMove.LABEL_DIRECTED: 5,
}

#: The NumPy gradient-informed sweeps cost seconds; 4,000 sweeps still expect
#: 250 per cell of sixteen, which the chi-square needs.
BALANCED_SWEEPS = 4_000
SWEEPS_BY_MOVE = {
    PottsMove.SINGLE_SITE: SWEEPS,
    PottsMove.SWENDSEN_WANG: SWEEPS,
    PottsMove.WOLFF: SWEEPS,
    PottsMove.NIEDERMAYER: SWEEPS,
    PottsMove.GHOST_SPIN: SWEEPS,
    PottsMove.LABEL_DIRECTED: SWEEPS,
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
        called.append(None)
        state[members] = int(rng.integers(rows.shape[1]))
        return potts_mcmc.Recolour(proposed=True, accepted=True)

    # The sweeps read `_recolour` from `sweeps`' globals; the package's copy
    # of a name is not the one they call (issue #1010).
    called: list[None] = []
    monkeypatch.setattr(sweeps, "_recolour", unconditional)

    assert _goodness_of_fit(move, FIELDS[field]) < SIGNIFICANCE
    assert called, "the stub never ran: the patch missed the sweep"


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
    # The refusal is the cluster construction's: the heat bath at J = -0.5 is
    # still exact, p = 0.524 and 0.608 over two seeds (0.001).
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
# Zanella (2020) and Grathwohl et al. (2021) are one kernel on this energy: the
# relaxed log weight is affine per site's row, so the Taylor estimate is the
# single-flip difference, pinned three ways below.


def _wider_lattice() -> PottsGraph:
    """The 3x3 open square: 512 configurations, and an interior site of degree 4.

    Weaker coupling spreads the law over the cells, so chi-square measures sampling.
    """
    return lattice_graph((3, 3), BoundaryCondition.OPEN, 0.4)


def _frustrated_lattice() -> PottsGraph:
    """The 3x3 periodic triangular antiferromagnet: every coupling negative.

    Both cluster moves refuse it; a frustrated law tests the gradient-informed moves.
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
# Issue #756. Wolff is refused on the frustrated lattice. Niedermayer (1988)
# activates bonds against a threshold `E_0`, with a Metropolis ratio;
# Houdayer (2001) moves a replica pair, isoenergetic, accepted at 1 by identity.
# Referees: the 512-configuration laws, ablations, and the two identities.


#: The frustrated lattice with half its bonds ferromagnetic: its threshold
#: cluster percolates (8.94 of 9 sites), so a chain is rejected at p = 9e-218;
#: refereed by reversibility instead.
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
    # Enumeration by `log_weights` on the instances Wolff refuses; `release`
    # (275,000 Python cluster growths), with the 2x2 sibling per PR. Realized
    # p over two seeds: 3x3-open 0.0221, 0.1116; triangular 0.0936, 0.9945.
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
    # One replica's marginal: the product law is invariant under the
    # involution. Both replicas read. p per replica over two seeds:
    # single-site 3x3-open 0.5086/0.1672, 0.9580/0.0171; triangular
    # 0.0039/0.9058, 0.8477/0.9913. Niedermayer 3x3-open 0.5061/0.1717,
    # 0.1394/0.7302; triangular 0.3116/0.9231, 0.3202/0.0386.
    graph = ENUMERABLE[instance]()

    first, second = _pair_chi_square(graph, WITH_FIELD, move, SEED)

    assert min(first, second) > SIGNIFICANCE, (first, second)


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize(
    ("name", "coupling", "above"),
    [
        ("ferromagnet", (0.8, 0.8, 0.8, 0.8), 0.0),
        ("antiferromagnet", (-1.0, -1.0, -1.0, -1.0), 0.0),
        ("mixed", (0.7, -1.0, 0.7, -1.0), 0.0),
        # Above the threshold unlike neighbours bond too, which is what the
        # #1046 search varies on the notebook's ferromagnet.
        ("ferromagnet, E_0 = 0.4", (0.8, 0.8, 0.8, 0.8), 0.4),
        ("ferromagnet, E_0 = 1.6", (0.8, 0.8, 0.8, 0.8), 1.6),
    ],
)
def test_niedermayers_kernel_is_reversible_against_the_enumerated_law(
    name: str, coupling: tuple[float, ...], above: float
) -> None:
    # `pi(s) K(s, s')` against its transpose, `K` from 12,500 moves out of each
    # of 16 four-cycle states. Max asymmetry: ferromagnet 0.00137,
    # antiferromagnet 0.00080, mixed 0.00160, ferromagnet at E_0 = 0.4
    # 0.00124 and at 1.6 0.00045; the bound is the Monte Carlo error
    # 1 / sqrt(12,500) = 0.0089.
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
    threshold = niedermayer_threshold(couplings) + above
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
    # Mean cluster over 4,000: mixed 8.94 of 9 sites, antiferromagnet 8.31,
    # ferromagnet 2.25; a whole-lattice cluster is a global reversal.
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
    # At `E_0 = 0` an antiferromagnet forms no bond: the ratio is the exact
    # single-flip difference, read against `energies`.
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
    # Ablation (#756): at the threshold the boundary terms cancel, so this
    # removes the field term; p = 0.0 against 0.0936 unablated.
    def unconditional(delta: float, beta: float, rng: np.random.Generator) -> bool:
        # The accept step gone, both terms with it: the field difference the
        # ferromagnetic case also carries, and the boundary terms only a mixed
        # instance has.
        del delta, beta, rng
        called.append(None)
        return True

    called: list[None] = []
    monkeypatch.setattr(sweeps, "_niedermayer_accept", unconditional)

    p_value = _chi_square_against(
        _frustrated_lattice(),
        WITH_FIELD,
        PottsMove.NIEDERMAYER,
        SEED,
        sweeps=ABLATION_SWEEPS,
    )

    assert p_value < SIGNIFICANCE, p_value
    assert called, "the stub never ran: the patch missed the sweep"


@pytest.mark.smoke
def test_swapping_a_site_rather_than_a_component_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Singletons instead of Houdayer's component break isoenergy: p = 0.0 on
    # the first replica against 0.0039 unablated.
    def singletons(
        first: np.ndarray,
        second: np.ndarray,
        offsets: np.ndarray,
        neighbours: np.ndarray,
    ) -> np.ndarray:
        del first, second, neighbours
        called.append(None)
        return np.arange(int(offsets.shape[0]) - 1)

    called: list[None] = []
    monkeypatch.setattr(sweeps, "houdayer_cluster", singletons)

    first, second = _pair_chi_square(
        _frustrated_lattice(), WITH_FIELD, PottsMove.SINGLE_SITE, SEED
    )

    assert min(first, second) < SIGNIFICANCE, (first, second)
    assert called, "the stub never ran: the patch missed the move"


@pytest.mark.analytic
@pytest.mark.critical
def test_houdayers_move_leaves_the_pairs_energy_where_it_found_it() -> None:
    # Inside bonds, leaving bonds and field terms are exchanged: worst |dE|
    # over 500 pairs 3.6e-15 at energies ~20, the double-precision floor.
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
    # Houdayer alone keeps each site's unordered pair, confined to a 2**n orbit:
    # hence the within-replica move set.
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
def test_niedermayers_rule_is_wolffs_bitwise_on_a_ferromagnet() -> None:
    # On a ferromagnet the threshold is 0 and Niedermayer is Wolff: equal
    # labellings over 300 draws at three temperatures, `q = 3`, 4x4 open,
    # root and colour named.
    def check(temperature: float) -> None:
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

    every_value([0.25, 1.0, 4.0], check)


@pytest.mark.analytic
@pytest.mark.critical
def test_the_threshold_is_where_the_antiferromagnets_bonds_become_a_probability() -> (
    None
):
    # `E_0 = max(0, -min J)` is the smallest threshold at which every bond
    # probability is defined.
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
    # Frustrated: Wolff refused, Niedermayer's cluster 8.31 of 9 sites; on the
    # ferromagnet the two agree on both readings.
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
    # Enumeration by `log_weights`; `release` (360,000 NumPy proposals), 2x2
    # sibling per PR. p over two seeds: 3x3-open 0.0240, 0.2629; triangular
    # 0.0758, 0.9926.
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
    # The Taylor-weighted route reaches the same law: p identical to four
    # figures on every instance and seed (same uniforms, same weights: the
    # same chain).
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
    # Refereed by the heat bath's conditional, the tape's gradient and the
    # enumerated energy: none is the sampler's arithmetic.
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
    # Zanella: `g(t) = t g(1 / t)` collapses the ratio to `Z(s) / Z(s')` for
    # both balancing functions; the general form is checked to equal it.
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
    # Unconditional acceptance is stationary at `pi(s) Z(s)`, which varies over
    # the 16 states in a field: the enumeration rejects it.
    def always_accept(*arguments: float) -> float:
        del arguments
        called.append(None)
        return 0.0

    called: list[None] = []
    monkeypatch.setattr(sweeps, "log_metropolis_ratio", always_accept)

    assert _goodness_of_fit(move, WITH_FIELD) < SIGNIFICANCE
    assert called, "the stub never ran: the patch missed the sweep"


# `potts_lattice/stress`: 12x12 open at the 3-state transition, zero field,
# shared with `docs/nb/potts_chain.ipynb` section 9 (issue #413).
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
    """Energy autocorrelation time in site updates: a one-cluster Wolff sweep is not free."""
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
    # The ordering is claimed, seeded values pinned; the ratio widens with
    # extent (docs/experiments/001-potts-cluster-autocorrelation.md).
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
    # The registry lattice is 9 sites at J = 0.6, below J_c = 1.005: no
    # critical slowing, clusters buy nothing (`docs/nb/potts_chain.ipynb`).
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
    # Single-site clusters would make Wolff an expensive single-site sampler;
    # extent 8, a quarter of the sweeps of 12.
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
    # `enumerated_law` over 16 states; hot and cold, in a field, every move
    # set: p over two seeds 0.016 to 0.89 (0.001).
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


@pytest.mark.smoke
@pytest.mark.patch
def test_annealing_at_the_graphs_own_threshold_is_the_default_bitwise() -> None:
    # `threshold=None` is `niedermayer_threshold`, the value before the
    # parameter existed (#1046); a threshold above it is another chain.
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, COUPLING)
    field = np.array([0.6, -0.4, 0.1])
    schedule = ExponentialTempSchedule(2.0, 0.05, 200)

    def run(threshold: float | None) -> np.ndarray:
        return anneal_potts(
            graph,
            field,
            schedule,
            np.random.default_rng(SEED),
            move=PottsMove.NIEDERMAYER,
            threshold=threshold,
        ).final

    default = run(None)

    assert np.array_equal(default, run(0.0))
    assert not np.array_equal(default, run(0.5))


@pytest.mark.smoke
@pytest.mark.warning
def test_a_threshold_is_refused_for_a_move_without_one() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, COUPLING)

    with pytest.raises(ValueError, match="Niedermayer"):
        anneal_potts(
            graph,
            WITH_FIELD,
            ExponentialTempSchedule(2.0, 0.05, 5),
            np.random.default_rng(SEED),
            move=PottsMove.WOLFF,
            threshold=0.5,
        )


@pytest.mark.oracle
@pytest.mark.release
def test_annealing_reaches_the_closed_form_ground_energy_where_descent_does_not() -> (
    None
):
    # Periodic triangular antiferromagnet (`minimum_frustrated_edges`), 9x9, 20
    # seeds, 200 sweeps: annealing 20/20, ICM 2/20, constant T = 1 7/20. Not
    # equal budget: ICM converges in 2.6 sweeps, and the best of 78 restarts
    # also reaches 20/20 (#267).
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
    # Each replica against exp(-E / T_r) from the unscaled model: p over two
    # seeds 0.024 to 0.70 (0.001); exchange acceptance 0.78 and 0.57.
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
        called.append(None)
        return 0.0

    # `parallel_tempering` reads `swap_log_ratio` from `chains`' globals.
    called: list[None] = []
    monkeypatch.setattr(chains, "swap_log_ratio", always_exchange)
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
    assert called, "the stub never ran: the patch missed the exchange"


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
    # The rung below (#734): ground energy |J| * N. 200 sweeps as one annealed
    # chain or four replicas of 50; the coldest replica, not `best_energy`.
    # Over six seeds every run at 81.0 exactly (0.0, 1e-12). Cold pair
    # exchanges 0.26-0.38; the middle pair reaches 0.0 on one seed.
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
    # The planted Viana-Bray glass (60 sites, degree 4, frustration 0.2);
    # 400 heat-bath sweeps per method (`opt.budget.compare`, #281): annealing
    # one chain, tempering 4 x 100, descent 100 restarts of <= 4. Over 12
    # instances against the best found: tempering 12/12, annealing 10/12,
    # restarts 4/12 (mean gap 0.75); the plan's prediction is retracted.
    # Asserted: restarts below both; both tempered at or below the planted energy.
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

    with pytest.raises(ValueError, match="runs on python or rust, not numba"):
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
    # 9x9 triangular antiferromagnet: the warm-up's ladder is in band, and a
    # fresh run (another seed, 4x sweeps) exchanges in band widened by noise.
    # Over 20 seeds: 5 to 8 rungs, fresh 0.16 to 0.72; the hand ladder's pairs
    # 0.28, 0.15, 0.12, two below the band.
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
    # Referee `minimum_frustrated_edges` (|J| * N). 2400 sweeps per seed, the
    # warm-up charged (895 mean, 500-1900); hand ladder 600 per replica on
    # four. Over 20 seeds: 20/20 both; at 1600, 19/20 against 20/20; hand
    # 18/20 at 100. The instance does not separate them; the band does.
    graph = frustrated_triangular_lattice((9, 9), BoundaryCondition.PERIODIC, -1.0)
    field = np.zeros(2)
    ground = float(minimum_frustrated_edges(graph))
    budget = 2400

    adapted_hits = 0
    for seed in range(n_seeds):
        # Run here, not shared: the warm-up is half of what is judged, and a
        # shared cache moved `adapt_ladder_potts` out of the judged set (#745).
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
    """Five sweeps from one seed, on the backend named, over compressed adjacency (#277)."""
    offsets, neighbours, couplings = graph.compressed_adjacency()
    sweep = sweeps.sweep_at(rows, offsets, neighbours, couplings, backend)
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    rng = np.random.default_rng(7)
    for _ in range(5):
        sweep(state, rng, beta)
    return state


@pytest.mark.oracle
@pytest.mark.parametrize("field", ["shared", "per_site"])
def test_the_backends_agree_bitwise_at_every_temperature(field: str) -> None:
    # The per-site field (#571), and `beta` inside the kernel: `(h + sum J) *
    # beta`, not `beta * h + sum (beta * J)`, which differ in the last bits
    # away from beta = 1.0.
    def check(beta: float) -> None:
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

    every_value([1.0, 0.37, 2.5], check)


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
