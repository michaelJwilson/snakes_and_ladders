"""Niedermayer's rule and Houdayer's pair with it, held to the enumerated laws, moved with the move to the sandbox (issues #756, #1314, #1315, #1365)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sal.backend import Backend
from sal.likelihood.potts import log_weights
from sal.sample import potts_mcmc
from sal.sample.potts_mcmc import PottsMove, energies
from sal.sample.statistics import chi_square_p_value
from sal.sandbox import potts_moves
from sal.sandbox.potts_moves import SandboxMove, niedermayer_threshold
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, site_field

from tests._chains import cell_counts, enumerated_law
from tests._rows import every_value
from tests.regression.sample.test_potts_mcmc import (
    ABLATION_SWEEPS,
    COUPLING,
    ENUMERABLE,
    KERNEL_TRIALS,
    POOL_BELOW,
    SEED,
    SHAPE,
    SIGNIFICANCE,
    SITE_FIELD,
    SITE_FIELD_RECORDS,
    SITE_FIELD_SHAPE,
    SITE_FIELD_THIN,
    WIDE_SWEEPS,
    WITH_FIELD,
    _chi_square_against,
    _cluster_counter,
    _frustrated_lattice,
    _mixed_lattice,
    _pair_chi_square,
    _wider_lattice,
)


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
        graph, WITH_FIELD, SandboxMove.NIEDERMAYER, SEED, sweeps=WIDE_SWEEPS
    )

    assert p_value > SIGNIFICANCE, p_value


@pytest.mark.oracle
@pytest.mark.release
@pytest.mark.parametrize("instance", sorted(ENUMERABLE))
@pytest.mark.parametrize("move", [PottsMove.SINGLE_SITE, SandboxMove.NIEDERMAYER])
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
    # `pi(s) K(s, s')` against its transpose, `K` from 12,500 moves out of each
    # of 16 four-cycle states. Max asymmetry: ferromagnet 0.00137,
    # antiferromagnet 0.00080, mixed 0.00160; the bound is the Monte Carlo
    # error 1 / sqrt(12,500) = 0.0089.
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
            potts_moves.niedermayer_sweep(
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


@pytest.mark.oracle
def test_niedermayer_is_exact_under_a_per_site_field_above_the_transition() -> None:
    # The law #1278's 64 x 64 corner is read against: #1142 enumerates the
    # uniform field only. Declared before the run: pooled chi-square p above
    # SIGNIFICANCE. The 64 x 64 gap is a frozen chain, not this law (#1314).
    graph = lattice_graph(SITE_FIELD_SHAPE, BoundaryCondition.OPEN, 1.0)
    beta = 1.5 * critical_coupling(3)
    index, probability = enumerated_law(graph, SITE_FIELD, temperature=1.0 / beta)
    rows = site_field(SITE_FIELD, graph.n_nodes, n_states=3)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    advance = potts_moves.sweep_for(
        SandboxMove.NIEDERMAYER, graph, rows, offsets, neighbours, couplings
    )
    rng = np.random.default_rng(SEED)
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    # Burn-in: a tenth of the recorded steps, as `_chi_square_against` discards.
    for _ in range(SITE_FIELD_RECORDS * SITE_FIELD_THIN // 10):
        advance(state, rng, beta)
    visits = []
    for _ in range(SITE_FIELD_RECORDS):
        for _ in range(SITE_FIELD_THIN):
            advance(state, rng, beta)
        visits.append(tuple(state.tolist()))

    observed = cell_counts(index, visits)
    expected = probability * SITE_FIELD_RECORDS
    kept = expected >= POOL_BELOW
    p_value = chi_square_p_value(
        np.append(observed[kept], observed[~kept].sum()),
        np.append(expected[kept], expected[~kept].sum()),
    )

    assert p_value > SIGNIFICANCE, p_value


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
        sizes[name] = _cluster_counter(graph, SandboxMove.NIEDERMAYER).mean_size

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
        size = potts_moves.niedermayer_sweep(
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
    monkeypatch.setattr(potts_moves, "_niedermayer_accept", unconditional)

    p_value = _chi_square_against(
        _frustrated_lattice(),
        WITH_FIELD,
        SandboxMove.NIEDERMAYER,
        SEED,
        sweeps=ABLATION_SWEEPS,
    )

    assert p_value < SIGNIFICANCE, p_value
    assert called, "the stub never ran: the patch missed the sweep"


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
                # Niedermayer has the oracle's stream alone (#1362).
                backend=Backend.PYTHON,
            )
            ours = potts_moves.niedermayer_sweep(
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


@pytest.mark.analytic
@pytest.mark.critical
def test_the_accepted_fraction_and_the_cluster_are_what_the_instance_makes_them() -> (
    None
):
    # Frustrated: Wolff refused, Niedermayer's cluster 8.31 of 9 sites; on the
    # ferromagnet the two agree on both readings.
    frustrated, ferromagnet = _frustrated_lattice(), _wider_lattice()
    readings = {
        "frustrated": _cluster_counter(frustrated, SandboxMove.NIEDERMAYER),
        "ferro-niedermayer": _cluster_counter(ferromagnet, SandboxMove.NIEDERMAYER),
        "ferro-wolff": _cluster_counter(ferromagnet, PottsMove.WOLFF),
    }

    with pytest.raises(ValueError, match="needs every coupling >= 0"):
        potts_moves.sample_potts(
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


@pytest.mark.smoke
@pytest.mark.backend
@pytest.mark.parametrize("move", [PottsMove.WOLFF, SandboxMove.NIEDERMAYER])
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
        else potts_moves.niedermayer_sweep
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
