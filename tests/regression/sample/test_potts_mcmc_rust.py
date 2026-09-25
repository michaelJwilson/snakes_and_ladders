"""The Rust single-site sampler, refereed twice: by the distribution it
converges to, and by the oracle's own chain.

Issue #246: against the exact enumerated Boltzmann law (`search/CLAUDE.md`).
State for state against the oracle (#599): Rust's `f64::exp` is within an ulp
of NumPy's, so a site is decided only where the draw clears every boundary by
`potts_mcmc.sweeps.GUARD` ulps per state, else NumPy decides it. Fixture,
significance and thinning are `test_potts_mcmc.py`'s: 16 configurations,
10,000 sweeps thinned by 5.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample.potts_mcmc import PottsChain, PottsMove, sweeps
from sal.sample.potts_mcmc import sample_potts as oracle_sample_potts
from sal.sample.potts_mcmc.sweeps import GUARD
from sal.sample.statistics import chi_square_p_value
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import site_field

from tests._chains import enumerated_law, fit_p_value
from tests._rows import every_row


def sample_potts(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
) -> PottsChain:
    # The name issue #246 published, as one call: the single-site move on
    # the extension, which is `sample_potts`'s default backend.
    return oracle_sample_potts(
        graph,
        field,
        PottsMove.SINGLE_SITE,
        rng,
        n_sweeps,
        burn_in,
        thin,
        backend=Backend.RUST,
    )


SIGNIFICANCE = 0.001
SWEEPS = 10_000
THINNING = 5
SEED = 4242

SHAPE = (2, 2)
COUPLING = 0.8
NO_FIELD = np.zeros(2)
WITH_FIELD = np.array([0.6, -0.4])


def _goodness_of_fit(field: np.ndarray, seed: int = SEED) -> float:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, probability = enumerated_law(graph, field)

    chain = sample_potts(
        graph,
        field,
        np.random.default_rng(seed),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING,
    )

    return fit_p_value(index, probability, chain.states, SWEEPS)


@pytest.mark.oracle
def test_the_rust_chain_is_drawn_from_the_exact_boltzmann_distribution() -> None:
    """The claim the port has to earn, against enumeration."""
    assert _goodness_of_fit(NO_FIELD) > SIGNIFICANCE


@pytest.mark.oracle
def test_the_rust_chain_is_still_exact_in_an_external_field() -> None:
    """A field is not optional here (`search/CLAUDE.md`): zero field is symmetric."""
    assert _goodness_of_fit(WITH_FIELD) > SIGNIFICANCE


@pytest.mark.smoke
def test_the_test_would_catch_a_sampler_that_ignored_the_field() -> None:
    """Evidence the two tests above have the power they claim.

    A no-field chain scored against the with-field truth must be rejected.
    """
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, with_field_truth = enumerated_law(graph, WITH_FIELD)

    chain = sample_potts(
        graph,
        NO_FIELD,
        np.random.default_rng(SEED),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING,
    )

    observed = np.zeros(len(with_field_truth))
    for row in chain.states:
        observed[index[tuple(row)]] += 1

    assert chi_square_p_value(observed, with_field_truth * SWEEPS) < SIGNIFICANCE


@pytest.mark.smoke
def test_the_kernel_reads_the_graph_s_own_adjacency() -> None:
    """Both backends read the same neighbour structure, because there is one.

    `PottsGraph.compressed_adjacency` since #277 retired `flatten_adjacency`.
    """
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.4)
    offsets, neighbours, couplings = graph.compressed_adjacency()

    degree = np.zeros(graph.n_nodes, dtype=np.int64)
    for first, second in graph.edges:
        degree[first] += 1
        degree[second] += 1

    assert len(offsets) == graph.n_nodes + 1
    assert int(offsets[-1]) == 2 * len(graph.edges)
    assert np.array_equal(np.diff(offsets), degree)
    for node in range(graph.n_nodes):
        start, end = int(offsets[node]), int(offsets[node + 1])
        for neighbour, coupling in zip(
            neighbours[start:end], couplings[start:end], strict=True
        ):
            assert (node, int(neighbour)) in graph.edges or (
                int(neighbour),
                node,
            ) in graph.edges
            assert float(coupling) in graph.coupling


@pytest.mark.smoke
def test_a_chain_is_reproducible_from_its_generator() -> None:
    """A declared seed still determines the run: every uniform is drawn here."""
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    first = sample_potts(graph, NO_FIELD, np.random.default_rng(11), 20)
    second = sample_potts(graph, NO_FIELD, np.random.default_rng(11), 20)

    assert np.array_equal(first.states, second.states)


@pytest.mark.smoke
def test_a_state_outside_the_alphabet_is_refused() -> None:
    """The kernel's own precondition, surfaced as a Python error (via the extension)."""
    from sal import oxisal

    with pytest.raises(ValueError, match=r"expected \[0, 2\)"):
        oxisal.single_site_sweeps(
            np.array([5], dtype=np.int64),
            np.zeros((1, 2)),
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([0.5]),
            1,
            1.0,
            GUARD,
            0,
        )


# --- the exact pin: the same chain, not a chain of the same law (#599) -------

#: The extents and seeds #561 pinned the Gibbs kernel over, at the alphabets
#: and fields this sampler is declared on. 20 sweeps each, so a divergence has
#: somewhere to show: one draw across a moved boundary sends the two chains
#: apart and every later sweep differs, not just the site that crossed.
PIN_EXTENTS = (8, 16)
PIN_SEEDS = (1, 2, 3, 4)
PIN_SWEEPS = 20


@pytest.mark.oracle
def test_the_rust_chain_is_the_oracle_s_chain_state_for_state() -> None:
    """Every recorded configuration, not a distribution over them.

    So the Rust default moves no autocorrelation figure `STATUS.md` pins.
    """

    def check(extent: int, seed: int, states: int) -> None:
        graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 0.4)
        field = np.linspace(0.6, -0.4, states)

        python = oracle_sample_potts(
            graph,
            field,
            PottsMove.SINGLE_SITE,
            np.random.default_rng(seed),
            PIN_SWEEPS,
            backend=Backend.PYTHON,
        )
        rust = oracle_sample_potts(
            graph,
            field,
            PottsMove.SINGLE_SITE,
            np.random.default_rng(seed),
            PIN_SWEEPS,
            backend=Backend.RUST,
        )

        np.testing.assert_array_equal(python.states, rust.states)

    every_row(product(PIN_EXTENTS, PIN_SEEDS, [2, 3]), check)


@pytest.mark.smoke
def test_a_guard_wide_enough_hands_every_site_back() -> None:
    """The hand-back path itself, which no realistic draw reaches.

    A guard over the whole sum hands every site to NumPy: still the oracle's chain (#599).
    """
    from sal import oxisal

    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.7)
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()

    rng = np.random.default_rng(3)
    state = np.ascontiguousarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
    handed_back = 0
    for _ in range(5):
        draws = np.ascontiguousarray(rng.random(graph.n_nodes), dtype=np.float64)
        node = 0
        while node < graph.n_nodes:
            node = oxisal.single_site_sweeps(
                state, rows, offsets, neighbours, couplings, draws, 1, 1.0, 1e18, node
            )
            if node < graph.n_nodes:
                handed_back += 1
                sweeps.site_update(
                    state,
                    rows,
                    neighbours.tolist(),
                    couplings.tolist(),
                    offsets.tolist(),
                    node,
                    float(draws[node]),
                    1.0,
                )
                node += 1

    assert handed_back == 5 * graph.n_nodes

    expected = oracle_sample_potts(
        graph,
        WITH_FIELD,
        PottsMove.SINGLE_SITE,
        np.random.default_rng(3),
        5,
        backend=Backend.PYTHON,
    )
    np.testing.assert_array_equal(state, expected.states[-1])


@pytest.mark.smoke
def test_the_kernel_refuses_a_negative_guard() -> None:
    """A guard is a width, and a negative one would decide every site."""
    from sal import oxisal

    with pytest.raises(ValueError, match="guard must be >= 0"):
        oxisal.single_site_sweeps(
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 2)),
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([0.5]),
            1,
            1.0,
            -1.0,
            0,
        )


@pytest.mark.smoke
def test_the_default_guard_hands_nothing_back_on_a_realistic_chain() -> None:
    """The rate the port is worth measuring at, pinned as an absence.

    The kernel decided every site; a hand-back would be a correctness change.
    """
    from sal import oxisal

    graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.4)
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()

    rng = np.random.default_rng(11)
    state = np.ascontiguousarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
    for _ in range(50):
        draws = np.ascontiguousarray(rng.random(graph.n_nodes), dtype=np.float64)
        assert (
            oxisal.single_site_sweeps(
                state, rows, offsets, neighbours, couplings, draws, 1, 1.0, GUARD, 0
            )
            == graph.n_nodes
        )
