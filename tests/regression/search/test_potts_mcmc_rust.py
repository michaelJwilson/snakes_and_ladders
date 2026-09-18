"""The Rust single-site sampler, refereed twice: by the distribution it
converges to, and by the oracle's own chain.

Issue #246. `search/CLAUDE.md`: a sampler is validated by the distribution it
converges to, never by inspection -- a chain that visibly moves is what a
sampler with a broken accept step also does. So this compares the Rust backend
against the *exact enumerated* Boltzmann distribution.

**And state for state against the oracle, which issue #599 is what buys.**
Rust's `f64::exp` agrees with NumPy's to within a unit in the last place
rather than exactly, and `np.searchsorted` is a threshold, so the kernel
decides a site only where the draw clears every cumulative boundary by
`potts_mcmc._GUARD` units of the last place per state and hands the rest to
NumPy. The enumeration test stays, because it is what would catch a guard
that decided the wrong site rather than declining it.

The fixture, the significance and the thinning are `test_potts_mcmc.py`'s, so
the two backends are held to one standard: a 2x2 two-state lattice has 16
configurations, and 10,000 sweeps thinned by 5 puts every chi-square cell in
the hundreds.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.search import potts_mcmc
from snakes_and_ladders.search.potts_mcmc import _GUARD, PottsChain, PottsMove
from snakes_and_ladders.search.potts_mcmc import sample_potts as oracle_sample_potts
from snakes_and_ladders.search.statistics import chi_square_p_value
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import site_field


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


def _goodness_of_fit(field: np.ndarray, seed: int = SEED) -> float:
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, probability = _exact_distribution(graph, field)

    chain = sample_potts(
        graph,
        field,
        np.random.default_rng(seed),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING,
    )

    observed = np.zeros(len(probability))
    for row in chain.states:
        observed[index[tuple(row)]] += 1
    return chi_square_p_value(observed, probability * SWEEPS)


@pytest.mark.oracle
def test_the_rust_chain_is_drawn_from_the_exact_boltzmann_distribution() -> None:
    """The claim the port has to earn, against enumeration."""
    assert _goodness_of_fit(NO_FIELD) > SIGNIFICANCE


@pytest.mark.oracle
def test_the_rust_chain_is_still_exact_in_an_external_field() -> None:
    """A field is not optional here.

    `search/CLAUDE.md` requires every distributional test to run with a field
    as well as without: a zero field is symmetric between states, so a sampler
    that mishandled the field term entirely could still pass the test above.
    """
    assert _goodness_of_fit(WITH_FIELD) > SIGNIFICANCE


@pytest.mark.smoke
def test_the_test_would_catch_a_sampler_that_ignored_the_field() -> None:
    """Evidence the two tests above have the power they claim.

    Without this, "the sampler passes a chi-square" and "the chi-square could
    not tell" are indistinguishable -- the same argument
    `test_dropping_the_field_accept_step_is_caught` makes for the cluster
    moves. A chain drawn under no field, scored against the *with-field*
    truth, must be rejected.
    """
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, with_field_truth = _exact_distribution(graph, WITH_FIELD)

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

    The kernel took a second builder (`flatten_adjacency`) until issue #277
    retired it; it now takes `PottsGraph.compressed_adjacency`, which the
    Python sweep indexes. Two builders cannot disagree about which nodes are
    adjacent when there is one -- a disagreement that would have shown up as
    a distributional failure with no indication of where it came from.
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
    """A declared seed still determines the run.

    The kernel holds no generator: every uniform it consumes is drawn here and
    passed down, which is what keeps `snakes_and_ladders.sim`'s reproducibility
    contract intact across the boundary.
    """
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)

    first = sample_potts(graph, NO_FIELD, np.random.default_rng(11), 20)
    second = sample_potts(graph, NO_FIELD, np.random.default_rng(11), 20)

    assert np.array_equal(first.states, second.states)


@pytest.mark.smoke
def test_a_state_outside_the_alphabet_is_refused() -> None:
    """The kernel's own precondition, surfaced as a Python error.

    Reached through the extension rather than the wrapper, since the wrapper
    draws a valid start itself: this asserts the boundary refuses rather than
    reading past the field.
    """
    from snakes_and_ladders import oxi_snakes_and_ladders

    with pytest.raises(ValueError, match=r"expected \[0, 2\)"):
        oxi_snakes_and_ladders.single_site_sweeps(
            np.array([5], dtype=np.int64),
            np.zeros((1, 2)),
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([0.5]),
            1,
            1.0,
            _GUARD,
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
@pytest.mark.parametrize("extent", PIN_EXTENTS, ids=lambda e: f"{e}x{e}")
@pytest.mark.parametrize("seed", PIN_SEEDS)
@pytest.mark.parametrize("states", [2, 3])
def test_the_rust_chain_is_the_oracle_s_chain_state_for_state(
    extent: int, seed: int, states: int
) -> None:
    """Every recorded configuration, not a distribution over them.

    This is what makes :data:`~snakes_and_ladders.backend.Backend.RUST`
    the default without moving a committed number: a chain of the same law
    would move every autocorrelation figure `STATUS.md` pins.
    """
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


@pytest.mark.smoke
def test_a_guard_wide_enough_hands_every_site_back() -> None:
    """The hand-back path itself, which no realistic draw reaches.

    A guard covering the whole cumulative sum leaves the kernel unable to
    decide any site, so it returns the position it started at every time and
    NumPy decides all of them. The chain that comes out is still the oracle's,
    which is what says the two halves of the sweep join up (issue #599, and
    #561's test of the same shape).
    """
    from snakes_and_ladders import oxi_snakes_and_ladders

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
            node = oxi_snakes_and_ladders.single_site_sweeps(
                state, rows, offsets, neighbours, couplings, draws, 1, 1.0, 1e18, node
            )
            if node < graph.n_nodes:
                handed_back += 1
                potts_mcmc._site_update(
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
    from snakes_and_ladders import oxi_snakes_and_ladders

    with pytest.raises(ValueError, match="guard must be >= 0"):
        oxi_snakes_and_ladders.single_site_sweeps(
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

    `_GUARD` is a hand-back *threshold*, so narrowing it toward the derived
    bound of four units per state is what would buy speed. It buys nothing:
    the kernel decided every one of these sites itself. A regression that
    started handing sites back would be a correctness change dressed as a
    slowdown, so it is asserted rather than left to the benchmark.
    """
    from snakes_and_ladders import oxi_snakes_and_ladders

    graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.4)
    rows = site_field(WITH_FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()

    rng = np.random.default_rng(11)
    state = np.ascontiguousarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
    for _ in range(50):
        draws = np.ascontiguousarray(rng.random(graph.n_nodes), dtype=np.float64)
        assert (
            oxi_snakes_and_ladders.single_site_sweeps(
                state, rows, offsets, neighbours, couplings, draws, 1, 1.0, _GUARD, 0
            )
            == graph.n_nodes
        )
