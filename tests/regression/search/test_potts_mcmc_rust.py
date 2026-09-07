"""The Rust single-site sampler, refereed by the distribution it converges to.

Issue #246. `search/CLAUDE.md`: a sampler is validated by the distribution it
converges to, never by inspection -- a chain that visibly moves is what a
sampler with a broken accept step also does. So this compares the Rust backend
against the *exact enumerated* Boltzmann distribution, not against the Python
oracle's chain.

**Not bitwise, and not against the oracle's output.** Rust's `f64::exp` agrees
with NumPy's to within a unit in the last place rather than exactly, and
`np.searchsorted` is a threshold: one draw across a boundary that moved by 1
ulp picks a different state, and from that step the chains are unrelated. Two
implementations of a stochastic process can only be compared by what they
converge to, which is why this file enumerates the truth rather than diffing
two chains.

The fixture, the significance and the thinning are `test_potts_mcmc.py`'s, so
the two backends are held to one standard: a 2x2 two-state lattice has 16
configurations, and 10,000 sweeps thinned by 5 puts every chi-square cell in
the hundreds.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.search.potts_mcmc import _adjacency
from snakes_and_ladders.search.potts_mcmc_rust import flatten_adjacency, sample_potts
from snakes_and_ladders.search.statistics import chi_square_p_value
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

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


@pytest.mark.structural
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


@pytest.mark.structural
def test_the_flattened_adjacency_is_the_oracle_s_own() -> None:
    """Both backends read the same neighbour structure.

    `flatten_adjacency` is built from `potts_mcmc._adjacency` rather than from
    the edge list, so the two cannot disagree about which nodes are adjacent --
    a disagreement that would show up as a distributional failure with no
    indication of where it came from.
    """
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.4)
    offsets, neighbours, couplings = flatten_adjacency(graph)
    expected = _adjacency(graph)

    assert len(offsets) == graph.n_nodes + 1
    assert int(offsets[-1]) == sum(len(incident) for incident in expected)
    for node, incident in enumerate(expected):
        start, end = int(offsets[node]), int(offsets[node + 1])
        assert [
            (int(n), float(c))
            for n, c in zip(neighbours[start:end], couplings[start:end], strict=True)
        ] == incident


@pytest.mark.structural
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


@pytest.mark.edge_case
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
            np.zeros(2),
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([0.5]),
            1,
        )
