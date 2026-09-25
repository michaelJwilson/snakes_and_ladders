"""The three compiled kernels against referees outside the package (issue #729).

62 of `search/kernels.py`'s 71 statements sat behind `njit`, unseen by
coverage; each kernel is entered through ``py_func`` and judged outside the
package: :func:`factor_graph_log_density` against
:func:`sal.likelihood.potts.log_weights` and `enumerate_potts`
on a 2x2 lattice; :func:`icm_sweeps` against enumeration where descent is
exact, a hand-stepped two-node sweep, and a brute-forced local-minimum
certificate; :func:`gibbs_sweep_sites` against the heat-bath conditional and
10,000 sweeps of enumerated marginals. Compiled equals ``py_func`` bitwise.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sal.likelihood.potts import enumerate_potts, log_weights
from sal.numerics import logsumexp
from sal.sample.gibbs import _GUARD, Indexed
from sal.sample.numba.factor_graph import factor_graph_log_density
from sal.sample.numba.gibbs import gibbs_sweep_sites
from sal.sample.statistics import chi_square_p_value
from sal.search.numba.icm import icm_sweeps
from sal.sim.factor_graph import from_potts
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energies, site_field

#: The 2x2 open square in two states: 16 configurations, so an enumeration is
#: the referee and a chi-square cell holds hundreds of counts at the chain
#: length below.
SHAPE = (2, 2)
COUPLING = 0.8
FIELD = np.array([0.6, -0.4])

#: The order `energies` and `log_weights` sum the edge terms in differs from
#: the kernel's, so the two agree relatively rather than bitwise. The bound is
#: `sim.potts.energies`' own declared 1e-12; realized worst over the 16
#: configurations is 0.0.
SUM_ORDER_RTOL = 1e-12

#: The significance `test_potts_mcmc.py` declares, on the same argument: at
#: this chain length it does not reject a correct sampler. Realized worst
#: p-value over the four sites is 0.070, 70 times the significance.
SIGNIFICANCE = 0.001
SWEEPS = 10_000
THIN = 5


def _lattice() -> tuple[PottsGraph, Indexed]:
    """The 2x2 instance and its edge layout."""
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    return graph, Indexed(from_potts(graph, FIELD))


def _configurations(n_nodes: int, n_states: int) -> np.ndarray:
    return np.array(list(product(range(n_states), repeat=n_nodes)), dtype=np.int64)


# --- factor_graph_log_density -------------------------------------------


@pytest.mark.oracle
def test_the_density_kernel_is_the_enumerated_boltzmann_log_weight() -> None:
    # `log_weights` reads edges and field, not the flat factor tables: all 16
    # configurations and log Z, realized 0.0 (declared tolerance, since sums
    # reorder); compiled equals Python bitwise.
    graph, indexed = _lattice()
    layout = indexed.layout()
    states = _configurations(graph.n_nodes, int(FIELD.shape[0]))
    expected = log_weights(graph, FIELD, states)

    density = np.array(
        [
            factor_graph_log_density.py_func(
                state,
                layout.tables,
                layout.factor_start,
                layout.factor_offsets,
                layout.factor_column,
                layout.factor_stride,
            )
            for state in states
        ]
    )
    compiled = np.array(
        [
            factor_graph_log_density(
                state,
                layout.tables,
                layout.factor_start,
                layout.factor_offsets,
                layout.factor_column,
                layout.factor_stride,
            )
            for state in states
        ]
    )

    assert np.array_equal(density, compiled)
    assert_allclose(density, expected, rtol=SUM_ORDER_RTOL)
    assert_allclose(
        logsumexp(density, axis=0),
        enumerate_potts(graph, FIELD).log_partition,
        rtol=SUM_ORDER_RTOL,
    )


# --- icm_sweeps ----------------------------------------------------------


def _descend(
    graph: PottsGraph,
    values: np.ndarray,
    labelling: np.ndarray,
    max_sweeps: int,
) -> tuple[np.ndarray, int, int]:
    """Both forms of the kernel on one start; the labelling and both sweep counts."""
    offsets, neighbours, couplings = graph.compressed_adjacency()
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    interpreted = labelling.copy()
    compiled = labelling.copy()
    # Index order (no rows of `orders`) and no floor (no draws), which is the
    # descent the two kernels before #1055 ran.
    unfloored = (
        np.empty((0, labelling.size), dtype=np.int64),
        np.empty(0, dtype=np.float64),
        max_sweeps,
        True,
        0,
    )
    sweeps = icm_sweeps.py_func(
        interpreted, contiguous, offsets, neighbours, couplings, *unfloored
    )
    other = icm_sweeps(compiled, contiguous, offsets, neighbours, couplings, *unfloored)
    assert np.array_equal(interpreted, compiled)
    return interpreted, int(sweeps), int(other)


@pytest.mark.oracle
def test_the_descent_kernel_reaches_the_enumerated_minimum_where_it_provably_can() -> (
    None
):
    # Descent is exact on a decoupled field (from all 16 starts, 1 or 2 sweeps)
    # and on an aligned ferromagnet (1 sweep); energies equal the enumerated
    # minimum to 0.0.
    n_states = 2
    decoupled = lattice_graph(SHAPE, BoundaryCondition.OPEN, 0.0)
    rng = np.random.default_rng(11)
    values = site_field(rng.normal(size=(decoupled.n_nodes, n_states)), 4)
    states = _configurations(decoupled.n_nodes, n_states)
    exact = states[int(np.argmin(energies(decoupled, values, states)))]

    for start in states:
        labelling, sweeps, compiled_sweeps = _descend(decoupled, values, start, 200)
        assert np.array_equal(labelling, exact)
        # One sweep where the start is already the minimum, since the sweep
        # that changes nothing is the one that stops; two otherwise.
        assert sweeps == compiled_sweeps == (1 if np.array_equal(start, exact) else 2)

    ferromagnet = lattice_graph(SHAPE, BoundaryCondition.OPEN, 1.0)
    flat = site_field(np.zeros(n_states), ferromagnet.n_nodes)
    aligned = np.ones(ferromagnet.n_nodes, dtype=np.int64)
    labelling, sweeps, compiled_sweeps = _descend(ferromagnet, flat, aligned, 200)

    assert np.array_equal(labelling, aligned)
    assert sweeps == compiled_sweeps == 1
    assert float(energies(ferromagnet, flat, labelling[None])[0]) == float(
        energies(ferromagnet, flat, states).min()
    )


@pytest.mark.oracle
def test_every_descent_the_kernel_settles_on_is_a_certified_local_minimum() -> None:
    # Where descent is inexact: no single relabel lowers the energy (checked by
    # `sim.potts.energies`) and it is at or above the minimum. 32 starts on the
    # 3-state antiferromagnetic 3x3, 19,683 configurations: all certified,
    # minimum from 3, missed from 29, worst gap 1.328.
    n_states = 3
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.7)
    rng = np.random.default_rng(3)
    values = site_field(rng.normal(size=(graph.n_nodes, n_states)) * 0.4, graph.n_nodes)
    states = _configurations(graph.n_nodes, n_states)
    minimum = float(energies(graph, values, states).min())

    for _ in range(32):
        start = rng.integers(0, n_states, size=graph.n_nodes)
        labelling, _, _ = _descend(graph, values, start, 200)
        settled = float(energies(graph, values, labelling[None])[0])
        assert settled >= minimum - SUM_ORDER_RTOL * abs(minimum)
        for node in range(graph.n_nodes):
            for label in range(n_states):
                if label == labelling[node]:
                    continue
                moved = labelling.copy()
                moved[node] = label
                assert float(energies(graph, values, moved[None])[0]) >= settled


@pytest.mark.oracle
def test_the_descent_kernel_reproduces_a_sweep_stepped_by_hand() -> None:
    # J = 0.5, h = [[0.0, 0.3], [0.2, 0.0]], start [0, 1]: site 0 takes 1
    # (0.0 against -0.8), site 1 keeps 1 (-0.2 against -0.5); [1, 1] at 2
    # sweeps, bitwise; `max_sweeps = 1` returns the first sweep and 1.
    graph = lattice_graph((2,), BoundaryCondition.OPEN, 0.5)
    values = np.array([[0.0, 0.3], [0.2, 0.0]])

    labelling, sweeps, compiled_sweeps = _descend(
        graph, values, np.array([0, 1], dtype=np.int64), 200
    )
    assert np.array_equal(labelling, np.array([1, 1]))
    assert sweeps == compiled_sweeps == 2

    capped, sweeps, compiled_sweeps = _descend(
        graph, values, np.array([0, 1], dtype=np.int64), 1
    )
    assert np.array_equal(capped, np.array([1, 1]))
    assert sweeps == compiled_sweeps == 1

    # The tie rule the oracle shares: with no field and no coupling every
    # label scores 0.0, and the first minimum is label 0, so any start
    # collapses to zeros in one sweep and settles in the next.
    isolated = lattice_graph((2,), BoundaryCondition.OPEN, 0.0)
    tied, sweeps, compiled_sweeps = _descend(
        isolated, np.zeros((2, 3)), np.array([2, 1], dtype=np.int64), 200
    )
    assert np.array_equal(tied, np.zeros(2, dtype=np.int64))
    assert sweeps == compiled_sweeps == 2


# --- gibbs_sweep_sites ---------------------------------------------------


def _sweep(
    indexed: Indexed,
    state: np.ndarray,
    draws: np.ndarray,
    *,
    interpreted: bool,
) -> int:
    """One kernel sweep over every site, in the form the flag names."""
    layout = indexed.layout()
    local = np.empty(int(indexed.cardinality.max()), dtype=np.float64)
    kernel = gibbs_sweep_sites.py_func if interpreted else gibbs_sweep_sites
    return int(
        kernel(
            state,
            draws,
            indexed.cardinality,
            layout.tables,
            layout.entry_offsets,
            layout.entry_start,
            layout.entry_stride,
            layout.term_offsets,
            layout.term_column,
            layout.term_stride,
            local,
            1.0,
            _GUARD,
            0,
        )
    )


def _heat_bath(
    graph: PottsGraph, state: np.ndarray, position: int
) -> tuple[np.ndarray, np.ndarray]:
    """One site's exact conditional from the model, rescoring each label: no shared table."""
    n_states = int(FIELD.shape[0])
    candidates = np.tile(state, (n_states, 1))
    candidates[:, position] = np.arange(n_states)
    weights = log_weights(graph, FIELD, candidates)
    probability = np.exp(weights - weights.max())
    return weights, probability / probability.sum()


@pytest.mark.oracle
def test_the_heat_bath_kernel_takes_the_label_the_exact_conditional_names() -> None:
    # Over 200 sweeps x 4 sites the kernel takes the cumulative-sum label every
    # time and decides every site (200 of 200 returned 4); conditionals agree
    # to 2.4e-16.
    graph, indexed = _lattice()
    rng = np.random.default_rng(19)
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    decisions = 0
    for _ in range(200):
        draws = rng.random(graph.n_nodes)
        expected = state.copy()
        for position in range(graph.n_nodes):
            _, probability = _heat_bath(graph, expected, position)
            expected[position] = int(
                np.searchsorted(np.cumsum(probability), draws[position])
            )
            decisions += 1

        interpreted = state.copy()
        compiled = state.copy()
        assert _sweep(indexed, interpreted, draws, interpreted=True) == graph.n_nodes
        assert _sweep(indexed, compiled, draws, interpreted=False) == graph.n_nodes

        assert np.array_equal(interpreted, compiled)
        assert np.array_equal(interpreted, expected)
        state = interpreted
    assert decisions == 800


@pytest.mark.oracle
def test_the_heat_bath_kernel_reproduces_the_enumerated_boltzmann_marginals() -> None:
    # The chain the kernel alone drives, against `enumerate_potts`' exact
    # single-site marginals: 10,000 sweeps thinned by 5, a chi-square per site
    # at the significance `test_potts_mcmc.py` declares on the same argument.
    # Realized p-values over the four sites: 0.699, 0.070, 0.570, 0.945.
    graph, indexed = _lattice()
    exact = enumerate_potts(graph, FIELD).single_site
    rng = np.random.default_rng(97)
    state = np.zeros(graph.n_nodes, dtype=np.int64)
    counts = np.zeros_like(exact)

    for sweep in range(SWEEPS * THIN):
        assert (
            _sweep(indexed, state, rng.random(graph.n_nodes), interpreted=False)
            == graph.n_nodes
        )
        if sweep % THIN == 0:
            counts[np.arange(graph.n_nodes), state] += 1

    for node in range(graph.n_nodes):
        assert chi_square_p_value(counts[node], exact[node] * SWEEPS) > SIGNIFICANCE, (
            node
        )
