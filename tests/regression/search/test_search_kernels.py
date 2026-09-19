"""The three compiled kernels against referees outside the package (issue #729).

`search/kernels.py` carried 62 of its 71 statements behind `njit`, where no
test could enter them: a dispatcher runs machine code and `coverage` sees the
`def` line alone. Each kernel is entered here through ``py_func`` --- the
Python the compiler was handed --- and judged against a referee that is not
this package's sweep:

* :func:`factor_graph_log_density` against
  :func:`snakes_and_ladders.likelihood.potts.log_weights` on every
  configuration of a 2x2 lattice, and its normalization against
  :func:`snakes_and_ladders.likelihood.potts.enumerate_potts`.
* :func:`icm_sweeps` against the enumerated ground state on the two instances
  where single-site descent provably reaches it --- a decoupled field and a
  ferromagnet from an aligned start --- against a sweep stepped by hand on
  two nodes, and against a local-minimum certificate brute-forced per site.
* :func:`gibbs_sweep_sites` against the heat-bath conditional written out
  from the model in this file, and over 10,000 sweeps against the enumerated
  single-site Boltzmann marginals.

Each kernel's compiled form is pinned to the ``py_func`` it was compiled
from, which is the bitwise reproduction the module's docstring claims.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood.potts import enumerate_potts, log_weights
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sample.gibbs import _GUARD, _Indexed
from snakes_and_ladders.search.kernels import (
    factor_graph_log_density,
    gibbs_sweep_sites,
    icm_sweeps,
)
from snakes_and_ladders.sample.statistics import chi_square_p_value
from snakes_and_ladders.sim.factor_graph import from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import energies, site_field

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


def _lattice() -> tuple[PottsGraph, _Indexed]:
    """The 2x2 instance and its edge layout."""
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    return graph, _Indexed(from_potts(graph, FIELD))


def _configurations(n_nodes: int, n_states: int) -> np.ndarray:
    return np.array(list(product(range(n_states), repeat=n_nodes)), dtype=np.int64)


# --- factor_graph_log_density -------------------------------------------


@pytest.mark.oracle
def test_the_density_kernel_is_the_enumerated_boltzmann_log_weight() -> None:
    # The referee is `likelihood.potts.log_weights`, which shares no code with
    # the edge layout: it reads the graph's edges and the field, where the
    # kernel reads one flat array of factor tables. Every one of the 16
    # configurations, and the normalization of the 16 against
    # `enumerate_potts`. Realized worst relative deviation 0.0 on the weights
    # and 0.0 on log Z, held at the declared tolerance rather than at equality
    # because the two sum the edge terms in different orders; the compiled
    # kernel equals the Python it was compiled from on all 16, bitwise.
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
    sweeps = icm_sweeps.py_func(
        interpreted, contiguous, offsets, neighbours, couplings, max_sweeps
    )
    other = icm_sweeps(compiled, contiguous, offsets, neighbours, couplings, max_sweeps)
    assert np.array_equal(interpreted, compiled)
    return interpreted, int(sweeps), int(other)


@pytest.mark.oracle
def test_the_descent_kernel_reaches_the_enumerated_minimum_where_it_provably_can() -> (
    None
):
    # Two instances where single-site descent is exact, so enumeration judges
    # the answer rather than bounding it. A decoupled field -- every coupling
    # zero -- makes each site an independent argmin, reached from any start in
    # two sweeps; a ferromagnet in zero field from an aligned start is already
    # a global minimum and must not move. Realized: the decoupled minimum
    # recovered from all 16 starts, in 1 sweep from the start that is already
    # it and 2 from the other 15, the ferromagnet in 1, and both energies
    # equal to the enumerated minimum to 0.0 relative.
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
    # Where descent is not exact, the claim it does make is checkable by brute
    # force: no single site can be relabelled without raising the energy, and
    # the energy is at or above the enumerated global minimum. Certified per
    # site by `sim.potts.energies` rather than by the kernel's own local
    # deltas. Realized over 32 starts on the 3-state antiferromagnetic 3x3,
    # against all 19,683 configurations: every settled labelling certified,
    # the enumerated minimum reached from 3 starts and missed from 29, worst
    # gap 1.328. So both assertions bind -- a descent that stopped early
    # fails the certificate and one that undershot the minimum fails the
    # bound.
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
    # Two nodes and one edge, where the update the docstring states can be
    # written out: with J = 0.5 and h = [[0.0, 0.3], [0.2, 0.0]] from the
    # start [0, 1], site 0 compares 0.0 against -0.8 and takes 1, site 1
    # compares -0.2 against -0.5 and keeps 1, and the second sweep changes
    # nothing, so the kernel returns [1, 1] at 2 sweeps. Bitwise, and the
    # budget is pinned by the same start at `max_sweeps = 1`, which returns
    # the first sweep's labelling and 1.
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
    indexed: _Indexed,
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
    """One site's exact conditional from the model: the log weights and the probabilities.

    The definition `likelihood.potts.log_weights` states, evaluated by
    rescoring the whole configuration at each label of one site, so it shares
    no gather, stride or table with the kernel it referees.
    """
    n_states = int(FIELD.shape[0])
    candidates = np.tile(state, (n_states, 1))
    candidates[:, position] = np.arange(n_states)
    weights = log_weights(graph, FIELD, candidates)
    probability = np.exp(weights - weights.max())
    return weights, probability / probability.sum()


@pytest.mark.oracle
def test_the_heat_bath_kernel_takes_the_label_the_exact_conditional_names() -> None:
    # The conditional is written out here from `log_weights` by rescoring the
    # configuration at each label, and the label the draw selects is read off
    # its cumulative sum. Over 200 sweeps of four sites -- 800 decisions --
    # the kernel takes that label every time, and decides every site rather
    # than handing one back (realized: 200 of 200 sweeps returned 4).
    # Realized worst relative deviation between the conditional here and the
    # one the kernel normalizes is 2.4e-16.
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
