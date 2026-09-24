"""The Rust Swendsen-Wang pass, refereed twice: by the oracle's own pass on
the same draws, and by the distribution it converges to.

Issue #754; the enumeration test is at `test_potts_mcmc.py`'s 2x2 fixture,
with and without a field (`search/CLAUDE.md`). The oracle's draw order depends
on outcomes, so its chain cannot be replayed; :class:`_ScriptedDraws` hands
``swendsen_wang_sweep`` the kernel's own bond uniforms, colours and accept
uniforms, so the comparison runs the oracle itself, bitwise. The kernel sums
field terms left to right where NumPy sums pairwise, so a cluster inside that
width is handed back to NumPy; one test forces the hand-back on every cluster.
"""

from __future__ import annotations

import itertools
from itertools import product
from typing import Any, cast

import numpy as np
import pytest
from snakes_and_ladders import oxisal
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.sample import potts_mcmc
from snakes_and_ladders.sample.potts_mcmc import (
    ClusterCounter,
    PottsMove,
    sample_potts,
    swendsen_wang_sweep,
)
from snakes_and_ladders.sample.potts_mcmc.sweeps import GUARD, bond_probability
from snakes_and_ladders.sample.statistics import chi_square_p_value
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

from tests._chains import enumerated_law, fit_p_value
from tests._rows import every_row, every_value

# `test_potts_mcmc.py`'s fixture, significance and thinning, so the two routes
# are held to one standard: 16 configurations, 10,000 sweeps thinned by 5.
SIGNIFICANCE = 0.001
SWEEPS = 10_000
THINNING = 5
SEEDS = (4242, 7)

SHAPE = (2, 2)
COUPLING = 0.8
NO_FIELD = np.zeros(2)
WITH_FIELD = np.array([0.6, -0.4])

#: The lattices the bitwise pin runs on, and the temperatures. 6x6 at three
#: states is past enumeration and is where a pass makes hundreds of clusters,
#: which is the arithmetic the guard is about.
PIN_SHAPES = ((3, 3), (6, 6))
PIN_TEMPERATURES = (0.5, 1.0, 2.0)


class _ScriptedDraws:
    """The generator ``swendsen_wang_sweep`` draws from, replaying one pass.

    Bond uniforms, then per cluster by root a colour and, if needed, a uniform.
    """

    def __init__(
        self, bond: np.ndarray, colours: np.ndarray, accepts: np.ndarray
    ) -> None:
        self._bond = bond
        self._colours = colours
        self._accepts = accepts
        self.clusters = 0

    def random(self, size: int | None = None) -> Any:
        if size is None:
            return float(self._accepts[self.clusters - 1])
        assert size == self._bond.size, "the pass drew a bond array of another size"
        return self._bond

    def integers(self, high: int) -> int:
        assert high == self._colours.max() + 1, "the pass drew from another alphabet"
        self.clusters += 1
        return int(self._colours[self.clusters - 1])


def _instance(
    shape: tuple[int, int], n_states: int, seed: int
) -> tuple[
    PottsGraph,
    np.ndarray,
    np.ndarray,
]:
    """A lattice, a per-site field and a starting configuration."""
    graph = lattice_graph(shape, BoundaryCondition.OPEN, COUPLING)
    rows = np.random.default_rng(seed).normal(size=(graph.n_nodes, n_states))
    state = np.ascontiguousarray(
        np.random.default_rng(seed + 1).integers(0, n_states, size=graph.n_nodes),
        dtype=np.int64,
    )
    return graph, rows, state


def _draws(
    graph: PottsGraph, n_states: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three arrays the Rust route draws, in its own order."""
    rng = np.random.default_rng(seed)
    return (
        rng.random(len(graph.edges)),
        rng.integers(0, n_states, size=graph.n_nodes),
        rng.random(graph.n_nodes),
    )


def _kernel_pass(
    graph: PottsGraph,
    rows: np.ndarray,
    state: np.ndarray,
    beta: float,
    draws: tuple[np.ndarray, np.ndarray, np.ndarray],
    guard: float = GUARD,
) -> tuple[np.ndarray, int, int]:
    """One pass through the binding, with the draws supplied rather than drawn."""
    bond, colours, accepts = draws
    labels = np.empty(graph.n_nodes, dtype=np.int64)
    n_clusters, stop = oxisal.swendsen_wang_sweep(
        state,
        np.ascontiguousarray(beta * rows),
        np.ascontiguousarray(graph.edge_index).reshape(-1),
        np.ascontiguousarray(bond_probability(graph, beta)),
        bond,
        colours,
        accepts,
        labels,
        guard,
        0,
    )
    return labels, n_clusters, stop


@pytest.mark.oracle
@pytest.mark.backend
def test_the_rust_pass_is_the_oracles_pass_bitwise_on_the_same_draws() -> None:
    """The pin: same bonds, same clusters, same recolourings, bit for bit."""

    def check(shape: tuple[int, int], temperature: float, seed: int) -> None:
        n_states, beta = 3, 1.0 / temperature
        graph, rows, state = _instance(shape, n_states, seed)
        bond, colours, accepts = _draws(graph, n_states, seed)

        mine = state.copy()
        labels, n_clusters, _ = _kernel_pass(
            graph, rows, mine, beta, (bond, colours, accepts)
        )

        theirs = state.copy()
        scripted = _ScriptedDraws(bond, colours, accepts)
        swendsen_wang_sweep(
            theirs, graph, rows, cast(np.random.Generator, scripted), None, beta
        )

        # The partition, then what it did with it: a pass that agreed on the
        # configuration while building other clusters would pass the second
        # assertion and not the first, and the test below pins the roots
        # themselves.
        assert scripted.clusters == n_clusters == np.unique(labels).size
        assert np.array_equal(mine, theirs), "the recolourings differ"

    every_row(product(PIN_SHAPES, PIN_TEMPERATURES, SEEDS), check)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_labels_are_the_oracles_own_components() -> None:
    """The bond pass and the union-find, against ``find_root`` on the same bonds.

    Root identity, not only partition: it orders the recolourings.
    """

    def check(seed: int) -> None:
        n_states, beta = 3, 1.0
        graph, rows, state = _instance((6, 6), n_states, seed)
        bond, colours, accepts = _draws(graph, n_states, seed)

        labels, n_clusters, _ = _kernel_pass(
            graph, rows, state.copy(), beta, (bond, colours, accepts)
        )

        first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
        active = (state[first] == state[second]) & (
            bond < bond_probability(graph, beta)
        )
        parent = np.arange(graph.n_nodes)
        for edge in np.flatnonzero(active):
            potts_mcmc.union_roots(parent, int(first[edge]), int(second[edge]))
        expected = np.array(
            [potts_mcmc.find_root(parent, node) for node in range(graph.n_nodes)]
        )

        assert np.array_equal(labels, expected)
        assert n_clusters == np.unique(expected).size

    every_value(SEEDS, check)


@pytest.mark.oracle
@pytest.mark.backend
def test_a_guard_that_hands_every_cluster_back_changes_no_recolouring() -> None:
    """The hand-back path, forced and then checked against the pass without it.

    A guard wide enough hands back every fielded cluster; the result is unchanged.
    """
    n_states, beta = 3, 1.0
    graph, rows, state = _instance((6, 6), n_states, SEEDS[0])
    draws = _draws(graph, n_states, SEEDS[0])

    decided = state.copy()
    _, n_clusters, stop = _kernel_pass(graph, rows, decided, beta, draws)
    assert stop == n_clusters, "the default guard handed a cluster back"

    # Wide enough that the slack covers a field difference of order one:
    # the width is `guard * members * 2**-52 * sum |term|`, so a guard of
    # 1e18 puts it at hundreds where the differences here are single digits.
    handed_back = state.copy()
    _, _, first_stop = _kernel_pass(graph, rows, handed_back, beta, draws, 1e18)
    assert first_stop < n_clusters, "the wide guard decided a cluster anyway"

    # The whole pass through the adapter, which resumes after each hand-back.
    rebuilt = state.copy()
    bond, colours, accepts = draws
    scripted = _ScriptedDraws(bond, colours, accepts)
    swendsen_wang_sweep(
        rebuilt, graph, rows, cast(np.random.Generator, scripted), None, beta
    )
    assert np.array_equal(decided, rebuilt)


@pytest.mark.smoke
def test_the_default_guard_hands_nothing_back_on_a_realistic_pass() -> None:
    """What the guard costs, measured rather than assumed.

    Twelve passes at 16x16, three states: every cluster decided in Rust.
    """
    n_states = 3
    handed_back = 0
    for seed in range(12):
        graph, rows, state = _instance((16, 16), n_states, seed)
        _, n_clusters, stop = _kernel_pass(
            graph, rows, state.copy(), 1.0, _draws(graph, n_states, seed)
        )
        handed_back += n_clusters - stop

    assert handed_back == 0


def _goodness_of_fit(field: np.ndarray, seed: int) -> float:
    """The Rust pass's chain against the exact enumerated distribution."""
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    index, probability = enumerated_law(graph, field)

    chain = sample_potts(
        graph,
        field,
        PottsMove.SWENDSEN_WANG,
        np.random.default_rng(seed),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING,
        cluster_backend=Backend.RUST,
    )

    return fit_p_value(index, probability, chain.states, SWEEPS)


@pytest.mark.oracle
def test_the_rust_pass_draws_the_exact_boltzmann_distribution() -> None:
    """The claim the port has to earn, against enumeration."""

    def check(seed: int) -> None:
        assert _goodness_of_fit(NO_FIELD, seed) > SIGNIFICANCE

    every_value(SEEDS, check)


@pytest.mark.oracle
def test_the_rust_pass_is_still_exact_in_an_external_field() -> None:
    """A field is not optional here: zero field cannot see a dropped accept step."""

    def check(seed: int) -> None:
        assert _goodness_of_fit(WITH_FIELD, seed) > SIGNIFICANCE

    every_value(SEEDS, check)


@pytest.mark.smoke
def test_the_test_would_catch_a_pass_that_ignored_the_field() -> None:
    """Evidence the two tests above have the power they claim.

    A zero-field pass samples a different law; the chi-square must reject it.
    """
    graph = lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)
    n_states = int(WITH_FIELD.shape[0])
    configurations = np.array(
        list(itertools.product(range(n_states), repeat=graph.n_nodes)), dtype=np.int64
    )
    weights = log_weights(graph, WITH_FIELD, configurations)
    weights -= weights.max()
    probability = np.exp(weights)
    probability /= probability.sum()
    index = {tuple(row): position for position, row in enumerate(configurations)}

    chain = sample_potts(
        graph,
        NO_FIELD,
        PottsMove.SWENDSEN_WANG,
        np.random.default_rng(SEEDS[0]),
        SWEEPS,
        burn_in=SWEEPS // 10,
        thin=THINNING,
        cluster_backend=Backend.RUST,
    )
    observed = np.zeros(len(probability))
    for row in chain.states:
        observed[index[tuple(row)]] += 1

    assert chi_square_p_value(observed, probability * SWEEPS) < SIGNIFICANCE


@pytest.mark.smoke
def test_the_rust_route_refuses_a_counter() -> None:
    """Instrumentation is refused rather than silently dropped.

    `ClusterCounter` needs the gather the port removes; silence would zero #551.
    """
    graph, rows, state = _instance((3, 3), 3, SEEDS[0])

    with pytest.raises(ValueError, match="takes no counter"):
        swendsen_wang_sweep(
            state,
            graph,
            rows,
            np.random.default_rng(0),
            ClusterCounter(),
            1.0,
            Backend.RUST,
        )


@pytest.mark.smoke
def test_the_pass_refuses_a_backend_it_does_not_have() -> None:
    graph, rows, state = _instance((3, 3), 3, SEEDS[0])

    with pytest.raises(ValueError, match="runs on python or rust, not numba"):
        swendsen_wang_sweep(
            state, graph, rows, np.random.default_rng(0), None, 1.0, Backend.NUMBA
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ("colour", r"expected \[0, 3\)"),
        ("bond", "expected 12 each"),
        ("labels", "one root per site"),
    ],
)
def test_the_kernel_refuses_a_malformed_call(broken: str, message: str) -> None:
    """The preconditions, named in the error rather than read off a crash."""
    graph, rows, state = _instance((3, 3), 3, SEEDS[0])
    bond, colours, accepts = _draws(graph, 3, SEEDS[0])
    labels = np.empty(graph.n_nodes, dtype=np.int64)
    if broken == "colour":
        colours = np.full(graph.n_nodes, 3, dtype=np.int64)
    elif broken == "bond":
        bond = bond[:-1]
    else:
        labels = np.empty(graph.n_nodes - 1, dtype=np.int64)

    with pytest.raises(ValueError, match=message):
        oxisal.swendsen_wang_sweep(
            state,
            np.ascontiguousarray(rows),
            np.ascontiguousarray(graph.edge_index).reshape(-1),
            np.ascontiguousarray(bond_probability(graph, 1.0)),
            bond,
            colours,
            accepts,
            labels,
            GUARD,
            0,
        )
