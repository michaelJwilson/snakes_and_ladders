"""The Rust Swendsen-Wang pass, refereed twice: by the oracle's own pass on
the same draws, and by the distribution it converges to.

Issue #754. `search/CLAUDE.md`: a sampler is validated by the distribution it
converges to, never by inspection. So the enumeration test is here, at the
2x2 fixture `test_potts_mcmc.py` declares, with and without a field.

**The oracle's chain is not reachable, and its pass is.** The oracle draws a
cluster's colour and then, only where the field difference is negative, its
accept uniform --- what the next draw *is* depends on the last one's outcome,
so no array replays that stream and the Rust route draws its colours and
uniforms in bulk instead (``potts_mcmc._cluster_pass_rust``). What *is*
reachable is the oracle running on the draws the kernel was given:
:class:`_ScriptedDraws` hands ``swendsen_wang_sweep`` the same bond uniforms,
the same colour per cluster and the same uniform per cluster, in the order the
kernel indexes them. The two then have the same input and the comparison is
bitwise --- which is the pin, and it is stronger than a second implementation
written here would be, because what it runs *is* the oracle.

The guard is what makes that bitwise rather than approximate: the kernel sums
a cluster's field terms left to right where NumPy sums them pairwise, and
``exp`` of the difference is a threshold. A cluster whose decision sits inside
that width is handed back and decided by NumPy, and
:func:`test_a_guard_that_hands_every_cluster_back_changes_no_recolouring`
runs the pass with the hand-back forced on every cluster.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np
import pytest
from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.potts import log_weights
from snakes_and_ladders.sample import potts_mcmc
from snakes_and_ladders.sample.potts_mcmc import (
    _GUARD,
    ClusterCounter,
    PottsMove,
    _bond_probability,
    sample_potts,
    swendsen_wang_sweep,
)
from snakes_and_ladders.sample.statistics import chi_square_p_value
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

from tests._chains import enumerated_law, fit_p_value

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

    The oracle draws in this order: one array of bond uniforms, then per
    cluster in increasing root order one colour and --- where the field
    difference is negative --- one uniform. Every cluster draws its colour,
    so counting ``integers`` counts clusters, and the accept uniform a cluster
    asks for is the one the kernel indexes by that cluster's rank. That is
    what makes the comparison a comparison of *implementations* rather than of
    generators.
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
    guard: float = _GUARD,
) -> tuple[np.ndarray, int, int]:
    """One pass through the binding, with the draws supplied rather than drawn."""
    bond, colours, accepts = draws
    labels = np.empty(graph.n_nodes, dtype=np.int64)
    n_clusters, stop = oxi_snakes_and_ladders.swendsen_wang_sweep(
        state,
        np.ascontiguousarray(beta * rows),
        np.ascontiguousarray(graph.edge_index).reshape(-1),
        np.ascontiguousarray(_bond_probability(graph, beta)),
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
@pytest.mark.parametrize("shape", PIN_SHAPES)
@pytest.mark.parametrize("temperature", PIN_TEMPERATURES)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_rust_pass_is_the_oracles_pass_bitwise_on_the_same_draws(
    shape: tuple[int, int], temperature: float, seed: int
) -> None:
    """The pin: same bonds, same clusters, same recolourings, bit for bit."""
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


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("seed", SEEDS)
def test_the_labels_are_the_oracles_own_components(seed: int) -> None:
    """The bond pass and the union-find, against ``find_root`` on the same bonds.

    Root identity and not only the partition: the oracle's ``union_roots`` keeps
    the first edge end's root, and which node labels a component is what puts
    the recolourings in their order --- so a kernel that partitioned the same
    way under another rule would draw a different colour for each cluster.
    """
    n_states, beta = 3, 1.0
    graph, rows, state = _instance((6, 6), n_states, seed)
    bond, colours, accepts = _draws(graph, n_states, seed)

    labels, n_clusters, _ = _kernel_pass(
        graph, rows, state.copy(), beta, (bond, colours, accepts)
    )

    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    active = (state[first] == state[second]) & (bond < _bond_probability(graph, beta))
    parent = np.arange(graph.n_nodes)
    for edge in np.flatnonzero(active):
        potts_mcmc.union_roots(parent, int(first[edge]), int(second[edge]))
    expected = np.array(
        [potts_mcmc.find_root(parent, node) for node in range(graph.n_nodes)]
    )

    assert np.array_equal(labels, expected)
    assert n_clusters == np.unique(expected).size


@pytest.mark.oracle
@pytest.mark.backend
def test_a_guard_that_hands_every_cluster_back_changes_no_recolouring() -> None:
    """The hand-back path, forced and then checked against the pass without it.

    The default guard hands nothing back on a realistic pass, so the path that
    decides a borderline cluster in NumPy would otherwise never run. A guard
    wide enough refuses every cluster with a field to weigh, and the pass must
    come back with the same configuration --- which is what says the two
    halves of the construction agree rather than only the fast one working.
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

    The width is a bound on two summation orders and two ``exp``
    implementations, not a rate: if it were routinely met the pass would be
    running in NumPy while carrying a Rust kernel. Over twelve passes at 16x16
    and three states, every cluster was decided in Rust.
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
@pytest.mark.parametrize("seed", SEEDS)
def test_the_rust_pass_draws_the_exact_boltzmann_distribution(seed: int) -> None:
    """The claim the port has to earn, against enumeration."""
    assert _goodness_of_fit(NO_FIELD, seed) > SIGNIFICANCE


@pytest.mark.oracle
@pytest.mark.parametrize("seed", SEEDS)
def test_the_rust_pass_is_still_exact_in_an_external_field(seed: int) -> None:
    """A field is not optional here.

    The Fortuin-Kasteleyn construction is exact at zero field and the accept
    step is what extends it, so a zero field alone could not tell a pass that
    dropped the accept step from one that kept it.
    """
    assert _goodness_of_fit(WITH_FIELD, seed) > SIGNIFICANCE


@pytest.mark.smoke
def test_the_test_would_catch_a_pass_that_ignored_the_field() -> None:
    """Evidence the two tests above have the power they claim.

    A pass handed a zero field samples the zero-field law, which is a
    *different* law from the one the enumeration above is taken in: if the
    chi-square could not tell them apart it could not tell a broken accept
    step either.
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

    `ClusterCounter` reads every cluster's members and whether its accept step
    passed, which is the gather the port removes. A route that accepted the
    argument and recorded nothing would make issue #551's measurement read
    zero clusters and say so nowhere.
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

    with pytest.raises(ValueError, match="no .* backend"):
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
        oxi_snakes_and_ladders.swendsen_wang_sweep(
            state,
            np.ascontiguousarray(rows),
            np.ascontiguousarray(graph.edge_index).reshape(-1),
            np.ascontiguousarray(_bond_probability(graph, 1.0)),
            bond,
            colours,
            accepts,
            labels,
            _GUARD,
            0,
        )
