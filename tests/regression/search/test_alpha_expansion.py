"""Alpha expansion, checked against an exact solver before a bound is claimed.

At two labels one expansion offers every site the other label, so it must
reproduce `snakes_and_ladders.search.maxflow`'s exact cut, energy for energy.
That reduction caught swapped terminal capacities and auxiliary capacities
ignoring an endpoint already at alpha, while enumeration passed (off by up to
2.55). Then the bound, measured; and two oracle-free invariants: energy never
rises, and the loop terminates.
"""

from __future__ import annotations

import itertools
import sys
from itertools import product

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search.alpha_expansion import (
    UNIFORM_POTTS_BOUND,
    SweepOrder,
    _expansion_network,
    _infinite_capacity,
    _swap_arcs,
    _terminal_capacities,
    alpha_beta_swap,
    alpha_expansion,
    expand,
    iterated_conditional_modes,
    swap,
)
from snakes_and_ladders.search.maxflow import FlowNetwork, ising_ground_state
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    erdos_renyi_graph,
    lattice_graph,
)
from snakes_and_ladders.sim.potts import energy, site_field

from tests._rows import every_row, every_value

sys.setrecursionlimit(50_000)


def _enumerated(graph: PottsGraph, field_values: np.ndarray, n_states: int) -> float:
    return min(
        energy(graph, field_values, np.array(labelling, dtype=np.int64))
        for labelling in itertools.product(range(n_states), repeat=graph.n_nodes)
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_two_labels_reproduce_the_exact_minimum_cut() -> None:
    # At k = 2 one expansion is exact, so this compares against an
    # independently validated exact solver rather than enumeration, and fails
    # on a construction error enumeration at these sizes does not notice.
    def check(shape: tuple[int, int], coupling: float) -> None:
        rng = np.random.default_rng(0)
        graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)

        for _ in range(3):
            field_values = rng.normal(size=(graph.n_nodes, 2))

            realized = alpha_expansion(graph, field_values, 2).energy
            _, exact = ising_ground_state(graph, field_values)

            assert realized == pytest.approx(exact, abs=1e-9)

    every_row(product([(2, 2), (3, 3), (4, 3)], [0.0, 0.5, 1.5]), check)


@pytest.mark.critical
@pytest.mark.analytic
def test_the_energy_never_rises_across_an_expansion() -> None:
    # An invariant needing no oracle, and the one a sign error breaks
    # immediately. Checked move by move, so a rise followed by a larger fall
    # cannot hide.
    rng = np.random.default_rng(9)
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 1.1)
    field_values = rng.normal(size=(graph.n_nodes, 3))

    labelling = field_values.argmax(axis=1).astype(np.int64)
    previous = energy(graph, field_values, labelling)
    for _ in range(4):
        for alpha in range(3):
            labelling, current = expand(graph, field_values, labelling, alpha)

            assert current <= previous + 1e-12
            previous = current


@pytest.mark.smoke
def test_the_cycle_terminates_well_inside_its_cap() -> None:
    # Termination follows from monotonicity over a finite state space, so
    # reaching the cap would be a defect rather than a budget. Measured: two
    # or three cycles on every fixture here.
    rng = np.random.default_rng(12)
    graph = lattice_graph((6, 6), BoundaryCondition.OPEN, 1.2)

    for _ in range(3):
        result = alpha_expansion(graph, rng.normal(size=(graph.n_nodes, 4)), 4)

        assert result.cycles <= 6


@pytest.mark.analytic
@pytest.mark.parametrize("coupling", [0.3, 0.8, 1.5, 3.0])
def test_the_realized_energy_is_inside_the_proved_bound(coupling: float) -> None:
    # The bound `2 c_max / c_min` (2 at uniform coupling) holds at every size.
    # Measured at 3x3 over 40 runs: optimum 39 times, 99.554% of the
    # improvement in the miss; not tight.
    rng = np.random.default_rng(7)
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, coupling)

    for _ in range(4):
        field_values = rng.normal(size=(graph.n_nodes, 3))
        optimum = _enumerated(graph, field_values, 3)
        worst = max(
            energy(graph, field_values, np.array(labelling, dtype=np.int64))
            for labelling in itertools.product(range(3), repeat=graph.n_nodes)
        )

        realized = alpha_expansion(graph, field_values, 3).energy

        # Both energies are negative, so the ratio is taken on the improvement
        # over the worst labelling, which is positive and is what the bound is
        # about.
        achieved = (worst - realized) / (worst - optimum)
        assert achieved >= 1.0 / UNIFORM_POTTS_BOUND
        assert achieved <= 1.0 + 1e-9


@pytest.mark.critical
@pytest.mark.end2end
def test_expansion_beats_single_site_descent_past_enumeration() -> None:
    # At 3x3, three labels both find the optimum 31 of 32; at 8x8, four labels,
    # expansion beat the best of eight descents on every trial by 1.8 to 11.0.
    rng = np.random.default_rng(7)
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, 1.2)

    for _ in range(2):
        field_values = rng.normal(size=(graph.n_nodes, 4))

        expansion = alpha_expansion(graph, field_values, 4).energy
        descent = min(
            iterated_conditional_modes(
                graph, field_values, 4, np.random.default_rng(seed)
            ).energy
            for seed in range(8)
        )

        assert expansion < descent


@pytest.mark.analytic
def test_single_site_descent_settles_at_a_local_minimum() -> None:
    # A baseline that stopped early would make beating it say nothing. On
    # termination no single site can improve, which defines the move set it
    # represents.
    rng = np.random.default_rng(21)
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 0.9)
    field_values = rng.normal(size=(graph.n_nodes, 3))

    labelling, settled = iterated_conditional_modes(
        graph, field_values, 3, np.random.default_rng(1)
    )

    for node in range(graph.n_nodes):
        for label in range(3):
            candidate = labelling.copy()
            candidate[node] = label

            assert energy(graph, field_values, candidate) >= settled - 1e-12


@pytest.mark.critical
@pytest.mark.oracle
def test_a_zero_coupling_problem_is_solved_exactly_by_the_data_term() -> None:
    # With no bonds the sites decouple and the optimum is `argmax` per node,
    # so the answer is known without enumerating or cutting anything.
    rng = np.random.default_rng(5)
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 0.0)
    field_values = rng.normal(size=(graph.n_nodes, 4))

    result = alpha_expansion(graph, field_values, 4)

    np.testing.assert_array_equal(result.labelling, field_values.argmax(axis=1))


@pytest.mark.oracle
def test_a_dominant_coupling_drives_every_site_to_one_label() -> None:
    # At J = 50 the minimizer is constant at the summed field's argmax, a closed
    # form from the model. Realized: constant at label 1, that argmax.
    rng = np.random.default_rng(6)
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 50.0)
    field_values = rng.normal(size=(graph.n_nodes, 3))

    result = alpha_expansion(graph, field_values, 3)

    assert len(set(result.labelling.tolist())) == 1
    assert int(result.labelling[0]) == int(field_values.sum(axis=0).argmax())


@pytest.mark.smoke
def test_a_negative_coupling_is_refused() -> None:
    # The metric condition the bound rests on. Without it the binary
    # sub-problem is not submodular, the cut does not solve it, and the
    # guarantee does not hold.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.4)

    with pytest.raises(ValueError, match="metric only then"):
        alpha_expansion(graph, np.zeros(3), 3)


@pytest.mark.smoke
def test_an_already_optimal_start_makes_no_moves() -> None:
    # Zero moves is information rather than a failure: it says the starting
    # labelling was already expansion-optimal, which is what the run reports.
    rng = np.random.default_rng(8)
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.8)
    field_values = rng.normal(size=(graph.n_nodes, 3))

    settled = alpha_expansion(graph, field_values, 3)
    again = alpha_expansion(graph, field_values, 3, start=settled.labelling)

    assert again.moves == 0
    assert again.energy == pytest.approx(settled.energy, abs=1e-12)


# --- the compiled descent kernel -----------------------------------------------


@pytest.mark.critical
@pytest.mark.oracle
def test_the_numba_descent_reproduces_the_python_one_bitwise() -> None:
    # Same update, order and tie rule (#264): identical, not close. A rugged
    # per-node field makes the start decide the optimum; 20 of 20 agree at 32x32.
    def check(seed: int) -> None:
        graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.5)
        field = np.random.default_rng(100 + seed).normal(size=(graph.n_nodes, 3))

        python = iterated_conditional_modes(
            graph, field, 3, np.random.default_rng(seed), backend=Backend.PYTHON
        )
        compiled = iterated_conditional_modes(
            graph, field, 3, np.random.default_rng(seed), backend=Backend.NUMBA
        )

        assert np.array_equal(python.labelling, compiled.labelling)
        assert python.energy == compiled.energy

    every_value(range(6), check)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("sweep_order", [SweepOrder.RANDOM, SweepOrder.INDEX])
def test_the_numba_descent_in_any_order_every_sweep_run_is_the_python_one(
    sweep_order: SweepOrder,
) -> None:
    # Issue #923: ICM in a random order with every sweep run, the heat bath
    # at T = 0 (`ground_state.run_icm_random`). The compiled sweep takes the
    # permutations the Python sweep draws, in its order, so the labelling,
    # the energy and the generator's state after are the Python sweep's.
    def check(seed: int) -> None:
        graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.5)
        field = np.random.default_rng(300 + seed).normal(size=(graph.n_nodes, 3))
        streams = [np.random.default_rng(seed) for _ in range(2)]
        python, compiled = (
            iterated_conditional_modes(
                graph,
                field,
                3,
                stream,
                max_sweeps=12,
                sweep_order=sweep_order,
                stop_when_clean=False,
                backend=backend,
            )
            for stream, backend in zip(
                streams, (Backend.PYTHON, Backend.NUMBA), strict=True
            )
        )
        assert np.array_equal(python.labelling, compiled.labelling)
        assert python.energy == compiled.energy
        assert streams[0].integers(1 << 62) == streams[1].integers(1 << 62)

    every_value(range(6), check)


@pytest.mark.smoke
def test_descent_has_no_rust_backend() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match="runs on numba or python, not rust"):
        iterated_conditional_modes(
            graph, np.zeros(3), 3, np.random.default_rng(0), backend=Backend.RUST
        )


@pytest.mark.critical
@pytest.mark.oracle
def test_the_rust_cut_reproduces_the_python_expansion() -> None:
    # #528: the binding returns its source side; the solvers agree on the
    # labelling, not only the energy. Realized 12 of 12.
    def check(seed: int, n_states: int) -> None:
        graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.5)
        field = np.random.default_rng(500 + seed).normal(size=(graph.n_nodes, n_states))

        python = alpha_expansion(graph, field, n_states, backend=Backend.PYTHON)
        compiled = alpha_expansion(graph, field, n_states, backend=Backend.RUST)

        assert np.array_equal(python.labelling, compiled.labelling)
        assert python.energy == compiled.energy
        assert (python.cycles, python.moves) == (compiled.cycles, compiled.moves)

    every_row(product(range(6), [2, 4]), check)


@pytest.mark.smoke
def test_expansion_has_no_numba_backend() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(
        ValueError, match="minimum cut runs on python or rust, not numba"
    ):
        alpha_expansion(graph, np.zeros(3), 3, backend=Backend.NUMBA)


def _network_by_hand(
    graph: PottsGraph, values: np.ndarray, labelling: np.ndarray, alpha: int
) -> FlowNetwork:
    """The ``add_edge`` loop `_expansion_network` replaces (#598), kept as its oracle."""
    disagreeing = {
        position
        for position, (first, second) in enumerate(graph.edges)
        if labelling[first] != labelling[second]
    }
    source, sink = graph.n_nodes, graph.n_nodes + 1
    network = FlowNetwork(n_nodes=graph.n_nodes + 2 + len(disagreeing))
    infinite = _infinite_capacity(graph, values)
    for node in range(graph.n_nodes):
        switch_cost = -float(values[node, alpha])
        keep_cost = (
            infinite
            if labelling[node] == alpha
            else -float(values[node, labelling[node]])
        )
        offset = min(keep_cost, switch_cost)
        network.add_edge(source, node, switch_cost - offset)
        network.add_edge(node, sink, keep_cost - offset)
    auxiliary = graph.n_nodes + 2
    for position, ((first, second), coupling) in enumerate(graph.weighted_edges()):
        first_differs = coupling if labelling[first] != alpha else 0.0
        second_differs = coupling if labelling[second] != alpha else 0.0
        if position not in disagreeing:
            network.add_edge(first, second, first_differs, reverse=first_differs)
            continue
        network.add_edge(first, auxiliary, first_differs, reverse=first_differs)
        network.add_edge(second, auxiliary, second_differs, reverse=second_differs)
        network.add_edge(auxiliary, sink, coupling)
        auxiliary += 1
    return network


@pytest.mark.critical
@pytest.mark.oracle
def test_the_vectorized_network_is_the_loops_network_arc_for_arc() -> None:
    rng = np.random.default_rng(598)
    checked = 0
    for shape in ((3, 3), (4, 4), (5, 4)):
        for boundary in (BoundaryCondition.OPEN, BoundaryCondition.PERIODIC):
            graph = lattice_graph(shape, boundary, 0.8)
            for n_states in (2, 3, 5):
                values = site_field(
                    rng.normal(size=(graph.n_nodes, n_states)), graph.n_nodes
                )
                for _ in range(3):
                    labelling = rng.integers(0, n_states, size=graph.n_nodes)
                    for alpha in range(n_states):
                        wanted = _network_by_hand(graph, values, labelling, alpha)
                        built = _expansion_network(graph, values, labelling, alpha)
                        assert built.n_nodes == wanted.n_nodes
                        assert built.target == wanted.target
                        assert built.capacity == wanted.capacity
                        assert built.outgoing == wanted.outgoing
                        checked += 1
    assert checked == 3 * 2 * (2 + 3 + 5) * 3, checked


@pytest.mark.critical
@pytest.mark.oracle
def test_the_vectorized_network_holds_on_a_graph_that_is_not_a_lattice() -> None:
    # The lattice is regular; the build indexes by edge position, so an
    # irregular graph is where an off-by-one in the auxiliary numbering shows.
    rng = np.random.default_rng(1598)
    for _ in range(15):
        graph = erdos_renyi_graph(12, 0.35, 0.6, rng)
        values = site_field(rng.normal(size=(graph.n_nodes, 4)), graph.n_nodes)
        labelling = rng.integers(0, 4, size=graph.n_nodes)
        for alpha in range(4):
            wanted = _network_by_hand(graph, values, labelling, alpha)
            built = _expansion_network(graph, values, labelling, alpha)
            assert built.target == wanted.target
            assert built.capacity == wanted.capacity
            assert built.outgoing == wanted.outgoing


def _recomputing_descent(
    graph: PottsGraph,
    values: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    start: np.ndarray,
) -> np.ndarray:
    """The loop `search.spatio_sequential.label_step` ran, kept as the sweep's oracle.

    A full energy per candidate label: the same argmin as the local delta, the long way.
    """
    current = np.asarray(start, dtype=np.int64).copy()
    best = energy(graph, values, current)
    for _ in range(200):
        moved = False
        for node in rng.permutation(graph.n_nodes):
            for label in range(n_states):
                if label == current[node]:
                    continue
                trial = current.copy()
                trial[node] = label
                value = energy(graph, values, trial)
                if value < best - 1e-12:
                    best, current, moved = value, trial, True
        if not moved:
            break
    return current


@pytest.mark.critical
@pytest.mark.oracle
def test_the_local_delta_sweep_is_the_recomputing_descent() -> None:
    # What lets `label_step` call the sweep (#858): the same site order from
    # the same start, and the argmin read off the site's own field and
    # incident edges rather than off `O(n_edges)` of energy per candidate.
    # Realized: 6 of 6 seeds agree site for site, at 3 labels on 36 sites.
    def check(seed: int) -> None:
        graph = lattice_graph((6, 6), BoundaryCondition.OPEN, 0.7)
        field = np.random.default_rng(700 + seed).normal(size=(graph.n_nodes, 3))
        start = np.random.default_rng(seed).integers(0, 3, size=graph.n_nodes)

        swept, _ = iterated_conditional_modes(
            graph,
            field,
            3,
            np.random.default_rng(seed),
            start=start,
            sweep_order=SweepOrder.RANDOM,
            backend=Backend.PYTHON,
        )
        recomputed = _recomputing_descent(
            graph,
            site_field(field, graph.n_nodes),
            3,
            np.random.default_rng(seed),
            start,
        )

        assert np.array_equal(swept, recomputed)

    every_value(range(6), check)


@pytest.mark.analytic
def test_a_random_order_runs_every_sweep_when_a_clean_one_does_not_end_it() -> None:
    # Gibbs at T = 0 is charged a fixed budget, so it spends every sweep; the
    # descent is still monotone, since each visit takes an argmin.
    graph = lattice_graph((5, 5), BoundaryCondition.PERIODIC, 0.9)
    field = np.random.default_rng(31).normal(size=(graph.n_nodes, 3))

    labelling, value = iterated_conditional_modes(
        graph,
        field,
        3,
        np.random.default_rng(4),
        max_sweeps=25,
        sweep_order=SweepOrder.RANDOM,
        stop_when_clean=False,
        backend=Backend.PYTHON,
    )
    settled, settled_value = iterated_conditional_modes(
        graph,
        field,
        3,
        np.random.default_rng(4),
        max_sweeps=25,
        sweep_order=SweepOrder.RANDOM,
        backend=Backend.PYTHON,
    )

    assert value <= energy(graph, field, labelling) + 1e-12
    assert settled_value <= value + 1e-12
    assert settled.shape == labelling.shape


@pytest.mark.smoke
def test_the_compiled_sweep_refuses_an_order_it_does_not_walk() -> None:
    # Since #923 the compiled sweep runs any order when every sweep runs; a
    # random order that stops on a clean sweep would spend the generator
    # past the stop, and is refused.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match="stop_when_clean=True needs python"):
        iterated_conditional_modes(
            graph,
            np.zeros(3),
            3,
            np.random.default_rng(0),
            sweep_order=SweepOrder.RANDOM,
        )


# --- the one expansion template (issue #858) ----------------------------------

#: Both moves on a 4x4 open lattice at 0.8, recorded before the fold: the same
#: labelling in 2 moves (expansion) and 3 (swap), so the counts check the template.
#: The energy moved one ulp (7.1e-15) when `energies` left BLAS (#1044).
RECORDED_LABELLING = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
RECORDED_ENERGY = -25.22872289815644
RECORDED_CYCLES = 2
RECORDED_MOVES = {"expansion": 2, "swap": 3}


@pytest.mark.smoke
@pytest.mark.snapshot
@pytest.mark.parametrize(
    ("move", "method"), [("expansion", alpha_expansion), ("swap", alpha_beta_swap)]
)
def test_a_move_through_the_template_returns_its_recorded_result(
    move: str, method: object
) -> None:
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.8)
    field = np.random.default_rng(858).normal(size=(graph.n_nodes, 3))

    result = method(graph, field, 3)  # type: ignore[operator]

    assert list(result.labelling) == RECORDED_LABELLING
    assert result.energy == RECORDED_ENERGY
    assert (result.cycles, result.moves) == (RECORDED_CYCLES, RECORDED_MOVES[move])


@pytest.mark.smoke
def test_the_one_coupling_guard_keeps_each_move_its_own_reason() -> None:
    # Three constructions, three reasons, one refusal: what the sign buys is
    # the caller's and stays in the message that names it.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.4)
    prefix = "every coupling must be non-negative, got -0.4: "

    with pytest.raises(ValueError, match="metric only then") as expansion:
        alpha_expansion(graph, np.zeros(3), 3)
    with pytest.raises(ValueError, match="submodular only then") as swap_refusal:
        alpha_beta_swap(graph, np.zeros(3), 3)
    with pytest.raises(ValueError, match="non-submodular") as cut:
        ising_ground_state(graph, np.zeros(2))

    assert str(expansion.value) == prefix + (
        "the Potts pairwise term is a metric only then, and the factor-2 "
        "bound rests on it"
    )
    assert str(swap_refusal.value) == prefix + (
        "the swap's binary sub-problem is submodular only then"
    )
    assert str(cut.value) == prefix + (
        "a negative coupling makes the energy non-submodular, the ground "
        "state NP-hard, and this construction inapplicable rather than slow"
    )


def _swap_network_by_loop(
    graph: PottsGraph, values: np.ndarray, labelling: np.ndarray, alpha: int, beta: int
) -> FlowNetwork:
    """The ``add_edge`` loop the swap's arcs replace (issue #935), kept as their oracle."""
    moving = np.flatnonzero((labelling == alpha) | (labelling == beta))
    position = {int(node): index for index, node in enumerate(moving)}
    source, sink = moving.size, moving.size + 1
    network = FlowNetwork(n_nodes=moving.size + 2)
    from_source, to_sink = _terminal_capacities(
        -values[moving, alpha].astype(float), -values[moving, beta].astype(float)
    )
    for index in range(moving.size):
        network.add_edge(source, index, float(from_source[index]))
        network.add_edge(index, sink, float(to_sink[index]))
    for (first, second), coupling in graph.weighted_edges():
        if first in position and second in position:
            network.add_edge(
                position[first], position[second], coupling, reverse=coupling
            )
    return network


@pytest.mark.oracle
def test_the_swap_arcs_are_the_add_edge_loop_arc_for_arc() -> None:
    # Issue #935: the swap network is built as arrays, in the order the loop
    # appended its arcs, since a minimum cut need not be unique and the two
    # solvers are pinned to one labelling through one network.
    def check(seed: int) -> None:
        graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.5)
        rng = np.random.default_rng(935 + seed)
        values = rng.normal(size=(graph.n_nodes, 4))
        labelling = rng.integers(0, 4, graph.n_nodes)
        moving = np.flatnonzero((labelling == 1) | (labelling == 3))
        network = _swap_arcs(graph, values, moving, 1, 3).network()
        loop = _swap_network_by_loop(graph, values, labelling, 1, 3)
        assert network.target == loop.target
        assert network.capacity == loop.capacity
        assert network.outgoing == loop.outgoing
        python = swap(graph, values, labelling, 1, 3, backend=Backend.PYTHON)
        compiled = swap(graph, values, labelling, 1, 3, backend=Backend.RUST)
        assert np.array_equal(python.labelling, compiled.labelling)
        assert python.energy == compiled.energy

    every_value(range(4), check)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_cut_moves_agree_across_solvers_on_tied_fields() -> None:
    # #935: one-decimal fields tie with residuals of 1e-16 against 0; at the
    # shared saturation floor both cuts are minimal and agree bitwise (7 of 120
    # split before). The Rust network differs (no auxiliary node) and agrees.
    def check(seed: int) -> None:
        rng = np.random.default_rng(935 + seed)
        graph = lattice_graph((12, 12), BoundaryCondition.OPEN, 0.7)
        n_states = int(rng.choice([2, 3, 5]))
        field = np.round(rng.normal(size=(graph.n_nodes, n_states)), 1)
        for solve in (alpha_expansion, alpha_beta_swap):
            python = solve(graph, field, n_states, backend=Backend.PYTHON)
            compiled = solve(graph, field, n_states, backend=Backend.RUST)
            assert np.array_equal(python.labelling, compiled.labelling), solve.__name__
            assert python.energy == compiled.energy

    every_value(range(8), check)
