"""Alpha expansion, checked against an exact solver before a bound is claimed.

The strongest test here is a *reduction*, not an enumeration. At two labels a
single expansion offers every site the only other label, so alpha expansion is
not an approximation at all --- it must reproduce `phylo.search.maxflow`'s
exact minimum cut, energy for energy. That pins the multi-label construction
against something independently validated, and it is what caught the two
errors this module was written with: the terminal capacities were swapped, and
the auxiliary-node capacities ignored the case where an endpoint's current
label already equals alpha.

Neither error breaks loudly. Both produce a labelling that is merely *worse*,
which is indistinguishable from the algorithm doing badly on a hard problem
--- the enumeration tests below passed while the reduction failed by up to
2.55 in energy.

Then the bound, measured rather than assumed, and two invariants that need no
oracle at all: the energy never rises, and the loop terminates.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest
from phylo.search.alpha_expansion import (
    UNIFORM_POTTS_BOUND,
    alpha_expansion,
    energy,
    expand,
    iterated_conditional_modes,
)
from phylo.search.maxflow import energy as binary_energy
from phylo.search.maxflow import ising_ground_state
from phylo.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

sys.setrecursionlimit(50_000)


def _enumerated(graph: PottsGraph, field_values: np.ndarray, n_states: int) -> float:
    return min(
        energy(graph, field_values, np.array(labelling, dtype=np.int64))
        for labelling in itertools.product(range(n_states), repeat=graph.n_nodes)
    )


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 3)])
@pytest.mark.parametrize("coupling", [0.0, 0.5, 1.5])
def test_two_labels_reproduce_the_exact_minimum_cut(
    shape: tuple[int, int], coupling: float
) -> None:
    # The test that matters. At k = 2 one expansion is exact, so this compares
    # against an independently validated exact solver rather than against
    # enumeration -- and it fails on a construction error that enumeration at
    # these sizes does not notice.
    rng = np.random.default_rng(0)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)

    for _ in range(3):
        field_values = rng.normal(size=(graph.n_nodes, 2))

        realized = alpha_expansion(graph, field_values, 2).energy
        _, exact = ising_ground_state(graph, field_values)

        assert realized == pytest.approx(exact, abs=1e-9)


@pytest.mark.oracle
def test_the_multi_state_energy_agrees_with_the_two_state_one() -> None:
    # The generalization must reduce to what `maxflow` already validates, or
    # the two modules are optimizing different objectives accurately.
    rng = np.random.default_rng(4)
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.7)
    field_values = rng.normal(size=(graph.n_nodes, 2))

    for labelling in itertools.product(range(2), repeat=graph.n_nodes):
        states = np.array(labelling, dtype=np.int64)

        assert energy(graph, field_values, states) == pytest.approx(
            float(binary_energy(graph, field_values, states)), abs=1e-12
        )


@pytest.mark.mathematical
def test_the_energy_never_rises_across_an_expansion() -> None:
    # An invariant that needs no oracle, and the one a sign error in the
    # construction breaks immediately. Checked move by move rather than only
    # end to end, so a rise followed by a larger fall cannot hide.
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


@pytest.mark.structural
def test_the_cycle_terminates_well_inside_its_cap() -> None:
    # Termination follows from monotonicity over a finite state space, so
    # reaching the cap would be a defect rather than a budget. Measured: two
    # or three cycles on every fixture here.
    rng = np.random.default_rng(12)
    graph = lattice_graph((6, 6), BoundaryCondition.OPEN, 1.2)

    for _ in range(3):
        result = alpha_expansion(graph, rng.normal(size=(graph.n_nodes, 4)), 4)

        assert result.cycles <= 6


@pytest.mark.mathematical
@pytest.mark.parametrize("coupling", [0.3, 0.8, 1.5, 3.0])
def test_the_realized_energy_is_inside_the_proved_bound(coupling: float) -> None:
    # The bound is `2 c_max / c_min`, exactly 2 for a uniform coupling. It is
    # the first claim in this repository that holds at *every* size rather
    # than only where enumeration reaches, which is the reason for the ticket.
    #
    # Measured at 3x3 over 40 runs: alpha expansion found the global optimum
    # 39 times, and recovered 99.554% of the achievable improvement in the
    # one miss. The bound is not tight here and is not expected to be.
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


@pytest.mark.simulated_truth
def test_expansion_beats_single_site_descent_past_enumeration() -> None:
    # Where the move set earns its complexity. At the sizes enumeration
    # reaches, the two are indistinguishable -- 3x3 with three labels had both
    # finding the optimum in 31 of 32 runs between them. The separation
    # appears at 8x8 with four labels, past enumeration, where expansion beat
    # the best of eight single-site descents on every trial by 1.8 to 11.0 in
    # energy.
    rng = np.random.default_rng(7)
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, 1.2)

    for _ in range(2):
        field_values = rng.normal(size=(graph.n_nodes, 4))

        expansion = alpha_expansion(graph, field_values, 4).energy
        descent = min(
            iterated_conditional_modes(graph, field_values, 4, seed)[1]
            for seed in range(8)
        )

        assert expansion < descent


@pytest.mark.mathematical
def test_single_site_descent_settles_at_a_local_minimum() -> None:
    # The baseline has to be a fair one: if it stopped early, beating it would
    # say nothing. On termination no single site can improve, which is the
    # definition of the move set it represents.
    rng = np.random.default_rng(21)
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 0.9)
    field_values = rng.normal(size=(graph.n_nodes, 3))

    labelling, settled = iterated_conditional_modes(graph, field_values, 3, 1)

    for node in range(graph.n_nodes):
        for label in range(3):
            candidate = labelling.copy()
            candidate[node] = label

            assert energy(graph, field_values, candidate) >= settled - 1e-12


@pytest.mark.edge_case
@pytest.mark.oracle
def test_a_zero_coupling_problem_is_solved_exactly_by_the_data_term() -> None:
    # With no bonds the sites decouple and the optimum is `argmax` per node,
    # so the answer is known without enumerating or cutting anything.
    rng = np.random.default_rng(5)
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 0.0)
    field_values = rng.normal(size=(graph.n_nodes, 4))

    result = alpha_expansion(graph, field_values, 4)

    np.testing.assert_array_equal(result.labelling, field_values.argmax(axis=1))


@pytest.mark.edge_case
def test_a_dominant_coupling_drives_every_site_to_one_label() -> None:
    # The opposite corner: a coupling large enough that any disagreement costs
    # more than the whole field can repay, so the optimum is constant and
    # equals whichever label the summed field prefers.
    rng = np.random.default_rng(6)
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 50.0)
    field_values = rng.normal(size=(graph.n_nodes, 3))

    result = alpha_expansion(graph, field_values, 3)

    assert len(set(result.labelling.tolist())) == 1
    assert int(result.labelling[0]) == int(field_values.sum(axis=0).argmax())


@pytest.mark.edge_case
def test_a_negative_coupling_is_refused() -> None:
    # The metric condition the bound rests on. Without it the pairwise term is
    # not a metric, the binary sub-problem is not submodular, and the cut does
    # not solve it -- so the guarantee this ticket exists for does not hold.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.4)

    with pytest.raises(ValueError, match="metric only then"):
        alpha_expansion(graph, np.zeros(3), 3)


@pytest.mark.edge_case
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
