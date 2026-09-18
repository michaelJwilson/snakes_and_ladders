"""The spectral bound asserted as a bound, and the start measured as a start.

Issue #718. The bound is a theorem, so it is asserted on every instance ---
against the enumerated minimum where enumeration fits, against the exact cut
where the couplings are attractive, and against the closed-form ground state
of the frustrated lattice at every size. Its *tightness* is a measurement and
is reported, not asserted. The start is measured against random labellings on
the same instances: a start that is no better than a draw is not a start.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.search.spectral import spectral_bound, spectral_start
from snakes_and_ladders.sim.canonical import minimum_frustrated_edges
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import energies

PARAMS = fixture("planted_glass", "ci").params


def _enumerated_minimum(graph: PottsGraph, field_values: np.ndarray) -> float:
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    return float(energies(graph, field_values, configurations).min())


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 3)])
@pytest.mark.parametrize("coupling", [-1.2, -0.4, 0.0, 0.4, 1.2])
def test_the_bound_never_exceeds_the_enumerated_minimum(
    shape: tuple[int, int], coupling: float
) -> None:
    graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)
    for seed in range(3):
        field_values = np.random.default_rng(seed).normal(size=(graph.n_nodes, 2))
        exact = _enumerated_minimum(graph, field_values)

        result = spectral_bound(graph, field_values)

        assert result.bound <= exact + 1e-12


@pytest.mark.oracle
def test_the_bound_never_exceeds_the_exact_cut_where_the_cut_applies() -> None:
    # Attractive couplings: the minimum cut is the exact ground state at any
    # size, so the bound is checked far past enumeration.
    graph = lattice_graph((24, 24), BoundaryCondition.OPEN, 0.6)
    field_values = np.random.default_rng(718).normal(size=(graph.n_nodes, 2))
    _, exact = ising_ground_state(graph, field_values)

    result = spectral_bound(graph, field_values)

    assert result.bound <= exact + 1e-9
    assert result.lifted


@pytest.mark.oracle
@pytest.mark.parametrize("extent", [3, 6, 9])
def test_the_bound_holds_on_the_frustrated_lattice_at_every_size(extent: int) -> None:
    # The closed form: `N` of `3N` edges frustrated in any ground state, and
    # under `energies`' convention an agreeing edge on a repulsive coupling
    # costs `|J|` while a disagreeing one costs nothing, so the optimum is
    # `|J|` per frustrated edge.
    graph = PARAMS.lattice() if extent == 3 else _triangular(extent)
    exact = float(minimum_frustrated_edges(graph)) * abs(PARAMS.coupling)
    result = spectral_bound(graph, np.zeros(2))

    assert not result.lifted
    assert result.bound <= exact + 1e-12


def _triangular(extent: int) -> PottsGraph:
    from snakes_and_ladders.sim.graph import triangular_lattice_graph

    return triangular_lattice_graph(
        (extent, extent), BoundaryCondition.PERIODIC, PARAMS.coupling
    )


@pytest.mark.oracle
def test_the_bound_holds_on_the_planted_glass() -> None:
    glass = PARAMS.glass(
        PARAMS.glass_frustrations[0], np.random.default_rng(PARAMS.seed)
    )

    result = spectral_bound(glass.graph, np.zeros(2))

    # The planted energy is an upper bound on the optimum, so the two
    # bracket it; the bracket's width is what `STATUS.md` reports.
    assert result.bound <= glass.planted_energy + 1e-12


@pytest.mark.mathematical
def test_the_ghost_spin_reproduces_the_unlifted_bound_at_zero_field() -> None:
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 0.8)
    unlifted = spectral_bound(graph, np.zeros(2))
    almost = spectral_bound(graph, np.full((graph.n_nodes, 2), 0.3))

    # A field equal across states shifts the energy by its sum and lifts
    # nothing; a zero field lifts nothing.
    assert not unlifted.lifted
    assert not almost.lifted
    assert almost.bound == pytest.approx(
        unlifted.bound - 0.3 * graph.n_nodes, abs=1e-12
    )


@pytest.mark.simulated_truth
def test_the_start_beats_random_labellings_on_the_ferromagnet() -> None:
    graph = lattice_graph((12, 12), BoundaryCondition.OPEN, 0.6)
    rng = np.random.default_rng(718)
    field_values = 0.3 * rng.normal(size=(graph.n_nodes, 2))
    start = spectral_start(graph, field_values)
    draws = rng.integers(0, 2, size=(200, graph.n_nodes))

    started = float(energies(graph, field_values, start[None])[0])
    random = energies(graph, field_values, draws)
    _, exact = ising_ground_state(graph, field_values)

    assert start.shape == (graph.n_nodes,)
    assert started < random.min()
    assert exact <= started


@pytest.mark.simulated_truth
def test_the_start_on_the_planted_glass_beats_random_labellings_and_sits_in_the_bracket() -> (
    None
):
    # The planted state is a state of known energy and not the optimum
    # (`sim/CLAUDE.md`: a known energy is not a known optimum), so its
    # overlap with the start referees nothing: on the fixture's seed the
    # enumerated optimum is -14 against the planted -7. What is asserted is
    # what a start is for: no worse than the best of 2,000 random labellings,
    # and inside the bracket the bound and the planted energy make. Measured
    # on the fixture: bound -14.51 (-19.26 uncorrected), start -13,
    # enumerated optimum -14, best random draw -12, planted -7.
    glass = PARAMS.glass(
        PARAMS.glass_frustrations[0], np.random.default_rng(PARAMS.seed)
    )
    start = spectral_start(glass.graph, np.zeros(2))
    draws = np.random.default_rng(0).integers(0, 2, size=(2000, glass.graph.n_nodes))

    started = float(energies(glass.graph, np.zeros(2), start[None])[0])
    random = energies(glass.graph, np.zeros(2), draws)
    bound = spectral_bound(glass.graph, np.zeros(2)).bound

    assert bound <= started <= random.min()
    assert started <= glass.planted_energy


@pytest.mark.edge_case
def test_more_than_two_states_is_refused() -> None:
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0)
    with pytest.raises(ValueError, match="two-state"):
        spectral_bound(graph, np.zeros(3))
    with pytest.raises(ValueError, match="two-state"):
        spectral_start(graph, np.zeros((4, 3)))
