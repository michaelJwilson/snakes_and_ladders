"""Simulated bifurcation against the referees that have a guarantee (issue #823).

A relaxation with no bound is refereed by what has one. At two labels the
minimum cut is exact at any size, so the relaxation can never do better and the
test says so as a bound. At nine sites and on the declared glass enumeration is
the referee; on the glass a single descent reaches the ground state from
0.079 of starts (the fixture's record), and the relaxation is read against
that rate. The tensor path is one arithmetic with the NumPy one, so the two
labellings are pinned equal rather than close. Every rate asserted below is a
bound under the value measured, stated beside it, over a stated number of
seeds, because a heuristic's success is a rate and not a number.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal import oxisal
from sal.backend import Backend
from sal.fixtures import Scale
from sal.search import ground_state
from sal.search.bifurcation import (
    A_END,
    BifurcationResult,
    _integrate_numpy,
    simulated_bifurcation,
)
from sal.search.maxflow import ising_ground_state
from sal.sim.canonical import (
    FrustratedLatticeParams,
    PlantedSpinGlass,
    frustrated_triangular_lattice,
)
from sal.sim.fixtures import Baseline, baseline, fixture
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import SpatioOnlyParams, energy

#: The enumerated ground energy of the frustrated 3x3 triangular antiferromagnet
#: at two states: nine sites, every triangle carries one unsatisfied bond.
_FRUSTRATED_Q2 = 9.0
_SEEDS = 16
_EXACT = 1e-9


@pytest.fixture(scope="module")
def glass_params() -> FrustratedLatticeParams:
    params: FrustratedLatticeParams = fixture("planted_glass", "ci").params
    return params


@pytest.fixture(scope="module")
def glass(glass_params: FrustratedLatticeParams) -> PlantedSpinGlass:
    (frustration,) = glass_params.glass_frustrations
    return glass_params.glass(frustration, np.random.default_rng(glass_params.seed))


@pytest.fixture(scope="module")
def record() -> Baseline:
    return baseline("planted_glass", Scale.CI)


def _success(graph_energy: float, runs: list[BifurcationResult]) -> float:
    return float(np.mean([abs(run.energy - graph_energy) < _EXACT for run in runs]))


@pytest.mark.oracle
def test_the_relaxation_reaches_the_enumerated_ground_state_of_the_declared_glass(
    glass: PlantedSpinGlass, record: Baseline
) -> None:
    # The regime the method is for: couplings of both signs and no field, where
    # the cut does not apply and a single descent solves the instance from
    # 0.079 of starts (the record). One replica per seed, so the rate is the
    # method's and not a restart's. Realized 0.562 over 32 seeds.
    best = record.value("enumerated_ground_energy")
    field = np.zeros((glass.graph.n_nodes, 2))
    runs = [
        simulated_bifurcation(
            glass.graph, field, np.random.default_rng(seed), dt=0.25, n_states=2
        )
        for seed in range(_SEEDS)
    ]

    assert all(run.energy >= best - _EXACT for run in runs), (
        "below the enumerated minimum"
    )
    rate = _success(best, runs)
    assert rate > 0.3, f"realized {rate}"
    assert rate > record.value("descent_rate") + 4 * record.value(
        "descent_rate_half_width"
    )


@pytest.mark.oracle
def test_at_two_labels_the_relaxation_is_read_against_the_exact_cut() -> None:
    # The exact cut bounds the relaxation below, asserted; with a dominant
    # field it lands above (1 of 6 at the optimum, mean gap 1.09), as stated.
    rng = np.random.default_rng(0)
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.4)
    gaps = []
    for _ in range(6):
        field = rng.normal(size=(graph.n_nodes, 2))
        _, exact = ising_ground_state(graph, field)
        run = simulated_bifurcation(
            graph, field, np.random.default_rng(1), dt=0.1, n_replicas=4, n_states=2
        )
        assert run.energy >= exact - _EXACT
        gaps.append(run.energy - exact)

    assert min(gaps) < 3.0, "every run far from the cut: the dynamics are not searching"


@pytest.mark.oracle
def test_the_frustrated_antiferromagnet_ground_state_is_reached_where_the_cut_cannot() -> (
    None
):
    # A negative coupling refuses the cut; enumeration over 2^9 referees.
    # The declared instance, so the pairing is read from the fixture and not
    # from a literal that could drift from it. Realized 0.938 over 16 seeds.
    declared: FrustratedLatticeParams = fixture("frustrated_lattice", "ci").params
    graph = frustrated_triangular_lattice(
        declared.shape, declared.boundary, declared.coupling
    )
    field = np.zeros((graph.n_nodes, 2))
    exact = min(
        energy(graph, field, np.array(labels))
        for labels in itertools.product(range(2), repeat=graph.n_nodes)
    )
    assert exact == _FRUSTRATED_Q2
    runs = [
        simulated_bifurcation(
            graph, field, np.random.default_rng(seed), dt=0.25, n_states=2
        )
        for seed in range(_SEEDS)
    ]

    assert _success(exact, runs) > 0.7


@pytest.mark.oracle
def test_three_labels_reach_the_enumerated_minimum_at_nine_sites() -> None:
    # `spatio_only/ci`: nine sites, three classes, the rung `search.ground_state`
    # ranks every solver on. 3^9 configurations enumerate.
    params: SpatioOnlyParams = fixture("spatio_only", "ci").params
    field, _ = ground_state.rung_field(params, 3)
    exact = min(
        energy(params.graph, field, np.array(labels))
        for labels in itertools.product(range(3), repeat=params.graph.n_nodes)
    )
    run = simulated_bifurcation(
        params.graph, field, np.random.default_rng(2), dt=0.1, n_replicas=8, n_states=3
    )

    assert run.energy == pytest.approx(exact, abs=_EXACT)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_torch_path_returns_the_numpy_labelling() -> None:
    # One arithmetic over two array libraries. The scatter-add reduces in an
    # order neither promises, so the oscillators may differ in their last bits;
    # the labelling and its discrete energy may not.
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.4)
    field = np.random.default_rng(5).normal(size=(graph.n_nodes, 2))
    numpy_run = simulated_bifurcation(
        graph, field, np.random.default_rng(1), steps=300, n_states=2
    )
    torch_run = simulated_bifurcation(
        graph,
        field,
        np.random.default_rng(1),
        steps=300,
        backend=Backend.TORCH,
        n_states=2,
    )

    assert np.array_equal(numpy_run.labelling, torch_run.labelling)
    assert numpy_run.energy == torch_run.energy


@pytest.mark.analytic
def test_the_energy_reported_is_the_discrete_energy_of_the_labelling() -> None:
    # Never the relaxed score: a run is judged on the labelling it returns.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.6)
    field = np.random.default_rng(3).normal(size=(graph.n_nodes, 3))
    run = simulated_bifurcation(
        graph, field, np.random.default_rng(0), steps=200, n_states=3
    )

    assert run.labelling.shape == (graph.n_nodes,)
    assert run.energy == energy(graph, field, run.labelling)
    assert run.steps == 200
    assert run.n_replicas == 1


@pytest.mark.smoke
def test_the_relaxation_refuses_what_it_cannot_answer_for() -> None:
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.5)
    field = np.zeros((4, 2))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_states"):
        simulated_bifurcation(graph, np.zeros((4, 1)), rng, n_states=1)
    with pytest.raises(ValueError, match="steps and n_replicas"):
        simulated_bifurcation(graph, field, rng, steps=0, n_states=2)
    with pytest.raises(ValueError, match="dt"):
        simulated_bifurcation(graph, field, rng, dt=0.0, n_states=2)
    with pytest.raises(ValueError, match="runs on"):
        simulated_bifurcation(graph, field, rng, backend=Backend.NUMBA, n_states=2)


@pytest.mark.oracle
@pytest.mark.parametrize("discrete", [True, False])
def test_the_compiled_integration_is_the_numpy_one_bitwise(discrete: bool) -> None:
    # Issue #997: the same update in the same order over the graph's
    # compressed rows, whose per-row sums NumPy forms sequentially below
    # eight terms: the oscillators agree bit for bit, not only the labelling.
    graph = lattice_graph((9, 9), BoundaryCondition.PERIODIC, 0.4)
    rows = np.random.default_rng(997).normal(size=(graph.n_nodes, 3))
    start = np.random.default_rng(998).uniform(-0.1, 0.1, size=rows.shape)
    numpy_end = _integrate_numpy(
        rows,
        graph.incidence,
        start.copy(),
        steps=150,
        dt=0.2,
        c0=0.3,
        discrete=discrete,
    )
    compiled = start.copy()
    oxisal.bifurcation_integrate(
        rows.reshape(-1),
        graph.incidence.offsets,
        graph.incidence.neighbours,
        graph.incidence.couplings,
        compiled.reshape(-1),
        3,
        150,
        0.2,
        0.3,
        A_END,
        discrete,
    )
    assert np.array_equal(compiled, numpy_end)
