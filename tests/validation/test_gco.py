"""The package's alpha expansion beside gco's, run in a subprocess (issue #974).

gco (Veksler and Delong) is not an oracle for the labelling: expansion stops
at a local minimum. Checked: gco's labelling is a fixed point of our move
(energy bitwise unchanged) at 16² and 71², q = 3 and 10; on a 4x4 lattice at
q = 3 both are within Boykov, Veksler and Zabih's factor 2 of the minimum over
3^16 = 43,046,721 labellings (non-negative terms), and both reach it within
1e-12; at 71² the energies agree within 1% (#938 measured 0.13% at q = 10).
`external.ground_state` (#1282, step 3) returns the adapter's labelling
bitwise, expansion and swap, from gco's start and a given one, one-shot and
in a session; with forbidden labels (#1139) it holds none, is the adapter's
labelling on #1274's stand-in bitwise, and on the enumerable instances
reaches the constrained minimum within 1e-12.
With labels forbidden (#1139, #1274), on the finite stand-in
`_forbidden.stand_in` states: neither expansion holds a forbidden label,
gco's labelling is a fixed point of the package's move, and on 13 enumerable
instances both reach the constrained minimum within 1e-12.
Runtime goal: `test_goals.py`.
"""

from __future__ import annotations

from functools import partial
from itertools import product

import numpy as np
import pytest
from sal import external
from sal.backend import Backend
from sal.cost import Cost
from sal.enumeration import configurations
from sal.external import Solver
from sal.external.potts_inputs import stand_in
from sal.opt.budget import Budget
from sal.search.alpha_expansion import alpha_expansion
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, energies, energy, forbid
from sal.validation import gco

from tests._frameworks import requires
from tests._rows import every_row, every_value
from tests.regression.search.test_forbidden_labels import (
    ALLOWED,
    FINITE,
    GRAPH,
    SMALL,
    _small_problem,
)
from tests.validation._forbidden import check_expansion, enumerable, lattices

pytestmark = [
    pytest.mark.validation,
    requires("gco"),
]


def _potts(side: int, n_states: int, seed: int) -> tuple[PottsGraph, np.ndarray]:
    """An open lattice at its critical coupling under a standard normal field."""
    graph = lattice_graph(
        (side, side), BoundaryCondition.OPEN, critical_coupling(n_states)
    )
    field = np.random.default_rng(seed).normal(size=(graph.n_nodes, n_states))
    return graph, field


def _non_negative(graph: PottsGraph, field: np.ndarray, value: float) -> float:
    """``value`` shifted by ``sum J`` and each row's ``max h``: the bound's non-negative form."""
    return value + float(graph.edge_coupling.sum()) + float(field.max(axis=1).sum())


@pytest.mark.experiment
def test_gcos_labelling_is_a_fixed_point_of_the_package_move() -> None:
    def check(n_states: int, side: int) -> None:
        graph, field = _potts(side, n_states, 974)
        theirs = gco.alpha_expansion(graph, field, n_states=n_states)
        from_theirs = alpha_expansion(
            graph,
            field,
            start=theirs.labelling,
            backend=Backend.RUST,
            n_states=n_states,
        )
        assert from_theirs.moves == 0
        assert np.array_equal(from_theirs.labelling, theirs.labelling)
        assert from_theirs.energy == theirs.energy

    every_row(product([3, 10], [16, 71]), check)


@pytest.mark.oracle
@pytest.mark.release  # 22.6 s in the tier, over the 10 s cap (#1088)
def test_both_expansions_are_within_the_factor_two_bound_of_the_optimum() -> None:
    graph, field = _potts(4, 3, 974)
    # 3^16 in 81 blocks of 3^12, each block's tail enumerated once.
    rest = configurations(3, 12, limit=3**12)
    best = np.inf
    for head in configurations(3, 4):
        block = np.hstack([np.broadcast_to(head, (rest.shape[0], 4)), rest])
        best = min(best, float(energies(graph, field, block).min()))
    optimum = _non_negative(graph, field, best)
    ours = alpha_expansion(graph, field, backend=Backend.RUST, n_states=3)
    theirs = gco.alpha_expansion(graph, field, n_states=3)
    for found in (ours.energy, theirs.energy):
        shifted = _non_negative(graph, field, found)
        assert optimum <= shifted * (1.0 + 1e-12) <= 2.0 * optimum
        # Both reach the optimum on this instance, 7e-15 apart in the sum.
        assert found == pytest.approx(best, rel=1e-12)
    assert theirs.energy == energy(graph, field, theirs.labelling)
    assert optimum > 0.0


@pytest.mark.experiment
def test_the_two_energies_agree_within_one_per_cent_at_71() -> None:
    def check(n_states: int) -> None:
        graph, field = _potts(71, n_states, 974)
        ours = alpha_expansion(graph, field, backend=Backend.RUST, n_states=n_states)
        theirs = gco.alpha_expansion(graph, field, n_states=n_states)
        assert theirs.energy == pytest.approx(ours.energy, rel=1e-2)

    every_value([3, 10], check)


#: One call of gco, the unit `external.potts.UNITS` charges it in.
ONE_CALL = Budget(Cost.FITS, 1)

#: Agreement of an expansion's energy with the enumerated constrained
#: minimum: sums over the same terms in other orders (#1274).
ENERGY_AGREEMENT = 1e-12


def _moves(solver: Solver) -> str:
    return "swap" if solver is Solver.GCO_SWAP else "expansion"


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize("solver", [Solver.GCO_EXPANSION, Solver.GCO_SWAP], ids=str)
def test_external_ground_state_is_the_adapters_labelling(solver: Solver) -> None:
    # #1282, step 3: the same bytes reach gco by either path, so the
    # labelling and its energy are bitwise; from gco's own start and from a
    # drawn one.
    def check(n_states: int, side: int, seeded: bool) -> None:
        graph, field = _potts(side, n_states, 974)
        start = (
            np.random.default_rng(1282).integers(n_states, size=graph.n_nodes)
            if seeded
            else None
        )
        theirs = gco.alpha_expansion(
            graph, field, n_states, start=start, move=_moves(solver)
        )
        ours = external.ground_state(
            graph, field, solver, ONE_CALL, np.random.default_rng(0), start=start
        )
        assert np.array_equal(ours.labelling, theirs.labelling)
        assert ours.labelling.dtype == theirs.labelling.dtype
        assert ours.energy == theirs.energy
        assert (ours.spent, ours.converged) == (1, True)
        assert ours.termination.converged
        assert ours.provenance == external.provenance(solver)
        assert not ours.provenance.osi

    every_row(product([3, 10], [16], [False, True]), check)


@pytest.mark.analytic
@pytest.mark.patch
@pytest.mark.parametrize("solver", [Solver.GCO_EXPANSION, Solver.GCO_SWAP], ids=str)
def test_forbidden_labels_reach_gco_as_the_stand_in_and_none_is_returned(
    solver: Solver,
) -> None:
    # #1139 through #1274's rule: `-inf` in the field, the stand-in to gco.
    def check(graph: PottsGraph, finite: np.ndarray, allowed: np.ndarray) -> None:
        n_states = finite.shape[1]
        field = forbid(finite, allowed)
        ours = external.ground_state(
            graph, field, solver, ONE_CALL, np.random.default_rng(0)
        )
        theirs = gco.alpha_expansion(
            graph, stand_in(graph, finite, allowed), n_states, move=_moves(solver)
        )
        assert np.array_equal(ours.labelling, theirs.labelling)
        assert allowed[np.arange(graph.n_nodes), ours.labelling].all()
        assert ours.energy == energy(graph, field, ours.labelling)
        assert np.isfinite(ours.energy)

    rows = [(GRAPH, FINITE, ALLOWED)]
    rows += [(SMALL, *_small_problem(seed, 3)) for seed in range(6)]
    side = lattice_graph((16, 16), BoundaryCondition.OPEN, critical_coupling(10))
    rng = np.random.default_rng(1139)
    finite = rng.normal(size=(side.n_nodes, 10))
    allowed = rng.random((side.n_nodes, 10)) < 2 / 3
    allowed[np.arange(side.n_nodes), rng.integers(10, size=side.n_nodes)] = True
    rows.append((side, finite, allowed))
    every_row(rows, check)


@pytest.mark.oracle
def test_forbidden_expansion_reaches_the_enumerated_constrained_minimum() -> None:
    # The 3x3 q = 3 instance and the 2x4 ones at q = 2 and 3, as #1274 pinned.
    def check(graph: PottsGraph, finite: np.ndarray, allowed: np.ndarray) -> None:
        n_states = finite.shape[1]
        ours = external.ground_state(
            graph,
            forbid(finite, allowed),
            Solver.GCO_EXPANSION,
            ONE_CALL,
            np.random.default_rng(0),
        )
        every = configurations(n_states, graph.n_nodes)
        keep = allowed[np.arange(graph.n_nodes), every].all(axis=1)
        minimum = float(energies(graph, finite, every[keep]).min())
        scale = ENERGY_AGREEMENT * max(1.0, abs(minimum))
        assert abs(ours.energy - minimum) <= scale, (ours.energy, minimum)

    rows = [(GRAPH, FINITE, ALLOWED)]
    rows += [
        (SMALL, *_small_problem(seed, n_states))
        for n_states, seed in product((2, 3), range(6))
    ]
    every_row(rows, check)


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize("solver", [Solver.GCO_EXPANSION, Solver.GCO_SWAP], ids=str)
def test_a_session_serves_the_one_shot_ground_state(solver: Solver) -> None:
    # One worker, three fields: each labelling is the one-shot call's.
    problems = [_potts(16, 3, seed) for seed in (974, 975, 976)]
    rng = np.random.default_rng(0)
    with external.session(solver) as opened:
        for graph, field in problems:
            served = external.ground_state(
                graph, field, solver, ONE_CALL, rng, session=opened
            )
            once = external.ground_state(graph, field, solver, ONE_CALL, rng)
            assert np.array_equal(served.labelling, once.labelling)
            assert served.energy == once.energy


def _gco(graph: PottsGraph, field: np.ndarray, n_states: int) -> np.ndarray:
    return gco.alpha_expansion(graph, field, n_states=n_states).labelling


@pytest.mark.oracle
def test_with_forbidden_labels_both_expansions_reach_the_constrained_minimum() -> None:
    # Issue #1274, on #1139's forbidden labels: gco cuts the finite stand-in
    # `_forbidden.stand_in` states, the package the `-inf` field. Measured
    # 2026-10-06: all 13 instances reach the enumerated minimum, both.
    every_row(enumerable(), partial(check_expansion, theirs=_gco, exact=True))


@pytest.mark.experiment
def test_with_forbidden_labels_gcos_labelling_is_a_fixed_point_of_the_package_move() -> (
    None
):
    # 16² and 71² at 3 and 10 states. Measured 2026-10-06: gco's energy from
    # 0.82% below the package's (16², q = 10) to 0.27% above (71², q = 10).
    every_row(lattices(), partial(check_expansion, theirs=_gco, exact=False))
