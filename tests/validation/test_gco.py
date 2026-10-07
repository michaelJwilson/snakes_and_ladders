"""The package's alpha expansion beside gco's, run in a subprocess (issue #974).

gco (Veksler and Delong) is not an oracle for the labelling: expansion stops
at a local minimum. Checked: gco's labelling is a fixed point of our move
(energy bitwise unchanged) at 16² and 71², q = 3 and 10; on a 4x4 lattice at
q = 3 both are within Boykov, Veksler and Zabih's factor 2 of the minimum over
3^16 = 43,046,721 labellings (non-negative terms), and both reach it within
1e-12; at 71² the energies agree within 1% (#938 measured 0.13% at q = 10).
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
from sal.backend import Backend
from sal.enumeration import configurations
from sal.search.alpha_expansion import alpha_expansion
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, energies, energy
from sal.validation import gco

from tests._frameworks import requires
from tests._rows import every_row, every_value
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
