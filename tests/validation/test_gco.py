"""The package's alpha expansion beside gco's, run in a subprocess (issue #974).

gco (Veksler and Delong's C++) is not an oracle for the labelling: expansion
stops at a local minimum, and the two start differently and cut on different
costs. What a correct expansion must satisfy whatever it started from is
checked instead:

- a labelling gco returns is a fixed point of the package's own expansion
  move --- a full cycle from it makes no move and leaves the energy bitwise
  unchanged --- at 16² and 71², q = 3 and 10;
- on a 4x4 lattice at q = 3, both labellings are within the factor-2 bound
  Boykov, Veksler and Zabih prove for Potts, against the minimum over all
  3^16 = 43,046,721 labellings, the energy written with non-negative terms;
  on this instance both reach the minimum, within 1e-12;
- at 71² the two energies agree within 1 per cent, the spread #938's spike
  measured at 0.13 per cent at q = 10.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.enumeration import configurations
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling, energies, energy
from snakes_and_ladders.validation import gco
from snakes_and_ladders.validation.runner import available

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(not available("gco"), reason="gco is the validation-gco extra"),
]


def _potts(side: int, n_states: int, seed: int) -> tuple[PottsGraph, np.ndarray]:
    """An open lattice at its critical coupling under a standard normal field."""
    graph = lattice_graph(
        (side, side), BoundaryCondition.OPEN, critical_coupling(n_states)
    )
    field = np.random.default_rng(seed).normal(size=(graph.n_nodes, n_states))
    return graph, field


def _non_negative(graph: PottsGraph, field: np.ndarray, value: float) -> float:
    """``value`` with the constant removed that makes every term non-negative.

    ``E = sum_i D_i + sum J [s_i != s_j] - sum J`` with ``D_i = -h_i[s_i]``;
    adding ``sum J`` and each row's ``max h`` leaves the data costs shifted to
    their minimum of zero, which is the form the factor-2 bound is proved on.
    """
    return value + float(graph.edge_coupling.sum()) + float(field.max(axis=1).sum())


@pytest.mark.experiment
@pytest.mark.parametrize("n_states", [3, 10])
@pytest.mark.parametrize("side", [16, 71])
def test_gcos_labelling_is_a_fixed_point_of_the_package_move(
    side: int, n_states: int
) -> None:
    graph, field = _potts(side, n_states, 974)
    theirs = gco.alpha_expansion(graph, field, n_states)
    from_theirs = alpha_expansion(
        graph, field, n_states, start=theirs.labelling, backend=Backend.RUST
    )
    assert from_theirs.moves == 0
    assert np.array_equal(from_theirs.labelling, theirs.labelling)
    assert from_theirs.energy == theirs.energy


@pytest.mark.oracle
def test_both_expansions_are_within_the_factor_two_bound_of_the_optimum() -> None:
    graph, field = _potts(4, 3, 974)
    # 3^16 in 81 blocks of 3^12, each block's tail enumerated once.
    rest = configurations(3, 12, limit=3**12)
    best = np.inf
    for head in configurations(3, 4):
        block = np.hstack([np.broadcast_to(head, (rest.shape[0], 4)), rest])
        best = min(best, float(energies(graph, field, block).min()))
    optimum = _non_negative(graph, field, best)
    ours = alpha_expansion(graph, field, 3, backend=Backend.RUST)
    theirs = gco.alpha_expansion(graph, field, 3)
    for found in (ours.energy, theirs.energy):
        shifted = _non_negative(graph, field, found)
        assert optimum <= shifted * (1.0 + 1e-12) <= 2.0 * optimum
        # Both reach the optimum on this instance, 7e-15 apart in the sum.
        assert found == pytest.approx(best, rel=1e-12)
    assert theirs.energy == energy(graph, field, theirs.labelling)
    assert optimum > 0.0


@pytest.mark.experiment
@pytest.mark.parametrize("n_states", [3, 10])
def test_the_two_energies_agree_within_one_per_cent_at_71(n_states: int) -> None:
    graph, field = _potts(71, n_states, 974)
    ours = alpha_expansion(graph, field, n_states, backend=Backend.RUST)
    theirs = gco.alpha_expansion(graph, field, n_states)
    assert theirs.energy == pytest.approx(ours.energy, rel=1e-2)
