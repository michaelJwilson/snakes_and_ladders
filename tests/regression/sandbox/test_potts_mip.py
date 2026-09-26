"""The conserved MIP (`sal.sandbox.potts_mip`) against enumeration and at the tiling rung (issue #1069).

Release only: the route is conserved, not adopted, so no per-PR job pays for
it. Against enumeration on the nine lattices of `test_trws.py` and a 4x4
two-state lattice it is the minimum, proven, the three frustrated ones
included; held at an optimum it returns that optimum on any free region.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.sandbox import potts_mip
from sal.search.potts_starts import tiling_rung
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy
from sal.validation import highs

from tests._rows import every_row
from tests.regression.search.test_trws import _instances, _mixed, _optimum

pytestmark = pytest.mark.release

#: Agreement of a value with a labelling's energy, both sums over the same
#: terms in different orders.
ENERGY_AGREEMENT = 1e-12

Row = tuple[str, PottsGraph, np.ndarray, int]


def _mip_instances() -> list[Row]:
    """`_instances`, and a 4x4 open lattice at two states with mixed couplings."""
    grid = _mixed(lattice_graph((4, 4), BoundaryCondition.OPEN, 1.0), 1069)
    field = np.random.default_rng(1069).normal(size=(16, 2))
    return [*_instances(), ("grid-4x4", grid, field, 2)]


@pytest.mark.oracle
def test_the_mip_is_the_enumerated_minimum_where_the_lp_is_loose_too() -> None:
    # Integer node marginals over the local polytope: the frustrated
    # triangular lattices, where the LP is more than 1e-3 below the minimum,
    # are solved exactly as the rest are, and HiGHS proves each (#1069).
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        solved = potts_mip.mip(graph, field, time_limit=60.0)
        optimum = _optimum(graph, field, n_states)
        scale = ENERGY_AGREEMENT * max(1.0, abs(optimum))

        assert solved.proven, (name, solved.message)
        assert abs(solved.value - optimum) <= scale, (name, solved.value, optimum)
        assert abs(solved.dual_bound - optimum) <= scale, name
        assert abs(energy(graph, field, solved.labelling) - optimum) <= scale, name

    every_row(_mip_instances(), check)


@pytest.mark.oracle
def test_a_restricted_mip_folds_its_fixed_sites_into_the_free_ones() -> None:
    # Fixed at an optimum, any free region returns it: the folded field and
    # the constant are the terms the fixed sites carry. Fixed at the LP's
    # labelling, the free region is its fractional sites and their rings,
    # and the value is the energy `sim.potts.energy` gives the labelling.
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        optimum = _optimum(graph, field, n_states)
        scale = ENERGY_AGREEMENT * max(1.0, abs(optimum))
        best = potts_mip.mip(graph, field, time_limit=60.0).labelling
        half = np.random.default_rng(graph.n_nodes).random(graph.n_nodes) < 0.5
        held = potts_mip.mip(graph, field, fixed=best, free=half, time_limit=60.0)

        assert abs(held.value - optimum) <= scale, name
        assert np.array_equal(held.labelling[~half], best[~half]), name

        relaxed = highs.local_polytope(graph, field)
        for rings in (0, 1):
            free = potts_mip.free_region(graph, relaxed.node_marginals, rings)
            local = potts_mip.mip(
                graph, field, fixed=relaxed.labelling, free=free, time_limit=60.0
            )
            attained = energy(graph, field, local.labelling)

            assert local.proven, name
            assert abs(local.value - attained) <= scale, (name, rings)
            assert attained >= optimum - scale, (name, rings)
            if relaxed.integral:
                assert not free.any(), name

    every_row(_mip_instances(), check)


@pytest.mark.analytic
def test_the_free_region_grows_by_lattice_rings() -> None:
    # One fractional corner of an open 4x4 lattice: itself, then its two
    # neighbours, then the three sites two steps away.
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 1.0)
    marginals = np.zeros((16, 2))
    marginals[:, 0] = 1.0
    marginals[0] = (0.5, 0.5)

    assert [
        int(potts_mip.free_region(graph, marginals, r).sum()) for r in range(3)
    ] == [
        1,
        3,
        6,
    ]


#: The restricted MIP's energy on `spatio_tiling/release`, 2026-09-25: the
#: same at 0 to 4 rings around the LP's 95 fractional sites (95 to 703 free
#: sites, 4.6 s to 46.7 s), 0.02 below the best labelling before it (#1050).
TILING_MIP = -17022.1988


@pytest.mark.skip(
    reason="~5 min: the 268 s LP at 5,041 x 10 then three restricted MIPs; "
    "run by hand, its result is recorded in STATUS.md (#1069)"
)
@pytest.mark.oracle
def test_at_release_size_the_restricted_mip_does_not_move_with_its_rings() -> None:
    # The LP's labelling held at its integral sites and a MIP over the rest
    # (#1069). The value does not move from 0 to 2 rings, so the fixed
    # sites do not bind there; it is a labelling's energy, so it is at least
    # the LP value.
    tiling = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    relaxed = highs.local_polytope(tiling.graph, tiling.field, timeout=1800.0)

    assert relaxed.status == highs.OPTIMAL, relaxed.message
    for rings in (0, 1, 2):
        free = potts_mip.free_region(tiling.graph, relaxed.node_marginals, rings)
        local = potts_mip.mip(
            tiling.graph, tiling.field, fixed=relaxed.labelling, free=free
        )
        attained = energy(tiling.graph, tiling.field, local.labelling)

        assert local.proven, (rings, local.message)
        assert attained == pytest.approx(TILING_MIP, abs=1e-9), rings
        assert abs(local.value - attained) <= ENERGY_AGREEMENT * abs(attained)
        assert relaxed.value < attained
