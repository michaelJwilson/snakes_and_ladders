"""The TRW-S bound against HiGHS's optimum of the explicit local-polytope LP (issue #1063).

HiGHS (`linprog(method="highs")`) solves the primal LP with every node and
edge marginal written out; `search.trws` and `search.tightening.dual_bound`
ascend its dual by coordinate moves and share no code with it. Where TRW-S
converges its bound is the LP value to 1e-8 relative, and so is `dual_bound`'s
with no plaquettes: the six square and strip lattices of `test_trws.py`, one
frustrated triangular lattice, `potts_lattice/{ci,stress,release}`,
`spatio_only/{ci,stress}` and `spatio_tiling/ci`. On the other two frustrated
lattices both ascents stop below the LP, which is pinned. The primal's node
marginals are integral on every instance but the three frustrated ones, and
there the LP value is the labelling's energy and, where enumeration reaches,
the minimum.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.search.potts_starts import rung_of, spatio_rung, tiling_rung
from sal.search.tightening import dual_bound
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy
from sal.validation import highs

from tests._rows import every_row
from tests.regression.search.test_trws import (
    FRUSTRATED,
    STALLED,
    _instances,
    _mixed,
    _optimum,
)

pytestmark = pytest.mark.validation

#: Agreement asked of the LP value and a converged dual ascent: #1063's plan.
LP_AGREEMENT = 1e-8

#: Agreement of the LP value with a labelling's energy, both sums over the
#: same terms in different orders.
ENERGY_AGREEMENT = 1e-12

#: HiGHS's optimum on the two lattices where both ascents stall: TRW-S
#: converges 0.0160 and 0.0271 below it, `dual_bound` 7.4e-5 and 0.0204.
STALLED_LP = {"triangular-0": -1.1491051224864237, "triangular-2": -1.4002413745560185}

#: Largest state space enumeration is asked to walk.
ENUMERABLE = 600_000

Row = tuple[str, PottsGraph, np.ndarray, int]


def _fixtures() -> list[Row]:
    """The declared lattices up to `potts_lattice/release`, 2,500 sites at three states."""
    rows: list[Row] = []
    for tier in ("ci", "stress"):
        params = fixture("potts_lattice", tier).params
        graph = lattice_graph(params.shape, params.boundary, params.coupling)
        rows.append((f"potts_lattice/{tier}", graph, params.field, params.n_states))
    release = rung_of(fixture("potts_lattice", "release").params, "release")
    rows.append(
        ("potts_lattice/release", release.graph, release.field, release.n_states)
    )
    for tier in ("ci", "stress"):
        rung = spatio_rung(fixture("spatio_only", tier).params, tier)
        rows.append((f"spatio_only/{tier}", rung.graph, rung.field, rung.n_states))
    tiling = tiling_rung(fixture("spatio_tiling", "ci").params, "ci")
    rows.append(("spatio_tiling/ci", tiling.graph, tiling.field, tiling.n_states))
    return rows


def _solved(graph: PottsGraph, field: np.ndarray) -> highs.LocalPolytope:
    """HiGHS's optimum, refused unless `linprog` reports one."""
    result = highs.local_polytope(graph, field)
    assert result.status == highs.OPTIMAL, result.message
    return result


@pytest.mark.oracle
def test_where_trws_converges_its_bound_is_the_lp_value() -> None:
    # Two routes to one number: a primal simplex on the explicit LP and a
    # dual coordinate ascent. `dual_bound` is a third, skipped at
    # `potts_lattice/release`, where 5,000 sweeps take 215 s.
    rows = [row for row in _instances() if row[0] not in STALLED] + _fixtures()

    def check(name: str, graph: PottsGraph, field: np.ndarray, _n_states: int) -> None:
        value = _solved(graph, field).value
        result = trws(graph, field)
        scale = LP_AGREEMENT * abs(value)

        assert result.termination.converged, name
        assert abs(result.bound - value) <= scale, (name, result.bound, value)
        if graph.n_nodes <= 144:
            pairwise = dual_bound(graph, field, iterations=5000, plaquettes=())
            assert abs(pairwise.bound - value) <= scale, (name, pairwise.bound, value)

    every_row(rows, check)


@pytest.mark.oracle
@pytest.mark.warning
def test_where_trws_stalls_both_ascents_are_below_the_lp() -> None:
    # Coordinate ascent stops where no block move raises the dual, which need
    # not be its maximum (Kolmogorov 2006, weak tree agreement). The LP
    # value is pinned, and it is at most the enumerated minimum.
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        value = _solved(graph, field).value
        result = trws(graph, field)
        pairwise = dual_bound(graph, field, iterations=5000, plaquettes=())

        assert value == pytest.approx(STALLED_LP[name], rel=LP_AGREEMENT), name
        assert result.termination.converged, name
        assert result.bound < pairwise.bound < value - 5e-5, name
        assert value <= _optimum(graph, field, n_states), name

    every_row([row for row in _instances() if row[0] in STALLED], check)


@pytest.mark.oracle
def test_where_the_primal_is_integral_the_lp_is_the_minimum() -> None:
    # Integral node marginals fix every edge table, so the LP value is the
    # energy of the labelling they select; the LP bounds the minimum from
    # below, so that labelling is optimal. Enumeration checks it where it
    # reaches. The frustrated lattices are the fractional ones, and there
    # the LP is strictly below the minimum.
    fractional = []

    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        solved = _solved(graph, field)
        scale = ENERGY_AGREEMENT * max(1.0, abs(solved.value))
        enumerable = n_states**graph.n_nodes <= ENUMERABLE
        optimum = _optimum(graph, field, n_states) if enumerable else None
        if not solved.integral:
            fractional.append(name)
            assert optimum is not None, name
            assert solved.value < optimum - 1e-3, name
            return
        attained = energy(graph, field, solved.labelling)

        assert abs(attained - solved.value) <= scale, (name, attained, solved.value)
        if optimum is not None:
            assert abs(optimum - solved.value) <= scale, (name, optimum, solved.value)

    every_row(_instances() + _fixtures(), check)

    assert sorted(fractional) == sorted(FRUSTRATED)


#: `spatio_only/release`'s ten-state optimum, the two-state graph cut's
#: energy (`RELEASE_OPTIMUM` in `test_trws.py`, issue #1041).
SPATIO_OPTIMUM = -10454.1562900565

#: HiGHS's optimum on `spatio_tiling/release`, 2026-09-25: 268 s and 2.1 GB
#: peak, fractional at 95 of 5,041 sites. TRW-S converges 0.0496 below it.
TILING_LP = -17022.934121437305


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
        solved = highs.mip(graph, field, time_limit=60.0)
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
        best = highs.mip(graph, field, time_limit=60.0).labelling
        half = np.random.default_rng(graph.n_nodes).random(graph.n_nodes) < 0.5
        held = highs.mip(graph, field, fixed=best, free=half, time_limit=60.0)

        assert abs(held.value - optimum) <= scale, name
        assert np.array_equal(held.labelling[~half], best[~half]), name

        relaxed = _solved(graph, field)
        for rings in (0, 1):
            free = highs.free_region(graph, relaxed.node_marginals, rings)
            local = highs.mip(
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

    assert [int(highs.free_region(graph, marginals, r).sum()) for r in range(3)] == [
        1,
        3,
        6,
    ]


@pytest.mark.release
@pytest.mark.oracle
def test_at_release_size_the_lp_certifies_one_optimum_and_places_the_tiling_gap() -> (
    None
):
    # 5,041 sites at ten states: 50,410 node and 1,484,000 edge columns,
    # 344 s and 268 s on the reference host, 2.1 GB peak each. On
    # `spatio_only/release` the primal is integral and meets the graph cut's
    # optimum and TRW-S's bound. On `spatio_tiling/release` it is
    # fractional, so of the 0.81 between TRW-S's bound and the best labelling
    # (#1061) 0.05 is TRW-S stopping short and the rest is the relaxation.
    spatio = spatio_rung(fixture("spatio_only", "release").params, "release")
    solved = highs.local_polytope(spatio.graph, spatio.field, timeout=1800.0)
    scale = LP_AGREEMENT * abs(SPATIO_OPTIMUM)

    assert solved.status == highs.OPTIMAL, solved.message
    assert solved.integral
    assert abs(solved.value - SPATIO_OPTIMUM) <= scale
    assert (
        abs(energy(spatio.graph, spatio.field, solved.labelling) - SPATIO_OPTIMUM)
        <= scale
    )
    assert abs(trws(spatio.graph, spatio.field).bound - solved.value) <= scale

    tiling = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    loose = highs.local_polytope(tiling.graph, tiling.field, timeout=1800.0)
    result = trws(tiling.graph, tiling.field)

    assert loose.status == highs.OPTIMAL, loose.message
    assert not loose.integral
    assert loose.value == pytest.approx(TILING_LP, rel=LP_AGREEMENT)
    assert result.termination.converged
    assert result.bound < loose.value - 0.04
    assert loose.value < result.energy


#: The restricted MIP's energy on `spatio_tiling/release`, 2026-09-25: the
#: same at 0 to 4 rings around the LP's 95 fractional sites (95 to 703 free
#: sites, 4.6 s to 46.7 s), 0.02 below the best labelling before it (#1050).
TILING_MIP = -17022.1988


@pytest.mark.release
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
        free = highs.free_region(tiling.graph, relaxed.node_marginals, rings)
        local = highs.mip(
            tiling.graph, tiling.field, fixed=relaxed.labelling, free=free
        )
        attained = energy(tiling.graph, tiling.field, local.labelling)

        assert local.proven, (rings, local.message)
        assert attained == pytest.approx(TILING_MIP, abs=1e-9), rings
        assert abs(local.value - attained) <= ENERGY_AGREEMENT * abs(attained)
        assert relaxed.value < attained
