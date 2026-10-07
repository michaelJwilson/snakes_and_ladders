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
the minimum. `external.potts.lower_bound` (#1282, step 4) returns the
adapter's value as its bound and the adapter's labelling, bitwise, on every
instance here but `potts_lattice/release`; every other solve here is
`external.potts.lower_bound`, served by one session per module (step 7). The
adapter stays for that pin and for `tests/regression/sandbox/test_potts_mip.py`.
With integer node marginals, `lower_bound(..., integral=True)` solves the ILP
with HiGHS's `milp` (#1274): TRW-S's bound is at most the LP value, which is
at most the ILP optimum, the enumerated minimum where enumeration reaches;
TRW-S's labelling scores at least that optimum, and above it only on two
frustrated lattices.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from sal.external import Session, Solver, potts, session
from sal.external.potts import ExternalBound
from sal.search.potts_starts import rung_of, spatio_rung, tiling_rung
from sal.search.tightening import dual_bound
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.graph import PottsGraph, lattice_graph
from sal.sim.potts import energies, energy
from sal.validation import highs

from tests._rows import every_row
from tests.regression.search.test_trws import (
    FRUSTRATED,
    STALLED,
    _instances,
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


@pytest.fixture(scope="module")
def lp() -> Iterator[Session]:
    """One HiGHS worker for the module's solves, so each skips interpreter and import start-up."""
    with session(Solver.HIGHS_LP) as opened:
        yield opened


def _solved(
    graph: PottsGraph,
    field: np.ndarray,
    served: Session,
    *,
    integral: bool = False,
    timeout: float = 600.0,
) -> ExternalBound:
    """HiGHS's optimum through ``served``, refused unless HiGHS reports one.

    ``bound`` is the LP's value and ``labelling`` each site's largest node
    marginal; with ``integral`` the ILP's (#1274).
    """
    result = potts.lower_bound(
        graph,
        field,
        Solver.HIGHS_LP,
        integral=integral,
        timeout=timeout,
        session=served,
    )
    # A run HiGHS stopped short of its optimum is a termination on its limit.
    assert result.termination.converged, result.termination
    return result


@pytest.mark.oracle
def test_external_lower_bound_is_the_adapters_value_bitwise(lp: Session) -> None:
    # #1282, step 4: both pose the LP through `polytope_inputs`, so HiGHS
    # receives one set of bytes; `potts_lattice/release` is left to the
    # release-sized tests below.
    rows = _instances() + [
        row for row in _fixtures() if row[0] != "potts_lattice/release"
    ]

    def check(name: str, graph: PottsGraph, field: np.ndarray, _n_states: int) -> None:
        adapter = highs.local_polytope(graph, field)
        assert adapter.status == highs.OPTIMAL, adapter.message
        ours = _solved(graph, field, lp)

        assert ours.bound == adapter.value, (name, ours.bound, adapter.value)
        assert np.array_equal(ours.labelling, adapter.labelling), name
        assert ours.integral is adapter.integral, name
        assert ours.termination.iterations == adapter.iterations, name

    every_row(rows, check)


@pytest.mark.oracle
def test_where_trws_converges_its_bound_is_the_lp_value(lp: Session) -> None:
    # Two routes to one number: a primal simplex on the explicit LP and a
    # dual coordinate ascent. `dual_bound` is a third, skipped at
    # `potts_lattice/release`, where 5,000 sweeps take 215 s.
    rows = [row for row in _instances() if row[0] not in STALLED] + _fixtures()

    def check(name: str, graph: PottsGraph, field: np.ndarray, _n_states: int) -> None:
        value = _solved(graph, field, lp).bound
        result = trws(graph, field)
        scale = LP_AGREEMENT * abs(value)

        assert result.termination.converged, name
        assert abs(result.bound - value) <= scale, (name, result.bound, value)
        if graph.n_nodes <= 144:
            pairwise = dual_bound(graph, field, max_iterations=5000, plaquettes=())
            assert abs(pairwise.bound - value) <= scale, (name, pairwise.bound, value)

    every_row(rows, check)


@pytest.mark.oracle
@pytest.mark.warning
def test_where_trws_stalls_both_ascents_are_below_the_lp(lp: Session) -> None:
    # Coordinate ascent stops where no block move raises the dual, which need
    # not be its maximum (Kolmogorov 2006, weak tree agreement). The LP
    # value is pinned, and it is at most the enumerated minimum.
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        value = _solved(graph, field, lp).bound
        result = trws(graph, field)
        pairwise = dual_bound(graph, field, max_iterations=5000, plaquettes=())

        assert value == pytest.approx(STALLED_LP[name], rel=LP_AGREEMENT), name
        assert result.termination.converged, name
        assert result.bound < pairwise.bound < value - 5e-5, name
        assert value <= _optimum(graph, field, n_states), name

    every_row([row for row in _instances() if row[0] in STALLED], check)


@pytest.mark.oracle
def test_where_the_primal_is_integral_the_lp_is_the_minimum(lp: Session) -> None:
    # Integral node marginals fix every edge table, so the LP value is the
    # energy of the labelling they select; the LP bounds the minimum from
    # below, so that labelling is optimal. Enumeration checks it where it
    # reaches. The frustrated lattices are the fractional ones, and there
    # the LP is strictly below the minimum.
    fractional = []

    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        solved = _solved(graph, field, lp)
        scale = ENERGY_AGREEMENT * max(1.0, abs(solved.bound))
        enumerable = n_states**graph.n_nodes <= ENUMERABLE
        optimum = _optimum(graph, field, n_states) if enumerable else None
        if not solved.integral:
            fractional.append(name)
            assert optimum is not None, name
            assert solved.bound < optimum - 1e-3, name
            return
        attained = energy(graph, field, solved.labelling)

        assert abs(attained - solved.bound) <= scale, (name, attained, solved.bound)
        if optimum is not None:
            assert abs(optimum - solved.bound) <= scale, (name, optimum, solved.bound)

    every_row(_instances() + _fixtures(), check)

    assert sorted(fractional) == sorted(FRUSTRATED)


#: `spatio_only/release`'s ten-state optimum, the two-state graph cut's
#: energy (`RELEASE_OPTIMUM` in `test_trws.py`, issue #1041).
SPATIO_OPTIMUM = -10454.1562900565

#: HiGHS's optimum on `spatio_tiling/release`, 2026-09-25: 268 s and 2.1 GB
#: peak, fractional at 95 of 5,041 sites. TRW-S converges 0.0496 below it.
TILING_LP = -17022.934121437305


@pytest.mark.release
@pytest.mark.oracle
def test_at_release_size_the_lp_certifies_one_optimum_and_places_the_tiling_gap(
    lp: Session,
) -> None:
    # 5,041 sites at ten states: 50,410 node and 1,484,000 edge columns,
    # 344 s and 268 s on the reference host, 2.1 GB peak each. On
    # `spatio_only/release` the primal is integral and meets the graph cut's
    # optimum and TRW-S's bound. On `spatio_tiling/release` it is
    # fractional, so of the 0.81 between TRW-S's bound and the best labelling
    # (#1061) 0.05 is TRW-S stopping short and the rest is the relaxation.
    spatio = spatio_rung(fixture("spatio_only", "release").params, "release")
    solved = _solved(spatio.graph, spatio.field, lp, timeout=1800.0)
    scale = LP_AGREEMENT * abs(SPATIO_OPTIMUM)

    assert solved.integral
    assert abs(solved.bound - SPATIO_OPTIMUM) <= scale
    assert (
        abs(energy(spatio.graph, spatio.field, solved.labelling) - SPATIO_OPTIMUM)
        <= scale
    )
    assert abs(trws(spatio.graph, spatio.field).bound - solved.bound) <= scale

    tiling = tiling_rung(fixture("spatio_tiling", "release").params, "release")
    loose = _solved(tiling.graph, tiling.field, lp, timeout=1800.0)
    result = trws(tiling.graph, tiling.field)

    assert not loose.integral
    assert loose.bound == pytest.approx(TILING_LP, rel=LP_AGREEMENT)
    assert result.termination.converged
    assert result.bound < loose.bound - 0.04
    assert loose.bound < result.energy


#: The lattices where TRW-S's labelling scores above the ILP optimum,
#: 2026-10-06: by 1.4732 and 0.0388. On every other instance it is optimal.
GAPPED = ("triangular-0", "triangular-1")


@pytest.mark.oracle
def test_the_bound_the_lp_and_the_ilp_are_ordered_and_the_labelling_is_above(
    lp: Session,
) -> None:
    # Issue #1274 (a)-(c): bound <= LP <= ILP = enumerated minimum, and the
    # labelling TRW-S decodes, rescored by `energies`, at least the ILP
    # optimum. Rows: `test_trws.py`'s nine and `potts_lattice/stress`, 144
    # sites, the largest under the per-PR cap. `potts_lattice/release`, 2,500
    # sites, closes at the root node in 2.0 s with the labelling optimal.
    rows = _instances() + [
        row for row in _fixtures() if row[0] == "potts_lattice/stress"
    ]
    gapped = []

    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        # `_solved` refuses any run HiGHS did not report optimal.
        relaxed = _solved(graph, field, lp).bound
        ilp = _solved(graph, field, lp, integral=True)
        result = trws(graph, field)
        decoded = float(energies(graph, field, result.labelling[None])[0])
        scale = ENERGY_AGREEMENT * max(1.0, abs(ilp.bound))

        assert result.bound <= relaxed + LP_AGREEMENT * abs(relaxed), name
        assert relaxed <= ilp.bound + scale, (name, relaxed, ilp.bound)
        assert abs(energy(graph, field, ilp.labelling) - ilp.bound) <= scale, name
        if n_states**graph.n_nodes <= ENUMERABLE:
            optimum = _optimum(graph, field, n_states)
            assert abs(ilp.bound - optimum) <= scale, (name, ilp.bound, optimum)
        assert decoded >= ilp.bound - scale, (name, decoded, ilp.bound)
        if decoded > ilp.bound + scale:
            gapped.append(name)

    every_row(rows, check)

    assert sorted(gapped) == sorted(GAPPED)
