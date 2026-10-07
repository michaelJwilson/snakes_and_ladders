"""OpenGM's seven algorithms against enumeration, HiGHS, gco and the package's own (issue #1279).

OpenGM (Andres, Beier and Kappes; MIT) is built from source by
`infra/build_opengm.sh` and reached through `sal.external.potts`, one session
per solver for the module. Its inference shares no code with the package's.
The shared instances are `test_trws.py`'s nine enumerable lattices (square,
strip and frustrated triangular, couplings of either sign), #1274's
forbidden-label lattices, and open critical lattices at 16² and 32².

- **A\\*** is exact: its energy is the enumerated minimum to 1e-12 relative on
  every enumerable instance, forbidden labels included.
- **TRW-S** (OpenGM's `TRWSi`, Savchynskyy's implementation): every bound on
  its trace is at most the enumerated minimum, the trace does not fall, and
  where `search.trws` converges to the LP its limit is TRW-S's to 1e-8
  relative. The two decompose into different chains, so an iteration of one
  is not an iteration of the other: after equal iterations OpenGM's bound sat
  between 0.048 below and 0.017 above the package's on the nine instances. On the two
  frustrated lattices where both stall below the LP, the limits differ by
  0.0017 and 0.0166, each below HiGHS's value.
- **Dual decomposition** (subgradient, 100 steps): its bound is at most
  HiGHS's LP value, the best a pairwise decomposition reaches.
- **ICM** from a given start is `search.icm`'s index-order descent bitwise.
- **Loopy BP** on a chain is exact: its energy is the enumerated minimum.
- **Swap**: at q = 3 its labelling is a fixed point of
  `search.alpha_expansion`'s swap, and gco's where both reach the enumerated
  minimum. Its loop stops after q (q - 1) / 2 unproductive moves in total,
  not in a row, so at q = 10 it stops early: restarted from its own labelling
  it lowers the energy by 3.517 at 32² (pinned as a `bug`, OpenGM's).
- **Expansion** reaches the minimum where enumeration reaches; past it,
  OpenGM's auxiliary-node move under-charges a pair that keeps two labels
  other than alpha, so its labelling is not an expansion fixed point: the
  package's expansion from it lowers the energy by 1.351 at 16² (pinned as a
  `bug`, OpenGM's).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

import numpy as np
import pytest
from sal import external
from sal.cost import Cost
from sal.external import Session, Solver, potts
from sal.external.potts import ExternalBound, ExternalRun
from sal.opt.budget import Budget
from sal.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from sal.search.icm import iterated_conditional_modes
from sal.search.trws import trws
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, energy, forbid
from sal.validation import opengm

from tests._frameworks import requires
from tests._rows import every_row
from tests.regression.search.test_trws import STALLED, _instances, _optimum
from tests.validation._forbidden import constrained_minimum, enumerable, lattices

pytestmark = [pytest.mark.validation, requires("opengm")]

#: One call of OpenGM, the unit `external.potts.UNITS` charges it in.
ONE_CALL = Budget(Cost.FITS, 1)

#: Agreement of two energies of one labelling, or of an exact solver and
#: enumeration: sums over the same terms in other orders.
ENERGY_AGREEMENT = 1e-12

#: Agreement of two converged ascents of the same LP's dual: #1063's.
LP_AGREEMENT = 1e-8

#: HiGHS's LP optimum on the two stalled lattices, `test_highs.py`'s.
STALLED_LP = {"triangular-0": -1.1491051224864237, "triangular-2": -1.4002413745560185}

#: The energy the package's expansion takes off OpenGM's at 16², measured.
EXPANSION_DEFECT = 1.3507867890207308

#: The energy OpenGM's swap takes off its own labelling at 32², q = 10, when
#: restarted from it, measured.
SWAP_DEFECT = 3.516503096705999

GROUND = (
    Solver.OPENGM_ICM,
    Solver.OPENGM_LBP,
    Solver.OPENGM_ASTAR,
    Solver.OPENGM_EXPANSION,
    Solver.OPENGM_SWAP,
)


@pytest.fixture(scope="module")
def sessions() -> Iterator[dict[Solver, Session]]:
    """One worker per OpenGM solver, and HiGHS's, for the module's calls."""
    with ExitStack() as stack:
        yield {
            solver: stack.enter_context(external.session(solver))
            for solver in (
                *GROUND,
                Solver.OPENGM_TRWS,
                Solver.OPENGM_DD,
                Solver.HIGHS_LP,
            )
        }


def _critical(
    side: int | tuple[int, int], n_states: int, seed: int
) -> tuple[PottsGraph, np.ndarray]:
    """An open lattice, square where ``side`` is one number, at its critical coupling under a standard normal field."""
    shape = (side, side) if isinstance(side, int) else side
    graph = lattice_graph(shape, BoundaryCondition.OPEN, critical_coupling(n_states))
    field = np.random.default_rng(seed).normal(size=(graph.n_nodes, n_states))
    return graph, field


def _ground(
    graph: PottsGraph,
    field: np.ndarray,
    solver: Solver,
    sessions: dict[Solver, Session],
    *,
    start: np.ndarray | None = None,
) -> ExternalRun:
    # OpenGM is deterministic: the generator is the sibling's argument, never drawn.
    return potts.ground_state(
        graph,
        field,
        solver,
        ONE_CALL,
        np.random.default_rng(0),
        start=start,
        session=sessions[solver],
    )


def _at_most(value: float, limit: float) -> bool:
    """``value <= limit`` up to :data:`ENERGY_AGREEMENT`, relative to ``limit``."""
    return value <= limit + ENERGY_AGREEMENT * max(1.0, abs(limit))


def _bound(
    graph: PottsGraph,
    field: np.ndarray,
    solver: Solver,
    sessions: dict[Solver, Session],
) -> ExternalBound:
    return potts.lower_bound(graph, field, solver, session=sessions[solver])


@pytest.mark.oracle
def test_astar_is_the_enumerated_minimum(sessions: dict[Solver, Session]) -> None:
    def check(_name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        found = _ground(graph, field, Solver.OPENGM_ASTAR, sessions)
        optimum = _optimum(graph, field, n_states)
        assert found.energy == pytest.approx(optimum, rel=ENERGY_AGREEMENT, abs=0)
        assert found.energy == energy(graph, field, found.labelling)

    every_row(_instances(), check)


@pytest.mark.oracle
def test_astar_is_the_constrained_minimum_with_forbidden_labels(
    sessions: dict[Solver, Session],
) -> None:
    def check(
        _name: str, graph: PottsGraph, finite: np.ndarray, allowed: np.ndarray
    ) -> None:
        found = _ground(graph, forbid(finite, allowed), Solver.OPENGM_ASTAR, sessions)
        minimum = constrained_minimum(graph, finite, allowed)
        assert allowed[np.arange(graph.n_nodes), found.labelling].all()
        assert found.energy == pytest.approx(minimum, rel=ENERGY_AGREEMENT, abs=0)

    every_row(enumerable(), check)


@pytest.mark.oracle
def test_trws_bounds_the_minimum_every_iteration_and_meets_the_package_limit(
    sessions: dict[Solver, Session],
) -> None:
    def check(name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        optimum = _optimum(graph, field, n_states)
        # OpenGM reports the bound after each pass; two passes, forward and
        # backward, are one of the package's iterations.
        trace = opengm.bound_trace(
            graph, field, "trws", max_iterations=1000, tolerance=0.0
        )
        assert all(_at_most(value, optimum) for value in trace)
        assert np.all(np.diff(trace) >= -ENERGY_AGREEMENT * np.abs(trace[1:]))
        theirs = _bound(graph, field, Solver.OPENGM_TRWS, sessions)
        ours = trws(graph, field)
        assert theirs.termination.converged
        assert _at_most(theirs.bound, optimum)
        if name in STALLED:
            assert max(theirs.bound, ours.bound) < STALLED_LP[name]
        else:
            assert theirs.bound == pytest.approx(ours.bound, rel=LP_AGREEMENT)

    every_row(_instances(), check)


@pytest.mark.oracle
def test_trws_meets_the_package_limit_at_32(sessions: dict[Solver, Session]) -> None:
    graph, field = _critical(32, 3, 1279)
    theirs = _bound(graph, field, Solver.OPENGM_TRWS, sessions)
    ours = trws(graph, field)
    assert theirs.termination.converged
    assert ours.termination.converged
    # Measured 1.1e-15 relative apart.
    assert theirs.bound == pytest.approx(ours.bound, rel=LP_AGREEMENT)
    assert _at_most(theirs.bound, theirs.energy)


@pytest.mark.oracle
def test_dual_decomposition_bound_is_at_most_the_lp(
    sessions: dict[Solver, Session],
) -> None:
    rows = [*_instances(), ("16x16", *_critical(16, 3, 1279), 3)]

    def check(_name: str, graph: PottsGraph, field: np.ndarray, _n: int) -> None:
        theirs = _bound(graph, field, Solver.OPENGM_DD, sessions)
        lp = _bound(graph, field, Solver.HIGHS_LP, sessions)
        assert lp.termination.converged
        assert _at_most(theirs.bound, lp.bound)
        assert _at_most(theirs.bound, theirs.energy)

    every_row(rows, check)


@pytest.mark.oracle
def test_icm_is_the_package_descent_from_the_same_start(
    sessions: dict[Solver, Session],
) -> None:
    def check(side: int, n_states: int) -> None:
        graph, field = _critical(side, n_states, 1279)
        start = np.random.default_rng(5).integers(n_states, size=graph.n_nodes)
        theirs = _ground(graph, field, Solver.OPENGM_ICM, sessions, start=start)
        ours = iterated_conditional_modes(
            graph, field, np.random.default_rng(0), n_states=n_states, start=start
        )
        assert np.array_equal(theirs.labelling, ours.labelling)
        assert theirs.energy == energy(graph, field, ours.labelling)

    every_row([(8, 3), (32, 3), (32, 10)], check)


@pytest.mark.oracle
def test_loopy_bp_is_exact_on_a_chain(sessions: dict[Solver, Session]) -> None:
    def check(seed: int) -> None:
        rng = np.random.default_rng(1279 + seed)
        n_nodes = 11
        graph = PottsGraph(
            n_nodes,
            tuple((site, site + 1) for site in range(n_nodes - 1)),
            tuple(rng.normal(size=n_nodes - 1).tolist()),
        )
        field = rng.normal(size=(n_nodes, 3))
        found = _ground(graph, field, Solver.OPENGM_LBP, sessions)
        assert found.termination.converged
        optimum = _optimum(graph, field, 3)
        assert found.energy == pytest.approx(optimum, rel=ENERGY_AGREEMENT, abs=0)

    every_row([(seed,) for seed in range(3)], check)


@pytest.mark.oracle
def test_swap_is_a_fixed_point_of_the_package_swap(
    sessions: dict[Solver, Session],
) -> None:
    def check(side: int, n_states: int) -> None:
        graph, field = _critical(side, n_states, 1279)
        theirs = _ground(graph, field, Solver.OPENGM_SWAP, sessions)
        ours = alpha_beta_swap(graph, field, n_states=n_states, start=theirs.labelling)
        assert ours.moves == 0
        assert np.array_equal(ours.labelling, theirs.labelling)

    every_row([(16, 3), (32, 3)], check)


@pytest.mark.experiment
@pytest.mark.bug
def test_opengms_swap_stops_after_unproductive_moves_in_total(
    sessions: dict[Solver, Session],
) -> None:
    # OpenGM's swap counts moves that lower nothing and never resets the
    # count when one does, so it stops after q (q - 1) / 2 of them in total,
    # not in a row: 68 moves at q = 10, 23 of them productive. Restarted from
    # its own labelling it lowers the energy again. Fixed upstream, this fails.
    graph, field = _critical(32, 10, 1279)
    theirs = _ground(graph, field, Solver.OPENGM_SWAP, sessions)
    again = _ground(graph, field, Solver.OPENGM_SWAP, sessions, start=theirs.labelling)
    ours = alpha_beta_swap(graph, field, n_states=10, start=theirs.labelling)
    assert ours.moves > 0
    assert again.energy < theirs.energy
    assert theirs.energy - again.energy == pytest.approx(
        SWAP_DEFECT, rel=ENERGY_AGREEMENT
    )


@pytest.mark.oracle
@requires("gco")
def test_both_moves_are_gcos_and_the_minimum_at_3x4(
    sessions: dict[Solver, Session],
) -> None:
    graph, field = _critical((3, 4), 3, 1279)
    optimum = _optimum(graph, field, 3)
    for solver, reference in (
        (Solver.OPENGM_EXPANSION, Solver.GCO_EXPANSION),
        (Solver.OPENGM_SWAP, Solver.GCO_SWAP),
    ):
        theirs = _ground(graph, field, solver, sessions)
        gco = potts.ground_state(
            graph, field, reference, ONE_CALL, np.random.default_rng(0)
        )
        assert np.array_equal(theirs.labelling, gco.labelling)
        assert theirs.energy == pytest.approx(optimum, rel=ENERGY_AGREEMENT, abs=0)


@pytest.mark.oracle
def test_expansion_reaches_the_minimum_where_enumeration_does(
    sessions: dict[Solver, Session],
) -> None:
    def check(_name: str, graph: PottsGraph, field: np.ndarray, n_states: int) -> None:
        found = _ground(graph, field, Solver.OPENGM_EXPANSION, sessions)
        optimum = _optimum(graph, field, n_states)
        assert found.energy == pytest.approx(optimum, rel=ENERGY_AGREEMENT, abs=0)

    # Expansion needs a metric pairwise term: the instances whose couplings
    # are all non-negative.
    rows = [row for row in _instances() if (row[1].edge_coupling >= 0).all()]
    rows.append(("3x4", *_critical((3, 4), 3, 1279), 3))
    every_row(rows, check)


@pytest.mark.experiment
@pytest.mark.bug
def test_opengms_expansion_is_not_an_expansion_fixed_point(
    sessions: dict[Solver, Session],
) -> None:
    # OpenGM's auxiliary node for a pair holding two labels other than alpha
    # charges V(a, a) = 0 where the pair keeps both, not V(b, c): a move that
    # would cut such a boundary is under-valued. Fixed upstream, this fails.
    graph, field = _critical(16, 3, 1279)
    theirs = _ground(graph, field, Solver.OPENGM_EXPANSION, sessions)
    ours = alpha_expansion(graph, field, n_states=3, start=theirs.labelling)
    assert ours.moves > 0
    assert theirs.energy - ours.energy == pytest.approx(
        EXPANSION_DEFECT, rel=ENERGY_AGREEMENT
    )


@pytest.mark.oracle
def test_no_solver_returns_a_forbidden_label(sessions: dict[Solver, Session]) -> None:
    def check(
        _name: str, graph: PottsGraph, finite: np.ndarray, allowed: np.ndarray
    ) -> None:
        field = forbid(finite, allowed)
        for solver in (Solver.OPENGM_ICM, Solver.OPENGM_EXPANSION, Solver.OPENGM_SWAP):
            found = _ground(graph, field, solver, sessions)
            assert allowed[np.arange(graph.n_nodes), found.labelling].all(), solver
            assert np.isfinite(found.energy)
        bound = _bound(graph, field, Solver.OPENGM_TRWS, sessions)
        assert allowed[np.arange(graph.n_nodes), bound.labelling].all()
        assert _at_most(bound.bound, bound.energy)

    every_row([row for row in lattices() if row[0].startswith("16^2")], check)


@pytest.mark.smoke
@pytest.mark.patch
def test_a_session_returns_the_one_shot_answer_bitwise(
    sessions: dict[Solver, Session],
) -> None:
    # A* is exponential in the sites: 4x4 takes 1.8 ms, 8x8 does not end.
    graph, field = _critical(4, 3, 1279)
    for solver in GROUND:
        once = potts.ground_state(
            graph, field, solver, ONE_CALL, np.random.default_rng(0)
        )
        served = _ground(graph, field, solver, sessions)
        assert np.array_equal(once.labelling, served.labelling), solver
        assert once.energy == served.energy, solver
        assert once.provenance == served.provenance == external.provenance(solver)
        assert once.provenance.osi
