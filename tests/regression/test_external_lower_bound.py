"""`external.potts.lower_bound` takes `search.trws`'s arguments, refuses before it spawns, and orders its bounds (issue #1282, step 4).

Read against the sibling's signature, the `Capability` declarations, the
pre-#1282 bytes of the HiGHS adapter, TRW-S and enumeration. On
`test_trws.py`'s nine enumerable instances, TRW-S's bound <= the LP <= the
ILP = the enumerated minimum, and the LP is integral, its labelling attaining
the bound, on all but the three frustrated lattices; a forbidden label is
never chosen and the ILP is the allowed minimum. The bound on
`validation.highs`'s fixtures is pinned to the adapter's in
`tests/validation/test_highs.py`.
"""

from __future__ import annotations

import inspect
import subprocess
from typing import Any

import numpy as np
import pytest
from sal.cost import Cost
from sal.external import CapabilityRefused, Solver, potts, provenance, session
from sal.external.potts import BOUND_UNIT, ExternalBound
from sal.external.runner import run
from sal.search import trws as search
from sal.search.alpha_expansion import BoundedLabelling
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy, forbid

from tests._rows import every_row
from tests.regression.search.test_trws import FRUSTRATED, _instances, _optimum

GRAPH = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8)
FIELD = np.random.default_rng(1282).normal(size=(GRAPH.n_nodes, 3))

#: Agreement of two sums over the same terms in different orders: the LP or
#: ILP value against a labelling's energy, or the enumerated minimum.
ENERGY_AGREEMENT = 1e-12

#: Agreement of TRW-S's dual ascent with the LP, `test_highs.py`'s 1e-8.
LP_AGREEMENT = 1e-8


def _spawns(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every process the package would start, starting none."""
    started: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        started.append((args, kwargs))
        message = "no subprocess may start"
        raise AssertionError(message)

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    return started


@pytest.mark.critical
@pytest.mark.smoke
def test_the_signature_is_the_siblings_and_the_result_its_type() -> None:
    ours = list(inspect.signature(potts.lower_bound).parameters.values())
    theirs = inspect.signature(search.trws).parameters
    # `trws`'s `graph` and `field`, in its order and kind, then the solver;
    # its two loop keywords, which OpenGM reads and HiGHS refuses (#1279).
    assert [p.name for p in ours] == [
        "graph",
        "field",
        "solver",
        "integral",
        "max_iterations",
        "tolerance",
        "timeout",
        "session",
    ]
    assert ours[5].default == theirs["tolerance"].default
    for parameter in ours[:2]:
        assert parameter.kind is theirs[parameter.name].kind
    assert ours[2].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in ours[3:])
    assert issubclass(ExternalBound, BoundedLabelling)
    assert BOUND_UNIT is Cost.FITS


@pytest.mark.critical
@pytest.mark.analytic
@pytest.mark.parametrize(
    ("solver", "field", "integral"),
    [
        (Solver.GCO_EXPANSION, FIELD, False),
        (Solver.PYMAXFLOW_EXACT, FIELD[:, :2], False),
        (Solver.HMMLEARN, FIELD, True),
    ],
    ids=["gco", "pymaxflow", "hmmlearn"],
)
def test_a_solver_without_a_bound_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    solver: Solver,
    field: np.ndarray,
    integral: bool,
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(CapabilityRefused, match="does not offer: .*lower_bound"):
        potts.lower_bound(GRAPH, field, solver, integral=integral)
    assert started == []


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("field", "message"),
    [
        (FIELD[:4], "shape|one row per node"),
        (np.where(FIELD > 0, np.inf, FIELD), "finite or -inf"),
        (np.where(FIELD > 0, np.nan, FIELD), "finite or -inf"),
        (np.full_like(FIELD, -np.inf), "allows no label"),
    ],
    ids=["rows", "+inf", "nan", "no-label"],
)
def test_a_malformed_field_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, field: np.ndarray, message: str
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(ValueError, match=message):
        potts.lower_bound(GRAPH, field, Solver.HIGHS_LP)
    assert started == []


@pytest.mark.oracle
def test_the_bound_is_the_pre_1282_adapters_bytes() -> None:
    # The four inputs `validation.highs.local_polytope` sent before #1282
    # posed them in `polytope_inputs`: the LP value and the marginals are
    # bitwise those `lower_bound` reads.
    edges = GRAPH.edge_index
    before = run(
        "highs",
        {
            "unary": np.ascontiguousarray(-FIELD),
            "first": np.ascontiguousarray(edges[:, 0], dtype=np.int64),
            "second": np.ascontiguousarray(edges[:, 1], dtype=np.int64),
            "coupling": np.ascontiguousarray(GRAPH.edge_coupling, dtype=np.float64),
        },
    ).outputs
    ours = potts.lower_bound(GRAPH, FIELD, Solver.HIGHS_LP)
    assert ours.bound == float(before["value"])
    assert np.array_equal(ours.labelling, np.argmax(before["node_marginals"], axis=1))
    assert ours.spent == 1
    assert ours.termination.converged
    assert ours.termination.iterations == int(before["iterations"])
    assert ours.provenance == provenance(Solver.HIGHS_LP)


@pytest.mark.oracle
def test_trws_the_lp_the_ilp_and_enumeration_are_ordered() -> None:
    # TRW-S ascends the LP's dual, so its bound is at most the LP value; the
    # ILP restricts the LP, so it is at least it, and with a zero gap it is
    # the minimum. One session serves all eighteen calls.
    integral = []
    with session(Solver.HIGHS_LP) as opened:

        def check(
            name: str, graph: PottsGraph, field: np.ndarray, n_states: int
        ) -> None:
            lp = potts.lower_bound(graph, field, Solver.HIGHS_LP, session=opened)
            ilp = potts.lower_bound(
                graph, field, Solver.HIGHS_LP, integral=True, session=opened
            )
            optimum = _optimum(graph, field, n_states)
            scale = ENERGY_AGREEMENT * max(1.0, abs(optimum))

            assert lp.termination.converged, name
            assert ilp.termination.converged, name
            assert search.trws(graph, field).bound <= lp.bound + LP_AGREEMENT * abs(
                lp.bound
            ), name
            assert lp.bound <= ilp.bound + scale, (name, lp.bound, ilp.bound)
            assert abs(ilp.bound - optimum) <= scale, (name, ilp.bound, optimum)
            assert ilp.integral, name
            assert abs(ilp.energy - optimum) <= scale, name
            assert lp.energy >= optimum - scale, name
            if lp.integral:
                integral.append(name)
                assert abs(lp.energy - lp.bound) <= scale, name

        every_row(_instances(), check)

    assert sorted({row[0] for row in _instances()} - set(integral)) == sorted(
        FRUSTRATED
    )


@pytest.mark.oracle
def test_a_forbidden_label_is_never_chosen_and_the_ilp_is_the_allowed_minimum() -> None:
    # Label 0 forbidden on every site, label 2 on every other: the stand-in
    # keeps every allowed labelling's energy, so the ILP is the minimum over
    # the allowed ones, enumerated here, and the LP is at most it.
    allowed = np.ones_like(FIELD, dtype=bool)
    allowed[:, 0] = False
    allowed[::2, 2] = False
    field = forbid(FIELD, allowed)
    lp = potts.lower_bound(GRAPH, field, Solver.HIGHS_LP)
    ilp = potts.lower_bound(GRAPH, field, Solver.HIGHS_LP, integral=True)
    grid = np.stack(
        np.meshgrid(*[np.flatnonzero(row) for row in allowed], indexing="ij"), axis=-1
    ).reshape(-1, GRAPH.n_nodes)
    optimum = min(energy(GRAPH, FIELD, labelling) for labelling in grid)
    scale = ENERGY_AGREEMENT * max(1.0, abs(optimum))

    for result in (lp, ilp):
        assert allowed[np.arange(GRAPH.n_nodes), result.labelling].all()
        assert np.isfinite(result.energy)
    assert lp.bound <= ilp.bound + scale
    assert abs(ilp.bound - optimum) <= scale, (ilp.bound, optimum)
    assert abs(ilp.energy - optimum) <= scale


@pytest.mark.analytic
def test_a_loop_cap_is_refused_for_highs_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HiGHS runs its simplex to its own criterion: a cap would be ignored.
    started = _spawns(monkeypatch)
    with pytest.raises(ValueError, match="takes no max_iterations"):
        potts.lower_bound(GRAPH, FIELD, Solver.HIGHS_LP, max_iterations=10)
    with pytest.raises(ValueError, match="at least 1"):
        potts.lower_bound(GRAPH, FIELD, Solver.OPENGM_TRWS, max_iterations=0)
    assert started == []
