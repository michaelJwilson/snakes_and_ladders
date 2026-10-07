"""`external.potts.ground_state` takes `search.ground_state`'s arguments and refuses before it spawns (issue #1282, step 3).

Read against the sibling's signature and `Capability` declarations: a
problem with more than two states, or a forbidden label, is refused for
PyMaxflow naming the capability, and a budget in another unit, a start to
the cut and a malformed field are refused, each with no subprocess started.
The labellings are pinned against the adapters in
`tests/validation/test_gco.py` and `tests/validation/test_pymaxflow.py`.
"""

from __future__ import annotations

import inspect
import subprocess
from typing import Any

import numpy as np
import pytest
from sal import external
from sal.cost import Cost
from sal.external import CapabilityRefused, Solver, potts
from sal.external.potts import UNITS, ExternalRun
from sal.external.potts_inputs import allowed_by, stand_in
from sal.opt.budget import Budget
from sal.search import ground_state as search
from sal.search.ground_state import MethodRun
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energy, forbid

GRAPH = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8)
RNG = np.random.default_rng(1282)
TWO = RNG.normal(size=(GRAPH.n_nodes, 2))
THREE = RNG.normal(size=(GRAPH.n_nodes, 3))


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
    ours = list(inspect.signature(potts.ground_state).parameters.values())
    theirs = inspect.signature(search.ground_state).parameters
    # The sibling's method is a solver here; every other name, order and
    # kind is the sibling's, and `session` is the one addition.
    names = [p.name for p in ours]
    assert names == ["graph", "field", "solver", "budget", "rng", "start", "session"]
    for parameter in ours[:5]:
        sibling = theirs["method" if parameter.name == "solver" else parameter.name]
        assert parameter.kind is sibling.kind
    assert ours[5].kind is theirs["start"].kind is inspect.Parameter.KEYWORD_ONLY
    assert ours[6].kind is inspect.Parameter.KEYWORD_ONLY
    assert issubclass(ExternalRun, MethodRun)
    assert set(UNITS) == {
        solver
        for solver in Solver
        if external.Capability.GROUND_STATE in solver.capabilities
    }


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("field", "missing"),
    [
        (THREE, "multi_label"),
        (
            forbid(TWO, np.arange(2) == np.arange(GRAPH.n_nodes)[:, None] % 2),
            "forbidden_labels",
        ),
    ],
    ids=["q3", "forbidden"],
)
def test_pymaxflow_refuses_what_it_does_not_declare(
    monkeypatch: pytest.MonkeyPatch, field: np.ndarray, missing: str
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(CapabilityRefused, match=f"does not offer: {missing}$") as no:
        potts.ground_state(
            GRAPH, field, Solver.PYMAXFLOW_EXACT, Budget(Cost.PASS, 1), RNG
        )
    assert no.value.missing == frozenset({external.Capability(missing)})
    assert started == []


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("solver", "field", "keywords", "message"),
    [
        (Solver.GCO_EXPANSION, THREE, {"budget": Budget(Cost.SITE_VISITS, 9)}, "fits"),
        (Solver.PYMAXFLOW_EXACT, TWO, {"start": np.zeros(9, dtype=np.int64)}, "start"),
        (Solver.GCO_SWAP, THREE[:4], {}, "one row per node"),
        (Solver.GCO_SWAP, np.where(THREE > 0, np.inf, THREE), {}, "finite or -inf"),
        (Solver.GCO_SWAP, np.full_like(THREE, -np.inf), {}, "allows no label"),
        (Solver.GCO_EXPANSION, THREE, {"start": np.full(9, 3)}, "start"),
        (Solver.OPENGM_ASTAR, THREE, {"start": np.zeros(9, dtype=np.int64)}, "start"),
        (Solver.OPENGM_LBP, THREE, {"budget": Budget(Cost.PASS, 1)}, "fits"),
    ],
    ids=[
        "unit",
        "cut-start",
        "rows",
        "+inf",
        "no-label",
        "range",
        "astar-start",
        "lbp-unit",
    ],
)
def test_a_malformed_call_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    solver: Solver,
    field: np.ndarray,
    keywords: dict[str, Any],
    message: str,
) -> None:
    started = _spawns(monkeypatch)
    budget = keywords.pop("budget", Budget(UNITS[solver], 1))
    with pytest.raises(ValueError, match=message):
        potts.ground_state(GRAPH, field, solver, budget, RNG, **keywords)
    assert started == []


@pytest.mark.analytic
@pytest.mark.parametrize(
    "solver", [Solver.OPENGM_EXPANSION, Solver.OPENGM_SWAP], ids=str
)
def test_opengms_moves_refuse_a_negative_coupling_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, solver: Solver
) -> None:
    # Their auxiliary construction needs a metric pairwise term (#1279).
    started = _spawns(monkeypatch)
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.5)
    with pytest.raises(ValueError, match="metric"):
        potts.ground_state(graph, THREE, solver, Budget(UNITS[solver], 1), RNG)
    assert started == []


@pytest.mark.analytic
def test_the_stand_in_makes_every_forbidden_label_cost_more_than_its_couplings() -> (
    None
):
    # #1274's rule: at every site, moving off a forbidden label to the
    # site's best allowed one lowers the energy whatever its neighbours hold.
    allowed = np.ones_like(THREE, dtype=bool)
    allowed[:, 0] = False
    allowed[::2, 2] = False
    posed = stand_in(GRAPH, THREE, allowed)
    assert np.array_equal(posed[allowed], THREE[allowed])
    for labelling in np.random.default_rng(0).integers(3, size=(200, GRAPH.n_nodes)):
        best = np.where(allowed, posed, -np.inf).argmax(axis=1)
        for site in np.flatnonzero(~allowed[np.arange(GRAPH.n_nodes), labelling]):
            moved = labelling.copy()
            moved[site] = best[site]
            assert energy(GRAPH, posed, moved) < energy(GRAPH, posed, labelling)
    assert allowed_by(allowed, np.ones(GRAPH.n_nodes, dtype=np.int64))
    assert not allowed_by(allowed, np.zeros(GRAPH.n_nodes, dtype=np.int64))
