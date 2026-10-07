"""`external`'s solver-keyed calls are a `match` onto its explicit ones, which return the same result (issue #1304).

Read against the explicit calls themselves: for every `Solver` member that
`potts.ground_state`, `potts.lower_bound` or `hmm.fit` runs, the
solver-keyed call and the explicit call it names return equal results on one
fixture, every field bitwise but the seconds each run measured. The `match`
covers every member but `BLACKJAX_HMC`, which `hmc.sample` runs; a solver
without the task, a `by` no framework runs, and a keyword the explicit call
does not take are refused with no subprocess started.
"""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from sal.cost import Cost
from sal.emissions import CategoricalEmission
from sal.external import CapabilityRefused, Solver, hmm, potts
from sal.external.frameworks import FRAMEWORKS, Framework
from sal.external.solvers import Capability
from sal.opt.budget import Budget
from sal.opt.em import EmConfig
from sal.sim.graph import BoundaryCondition, lattice_graph

from tests._frameworks import requires

GRAPH = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8)
#: Two states, so every ground-state solver, PyMaxflow's cut among them, runs it.
BINARY = np.random.default_rng(1304).normal(size=(GRAPH.n_nodes, 2))
#: Three states for the bounds.
FIELD = np.random.default_rng(1282).normal(size=(GRAPH.n_nodes, 3))

LOG_INITIAL = torch.log(torch.tensor([0.4, 0.6], dtype=torch.float64))
LOG_TRANSITION = torch.log(torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64))
OBSERVATIONS = np.random.default_rng(1282).integers(0, 3, size=(4, 6))
CATEGORICAL = CategoricalEmission(np.array([[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]]))

#: Each ground-state solver's explicit call and the keywords the `match` adds.
GROUND: dict[Solver, tuple[str, dict[str, Any]]] = {
    Solver.GCO_EXPANSION: ("expansion", {"by": Framework.GCO}),
    Solver.OPENGM_EXPANSION: ("expansion", {"by": Framework.OPENGM}),
    Solver.GCO_SWAP: ("swap", {"by": Framework.GCO}),
    Solver.OPENGM_SWAP: ("swap", {"by": Framework.OPENGM}),
    Solver.PYMAXFLOW_EXACT: ("min_cut", {}),
    Solver.OPENGM_ICM: ("icm", {}),
    Solver.OPENGM_LBP: ("loopy_bp", {}),
    Solver.OPENGM_ASTAR: ("astar", {}),
}

#: Each bound solver's explicit call.
BOUND: dict[Solver, str] = {
    Solver.HIGHS_LP: "lp",
    Solver.OPENGM_TRWS: "trws",
    Solver.OPENGM_DD: "dual_decomposition",
}

#: The fit solver's explicit call.
FIT: dict[Solver, str] = {Solver.HMMLEARN: "baum_welch"}


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


def _same(ours: Any, theirs: Any) -> None:
    """Every field of two results equal, arrays and tensors bitwise, the seconds aside."""
    for item in dataclasses.fields(ours):
        if item.name == "seconds":
            continue
        left, right = getattr(ours, item.name), getattr(theirs, item.name)
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right), item.name
        elif isinstance(left, torch.Tensor):
            assert torch.equal(left, right), item.name
        elif isinstance(left, CategoricalEmission):
            # An emission family compares by identity: read its tensors.
            mine, other = left.named_parameters(), right.named_parameters()
            assert mine.keys() == other.keys(), item.name
            for key in mine:
                assert torch.equal(mine[key], other[key]), (item.name, key)
        elif dataclasses.is_dataclass(left) and not isinstance(left, type):
            _same(left, right)
        else:
            assert left == right, item.name


def _required(solver: Solver) -> Any:
    """``solver`` as a parameter that skips where its framework is absent."""
    return pytest.param(
        solver, marks=requires(str(solver.framework.name)), id=str(solver)
    )


@pytest.mark.critical
@pytest.mark.analytic
def test_the_match_covers_every_solver_once() -> None:
    # Every member is in exactly one `match`, or is `hmc.sample`'s; each
    # table's solvers are those that declare its task.
    tables: tuple[dict[Solver, Any], ...] = (GROUND, BOUND, FIT)
    for solver in Solver:
        held = [table for table in tables if solver in table]
        assert len(held) == (solver is not Solver.BLACKJAX_HMC), solver
    for table, task in zip(
        tables,
        (Capability.GROUND_STATE, Capability.LOWER_BOUND, Capability.HMM_FIT),
        strict=True,
    ):
        assert set(table) == {s for s in Solver if task in s.capabilities}, task
    assert set(Framework) == set(FRAMEWORKS)


@pytest.mark.critical
@pytest.mark.analytic
def test_each_explicit_call_takes_no_solver_and_holds_one_algorithm() -> None:
    # The explicit calls name the algorithm, so none takes a solver; `by`
    # sits only on the two that several frameworks run.
    for namespace, names in (
        (potts.ground_state, {name for name, _ in GROUND.values()}),
        (potts.lower_bound, set(BOUND.values())),
        (hmm.fit, set(FIT.values())),
    ):
        for name in names:
            parameters = inspect.signature(getattr(namespace, name)).parameters
            assert "solver" not in parameters, name
            assert ("by" in parameters) == (name in {"expansion", "swap"}), name


@pytest.mark.analytic
@pytest.mark.parametrize("solver", [_required(solver) for solver in GROUND])
def test_a_ground_state_by_solver_is_its_explicit_calls(solver: Solver) -> None:
    name, keywords = GROUND[solver]
    budget = Budget(potts.UNITS[solver], 4)
    rng = np.random.default_rng(0)
    ours = potts.ground_state(GRAPH, BINARY, solver, budget, rng)
    theirs = getattr(potts.ground_state, name)(GRAPH, BINARY, budget, rng, **keywords)
    assert ours.provenance.framework == solver.framework.name
    _same(ours, theirs)


@pytest.mark.analytic
@pytest.mark.parametrize("solver", [_required(solver) for solver in BOUND])
def test_a_bound_by_solver_is_its_explicit_calls(solver: Solver) -> None:
    ours = potts.lower_bound(GRAPH, FIELD, solver)
    theirs = getattr(potts.lower_bound, BOUND[solver])(GRAPH, FIELD)
    _same(ours, theirs)


@pytest.mark.analytic
@requires("highs")
def test_the_ilp_by_solver_is_the_explicit_lps_with_integral() -> None:
    ours = potts.lower_bound(GRAPH, FIELD, Solver.HIGHS_LP, integral=True)
    _same(ours, potts.lower_bound.lp(GRAPH, FIELD, integral=True))


@pytest.mark.analytic
@requires("opengm")
def test_a_loop_keyword_by_solver_reaches_the_explicit_call() -> None:
    ours = potts.lower_bound(
        GRAPH, FIELD, Solver.OPENGM_TRWS, max_iterations=3, tolerance=0.0
    )
    theirs = potts.lower_bound.trws(GRAPH, FIELD, max_iterations=3, tolerance=0.0)
    _same(ours, theirs)
    assert ours.termination.iterations <= 3


@pytest.mark.hmm
@pytest.mark.analytic
@requires("hmmlearn")
def test_a_fit_by_solver_is_baum_welchs() -> None:
    config = EmConfig(max_iterations=5, tolerance=0.0)
    problem = (OBSERVATIONS, LOG_INITIAL, LOG_TRANSITION, CATEGORICAL)
    ours = hmm.fit(*problem, Solver.HMMLEARN, config)
    _same(ours, hmm.fit.baum_welch(*problem, config))


@pytest.mark.critical
@pytest.mark.analytic
@pytest.mark.parametrize(
    ("call", "task"),
    [
        (
            lambda solver: potts.ground_state(
                GRAPH, BINARY, solver, Budget(Cost.FITS, 1), np.random.default_rng(0)
            ),
            "ground_state",
        ),
        (lambda solver: potts.lower_bound(GRAPH, FIELD, solver), "lower_bound"),
        (
            lambda solver: hmm.fit(
                OBSERVATIONS, LOG_INITIAL, LOG_TRANSITION, CATEGORICAL, solver
            ),
            "hmm_fit",
        ),
    ],
    ids=["ground_state", "lower_bound", "fit"],
)
def test_a_solver_without_the_task_is_refused_by_name_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[Solver], Any], task: str
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(CapabilityRefused, match=f"{Solver.BLACKJAX_HMC}.*{task}"):
        call(Solver.BLACKJAX_HMC)
    assert started == []


@pytest.mark.critical
@pytest.mark.analytic
@pytest.mark.parametrize("name", ["expansion", "swap"])
def test_a_by_no_framework_runs_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(
        ValueError, match=f"{name} is run by gco or opengm, not pymaxflow"
    ):
        getattr(potts.ground_state, name)(
            GRAPH,
            BINARY,
            Budget(Cost.FITS, 1),
            np.random.default_rng(0),
            by=Framework.PYMAXFLOW,
        )
    assert started == []


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: potts.ground_state(
                GRAPH,
                BINARY,
                Solver.OPENGM_LBP,
                Budget(Cost.FITS, 1),
                np.random.default_rng(0),
                start=np.zeros(GRAPH.n_nodes, dtype=np.int64),
            ),
            "runs loopy_bp, which takes no start",
        ),
        (
            lambda: potts.lower_bound(GRAPH, FIELD, Solver.HIGHS_LP, tolerance=1e-3),
            "runs lp, which takes no tolerance",
        ),
        (
            lambda: potts.lower_bound(GRAPH, FIELD, Solver.OPENGM_DD, integral=True),
            "runs dual_decomposition, which takes no integral",
        ),
    ],
    ids=["start-to-loopy_bp", "tolerance-to-lp", "integral-to-dd"],
)
def test_a_keyword_the_explicit_call_lacks_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], Any], message: str
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(ValueError, match=message):
        call()
    assert started == []
