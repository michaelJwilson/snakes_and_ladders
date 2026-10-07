"""`sal.external` names each external solver, refuses before it spawns, and is the one subprocess path (issue #1282).

Step 1 of #1282: the runner and protocol moved here from `sal.validation`,
and `Solver`, `Capability` and `Provenance` declare what each framework can be
asked and the licence its answer carries. Read against the registry
`FRAMEWORKS`, the installed distributions' metadata, and the runner itself:
a call `invoke` lets through returns `runner.run`'s arrays bitwise. The
one-spawner count is in `test_duplication_guards.py`.
"""

from __future__ import annotations

import ast
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sal
from sal import external
from sal.external import (
    Capability,
    CapabilityRefused,
    ExternalUnavailable,
    Provenance,
    Solver,
    available,
    invoke,
    protocol,
    provenance,
    runner,
    solvers,
)
from sal.external.frameworks import FRAMEWORKS

PACKAGE = Path(sal.__file__).parent

#: The packages that carry a hot path, as `test_validation.py` names them.
HOT_PATH = ("sim", "likelihood", "opt", "search", "learn")

#: The tasks a solver can be asked; every solver offers at least one.
TASKS = frozenset(
    {
        Capability.GROUND_STATE,
        Capability.LOWER_BOUND,
        Capability.HMM_FIT,
        Capability.VITERBI,
        Capability.LOG_LIKELIHOOD,
        Capability.HMC_SAMPLE,
    }
)


def _imports(path: Path, module: str) -> bool:
    """Whether ``path`` imports ``module`` or anything under it, lazily or not."""
    for node in ast.walk(ast.parse(path.read_text())):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        if any(name == module or name.startswith(module + ".") for name in names):
            return True
    return False


def _spawns(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every process the runner would start, starting none."""
    started: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        started.append((args, kwargs))
        message = "no subprocess may start"
        raise AssertionError(message)

    monkeypatch.setattr(subprocess, "run", refuse)
    return started


@pytest.mark.critical
@pytest.mark.infra
def test_every_solver_has_a_framework_and_a_provenance() -> None:
    assert set(solvers.DECLARED) == set(Solver)
    for solver in Solver:
        framework = solver.framework
        assert FRAMEWORKS[framework.name] is framework
        assert solver.capabilities & TASKS, solver
        if available(solver):
            found = provenance(solver)
            assert found == Provenance(
                framework=framework.name,
                version=importlib.metadata.version(framework.distribution),
                licence=framework.licence,
                osi=framework.osi,
            )
        else:
            with pytest.raises(ExternalUnavailable, match=framework.extra):
                provenance(solver)


@pytest.mark.infra
def test_the_licence_flags_are_the_registered_terms() -> None:
    # gco-v3.0 is research-use only, the one non-OSI term (#974); GPL-3.0 is
    # OSI-approved, and is why PyMaxflow stays in its subprocess.
    assert {name for name, f in FRAMEWORKS.items() if not f.osi} == {"gco"}
    assert Solver.PYMAXFLOW_EXACT.framework.licence == "GPL-3.0"
    # HiGHS ships in SciPy, a core dependency: always available, and its
    # version is SciPy's.
    assert available(Solver.HIGHS_LP)
    assert provenance(Solver.HIGHS_LP).version == importlib.metadata.version("scipy")


@pytest.mark.critical
@pytest.mark.analytic
@pytest.mark.parametrize(
    ("solver", "needs", "missing"),
    [
        (
            Solver.PYMAXFLOW_EXACT,
            {Capability.GROUND_STATE, Capability.MULTI_LABEL},
            "multi_label",
        ),
        (Solver.HIGHS_LP, {Capability.GROUND_STATE}, "ground_state"),
        (Solver.GCO_SWAP, {Capability.GROUND_STATE, Capability.EXACT}, "exact"),
        (Solver.HMMLEARN, {Capability.HMC_SAMPLE}, "hmc_sample"),
    ],
)
def test_a_refusal_names_the_capability_and_starts_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    solver: Solver,
    needs: set[Capability],
    missing: str,
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(
        CapabilityRefused, match=f"{solver} does not offer: {missing}$"
    ) as refused:
        invoke(solver, needs, {"values": np.zeros(2)})
    assert refused.value.missing == frozenset({Capability(missing)})
    assert started == []


@pytest.mark.analytic
def test_an_absent_framework_names_its_extra_and_starts_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _spawns(monkeypatch)

    def absent(module: str) -> bool:
        assert module in {f.module for f in FRAMEWORKS.values()}
        return False

    monkeypatch.setattr(solvers, "installed", absent)
    for solver in Solver:
        assert not available(solver)
        with pytest.raises(
            ExternalUnavailable, match=f"`{solver.framework.extra}` extra"
        ):
            invoke(solver, set(), {"values": np.zeros(2)})
    assert started == []


@pytest.mark.infra
def test_an_admitted_call_returns_the_runners_arrays_bitwise() -> None:
    # A two-site, two-state LP: the HiGHS script, reached by `invoke` and by
    # `runner.run`, on the same bytes; every output but the script's clock.
    inputs = {
        "unary": np.array([[0.0, 0.5], [0.3, 0.0]]),
        "first": np.array([0], dtype=np.int64),
        "second": np.array([1], dtype=np.int64),
        "coupling": np.array([0.7]),
    }
    through = invoke(Solver.HIGHS_LP, {Capability.LOWER_BOUND}, inputs)
    direct = runner.run("highs", inputs)
    assert through.outputs.keys() == direct.outputs.keys()
    for name, array in direct.outputs.items():
        if name.endswith("seconds"):
            continue
        assert np.array_equal(through.outputs[name], array), name
        assert through.outputs[name].dtype == array.dtype, name


@pytest.mark.critical
@pytest.mark.infra
def test_the_old_import_paths_resolve_to_the_moved_objects() -> None:
    # Kept for one release, as #1010's splits kept theirs.
    from sal import validation
    from sal.validation import protocol as old_protocol
    from sal.validation import runner as old_runner

    assert validation.FRAMEWORKS is FRAMEWORKS
    assert old_runner.run is runner.run
    assert old_runner.Run is runner.Run
    assert old_runner.ScriptError is runner.ScriptError
    assert old_runner.package is runner.package
    assert old_runner.available is runner.installed
    for name in old_protocol.__all__:
        assert getattr(old_protocol, name) is getattr(protocol, name), name


@pytest.mark.critical
@pytest.mark.infra
def test_the_import_direction_is_validation_to_external() -> None:
    home = PACKAGE / "external"
    assert [p.name for p in home.rglob("*.py") if _imports(p, "sal.validation")] == []
    offenders = sorted(
        str(path.relative_to(PACKAGE))
        for package in HOT_PATH
        for path in (PACKAGE / package).rglob("*.py")
        if _imports(path, "sal.external")
    )
    assert offenders == []
    assert not _imports(PACKAGE / "__init__.py", "sal.external")


@pytest.mark.infra
def test_the_home_states_its_rules_where_its_docstring_says() -> None:
    rules = (PACKAGE / "external" / "CLAUDE.md").read_text()
    assert "Every framework runs in a subprocess" in rules
    assert "A refusal comes before the subprocess" in rules
    assert external.__doc__ is not None
    assert "test_external.py" in external.__doc__


#: The modules the root re-exports from: the infrastructure, no problem family.
INFRASTRUCTURE = frozenset(
    {
        "sal.external.runner",
        "sal.external.sessions",
        "sal.external.solvers",
        "sal.external.transport",
    }
)


@pytest.mark.critical
@pytest.mark.infra
def test_the_root_exports_no_problem_specific_function() -> None:
    # `sal.external` is namespaced by problem family, as `sal` is: the Potts
    # calls are `external.potts.*`, and a bare `external.ground_state` would
    # not say which problem it solves. Every root export is infrastructure,
    # and no task's name resolves on the root.
    for name in external.__all__:
        assert getattr(external, name).__module__ in INFRASTRUCTURE, name
    for task in [*TASKS, "fit", "score", "sample"]:
        assert not hasattr(external, task), task
    assert external.potts.ground_state.__module__ == "sal.external.potts"
    assert external.potts.lower_bound.__module__ == "sal.external.potts"
