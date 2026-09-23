"""`snakes_and_ladders.validation` drives external frameworks, each in a subprocess (issue #972).

Three rules are read from the source tree, as `test_sandbox.py` reads its own,
so a lazy import inside a function is caught too:

- no hot-path package imports `validation`, and the package root does not
  re-export it;
- no module under `python/` or `tests/` imports a registered framework except
  the scripts under `validation/scripts/`, which run in their own interpreter;
- every `validation-*` extra names exactly one distribution, is registered in
  `FRAMEWORKS` with the module its script imports, and has an adapter, a
  script and a test module under the framework's name.

The runner is checked against `scripts/selftest.py`, which imports NumPy
alone: arrays come back bitwise, the time is the script's, and a failing
script raises with its standard error.

**This module stays at the top level**, beside `test_sandbox.py`, for the
reason that one does: what it asserts is an absence across five packages.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
import snakes_and_ladders
from snakes_and_ladders.validation import FRAMEWORKS
from snakes_and_ladders.validation.protocol import SECONDS
from snakes_and_ladders.validation.runner import ScriptError, available, run
from validation_extras import PREFIX, validation_extras

from tests._paths import REPO_ROOT
from tests.validation._goals import Goal, assert_meets, median_seconds

PACKAGE = Path(snakes_and_ladders.__file__).parent
VALIDATION = "snakes_and_ladders.validation"
SCRIPTS = PACKAGE / "validation" / "scripts"
TESTS = REPO_ROOT / "tests"

#: The packages that may not import the validation home: everything that
#: could carry a hot path, as `test_sandbox.py` names them.
FORBIDDEN_IMPORTERS = ("sim", "likelihood", "opt", "search", "learn")


def _imported(source: str) -> set[str]:
    """Every module name ``source`` imports, lazily or not."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module or "")
    return names


def _imports(source: str, module: str) -> bool:
    """Whether ``source`` imports ``module`` or anything under it."""
    return any(
        name == module or name.startswith(module + ".") for name in _imported(source)
    )


def _declared_extras() -> dict[str, list[str]]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        declared = tomllib.load(handle)["project"]["optional-dependencies"]
    return {name: declared[name] for name in validation_extras()}


def _distribution(requirement: str) -> str:
    """The distribution a requirement names, without version or marker."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    assert match is not None, requirement
    return match.group(1)


@pytest.mark.critical
@pytest.mark.infra
def test_no_hot_path_package_imports_the_validation_home() -> None:
    offenders = sorted(
        str(path.relative_to(PACKAGE))
        for package in FORBIDDEN_IMPORTERS
        for path in (PACKAGE / package).rglob("*.py")
        if _imports(path.read_text(), VALIDATION)
    )
    assert offenders == []
    assert not _imports((PACKAGE / "__init__.py").read_text(), VALIDATION)


@pytest.mark.critical
@pytest.mark.infra
def test_only_the_scripts_import_a_framework() -> None:
    modules = [framework.module for framework in FRAMEWORKS.values()]
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for root in (PACKAGE, TESTS)
        for path in root.rglob("*.py")
        if SCRIPTS not in path.parents
        and any(_imports(path.read_text(), module) for module in modules)
    )
    assert offenders == []


@pytest.mark.infra
def test_every_validation_extra_is_one_registered_framework_with_a_test() -> None:
    extras = _declared_extras()
    registered = {framework.extra: framework for framework in FRAMEWORKS.values()}
    assert set(extras) == set(registered)
    for extra, requirements in extras.items():
        framework = registered[extra]
        assert extra == PREFIX + framework.name
        assert FRAMEWORKS[framework.name] is framework
        assert len(requirements) == 1, (extra, requirements)
        assert _distribution(requirements[0]).lower() == (
            framework.distribution.lower()
        )
        assert (PACKAGE / "validation" / f"{framework.name}.py").exists()
        assert (SCRIPTS / f"{framework.name}.py").exists()
        assert (TESTS / "validation" / f"test_{framework.name}.py").exists()
        assert framework.source.startswith("https://")


@pytest.mark.infra
def test_the_guards_catch_every_spelling_of_the_import() -> None:
    # The guards pass vacuously while `FRAMEWORKS` is empty, so what they
    # would refuse is asserted on source text here.
    assert _imports("import maxflow", "maxflow")
    assert _imports("from maxflow import GraphFloat", "maxflow")
    assert _imports(
        "def f():\n    import maxflow.fastmin as m\n    return m", "maxflow"
    )
    assert not _imports("import maxflowing", "maxflow")
    assert not _imports("from . import maxflow", "maxflow")
    assert _imports("from snakes_and_ladders.validation import runner", VALIDATION)


@pytest.mark.critical
@pytest.mark.infra
def test_the_runner_returns_the_scripts_arrays_bitwise() -> None:
    values = np.random.default_rng(972).normal(size=(17, 3))
    labels = np.arange(11, dtype=np.int64)
    result = run("selftest", {"values": values, "labels": labels})
    assert np.array_equal(result.outputs["values"], values)
    assert result.outputs["values"].dtype == values.dtype
    assert np.array_equal(result.outputs["labels"], labels)
    assert np.array_equal(result.outputs["doubled"], 2.0 * values)
    assert SECONDS not in result.outputs
    # The script's figure is its own call, far below one interpreter start.
    assert 0.0 <= result.seconds < 0.05


@pytest.mark.infra
def test_a_failing_script_raises_with_its_standard_error() -> None:
    with pytest.raises(ScriptError, match="asked to fail with 3"):
        run("selftest", {"values": np.zeros(2), "fail": np.asarray(3, dtype=np.int64)})


@pytest.mark.infra
def test_availability_is_read_without_importing() -> None:
    assert available("numpy")
    assert not available("snakes_and_ladders_no_such_framework")
    assert not available("snakes_and_ladders_no_such_framework.sub")


@pytest.mark.infra
def test_the_home_states_its_rules_where_its_docstring_says() -> None:
    rules = (PACKAGE / "validation" / "CLAUDE.md").read_text()
    assert "Every framework runs in a subprocess" in rules
    assert "Only `tests/` imports from here" in rules
    assert snakes_and_ladders.validation.__doc__ is not None
    assert "test_validation.py" in snakes_and_ladders.validation.__doc__


@pytest.mark.infra
def test_a_goal_fails_by_how_far_the_package_is_off() -> None:
    goal = Goal("selftest", "a call", 1.0e-3, "a hardcoded figure")
    assert_meets(0.9e-3, goal)
    assert_meets(1.0e-3, goal)
    with pytest.raises(AssertionError, match=r"1\.50x"):
        assert_meets(1.5e-3, goal)
    assert median_seconds(lambda: None, repeats=3) >= 0.0
