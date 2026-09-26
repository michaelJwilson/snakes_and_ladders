"""`sal.sandbox` is the oracle home, and only tests and QA may read it.

Issue #322. A hot-path module importing its own oracle has not been replaced,
and a package root re-exporting it puts it on the surface; both are read from
the source, so a lazy import is caught. This stays at the top level (#516): it
asserts an absence in five packages, which `infra/select_tests.py` cannot
derive, so it is in `ALWAYS`, at 1.9 s.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import sal
import sal.sandbox

PACKAGE = Path(sal.__file__).parent
SANDBOX = "sal.sandbox"

# The packages the oracle home may not be imported from: everything that could
# carry a hot path. `qa` renders and may read an oracle; `tests/` pins against
# one. `sandbox` itself is excluded because an oracle may import a sibling.
FORBIDDEN_IMPORTERS = ("sim", "likelihood", "opt", "search", "learn")


def _imports_sandbox(source: str) -> bool:
    """Whether ``source`` imports the sandbox package or anything under it."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(name == SANDBOX or name.startswith(SANDBOX + ".") for name in names):
            return True
    return False


@pytest.mark.critical
@pytest.mark.infra
def test_no_hot_path_package_imports_the_sandbox() -> None:
    offenders = sorted(
        str(path.relative_to(PACKAGE))
        for package in FORBIDDEN_IMPORTERS
        for path in (PACKAGE / package).rglob("*.py")
        if _imports_sandbox(path.read_text())
    )
    assert offenders == []


@pytest.mark.critical
@pytest.mark.infra
def test_the_package_root_does_not_re_export_the_sandbox() -> None:
    # Root `CLAUDE.md`'s Package Surface rule, applied to the one package
    # whose contents are oracles rather than an API.
    assert not _imports_sandbox((PACKAGE / "__init__.py").read_text())


@pytest.mark.infra
def test_the_guard_has_a_subject_and_it_still_imports() -> None:
    # The walk above passes vacuously over an empty sandbox, so what it is
    # asserted against is named here and imported. An import that stopped
    # working is the bit-rot the conservation rule exists to prevent, and
    # nothing else in the package would notice it.
    # Walked, not globbed: a subpackage such as `surrogate.potts` (#1067) is
    # conserved code the top level alone would not name.
    sandbox = PACKAGE / "sandbox"
    modules = sorted(
        ".".join(path.relative_to(sandbox).with_suffix("").parts)
        for path in sandbox.rglob("*.py")
        if path.name != "__init__.py"
    )
    assert modules, "the sandbox carries nothing for the import guard to guard"
    for name in modules:
        importlib.import_module(f"{SANDBOX}.{name}")


@pytest.mark.infra
def test_the_sandbox_states_its_rules_where_the_root_says_they_live() -> None:
    # The docstring points at the rules file, which must carry the two rules
    # asserted here; the package-root rule is root `CLAUDE.md`'s.
    rules = (PACKAGE / "sandbox" / "CLAUDE.md").read_text()
    assert "deleted" in rules
    assert "Only `tests/` and `sal.qa` import from here" in rules
    assert sal.sandbox.__doc__ is not None
    assert "test_sandbox.py" in sal.sandbox.__doc__


@pytest.mark.infra
def test_the_guard_catches_every_spelling_of_the_import() -> None:
    # The guard's own trigger: a lazy import inside a function is the spelling
    # a reviewer misses, and the AST walk sees it.
    assert _imports_sandbox("import sal.sandbox")
    assert _imports_sandbox("from sal.sandbox import maxflow")
    assert _imports_sandbox("from sal.sandbox.maxflow import max_flow")
    assert _imports_sandbox(
        "def f():\n    import sal.sandbox.maxflow as m\n    return m"
    )
    assert not _imports_sandbox("from sal.search import maxflow")
    assert not _imports_sandbox("import sal.sandboxed")
