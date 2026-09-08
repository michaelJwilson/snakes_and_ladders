"""`snakes_and_ladders.sandbox` is the oracle home, and only tests and QA may read it.

Issue #322. An implementation a framework replaced on a hot path is kept there
to referee the framework. Two things follow that no reviewer would notice in a
diff: a hot-path module importing its own oracle has not been replaced, and a
package root re-exporting it would put it back on the package surface. Both
are asserted from the source tree rather than from the import system, so a
lazy import inside a function is caught too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import snakes_and_ladders
import snakes_and_ladders.sandbox

PACKAGE = Path(snakes_and_ladders.__file__).parent
SANDBOX = "snakes_and_ladders.sandbox"

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
@pytest.mark.structural
def test_no_hot_path_package_imports_the_sandbox() -> None:
    offenders = sorted(
        str(path.relative_to(PACKAGE))
        for package in FORBIDDEN_IMPORTERS
        for path in (PACKAGE / package).rglob("*.py")
        if _imports_sandbox(path.read_text())
    )
    assert offenders == []


@pytest.mark.critical
@pytest.mark.structural
def test_the_package_root_does_not_re_export_the_sandbox() -> None:
    # Root `CLAUDE.md`'s Package Surface rule, applied to the one package
    # whose contents are oracles rather than an API.
    assert not _imports_sandbox((PACKAGE / "__init__.py").read_text())


@pytest.mark.structural
def test_the_sandbox_states_its_rules_where_the_root_says_they_live() -> None:
    # The docstring names the file that carries the rules; the file must exist
    # and carry the three rules, or the pointer points at nothing.
    rules = (PACKAGE / "sandbox" / "CLAUDE.md").read_text()
    assert "Nothing is deleted" in rules
    assert "Only `tests/` and `snakes_and_ladders.qa` import from here" in rules
    assert "Not re-exported from the package root" in rules
    assert snakes_and_ladders.sandbox.__doc__ is not None
    assert "test_sandbox.py" in snakes_and_ladders.sandbox.__doc__


@pytest.mark.structural
def test_the_guard_catches_every_spelling_of_the_import() -> None:
    # The guard's own trigger: a lazy import inside a function is the spelling
    # a reviewer misses, and the AST walk sees it.
    assert _imports_sandbox("import snakes_and_ladders.sandbox")
    assert _imports_sandbox("from snakes_and_ladders.sandbox import maxflow")
    assert _imports_sandbox("from snakes_and_ladders.sandbox.maxflow import max_flow")
    assert _imports_sandbox(
        "def f():\n    import snakes_and_ladders.sandbox.maxflow as m\n    return m"
    )
    assert not _imports_sandbox("from snakes_and_ladders.search import maxflow")
    assert not _imports_sandbox("import snakes_and_ladders.sandboxed")
