"""`snakes_and_ladders.sandbox` is the conserved home, and both directions are asserted here.

Issue #322. An implementation a framework replaced on a hot path is kept there
to referee the framework. Two things follow that no reviewer would notice in a
diff: a hot-path module importing its own oracle has not been replaced, and a
package root re-exporting it would put it back on the package surface. Both
are asserted from the source tree rather than from the import system, so a
lazy import inside a function is caught too.

Issue #462 adds the opposite direction: every module in the sandbox is
imported by a test the suite collects. The rules above are structural and have
never been violated; this one was enforced by remembering, and three branches
in one day declined a route and conserved nothing. A module nothing imports is
either dead or a decline nobody kept, and both are findings.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import snakes_and_ladders
import snakes_and_ladders.sandbox
from tests.conftest import COLLECTED_FILES

PACKAGE = Path(snakes_and_ladders.__file__).parent
TESTS = Path(__file__).resolve().parents[1]
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


def _sandbox_modules() -> list[str]:
    """Every module under the sandbox, as a dotted name relative to it."""
    return sorted(
        path.relative_to(PACKAGE / "sandbox").with_suffix("").as_posix().replace("/", ".")
        for path in (PACKAGE / "sandbox").rglob("*.py")
        if path.name != "__init__.py"
    )


def _sandbox_modules_imported(source: str) -> set[str]:
    """The sandbox submodules ``source`` names in an import statement.

    Both spellings reduce to the same dotted suffix, so
    ``from snakes_and_ladders.sandbox import tropical`` and
    ``import snakes_and_ladders.sandbox.tropical`` are one answer.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        else:
            continue
        found.update(
            name[len(SANDBOX) + 1 :]
            for name in names
            if name.startswith(SANDBOX + ".")
        )
    return found


@pytest.mark.critical
@pytest.mark.structural
def test_every_sandbox_module_is_reached_by_a_collected_test(
    request: pytest.FixtureRequest,
) -> None:
    """Issue #462, the conservation rule's other half.

    The subject is the *collected* suite, not a text search of ``tests/``. A
    grep answers from a file pytest never runs -- one outside ``testpaths``,
    one named so pytest does not collect it, one that fails to import at all
    -- and every sandbox test ``importorskip``s a framework, so a guard that
    cannot tell those apart is the rot it is meant to catch. Whether a
    collected test then skips is a second question this guard does not
    answer, and is the one issue #447 is settling when it decides which job
    installs the ``frameworks`` extra.
    """
    if not _collected_the_whole_suite(request.config):
        pytest.skip(
            "the session collected a subset of tests/, so an unimported module "
            "cannot be told from an uncollected importer; run `pytest -m critical`"
        )
    reached: set[str] = set()
    for collected in sorted(request.config.stash[COLLECTED_FILES]):
        path = Path(collected)
        if TESTS not in path.parents:
            continue
        source = path.read_text()
        # An import of the sandbox spells the package out, so a file without
        # that substring cannot name a module in it and need not be parsed.
        # Parsing all 271 collected files costs 0.56 s; this costs 0.02 s.
        if SANDBOX in source:
            reached |= _sandbox_modules_imported(source)
    unreached = [
        name
        for name in _sandbox_modules()
        if name not in reached
        and not any(other.startswith(name + ".") for other in reached)
    ]
    assert unreached == [], (
        f"conserved but unreached by any collected test: {unreached}; "
        "sandbox/CLAUDE.md keeps these, so a test must import each"
    )


def _collected_the_whole_suite(config: pytest.Config) -> bool:
    """Whether this session collected ``tests/`` entire, rather than a subset."""
    invocation = config.invocation_params.dir
    return any(
        (invocation / str(argument).split("::")[0]).resolve() == TESTS
        for argument in config.args
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_package_root_does_not_re_export_the_sandbox() -> None:
    # Root `CLAUDE.md`'s Package Surface rule, applied to the one package
    # whose contents are oracles rather than an API.
    assert not _imports_sandbox((PACKAGE / "__init__.py").read_text())


@pytest.mark.structural
def test_the_guard_has_a_subject_and_it_still_imports() -> None:
    # The walk above passes vacuously over an empty sandbox, so what it is
    # asserted against is named here and imported. An import that stopped
    # working is the bit-rot the conservation rule exists to prevent, and
    # nothing else in the package would notice it.
    modules = sorted(
        path.stem
        for path in (PACKAGE / "sandbox").glob("*.py")
        if path.name != "__init__.py"
    )
    assert modules, "the sandbox carries nothing for the import guard to guard"
    for name in modules:
        importlib.import_module(f"{SANDBOX}.{name}")


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


@pytest.mark.structural
def test_the_conservation_guard_reads_both_spellings_and_nothing_else() -> None:
    # The #462 guard's own trigger. A module is reached however the import is
    # written, and the package alone reaches nothing -- importing `sandbox`
    # without naming a module is what an empty conservation looks like.
    assert _sandbox_modules_imported("import snakes_and_ladders.sandbox.tropical") == {
        "tropical"
    }
    assert _sandbox_modules_imported(
        "from snakes_and_ladders.sandbox import tropical"
    ) == {"tropical"}
    assert _sandbox_modules_imported(
        "from snakes_and_ladders.sandbox.tropical import quartet_scores"
    ) == {"tropical", "tropical.quartet_scores"}
    assert _sandbox_modules_imported("import snakes_and_ladders.sandbox") == set()
    assert _sandbox_modules_imported("from snakes_and_ladders.search import maxflow") == set()
