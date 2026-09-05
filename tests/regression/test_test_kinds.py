"""Every test says what it is checked against, and nothing checks that but this.

Issue #237. A suite of 855 assertions had no axis for *kind*: `select_tests.py`
chooses by module path, so there was no way to ask for "the checks that must
never break" independently of where a diff landed. The kind markers add that
axis, and a marker nobody applies is a marker that rots -- so the rule is
enforced here rather than asked for in a document.

Two axes, and they are independent. A *kind* says what a test is checked
against; `critical` says whether it gates early. A test is critical *and* an
oracle test, never instead of one.

The check reads the source rather than pytest's collected items: a decorator is
what a reviewer sees in the diff, and reading the tree means the guard cannot be
satisfied by a `conftest.py` applying markers invisibly.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: What a test is checked against. Exactly one home for the names: they are
#: registered in `pyproject.toml`, and `test_the_registered_markers_are_these`
#: asserts this tuple and that registration agree.
KINDS = (
    "oracle",
    "simulated_truth",
    "mathematical",
    "edge_case",
    "structural",
)

#: The second axis. Not a kind: it says when a test runs, not what it checks.
SCHEDULING = ("critical", "release")

#: Benchmarks measure rather than assert, so they carry no kind. Excluded here
#: rather than exempted case by case, because the exclusion is a property of the
#: directory and not of any test in it.
EXCLUDED_DIRECTORY = "benchmarks"


def _test_functions(path: Path) -> list[ast.FunctionDef]:
    """Every top-level ``test_`` function in one file."""
    tree = ast.parse(path.read_text())
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]


def _markers(node: ast.FunctionDef) -> set[str]:
    """The ``pytest.mark.<name>`` markers decorating ``node``."""
    found: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
        ):
            found.add(target.attr)
    return found


def _marked_test_files() -> list[Path]:
    """Every test file the kind rule applies to."""
    return sorted(
        path
        for path in (REPO_ROOT / "tests").rglob("test_*.py")
        if EXCLUDED_DIRECTORY not in path.parts
    )


@pytest.mark.critical
@pytest.mark.structural
def test_every_test_says_what_it_is_checked_against() -> None:
    """The rule itself: at least one kind, never exactly one.

    "At least" rather than "exactly": a recovery test that checks a fitted
    parameter against simulated truth *and* refuses a bad input is both, and
    splitting it would mean writing two worse tests. The scheme is tags, and a
    partition would force a choice the suite has no basis for.
    """
    unmarked = [
        f"{path.relative_to(REPO_ROOT)}::{node.name}"
        for path in _marked_test_files()
        for node in _test_functions(path)
        if not _markers(node) & set(KINDS)
    ]
    assert not unmarked, (
        f"{len(unmarked)} tests carry no kind marker: {unmarked[:10]}. "
        f"Every test outside tests/{EXCLUDED_DIRECTORY}/ takes at least one of "
        f"{KINDS}. A test that fits none of them is either a missing category "
        "or a test with nothing to assert, and both are findings."
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_guard_fails_on_an_unmarked_test(tmp_path: Path) -> None:
    """The guard rejects what it exists to reject.

    A guard that only passes on the current tree says nothing about the next
    module -- the rule this repository settled on after the documentation index
    needed four repairs by hand before a test closed it (#223).
    """
    unmarked = tmp_path / "test_unmarked.py"
    unmarked.write_text("def test_nothing() -> None:\n    assert True\n")
    marked = tmp_path / "test_marked.py"
    marked.write_text(
        "import pytest\n\n\n@pytest.mark.structural\n"
        "def test_something() -> None:\n    assert True\n"
    )

    assert not _markers(_test_functions(unmarked)[0]) & set(KINDS)
    assert _markers(_test_functions(marked)[0]) & set(KINDS) == {"structural"}


@pytest.mark.critical
@pytest.mark.structural
def test_the_registered_markers_are_these() -> None:
    """`pyproject.toml` and `KINDS` cannot drift apart.

    `--strict-markers` makes a typo fail collection rather than silently select
    nothing, which is the other half of the same protection: registration
    catches the misspelling, this catches a name registered and never enforced.
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pytest_config = config["tool"]["pytest"]["ini_options"]
    registered = {entry.split(":", 1)[0] for entry in pytest_config["markers"]}

    assert set(KINDS) | set(SCHEDULING) == registered
    assert "--strict-markers" in pytest_config["addopts"]


@pytest.mark.critical
@pytest.mark.structural
def test_critical_is_a_second_axis_and_not_a_kind() -> None:
    """Every critical test also says what it is checked against.

    If `critical` were a kind, a test would have to choose between saying it
    gates early and saying what it checks. It is not, so the set of critical
    tests is a strict subset of the kind-marked ones and carries no test of its
    own.
    """
    critical = [
        (path, node)
        for path in _marked_test_files()
        for node in _test_functions(path)
        if "critical" in _markers(node)
    ]
    assert critical, "nothing is marked critical, so the early gate selects nothing"
    for path, node in critical:
        assert _markers(node) & set(KINDS), (
            f"{path.relative_to(REPO_ROOT)}::{node.name} gates early but does not "
            "say what it is checked against"
        )


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_is_used(kind: str) -> None:
    """A category nothing carries is a category that has stopped being applied.

    The taxonomy was settled against a survey of the whole suite, so each name
    had tests when it was registered. This is what notices if one empties.
    """
    carriers = sum(
        1
        for path in _marked_test_files()
        for node in _test_functions(path)
        if kind in _markers(node)
    )
    assert carriers > 0, f"no test carries {kind!r}"
