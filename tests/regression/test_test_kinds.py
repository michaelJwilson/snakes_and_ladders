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

import tomllib
from pathlib import Path

import pytest

# The names live in `infra/test_kinds.py`, which the merge gate also reads:
# one definition, and `infra/` is already on `mypy_path`. See that module.
from test_kinds import (
    EXCLUDED_DIRECTORY,
    FINDINGS,
    KINDS,
    SCHEDULING,
    SUBJECTS,
    functions_of,
    markers,
)

from tests._paths import REPO_ROOT
from tests._problems import fixtures_named_in


def _marked_test_files() -> list[Path]:
    """Every test file the kind rule applies to."""
    return sorted(
        path
        for path in (REPO_ROOT / "tests").rglob("test_*.py")
        if EXCLUDED_DIRECTORY not in path.parts
    )


@pytest.mark.critical
@pytest.mark.infra
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
        for node in functions_of(path)
        if not markers(node) & set(KINDS)
    ]
    assert not unmarked, (
        f"{len(unmarked)} tests carry no kind marker: {unmarked[:10]}. "
        f"Every test outside tests/{EXCLUDED_DIRECTORY}/ takes at least one of "
        f"{KINDS}. A test that fits none of them is either a missing category "
        "or a test with nothing to assert, and both are findings."
    )


@pytest.mark.critical
@pytest.mark.infra
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
        "import pytest\n\n\n@pytest.mark.smoke\n"
        "def test_something() -> None:\n    assert True\n"
    )

    assert not markers(functions_of(unmarked)[0]) & set(KINDS)
    assert markers(functions_of(marked)[0]) & set(KINDS) == {"smoke"}


@pytest.mark.critical
@pytest.mark.infra
def test_the_registered_markers_are_these() -> None:
    """`pyproject.toml` and `KINDS` cannot drift apart.

    `--strict-markers` makes a typo fail collection rather than silently select
    nothing, which is the other half of the same protection: registration
    catches the misspelling, this catches a name registered and never enforced.
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pytest_config = config["tool"]["pytest"]["ini_options"]
    registered = {entry.split(":", 1)[0] for entry in pytest_config["markers"]}

    assert set(KINDS) | set(SCHEDULING) | set(SUBJECTS) | set(FINDINGS) == registered
    assert "--strict-markers" in pytest_config["addopts"]


@pytest.mark.critical
@pytest.mark.infra
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
        for node in functions_of(path)
        if "critical" in markers(node)
    ]
    assert critical, "nothing is marked critical, so the early gate selects nothing"
    for path, node in critical:
        assert markers(node) & set(KINDS), (
            f"{path.relative_to(REPO_ROOT)}::{node.name} gates early but does not "
            "say what it is checked against"
        )


@pytest.mark.critical
@pytest.mark.infra
@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_is_used(kind: str) -> None:
    """A category nothing carries is a category that has stopped being applied.

    The taxonomy was settled against a survey of the whole suite, so each name
    had tests when it was registered. This is what notices if one empties.
    """
    carriers = sum(
        1
        for path in _marked_test_files()
        for node in functions_of(path)
        if kind in markers(node)
    )
    assert carriers > 0, f"no test carries {kind!r}"


@pytest.mark.critical
@pytest.mark.infra
def test_a_written_infra_sits_in_a_module_naming_no_problem() -> None:
    """`infra` is a claim about the module, and a test may write it only there.

    Issue #729 made `infra` a kind an author writes on a test of the
    repository's own machinery. The collection hook adds the same marker where
    the problem scan finds nothing, and `test_problem_markers.py` holds that
    such a module imports no defining code; a written `infra` in a module that
    names a problem would count a test of the science as a test of the tree.
    """
    misplaced = [
        f"{path.relative_to(REPO_ROOT)}::{node.name}"
        for path in _marked_test_files()
        if fixtures_named_in(path)
        for node in functions_of(path)
        if "infra" in markers(node)
    ]

    assert misplaced == [], (
        f"{len(misplaced)} tests write `infra` in a module that names a problem: "
        f"{misplaced[:10]}. Such a test is `smoke`, or the kind its referee makes it."
    )


@pytest.mark.critical
@pytest.mark.infra
def test_a_finding_is_carried_beside_a_kind() -> None:
    """The second axis is never instead of the first.

    A test marked `bug` says what is wrong and not what decided that: the
    oracle, the planted truth or the contract is the kind beside it. Every
    finding marker is registered so `--strict-markers` accepts it; the audit
    issue #729 plans applies them, and this holds each carrier to a kind.
    """
    without_a_kind = [
        f"{path.relative_to(REPO_ROOT)}::{node.name}"
        for path in _marked_test_files()
        for node in functions_of(path)
        if markers(node) & set(FINDINGS) and not markers(node) & set(KINDS)
    ]

    assert without_a_kind == []
