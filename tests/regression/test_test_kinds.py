"""Every test says what it is checked against, and nothing checks that but this.

Issue #237. `select_tests.py` chooses by path, so kind is a second axis, and a
marker nobody applies rots. `critical` is independent: a test is critical and
an oracle test, never instead. The source is read rather than the collected
items, so a `conftest.py` cannot satisfy the guard invisibly.
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

    Tags, not a partition: a recovery test that also refuses bad input is both.
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

    The documentation index needed four hand repairs before this rule (#223).
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

    `--strict-markers` catches a misspelling; this, a name never enforced.
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

    `critical` is not a kind: critical tests are a strict subset of kind-marked ones.
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
    """A category nothing carries is a category that has stopped being applied."""
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

    Issue #729: a written `infra` beside a problem would count science as tree.
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

    A `bug` says what is wrong, the kind beside it what decided that (#729).
    """
    without_a_kind = [
        f"{path.relative_to(REPO_ROOT)}::{node.name}"
        for path in _marked_test_files()
        for node in functions_of(path)
        if markers(node) & set(FINDINGS) and not markers(node) & set(KINDS)
    ]

    assert without_a_kind == []
