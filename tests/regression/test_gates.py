"""Every copy of the gate policy still says what `infra/gates.py` says it says.

Issue #469. The policy -- which tests a gate selects, what it may cost, what
the markers mean -- is spread over twelve files, and they had already drifted:
`DEV.md` described `infra/review_gates.sh` as nine rows after issue #456
demoted one of them, and no check could see it. `infra/gates.py` writes the
policy down once; this reads the seven of those files that state a value it
carries -- `pyproject.toml`, `.github/workflows/ci.yml`, `infra/validate.sh`,
`infra/review_gates.sh`, `infra/release.sh`, `infra/test_kinds.py`, `DEV.md` --
back out of its own source text and asserts it matches.

Two rules make that worth its seconds. **Each copy is read from the file that
carries it**, never from the module under test: a test that imports a constant
and asserts it equals that constant passes forever, which is the coverage
theatre root `CLAUDE.md` forbids. And **a shell script or a workflow is
parsed, never executed**: the assertion is about the text a reader and a runner
both see, and running `review_gates.sh` to count its rows would cost 30 s to
learn something a regex has in a millisecond.

`infra/gates.py` is descriptive at this step, so a failure here does not say
which side is wrong -- only that the tree no longer agrees with itself. The
file that runs the gate wins; the module is corrected to match it.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# `infra/` is on `mypy_path` and is how the guards already reach their shared
# names (see `infra/test_kinds.py`).
sys.path.insert(0, str(REPO_ROOT / "infra"))

import gates  # noqa: E402
import test_kinds  # noqa: E402

PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
VALIDATE = REPO_ROOT / "infra" / "validate.sh"
REVIEW_GATES = REPO_ROOT / "infra" / "review_gates.sh"
RELEASE = REPO_ROOT / "infra" / "release.sh"
DEV = REPO_ROOT / "DEV.md"

#: `gate "critical tier passes"    critical_tier_passes` -- one row of the
#: review table, named as the reviewer reads it.
_ROW = re.compile(r'^gate\s+"([^"]+)"', re.MULTILINE)

#: `export SAL_DURATION_CAP="${SAL_DURATION_CAP:-10}"` -- a cap and the default
#: it takes when the caller sets nothing.
_CAP = re.compile(r'^export (SAL_\w+)="\$\{\1:-(\d+)\}"', re.MULTILINE)

#: `-m critical`, `-m "not release"` -- what a `pytest` invocation selects.
_MARKER_EXPRESSION = re.compile(r'-m\s+(?:"([^"]+)"|([^\s"$]+))')

#: `--cov-fail-under=90`.
_COVERAGE_FLOOR = re.compile(r"--cov-fail-under=(\d+)")

#: `**Eight rows in at most 39 s**` and `the other six rows`: the review
#: table's size, and the count that has to follow from it. The seconds in the
#: first are a measurement, not the budget, and the two coincided at the
#: nine-row reading -- which is why the budget is read from its own sentence.
_ROW_COUNT = re.compile(r"\*\*(\w+) rows in (?:at most )?\d+ s\*\*")
_REMAINING_ROWS = re.compile(r"the other (\w+) rows report")

#: `The budget is 30 s, above which a check belongs in CI`.
_REVIEW_BUDGET = re.compile(r"The budget is (\d+) s")

#: `DEV.md` writes a small count as a word, so the assertion has to read one.
_WORDS = {
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _registered_markers(path: Path) -> dict[str, str]:
    """The markers `pyproject.toml` registers, name to text."""
    config = tomllib.loads(path.read_text())
    entries: list[str] = config["tool"]["pytest"]["ini_options"]["markers"]
    registered: dict[str, str] = {}
    for entry in entries:
        name, text = entry.split(": ", 1)
        registered[name] = text
    return registered


def _exported_caps(path: Path) -> dict[str, int]:
    """The duration caps a shell script exports, variable to default seconds."""
    return {name: int(seconds) for name, seconds in _CAP.findall(path.read_text())}


def _review_rows(path: Path) -> tuple[str, ...]:
    """The rows `infra/review_gates.sh` runs, in order."""
    return tuple(_ROW.findall(path.read_text()))


def _pytest_command(path: Path, marker: str) -> str:
    """The one `uv run pytest` line in a shell script whose text holds ``marker``.

    Comments are dropped first. `infra/release.sh` explains in prose why it
    runs the unfiltered suite rather than `-m release`, and a search over the
    raw text would read the explanation as the command.
    """
    lines = [
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
        and "uv run pytest" in line
        and marker in line
    ]
    assert len(lines) == 1, (
        f"{path.name} has {len(lines)} pytest lines holding {marker!r}"
    )
    return lines[0]


def _selection(command: str) -> tuple[str | None, int | None]:
    """The marker expression and coverage floor a `pytest` invocation applies."""
    found = _MARKER_EXPRESSION.search(command)
    expression = (found.group(1) or found.group(2)) if found else None
    floor = _COVERAGE_FLOOR.search(command)
    return expression, int(floor.group(1)) if floor else None


def _workflow_steps(path: Path, job: str) -> dict[str, dict[str, object]]:
    """One job's steps, by step name."""
    workflow = yaml.safe_load(path.read_text())
    steps = workflow["jobs"][job]["steps"]
    return {step["name"]: step for step in steps if "name" in step}


def _tier_budgets(path: Path) -> dict[str, str]:
    """`DEV.md`'s tier table, tier name to the budget cell's text."""
    budgets: dict[str, str] = {}
    rows = iter(path.read_text().splitlines())
    for line in rows:
        if line.strip().startswith("| tier | marker |"):
            break
    else:  # pragma: no cover - the table is present or every test below fails
        pytest.fail("DEV.md carries no tier table")
    next(rows)  # the `| --- |` separator
    for line in rows:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            return budgets
        budgets[cells[0]] = cells[3]
    return budgets


def _seconds(budget: str) -> int | None:
    """A budget cell as seconds. `None` where the cell says unbounded."""
    if "unbounded" in budget:
        return None
    found = re.search(r"\*\*(\d+) (s|minutes?)\b", budget)
    assert found, f"unreadable budget cell: {budget!r}"
    return int(found.group(1)) * (60 if found.group(2).startswith("minute") else 1)


@pytest.mark.critical
@pytest.mark.structural
def test_pyproject_registers_the_markers_and_their_text() -> None:
    """Read from `pyproject.toml`: the nine markers and what each one claims.

    The text as well as the names, because the text is where a marker's meaning
    is written for anyone applying one, and `tests/regression/test_test_kinds.py`
    already holds the names against `infra/test_kinds.py`.
    """
    registered = _registered_markers(PYPROJECT)
    described = {**gates.KIND_MARKERS, **gates.SCHEDULING_MARKERS}

    assert registered == described


@pytest.mark.critical
@pytest.mark.structural
def test_the_kind_module_names_the_same_markers() -> None:
    """Read from `infra/test_kinds.py`: the two axes, and the exempt directory."""
    assert tuple(gates.KIND_MARKERS) == test_kinds.KINDS
    assert set(gates.SCHEDULING_MARKERS) == set(test_kinds.SCHEDULING)
    assert gates.KIND_EXEMPT_DIRECTORY == test_kinds.EXCLUDED_DIRECTORY


@pytest.mark.critical
@pytest.mark.structural
def test_the_workflow_runs_the_pull_request_and_main_gates() -> None:
    """Read from `.github/workflows/ci.yml`: the two gates and the job's cap.

    The condition on the second step is the whole of issue #455's trade: a pull
    request is judged by `critical`, and the selected tier runs on the push to
    `main`. A step that lost that condition would put 17 to 32 minutes back on
    every pull request, which is the drift this row exists to catch.
    """
    steps = _workflow_steps(WORKFLOW, "python-tests")
    pr, main = (gate for gate in gates.GATES if gate.name in {"pr", "main"})

    assert _selection(str(steps["Critical gate"]["run"])) == (
        pr.marker_expression,
        pr.coverage_floor,
    )
    assert "if" not in steps["Critical gate"]
    assert _selection(str(steps["Run the selected tests"]["run"])) == (
        main.marker_expression,
        main.coverage_floor,
    )
    assert (
        steps["Run the selected tests"]["if"] == "github.event_name != 'pull_request'"
    )

    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert workflow["jobs"]["python-tests"]["timeout-minutes"] == (
        gates.WORKFLOW_TIMEOUT_MINUTES
    )
    assert pr.runs_in == main.runs_in == str(WORKFLOW.relative_to(REPO_ROOT))


@pytest.mark.critical
@pytest.mark.structural
def test_the_release_gate_runs_every_tier() -> None:
    """Read from `infra/release.sh`: no marker filter, and the coverage floor."""
    (release,) = (gate for gate in gates.GATES if gate.name == "release")

    assert _selection(_pytest_command(RELEASE, "--cov-fail-under")) == (
        release.marker_expression,
        release.coverage_floor,
    )
    assert release.budget_seconds is None


@pytest.mark.critical
@pytest.mark.structural
def test_the_shell_scripts_export_the_caps() -> None:
    """Read from `infra/validate.sh` and `infra/review_gates.sh`: the two caps.

    `infra/validate.sh` exports both; `infra/review_gates.sh` exports the first
    alone, because the row it caps is the critical tier and no key test is in
    it. Both files are read, so a default changed in one and not the other is a
    failure rather than a silent difference between two local runs.
    """
    described = {cap.variable: cap.seconds for cap in gates.CAPS}
    validate = _exported_caps(VALIDATE)
    review = _exported_caps(REVIEW_GATES)

    assert validate == described
    assert review == {"SAL_DURATION_CAP": described["SAL_DURATION_CAP"]}


@pytest.mark.critical
@pytest.mark.structural
def test_the_review_gate_script_runs_the_rows() -> None:
    """Read from `infra/review_gates.sh`: the rows, in the order it runs them."""
    assert _review_rows(REVIEW_GATES) == gates.REVIEW_GATE.rows
    assert gates.REVIEW_GATE.script == str(REVIEW_GATES.relative_to(REPO_ROOT))


@pytest.mark.critical
@pytest.mark.structural
def test_dev_md_counts_the_review_gate_rows() -> None:
    """Read from `DEV.md`: the review table's size, its budget, and the remainder.

    The row this test was written for. `DEV.md` said nine rows after issue #456
    demoted the figure-stamp row to a function the script keeps and does not
    run, and the sentence's own arithmetic -- two rows named, the rest counted
    -- had drifted with it. The budget is read from the sentence that states
    it rather than from the measurement beside it: the two were the same
    number at nine rows, and reading the measurement would have made the
    budget whatever the host last managed.
    """
    text = DEV.read_text()
    stated = _ROW_COUNT.search(text)
    assert stated, "DEV.md no longer states the review table's size"
    remaining = _REMAINING_ROWS.search(text)
    assert remaining, "DEV.md no longer counts the rows outside the two it names"
    budget = _REVIEW_BUDGET.search(text)
    assert budget, "DEV.md no longer states the review table's budget"

    assert _WORDS[stated.group(1).lower()] == len(gates.REVIEW_GATE.rows)
    assert _WORDS[remaining.group(1).lower()] == len(gates.REVIEW_GATE.rows) - 2
    assert int(budget.group(1)) == gates.REVIEW_GATE.budget_seconds


@pytest.mark.critical
@pytest.mark.structural
def test_dev_md_states_the_tier_budgets() -> None:
    """Read from `DEV.md`: the tier table's budget column.

    The CI tier's budget is the budget both CI gates are held to, and the key
    tier's is `SAL_KEY_DURATION_CAP` written in a second place.
    """
    budgets = _tier_budgets(DEV)
    caps = {cap.variable: cap.seconds for cap in gates.CAPS}
    ci, key, release = (
        _seconds(budgets["CI"]),
        _seconds(budgets["key"]),
        _seconds(budgets["release"]),
    )

    assert [gate.budget_seconds for gate in gates.GATES] == [ci, ci, release]
    assert key == caps["SAL_KEY_DURATION_CAP"]
    assert "`SAL_KEY_DURATION_CAP`" in budgets["key"]


@pytest.mark.critical
@pytest.mark.structural
def test_the_guard_fails_on_a_copy_that_drifted(tmp_path: Path) -> None:
    """The guard rejects what it exists to reject.

    A drift guard that has never fired has not been tested, and this one exists
    only to fire. Each parser is given the source text with one value moved --
    a row deleted, a cap raised, a marker reworded -- and must report the moved
    value, which is what makes every assertion above a comparison rather than a
    restatement.
    """
    without_a_row = tmp_path / "review_gates.sh"
    without_a_row.write_text(
        REVIEW_GATES.read_text()
        .replace('gate "head carries the base"', ": # demoted")
        .replace("SAL_DURATION_CAP:-10", "SAL_DURATION_CAP:-11")
    )
    reworded = tmp_path / "pyproject.toml"
    reworded.write_text(
        PYPROJECT.read_text().replace(
            "edge_case: checked at a boundary", "edge_case: checked somewhere"
        )
    )

    rows = _review_rows(without_a_row)
    assert "head carries the base" not in rows
    assert len(rows) == len(gates.REVIEW_GATE.rows) - 1
    assert _exported_caps(without_a_row) == {"SAL_DURATION_CAP": 11}
    assert _registered_markers(reworded)["edge_case"] != gates.KIND_MARKERS["edge_case"]
