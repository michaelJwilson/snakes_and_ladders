"""Every copy of the gate policy still says what `infra/gates.py` says it says.

Issue #469. The policy is spread over twelve files and had drifted: `DEV.md`
counted nine review rows after #456 demoted one. This reads each copy
`infra/gates.py` cannot generate (#470 generates the marker list and tier
table) --- a workflow condition, a shell export, a reviewer's sentence --- from
the file that carries it, never from the module, and parses scripts rather
than running them (`review_gates.sh` costs 30 s). A failure says the tree
disagrees with itself; the file that runs the gate wins.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

# `infra/` is on `mypy_path` and is how the guards already reach their shared
# names (see `infra/test_kinds.py`).
import catalogue
import gates
import pytest
import test_kinds
import yaml

from tests._paths import REPO_ROOT

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

#: The review table's size and the count following from it. The seconds are
#: a measurement and not read, or the budget would be what the host last managed.
_ROW_COUNT = re.compile(r"\*\*(\w+) rows\b")
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

    Comments are dropped first: `infra/release.sh` explains its command in prose.
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
        cells = catalogue.cells(line)
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
@pytest.mark.infra
def test_the_kind_module_names_the_same_markers() -> None:
    """Read from `infra/test_kinds.py`: the two axes, and the exempt directory."""
    assert tuple(gates.KIND_MARKERS) == test_kinds.KINDS
    assert set(gates.SCHEDULING_MARKERS) == set(test_kinds.SCHEDULING)
    assert tuple(gates.FINDING_MARKERS) == test_kinds.FINDINGS
    assert gates.KIND_EXEMPT_DIRECTORY == test_kinds.EXCLUDED_DIRECTORY


@pytest.mark.critical
@pytest.mark.infra
def test_the_workflow_runs_the_pull_request_and_main_gates() -> None:
    """Read from `.github/workflows/ci.yml`: the two gates and the job's cap.

    Issue #455: a PR runs `critical`; losing the condition adds 17 to 32 minutes.
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
@pytest.mark.infra
def test_the_release_gate_runs_every_tier() -> None:
    """Read from `infra/release.sh`: no marker filter, and the coverage floor."""
    (release,) = (gate for gate in gates.GATES if gate.name == "release")

    assert _selection(_pytest_command(RELEASE, "--cov-fail-under")) == (
        release.marker_expression,
        release.coverage_floor,
    )
    assert release.budget_seconds is None


@pytest.mark.critical
@pytest.mark.infra
def test_the_shell_scripts_export_the_caps() -> None:
    """Read from `infra/validate.sh` and `infra/review_gates.sh`: the two caps.

    `review_gates.sh` exports only the first: its row is the critical tier.
    """
    described = {cap.variable: cap.seconds for cap in gates.CAPS}
    validate = _exported_caps(VALIDATE)
    review = _exported_caps(REVIEW_GATES)

    assert validate == described
    assert review == {"SAL_DURATION_CAP": described["SAL_DURATION_CAP"]}


@pytest.mark.critical
@pytest.mark.infra
def test_the_review_gate_script_runs_the_rows() -> None:
    """Read from `infra/review_gates.sh`: the rows, in the order it runs them."""
    assert _review_rows(REVIEW_GATES) == gates.REVIEW_GATE.rows
    assert gates.REVIEW_GATE.script == str(REVIEW_GATES.relative_to(REPO_ROOT))


@pytest.mark.critical
@pytest.mark.infra
def test_dev_md_counts_the_review_gate_rows() -> None:
    """Read from `DEV.md`: the review table's size, its budget, and the remainder.

    Budget from its sentence, not the measured seconds beside it (issue #456).
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
@pytest.mark.infra
def test_the_derived_blocks_are_current() -> None:
    """Regenerating the marker list and the tier table rewrites neither file.

    Against the committed files, never against the generator restating itself.
    """
    stale = [
        block.path
        for block in gates.DERIVED
        if (REPO_ROOT / block.path).read_text() != gates.regenerated(block)
    ]

    assert stale == [], f"edited where read rather than where written: {stale}"


@pytest.mark.critical
@pytest.mark.infra
def test_the_writer_restores_a_block_that_drifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One value moved in each generated block, and the writer puts both back.

    Read back through `tomllib` and the table reader, so unparseable output fails.
    """
    edits = {
        "pyproject.toml": (
            "smoke: checked against itself",
            "smoke: checked somewhere",
        ),
        "DEV.md": ("**120 s per test**", "**121 s per test**"),
    }
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    for block in gates.DERIVED:
        source = (REPO_ROOT / block.path).read_text()
        carried, moved = edits[block.path]
        drifted = source.replace(carried, moved, 1)
        assert drifted != source, f"{block.path} no longer carries {carried!r}"
        (tmp_path / block.path).write_text(drifted)

    for block in gates.DERIVED:
        (tmp_path / block.path).write_text(gates.regenerated(block))

    assert (tmp_path / "pyproject.toml").read_text() == PYPROJECT.read_text()
    assert (tmp_path / "DEV.md").read_text() == DEV.read_text()
    assert _registered_markers(tmp_path / "pyproject.toml") == {
        **gates.KIND_MARKERS,
        **gates.SCHEDULING_MARKERS,
        **gates.SUBJECT_MARKERS,
        **gates.FINDING_MARKERS,
    }
    budgets = _tier_budgets(tmp_path / "DEV.md")
    assert _seconds(budgets["key"]) == gates.KEY_DURATION_CAP.seconds
    assert [gate.budget_seconds for gate in gates.GATES] == [
        _seconds(budgets["CI"]),
        _seconds(budgets["CI"]),
        _seconds(budgets["release"]),
    ]


@pytest.mark.critical
@pytest.mark.infra
def test_the_guard_fails_on_a_copy_that_drifted(tmp_path: Path) -> None:
    """The guard rejects what it exists to reject.

    Each parser gets one moved value (row, cap, marker) and must report it.
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
            "smoke: checked against itself", "smoke: checked somewhere"
        )
    )

    rows = _review_rows(without_a_row)
    assert "head carries the base" not in rows
    assert len(rows) == len(gates.REVIEW_GATE.rows) - 1
    assert _exported_caps(without_a_row) == {"SAL_DURATION_CAP": 11}
    assert _registered_markers(reworded)["smoke"] != gates.KIND_MARKERS["smoke"]


@pytest.mark.infra
def test_the_default_marker_filter_is_the_tier_the_main_gate_runs() -> None:
    """`addopts` deselects `release`, and says so with the main gate's words.

    A bare `pytest` in `infra/release.sh` would otherwise silently run one tier.
    """
    configuration = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    addopts = configuration["tool"]["pytest"]["ini_options"]["addopts"]

    assert _selection(addopts) == (gates.MAIN_GATE.marker_expression, None), addopts
