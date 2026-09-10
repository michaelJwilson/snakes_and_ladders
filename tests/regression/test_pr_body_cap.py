"""The 40-line body cap refuses what it says it refuses (issue #521).

Structural guards over ``infra/check_pr_body.py``: a body one line over the
cap is refused with its own count in the message, a body exactly at the cap
passes, and the copies of the number a reader and an author see --- ``DEV.md``
and the issue templates --- still say what the script enforces.

Each copy is read from the file that carries it rather than from the module
under test, which is what makes the assertion worth its milliseconds: a test
that imports a constant and compares it against itself passes forever.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import check_pr_body  # noqa: E402

CAP = check_pr_body.BODY_LINE_CAP
DEV = REPO_ROOT / "DEV.md"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TEMPLATES = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"

#: `40 content lines`, `40-line cap` -- the two ways the number is written in
#: prose. Either counts as the file stating it.
_STATED = re.compile(r"\b(\d+)(?:-line cap| content lines)")


def _body(content: int, headings: int = 0) -> str:
    """A body of ``content`` charged lines, padded with uncharged structure."""
    parts = ["## Section"] * headings + [f"line {i}" for i in range(content)]
    return "\n\n".join(parts) + "\n"


@pytest.mark.edge_case
def test_a_body_one_line_over_the_cap_is_refused_with_its_count() -> None:
    found = check_pr_body.problem(_body(CAP + 1))
    assert found, f"a body of {CAP + 1} content lines was accepted"
    assert str(CAP + 1) in found, f"the message does not name the count: {found}"
    assert str(CAP) in found, f"the message does not name the cap: {found}"


@pytest.mark.edge_case
def test_a_body_at_the_cap_passes() -> None:
    assert check_pr_body.problem(_body(CAP)) == ""


@pytest.mark.edge_case
def test_headings_and_blank_lines_are_not_charged() -> None:
    """The cap charges content, so structure cannot push a body over it."""
    assert check_pr_body.problem(_body(CAP, headings=12)) == ""


@pytest.mark.structural
def test_the_committed_template_is_inside_the_cap() -> None:
    """A pull request that starts from the template has room left to write in.

    The template's instructions are HTML comments, which the counter excuses:
    charged, they alone are 64 lines and every conforming pull request would
    be refused unwritten.
    """
    template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text()
    assert check_pr_body.problem(template) == ""


@pytest.mark.structural
def test_dev_md_states_the_cap_the_script_enforces() -> None:
    stated = {int(m) for m in _STATED.findall(DEV.read_text())}
    assert CAP in stated, f"DEV.md states {sorted(stated)}, not the cap {CAP}"


@pytest.mark.structural
@pytest.mark.parametrize("name", ["task.yml", "release.yml", "documents.yml"])
def test_every_issue_template_states_the_cap(name: str) -> None:
    stated = {int(m) for m in _STATED.findall((TEMPLATES / name).read_text())}
    assert CAP in stated, f"{name} states {sorted(stated)}, not the cap {CAP}"


@pytest.mark.structural
def test_the_workflow_runs_the_check_on_the_pull_request_body() -> None:
    """The cap is enforced where the payload is, not only stated in prose."""
    text = WORKFLOW.read_text()
    assert "infra/check_pr_body.py" in text
    assert "github.event.pull_request.body" in text
