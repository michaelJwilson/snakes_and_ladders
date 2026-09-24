"""`ROADMAP.md` and `STATUS.md` plan the same work, so their
milestone names have to be the same names.

Issue #244. `STATUS.md` summarized Milestone 1.3 as not started while its
ledger recorded it Met at 0.981 coverage; Milestone 2.2 had no `STATUS.md`
section (#683, exposed by #804). Checked offline: the two name the same
milestones. Whether a claim is true of the code is left to review.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests._paths import REPO_ROOT

ROADMAP = REPO_ROOT / "ROADMAP.md"
STATUS = REPO_ROOT / "STATUS.md"

#: `Milestone 1.1`, `Milestone 2.4` -- the identifier the three share.
_MILESTONE = re.compile(r"Milestone (\d+\.\d+)")


def _milestones(path: Path) -> set[str]:
    """Every `N.M` milestone identifier ``path`` names."""
    return set(_MILESTONE.findall(path.read_text()))


def _bullets(path: Path) -> list[str]:
    """Top-level list items, with their continuation lines joined."""
    items: list[str] = []
    current: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith("- "):
            if current is not None:
                items.append(current)
            current = line[2:]
        elif current is not None and line.startswith("  "):
            current += " " + line.strip()
        elif not line.strip() or line.startswith("#"):
            if current is not None:
                items.append(current)
                current = None
    if current is not None:
        items.append(current)
    return items


@pytest.mark.critical
@pytest.mark.infra
def test_the_two_documents_name_the_same_milestones() -> None:
    """A milestone defined in one document and absent from the other is drift.

    Milestone 2.2 had no `STATUS.md` section until the third document went (#683, #804).
    """
    roadmap, status = _milestones(ROADMAP), _milestones(STATUS)

    assert status <= roadmap, (
        f"STATUS.md reports on {status - roadmap}, which ROADMAP.md does not define"
    )
    assert roadmap <= status, (
        f"ROADMAP.md defines {roadmap - status}, which STATUS.md does not "
        "report on -- a milestone nothing tracks"
    )


@pytest.mark.infra
def test_the_milestone_check_would_catch_a_document_that_drifted() -> None:
    """The comparison rejects the shape it exists to reject.

    The documentation index needed four hand repairs before this rule (#223).
    """
    assert _MILESTONE.findall("## Milestone 3.7 — Something New") == ["3.7"]
    assert set(_MILESTONE.findall("no milestones here")) == set()
    # A milestone one document defines and the other does not is the drift
    # the first test rejects, in both directions.
    assert not {"3.7"} <= {"1.1", "1.2"}
    assert not {"1.1", "3.7"} <= {"1.1"}


@pytest.mark.critical
@pytest.mark.infra
def test_no_tracked_file_points_a_reader_at_the_deleted_third_document() -> None:
    """`TICKETS.md` is gone (#804), so nothing may send a reader to it.

    History may keep it; elsewhere a mention must cite #804, explaining the deletion.
    """
    allowed = ("CHANGELOG.md", "changelog.d/", "docs/reviews/", "docs/experiments/")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = []
    for name in tracked:
        if name.startswith(allowed) or name.endswith((".pdf", ".png", ".so")):
            continue
        path = REPO_ROOT / name
        try:
            text = path.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if "TICKETS.md" in text and "#804" not in text:
            offenders.append(name)

    assert offenders == [], "these name `TICKETS.md`, which #804 deleted: " + ", ".join(
        offenders
    )
