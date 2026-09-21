"""`ROADMAP.md` and `STATUS.md` plan the same work, so their
milestone names have to be the same names.

Issue #244. They drift in ways no check could see. `STATUS.md`'s summary row
for Milestone 1.3 read "Potts lattice not started" while its own Requirements
Ledger recorded the lattice **Met** at 0.981 coverage, and the paragraph
carrying that evidence sat under Milestone 2.1 -- a continuous-optimization
result filed under reinforcement learning. Milestone 2.2 was the same shape:
`ROADMAP.md` defined it, `TICKETS.md` queued work under it, and `STATUS.md`
had no section for it (#683), which is what deleting the third document
exposed (#804).

What is checkable offline is the *shape*: that the two documents name the
same milestones. Whether a given claim is true of the code is not something a
string check can answer, and pretending otherwise would be the coverage
theatre root `CLAUDE.md` forbids -- so this asserts the structure and leaves
the reading to review.
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

    `ROADMAP.md` defines them and `STATUS.md` reports against them, so a name
    in one and not the other means a reader following the plan loses the
    thread. `TICKETS.md` was the third until #804 deleted it, and its going
    tightened this check rather than loosening it: a milestone it alone
    tracked --- 2.2, curriculum learning --- was a milestone `STATUS.md` had
    no section for (#683), and now must have one.
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

    Asserting the current tree passes says nothing about the next milestone
    somebody adds to one document and forgets in the others -- the rule this
    repository settled on after the documentation index needed four repairs by
    hand (#223).
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

    History keeps its references: `CHANGELOG.md` and a `changelog.d/` fragment
    record what a release said, and a dated review or an experiment file
    records what was true on its date. Elsewhere a file may name it only while also
    citing #804, which is what makes the mention an explanation of the
    deletion rather than a pointer -- and a pointer at a file that does not
    exist is worse than no pointer, since it reads as though the work is
    written down somewhere.
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
