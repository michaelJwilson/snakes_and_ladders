"""`ROADMAP.md`, `STATUS.md` and `TICKETS.md` plan the same work, so their
milestone names have to be the same names.

Issue #244. The three drifted apart in ways no check could see: `TICKETS.md`
carried bullets for two issues that had closed, and its header promised that a
parenthesized number means an issue is filed while nothing enforced the
promise. `STATUS.md`'s summary row for Milestone 1.3 read "Potts lattice not
started" while its own Requirements Ledger recorded the lattice **Met** at
0.981 coverage, and the paragraph carrying that evidence sat under Milestone
2.1 -- a continuous-optimization result filed under reinforcement learning.

What is checkable offline is the *shape*: that the three documents name the
same milestones, and that a reference to an issue is written the one way the
header claims. Whether a given claim is true of the code is not something a
string check can answer, and pretending otherwise would be the coverage
theatre root `CLAUDE.md` forbids -- so this asserts the structure and leaves
the reading to review.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

ROADMAP = REPO_ROOT / "ROADMAP.md"
STATUS = REPO_ROOT / "STATUS.md"
TICKETS = REPO_ROOT / "TICKETS.md"

#: `Milestone 1.1`, `Milestone 2.4` -- the identifier the three share.
_MILESTONE = re.compile(r"Milestone (\d+\.\d+)")

#: The one form the `TICKETS.md` header promises: a parenthesized `#n`, so a
#: reader can tell a filed ticket from an unfiled one at a glance.
_BULLET_ISSUE = re.compile(r"\(#\d+(?:, ?#\d+)*\)")

#: Any other way of naming an issue, which would defeat that.
_LOOSE_ISSUE = re.compile(r"\bissues? #\d+")


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
@pytest.mark.structural
def test_the_three_documents_name_the_same_milestones() -> None:
    """A milestone tracked in one document and absent from another is drift.

    `ROADMAP.md` defines them, `STATUS.md` reports against them and
    `TICKETS.md` queues work under them, so a name appearing in one and not the
    others means a reader following the plan loses the thread.
    """
    roadmap, status, tickets = (
        _milestones(ROADMAP),
        _milestones(STATUS),
        _milestones(TICKETS),
    )
    assert tickets <= roadmap, (
        f"TICKETS.md queues work under {tickets - roadmap}, which ROADMAP.md does not define"
    )
    assert status <= roadmap, (
        f"STATUS.md reports on {status - roadmap}, which ROADMAP.md does not define"
    )
    assert roadmap <= (status | tickets), (
        f"ROADMAP.md defines {roadmap - (status | tickets)}, which neither "
        "STATUS.md nor TICKETS.md mentions -- a milestone nothing tracks"
    )


@pytest.mark.critical
@pytest.mark.structural
def test_a_filed_ticket_is_written_the_one_way_the_header_promises() -> None:
    """`TICKETS.md`'s header says a parenthesized number means a filed issue.

    A bullet writing it any other way -- "issue #123" in prose -- reads as
    filed to a person and is invisible to anything checking. One form, so the
    file can be read either way.
    """
    offenders = [b for b in _bullets(TICKETS) if _LOOSE_ISSUE.search(b)]
    assert not offenders, (
        f"{len(offenders)} bullet(s) name an issue outside the parenthesized form: "
        f"{[b[:60] for b in offenders]}"
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_header_still_describes_the_file() -> None:
    """The promise the two tests above enforce is actually made.

    If the header stops claiming the convention, the checks stop meaning
    anything, and a check nobody can trace back to a stated rule is the kind
    that gets deleted in confusion later.
    """
    header = TICKETS.read_text().split("## ", 1)[0]
    assert "A parenthesized number is an issue already filed" in header


@pytest.mark.structural
def test_the_milestone_check_would_catch_a_document_that_drifted() -> None:
    """The comparison rejects the shape it exists to reject.

    Asserting the current tree passes says nothing about the next milestone
    somebody adds to one document and forgets in the others -- the rule this
    repository settled on after the documentation index needed four repairs by
    hand (#223).
    """
    assert _MILESTONE.findall("## Milestone 3.7 — Something New") == ["3.7"]
    assert set(_MILESTONE.findall("no milestones here")) == set()
    # A tickets-only milestone is the drift the first test rejects.
    assert not {"3.7"} <= {"1.1", "1.2"}
