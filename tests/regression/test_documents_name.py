"""`docs/tex/` builds two documents, and no live file may call them one.

Issue #377. Since #249 the build produces `docs/paper.pdf` and
`docs/textbook.pdf`, and nothing else; 77 references across ~30 files still
named a single artifact that had not existed since that split -- a CI job, a
build script, a `CLAUDE.md` heading, and docstrings that told a reader to look
for a document the tree does not contain. Renaming them once fixes the tree
today; this fixes the next line somebody copies, which is the rule the
repository settled on for the link guard (#250) and applies here.

Two exclusions, both history rather than instruction: `changelog.d/` fragments
and `CHANGELOG.md` record what landed under the name it had, and `STATUS.md`'s
dated consistency audits quote the name they found. Neither tells a reader
what to run.

A match is looked for across one line break as well as within a line, because
the phrase is wrapped in prose more often than not.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The retired name in every spelling it was written in: prose, the build
#: script's filename, and the CI job's id. `\s` rather than a literal space, so
#: a phrase wrapped across two lines is caught too.
_OLD_NAME = re.compile(r"technical[\s_-]+doc", re.IGNORECASE)

#: Where the name could be written. Tracked text only, so build output and
#: scratch files cannot fail the check.
SUFFIXES = (".md", ".py", ".sh", ".yml", ".yaml", ".tex", ".rst", ".toml", ".txt")

#: History, kept as written: what landed under the old name, and the audits
#: that found it.
HISTORY = ("CHANGELOG.md",)
HISTORY_DIRECTORY = "changelog.d/"

#: A section heading opening a dated audit, whose lines quote what it found.
_AUDIT_HEADING = re.compile(r"^(#+)\s+Consistency audit", re.IGNORECASE)
_HEADING = re.compile(r"^(#+)\s")


def _tracked_text_files() -> list[Path]:
    """Every tracked file the name could be written in, history excluded."""
    listing = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        REPO_ROOT / name
        for name in listing.stdout.splitlines()
        if name.endswith(SUFFIXES)
        and name not in HISTORY
        and not name.startswith(HISTORY_DIRECTORY)
    ]


def _audit_lines(lines: list[str]) -> set[int]:
    """Zero-based indices of the lines inside a consistency-audit section."""
    inside: set[int] = set()
    level = 0
    for index, line in enumerate(lines):
        if (audit := _AUDIT_HEADING.match(line)) is not None:
            level = len(audit.group(1))
            inside.add(index)
        elif level:
            heading = _HEADING.match(line)
            if heading is not None and len(heading.group(1)) <= level:
                level = 0
            else:
                inside.add(index)
    return inside


def _offending_lines(text: str) -> list[int]:
    """The one-based lines of ``text`` that name the retired artifact."""
    lines = text.splitlines()
    excluded = _audit_lines(lines)
    offenders: list[int] = []
    for index, line in enumerate(lines):
        if index in excluded:
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        window = f"{line}\n{following}"
        if any(match.start() < len(line) for match in _OLD_NAME.finditer(window)):
            offenders.append(index + 1)
    return offenders


@pytest.mark.critical
@pytest.mark.structural
def test_no_live_file_names_a_single_technical_artifact() -> None:
    """Every live reference names the documents, the paper, or the textbook.

    A reader following the old name looks for a build script and a CI job
    that no longer exist, and for one PDF where two are built.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): lines
        for path in _tracked_text_files()
        if (
            lines := _offending_lines(
                path.read_text(encoding="utf-8", errors="replace")
            )
        )
    }
    assert not offenders, (
        f"{len(offenders)} file(s) name the retired artifact at these lines: "
        f"{offenders}. `docs/tex/` builds docs/paper.pdf and docs/textbook.pdf; "
        "name the documents, the paper, or the textbook, and the script "
        "infra/build_documents.sh."
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_guard_rejects_the_retired_name_and_keeps_the_audits() -> None:
    """The guard rejects what it exists to reject, in each spelling.

    The name is assembled from parts rather than written out: this file is
    itself scanned, so a literal would fail the check above -- the guard
    working, but on its own test.
    """
    retired = "technical"
    assert _offending_lines(f"the {retired} document is built") == [1]
    assert _offending_lines(f"run infra/build_{retired}_doc.sh") == [1]
    assert _offending_lines(f"| `{retired}-doc` | LaTeX build |") == [1]
    # Wrapped in prose, which is how most of #377's 77 references were written.
    assert _offending_lines(f"every figure in the {retired}\ndocument") == [1]
    # The current names pass, and so does an unrelated use of the adjective.
    assert _offending_lines("infra/build_documents.sh builds the two PDFs") == []
    assert _offending_lines(f"{retired} detail belongs outside a CLAUDE.md") == []
    # A dated audit quotes what it found, and is history.
    audit = f"## Consistency audit at 0.4.0\n\nthe {retired} document was stale\n"
    assert _offending_lines(audit) == []
    assert _offending_lines(f"{audit}\n## Next\n\nthe {retired} document") == [7]
