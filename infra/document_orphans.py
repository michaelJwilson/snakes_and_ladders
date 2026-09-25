#!/usr/bin/env python
"""Report the orphans the two documents carry (issues #492, #495).

An orphan is a thing one side of the repository pays for and the other side
never reads. Two kinds are visible from here, and neither is gated --- both
belong to tickets of their own, and what the suite asserts is that the
*detector* works, not that the set is empty:

* a figure the QA manifest renders, paid for at the release gate, and cited
  by no document (issue #492);
* a problem statement the textbook carries with no box in the release
  checklist, whose fix the release follow-up owns.

The third orphan this module used to report --- a textbook statement no
``PROBLEMS.md`` row is code for, and a row keyed to no statement --- is gone
with the join (issue #640). ``PROBLEMS.md``'s ``Statement`` column still
carries the textbook section's ``\\label`` and still points one way only,
because the root ``CLAUDE.md`` splits the documents so the textbook can state
an algorithm without naming any code; that constraint is asserted, on the
hand-written sources, by ``test_document_orphans.py``.

Infrastructure, not science: it reads a LaTeX label, a manifest and a YAML
form, and knows nothing about what a Potts lattice is. Run::

    python infra/document_orphans.py    # print what it sees
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _paths import REPO_ROOT

TEX_DIR = REPO_ROOT / "docs" / "tex"
TEXTBOOK = TEX_DIR / "textbook.tex"
DOCUMENTS = (TEX_DIR / "paper.tex", TEXTBOOK)
RELEASE_TEMPLATE = REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "release.yml"
MANIFEST = REPO_ROOT / "python" / "sal" / "qa" / "manifest.py"


#: The shape a problem statement has. Read rather than a hand-kept list, so a
#: statement added to the textbook joins without a second edit here, and read
#: from the document's structure rather than from a prefix in its heading: the
#: statements are the sections of the problems part that open on a model, and
#: the part's other sections --- the notation, the two framing sections and
#: the applicability table --- do not.
_PROBLEMS_PART = re.compile(
    r"\\part\{The Problems\}(.*?)\\part\{The Algorithms\}", re.DOTALL
)
_STATEMENT = re.compile(
    r"\\section\{([^}]*)\}\s*\n\\label\{(sec:[a-z0-9]+)\}"
    r"(?:(?!\\section\{).)*?\\subsection\{The model\}",
    re.DOTALL,
)
#: A figure the manifest declares, and how a document refers to one --- the
#: two spellings ``sal.qa.manifest`` itself reads.
_MANIFEST_STEM = re.compile(r'FigureSpec\(\s*"([A-Za-z0-9_]+)"')
_FIGURE_REFERENCE = re.compile(r"figures/([A-Za-z0-9_]+)")
_CAPTION_SUFFIX = "_caption"
#: How the release checklist spells one problem: a ``- label:`` under the
#: ``problem-statements`` block.
_CHECKBOX = re.compile(r"^\s*- label:\s*(.+?)\s*$")


def statements(textbook: Path = TEXTBOOK) -> dict[str, str]:
    """``key -> title`` for every problem statement the textbook carries.

    Parameters
    ----------
    textbook : Path
        The textbook source.

    Returns
    -------
    dict[str, str]
        Keyed by the section's ``\\label``, in the order they are stated.
    """
    part = _PROBLEMS_PART.search(textbook.read_text())
    if part is None:
        return {}
    return {key: title.strip() for title, key in _STATEMENT.findall(part.group(1))}


def rendered_figures(manifest: Path = MANIFEST) -> set[str]:
    """The stem of every figure the QA manifest renders.

    Read as text rather than imported: nothing in ``infra/`` holds an
    application reference (``infra/CLAUDE.md``), and the manifest is one.

    Returns
    -------
    set[str]
    """
    return set(_MANIFEST_STEM.findall(manifest.read_text()))


def cited_figures(documents: tuple[Path, ...] = DOCUMENTS) -> set[str]:
    """Every figure stem the documents refer to, by a figure or a caption.

    The same two routes ``sal.qa.manifest`` reads, spelled
    here as a regular expression for the reason above.

    Returns
    -------
    set[str]
    """
    found = set()
    for document in documents:
        for stem in _FIGURE_REFERENCE.findall(document.read_text()):
            found.add(stem.removesuffix(_CAPTION_SUFFIX))
    return found


def uncited_figures(
    manifest: Path = MANIFEST, documents: tuple[Path, ...] = DOCUMENTS
) -> list[str]:
    """Rendered figures no document cites, sorted.

    An orphan from the other end: the figure is rendered, its cost is paid at
    the release gate, and nothing reads it. Whether each should exist is
    issue #492's question, so this is reported and not asserted.

    Returns
    -------
    list[str]
    """
    return sorted(rendered_figures(manifest) - cited_figures(documents))


def checklist_labels(template: Path = RELEASE_TEMPLATE) -> list[str]:
    """The release template's per-problem checkbox labels, in order.

    Returns
    -------
    list[str]
        Empty when the template carries no ``problem-statements`` block.
    """
    text = template.read_text()
    start = text.find("id: problem-statements")
    if start < 0:
        return []
    following = text.find("\n  - type:", start)
    block = text[start : following if following > 0 else len(text)]
    return [
        match.group(1) for match in map(_CHECKBOX.match, block.splitlines()) if match
    ]


def unchecked_statements(
    textbook: Path = TEXTBOOK, template: Path = RELEASE_TEMPLATE
) -> list[str]:
    """Problem statements the release checklist has no box for, sorted.

    Matched on the statement's own title, case-folded, since the checklist
    spells each in the textbook's words. Reported and not asserted: adding a
    box is the release follow-up's, not this ticket's (issue #495).

    Returns
    -------
    list[str]
    """
    boxed = {label.casefold() for label in checklist_labels(template)}
    return sorted(
        title
        for title in statements(textbook).values()
        if title.casefold() not in boxed
    )


def render(textbook: Path = TEXTBOOK) -> str:
    """What the two detectors see, as text.

    Returns
    -------
    str
    """
    lines = [f"{len(statements(textbook))} problem statements.", "", "seen, not gated:"]
    lines += [
        f"  rendered and cited by nothing (#492): {stem}" for stem in uncited_figures()
    ]
    lines += [
        f"  no box in the release checklist: {title}"
        for title in unchecked_statements(textbook)
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Entry point.

    Returns
    -------
    int
        Always 0: neither orphan gates.
    """
    sys.stdout.write(render(TEXTBOOK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
