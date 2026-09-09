#!/usr/bin/env python
"""Join the textbook's problem statements to ``PROBLEMS.md``'s rows (issue #495).

Two descriptions of one set of problems, in two vocabularies, and until now
nothing held them to each other: the textbook states the model in its own
notation, the catalogue names the code, and either could gain or lose a
problem without the other noticing. They had already drifted to eleven
statements against fifteen rows.

The join is on one **problem key**, and it points one way. The key is the
textbook section's own ``\\label``; ``PROBLEMS.md``'s ``Statement`` column
carries it. Nothing travels the other way, because the root ``CLAUDE.md``
splits the documents so the textbook can state an algorithm without naming
any code that implements it --- a join that put a module path in a statement,
or a formulation in the catalogue, would dissolve the separation it is meant
to check.

Two claims, and they are separate failures:

* every statement is keyed by at least one row, or it states a model nothing
  in the tree is code for;
* every row's key names a statement that exists, or it is code for a problem
  the textbook never states.

A key with several rows is not a failure. It is a *variant* --- the same model
under another substitution matrix, another graph, another size, another
emission family --- and the multiplicity is reported rather than asserted,
since what makes it a decision is that a new row must choose an existing key
or write a new statement.

``--report`` prints two further findings the join can see and does not gate.
Both are orphans of the same shape from other ends, and both belong to
tickets of their own: a rendered figure no document cites (issue #492), and a
problem statement with no box in the release checklist, whose fix the release
follow-up owns.

Infrastructure, not science: it reads a Markdown column, a LaTeX label and a
YAML form, and knows nothing about what a Potts lattice is. Run::

    python infra/problem_join.py --check    # exit 1 on either orphan
    python infra/problem_join.py --report   # the join, and what it sees
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = REPO_ROOT / "PROBLEMS.md"
TEX_DIR = REPO_ROOT / "docs" / "tex"
TEXTBOOK = TEX_DIR / "textbook.tex"
DOCUMENTS = (TEX_DIR / "paper.tex", TEXTBOOK)
RELEASE_TEMPLATE = REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "release.yml"
MANIFEST = REPO_ROOT / "python" / "snakes_and_ladders" / "qa" / "manifest.py"

#: The heading a problem statement carries. Read rather than a hand-kept list,
#: so a statement added to the textbook joins without a second edit here.
_STATEMENT = re.compile(
    r"\\section\{Problem Statement:\s*([^}]*)\}\s*\n\\label\{(sec:[a-z0-9]+)\}"
)
#: The key a catalogue row carries, in its ``Statement`` cell. Unbackticked,
#: because every backticked name in the table is resolved as a symbol by
#: ``tests/regression/test_problems_catalogue.py`` and a label is not one.
_KEY = re.compile(r"^sec:[a-z0-9]+$")
#: A figure the manifest declares, and how a document refers to one --- the
#: two spellings ``snakes_and_ladders.qa.manifest`` itself reads.
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
    return {
        key: title.strip() for title, key in _STATEMENT.findall(textbook.read_text())
    }


def registry(catalogue: Path = CATALOGUE) -> list[tuple[str, str]]:
    """``(row title, key)`` per table row of the catalogue, in order.

    A row whose ``Statement`` cell is not a key yields the empty string, which
    is an unkeyed row and fails the join the same way a wrong key does.

    Parameters
    ----------
    catalogue : Path
        ``PROBLEMS.md``.

    Returns
    -------
    list[tuple[str, str]]
    """
    found: list[tuple[str, str]] = []
    for line in catalogue.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Problem") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        key = cells[1] if len(cells) > 1 else ""
        found.append((cells[0], key if _KEY.match(key) else ""))
    return found


def orphans(
    catalogue: Path = CATALOGUE, textbook: Path = TEXTBOOK
) -> tuple[list[str], list[str]]:
    """The two directions the join fails in.

    Returns
    -------
    tuple[list[str], list[str]]
        The keys of statements no row is code for, and the titles of rows
        whose key names no statement. Both sorted.
    """
    stated = statements(textbook)
    rows = registry(catalogue)
    keyed = {key for _, key in rows if key}
    return (
        sorted(set(stated) - keyed),
        sorted(title for title, key in rows if key not in stated),
    )


def variants(
    catalogue: Path = CATALOGUE, textbook: Path = TEXTBOOK
) -> dict[str, list[str]]:
    """``key -> row titles``, in the textbook's order then the table's.

    Returns
    -------
    dict[str, list[str]]
        Every statement, including one no row keys, so the join reads as a
        table of the same length as the textbook.
    """
    found: dict[str, list[str]] = {key: [] for key in statements(textbook)}
    for title, key in registry(catalogue):
        found.setdefault(key, []).append(title)
    return found


# --- what the join sees and does not gate -------------------------------------


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

    The same two routes ``snakes_and_ladders.qa.manifest`` reads, spelled
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


# --- rendering ----------------------------------------------------------------


def render(catalogue: Path = CATALOGUE, textbook: Path = TEXTBOOK) -> str:
    """The join, and the two findings it does not gate, as text.

    Returns
    -------
    str
    """
    stated = statements(textbook)
    grouped = variants(catalogue, textbook)
    missing_rows, missing_statements = orphans(catalogue, textbook)

    lines = [
        f"{len(stated)} problem statements, {len(registry(catalogue))} registry rows.",
        "",
        "statement                  rows  code for",
    ]
    for key, titles in grouped.items():
        head = f"{key:<26} {len(titles):>4}  "
        lines.append(head + (titles[0] if titles else "-- no row --"))
        lines.extend(" " * len(head) + title for title in titles[1:])

    lines += ["", "orphans (these fail --check):"]
    lines += [f"  statement with no row: {key}" for key in missing_rows]
    lines += [f"  row with no statement: {title}" for title in missing_statements]
    if not missing_rows and not missing_statements:
        lines.append("  none")

    lines += ["", "seen, not gated:"]
    lines += [
        f"  rendered and cited by nothing (#492): {stem}" for stem in uncited_figures()
    ]
    lines += [
        f"  no box in the release checklist: {title}"
        for title in unchecked_statements(textbook)
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns
    -------
    int
        1 when either orphan direction is non-empty under ``--check``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 on an orphan")
    parser.add_argument("--report", action="store_true", help="print the join")
    arguments = parser.parse_args(argv)

    # The module globals are read here rather than through the defaults, so a
    # test can point the entry point at a tree of its own.
    if arguments.report or not arguments.check:
        sys.stdout.write(render(CATALOGUE, TEXTBOOK))
    if not arguments.check:
        return 0

    missing_rows, missing_statements = orphans(CATALOGUE, TEXTBOOK)
    for key in missing_rows:
        print(f"no PROBLEMS.md row is keyed to {key}", file=sys.stderr)
    for title in missing_statements:
        print(f"PROBLEMS.md row {title!r} names no problem statement", file=sys.stderr)
    return 1 if missing_rows or missing_statements else 0


if __name__ == "__main__":
    raise SystemExit(main())
