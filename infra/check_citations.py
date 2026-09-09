#!/usr/bin/env python
"""Citation integrity for the two documents: text and existence, no render.

Three claims, checked per document because the two build separately: a cited
figure exists on disk; a cited label is defined in that document's own input
closure; a ``\\cite`` key has an entry in ``docs/tex/references.bib``. A
fourth reads the bibliography itself --- an entry whose braces do not close
runs into the entry after it, which is how a merge dropped one and left the
survivor uncitable (issue #488).

``latexmk`` reports the first three in its log, and the ``documents`` job
greps for them, but only after a build: this runs before it, costs 55 ms
over both documents, and names the document, the line and the token rather
than a log line. A cross-document reference is the case it is here for --- a
``\\label`` in the textbook does not resolve a ``\\ref`` in the paper, and
nothing in either source says so (issue #249).

Usage: ``infra/check_citations.py [--document docs/tex/paper.tex ...]``;
default is both. Exits 1 if anything does not resolve, and lists every
finding in every document rather than stopping at the first.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX_DIR = REPO_ROOT / "docs" / "tex"
BIBLIOGRAPHY = "references.bib"
DEFAULT_DOCUMENTS = (TEX_DIR / "paper.tex", TEX_DIR / "textbook.tex")

#: Extensions ``\includegraphics`` resolves without being written down;
#: `graphicx` searches them in order and the documents commit PDFs.
GRAPHICS_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".eps")

#: A `%` opens a comment unless it is escaped. Stripping comments first keeps
#: a commented-out citation from being read as one.
COMMENT = re.compile(r"(?<!\\)%.*$")

INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REFERENCE = re.compile(r"\\(?:ref|eqref|autoref|[cC]ref|pageref|nameref)\{([^}]+)\}")
CITE = re.compile(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]+)\}")
BIB_ENTRY = re.compile(
    r"@(?!string\b|comment\b|preamble\b)[A-Za-z]+\s*\{\s*([^,\s}]+)\s*,"
)


def uncommented(text: str) -> str:
    """``text`` with LaTeX comments removed, its line numbering preserved."""
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def resolve(target: str, tex_dir: Path) -> Path:
    """The file ``\\input{target}`` reads, whether or not it exists."""
    path = tex_dir / target
    return path if path.suffix else path.with_suffix(".tex")


def closure(document: Path, tex_dir: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Every source ``document`` typesets, and the inputs that are missing.

    Returns the resolved files with their uncommented text, in the order the
    document reads them, and the ``\\input`` targets no file provides. The
    generated fragments are written by ``infra/build_documents.sh`` before it
    calls this, so a missing one is a finding rather than a file to skip.
    """
    seen: set[Path] = set()
    sources: list[tuple[Path, str]] = []
    missing: list[str] = []
    pending = [document]
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        seen.add(path)
        text = uncommented(path.read_text())
        sources.append((path, text))
        for target in INPUT.findall(text):
            nested = resolve(target, tex_dir)
            if nested.exists():
                pending.append(nested)
            elif target not in missing:
                missing.append(target)
    return sources, missing


def keys(argument: str) -> list[str]:
    """The comma-separated keys inside one ``\\ref`` or ``\\cite`` argument."""
    return [key.strip() for key in argument.split(",") if key.strip()]


def bibliography(tex_dir: Path) -> tuple[set[str], list[str]]:
    """The entry keys in the bibliography, and the entries that do not close.

    An entry whose closing brace is gone runs into the entries after it, so
    it stops resolving while the file still reads as text. The brace walk is
    what separates the two: an entry whose body reaches the next ``@`` is
    reported by name rather than going silently absent.
    """
    text = (tex_dir / BIBLIOGRAPHY).read_text()
    found: set[str] = set()
    unbalanced: list[str] = []
    starts = [(match.start(), match.group(1)) for match in BIB_ENTRY.finditer(text)]
    for index, (start, key) in enumerate(starts):
        opened = text.index("{", start)
        depth = 0
        end = len(text)
        for position in range(opened, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    end = position
                    break
        following = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        if depth != 0 or end > following:
            unbalanced.append(key)
        else:
            found.add(key)
    return found, unbalanced


def findings(document: Path, entries: set[str], tex_dir: Path) -> list[str]:
    """Every citation in ``document`` that does not resolve, as messages."""
    sources, missing_inputs = closure(document, tex_dir)
    out = [
        f"reads {target}, which no file provides "
        "(a generated fragment is written by infra/build_documents.sh)"
        for target in missing_inputs
    ]

    defined: set[str] = set()
    for _, text in sources:
        defined.update(LABEL.findall(text))

    for path, text in sources:
        where = path.name
        for number, line in enumerate(text.splitlines(), start=1):
            for target in GRAPHICS.findall(line):
                candidate = tex_dir / target
                if not candidate.exists() and not any(
                    candidate.with_suffix(suffix).exists()
                    for suffix in GRAPHICS_SUFFIXES
                ):
                    out.append(f"{where}:{number}: no figure file for {target}")
            for group in REFERENCE.findall(line):
                out += [
                    f"{where}:{number}: {label} is cited and this document defines no such label"
                    for label in keys(group)
                    if label not in defined
                ]
            for group in CITE.findall(line):
                out += [
                    f"{where}:{number}: {key} is cited and {BIBLIOGRAPHY} has no such entry"
                    for key in keys(group)
                    if key not in entries
                ]
    return out


def unresolved(documents: tuple[Path, ...], tex_dir: Path = TEX_DIR) -> list[str]:
    """Every finding over ``documents``, the bibliography's own first."""
    entries, unbalanced = bibliography(tex_dir)
    out = [
        f"{BIBLIOGRAPHY}: entry {key} does not close its braces, "
        "so it runs into the entries after it"
        for key in unbalanced
    ]
    for document in documents:
        out += [
            f"{document.name}: {message}"
            for message in findings(document, entries, tex_dir)
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Citation integrity for the documents."
    )
    parser.add_argument(
        "--document",
        action="append",
        type=Path,
        default=None,
        help="a document to check; repeatable, default both",
    )
    arguments = parser.parse_args(argv)
    documents = tuple(arguments.document) if arguments.document else DEFAULT_DOCUMENTS

    started = time.monotonic()
    out = unresolved(documents)
    elapsed = time.monotonic() - started
    scope = ", ".join(document.stem for document in documents)
    for message in out:
        print(message, file=sys.stderr)
    if out:
        print(f"citation integrity: {len(out)} unresolved in {scope} ({elapsed:.2f}s)")
        return 1
    print(f"citation integrity: {scope} resolve ({elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
