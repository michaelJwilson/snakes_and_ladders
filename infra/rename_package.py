"""Rename the import package: ``python infra/rename_package.py OLD NEW``.

A rename touches every import in the tree, so it conflicts with every pull
request open beside it. Written as a script, the rename is re-run on whatever
`main` is when it lands, rather than resolved by hand (issue #1048). The run is
idempotent: a second run over its own output moves and rewrites nothing, and
``--check`` reports without writing.

It does four things, in order:

1. ``git mv python/OLD python/NEW``, then moves what git does not track --- a
   built extension --- beside it and drops ``__pycache__``, so no directory is
   left for Python to import as a namespace package.
2. Rewrites every tracked text file outside `KEEP_FILES`, line by line: the
   alias ``import OLD as NEW`` becomes ``import NEW``, then each ``OLD`` that
   stands as a name of its own (``OLD.x``, ``python/OLD/``, ``"OLD"``, and the
   TeX-escaped spelling) becomes ``NEW``. A name that merely contains ``OLD``
   (``run_OLD``, ``oxi_OLD``) is left, and so is a line `KEEP_LINES` matches
   and a span `KEEP_SPANS` matches.
3. Re-sorts the imports and re-formats the Python files it rewrote, because the
   new name sorts and wraps differently (``ruff check --select I --fix``, then
   ``ruff format``).
4. Prints the count per file and per category: ``dotted`` (``OLD.x``),
   ``path`` (``OLD/`` or ``/OLD``), ``name`` (anything else), ``alias`` and
   ``tex``.

The keep-list is this repository's: the records that cite a path as it was,
and the places where the name is the repository's or the distribution's and
not the package's. `tests/regression/test_duplication_guards.py` reads it
through `pending` and fails on any line still to rewrite.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatch

from _paths import REPO_ROOT

#: Files kept whole.
KEEP_FILES = (
    # A record of a path as it was when it was written.
    "CHANGELOG.md",  # released sections
    "docs/reviews/*",  # dated reviews
    "docs/experiments/*",  # dated ledger entries, each at its commit
    # Unreleased fragments whose subject is a name: the rename itself, and the
    # two earlier ones it composes with in the release notes.
    "changelog.d/250.changed.md",
    "changelog.d/1003.changed.md",
    "changelog.d/1048.changed.md",
    # The distribution keeps its name, and the lock records it.
    "uv.lock",
    # This script, whose keep-list names what it keeps.
    "infra/rename_package.py",
)

#: Lines kept, as ``(path glob, pattern)``; ``{old}`` is the escaped old name.
#: The name here is the repository's or the distribution's, not the package's.
KEEP_LINES = (
    ("pyproject.toml", r'^name = "{old}"$'),
    ("README.md", r"^# {old} "),
    ("DEV.md", r"^# Developing {old}$"),
    ("CLAUDE.md", r"^`{old}` is a high-performance scientific repository"),
    ("docs/source/conf.py", r'^project = "{old}"$|^"""Sphinx configuration for {old}'),
    ("docs/templates/work_in_flight.html", r"{old} · work in flight"),
    ("tests/regression/test_repository_links.py", r'^REPOSITORY = "{old}"$'),
    ("*", r"rename-package: keep"),
)

#: Spans kept wherever they fall: the repository's slug and a checkout of it.
KEEP_SPANS = (r"michaelJwilson/{old}", r"/home/[\w.-]+/{old}")

CATEGORIES = ("dotted", "path", "name", "alias", "tex")


@dataclass
class Report:
    """What a run moved and rewrote."""

    moved: list[str] = field(default_factory=list)
    counts: dict[str, Counter[str]] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


def _git(*args: str) -> str:
    """Run git in the repository and return its standard output."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _tex(name: str) -> str:
    """The TeX-escaped spelling, ``snake\\_case``."""
    return name.replace("_", r"\_")


def _kept_file(path: str) -> bool:
    return any(fnmatch(path, pattern) for pattern in KEEP_FILES)


def rewrite_line(path: str, line: str, old: str, new: str) -> tuple[str, Counter[str]]:
    """``line`` with the old name rewritten, and the count per category."""
    counts: Counter[str] = Counter()
    quoted = re.escape(old)
    for glob, pattern in KEEP_LINES:
        if fnmatch(path, glob) and re.search(pattern.format(old=quoted), line):
            return line, counts

    def outside(match: re.Match[str]) -> bool:
        # Read against the line the match was found in, not the original.
        kept = [
            found.span()
            for pattern in KEEP_SPANS
            for found in re.finditer(pattern.format(old=quoted), match.string)
        ]
        return not any(low <= match.start() < high for low, high in kept)

    def alias(match: re.Match[str]) -> str:
        if not outside(match):
            return match.group(0)
        counts["alias"] += 1
        return f"import {new}"

    line = re.sub(rf"\bimport {quoted} as {re.escape(new)}\b", alias, line)

    def name(match: re.Match[str]) -> str:
        if not outside(match):
            return match.group(0)
        text = match.string
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        category = (
            "dotted" if after == "." else "path" if "/" in (before, after) else "name"
        )
        counts[category] += 1
        return new

    line = re.sub(rf"(?<!\w){quoted}(?!\w)", name, line)

    def tex(match: re.Match[str]) -> str:
        if not outside(match):
            return match.group(0)
        counts["tex"] += 1
        return _tex(new)

    line = re.sub(rf"(?<!\w){re.escape(_tex(old))}(?!\w)", tex, line)
    return line, counts


def _candidates(old: str) -> list[str]:
    """Tracked text files naming ``old`` in either spelling, keep-list excluded."""
    listed = subprocess.run(
        ["git", "grep", "-I", "-l", "-F", "-e", old, "-e", _tex(old)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return [path for path in listed if not _kept_file(path)]


def _move(old: str, new: str, report: Report, *, write: bool) -> None:
    """Move ``python/old`` to ``python/new``, leaving no importable remnant."""
    source, target = REPO_ROOT / "python" / old, REPO_ROOT / "python" / new
    tracked = _git("ls-files", f"python/{old}").split()
    if tracked and write and target.exists():
        # `git mv` into an existing directory nests the package inside it.
        message = f"python/{new} exists; remove it before the move"
        raise SystemExit(message)
    if tracked:
        if _git("ls-files", f"python/{new}").split():
            message = f"python/{old} and python/{new} are both tracked"
            raise SystemExit(message)
        report.moved.append(f"git mv python/{old} python/{new} ({len(tracked)} files)")
        if write:
            _git("mv", f"python/{old}", f"python/{new}")
    if not source.exists() or not write:
        return
    for leftover in sorted(source.rglob("*"), reverse=True):
        relative = leftover.relative_to(source)
        if "__pycache__" in relative.parts:
            if leftover.is_dir():
                shutil.rmtree(leftover)
            else:
                leftover.unlink()
        elif leftover.is_file():
            destination = target / relative
            if destination.exists():
                message = f"untracked {leftover} would overwrite {destination}"
                raise SystemExit(message)
            destination.parent.mkdir(parents=True, exist_ok=True)
            leftover.rename(destination)
            report.moved.append(f"untracked python/{old}/{relative} -> python/{new}/")
        elif leftover.is_dir():
            leftover.rmdir()
    source.rmdir()


def run(old: str, new: str, *, write: bool) -> Report:
    """Move and rewrite; with ``write`` false, only report what would change."""
    report = Report()
    _move(old, new, report, write=write)
    for path in _candidates(old):
        # Before a dry-run move, a file still lives under the old directory.
        file = REPO_ROOT / path
        text = file.read_text(encoding="utf-8")
        total: Counter[str] = Counter()
        rewritten = []
        for number, line in enumerate(text.splitlines(keepends=True), 1):
            changed, counts = rewrite_line(path, line, old, new)
            if counts:
                report.lines.append(f"{path}:{number}: {line.rstrip()}")
            total += counts
            rewritten.append(changed)
        if total:
            report.counts[path.replace(f"python/{old}/", f"python/{new}/")] = total
            if write:
                file.write_text("".join(rewritten), encoding="utf-8")
    if write:
        _format(sorted(report.counts))
    return report


def _format(paths: list[str]) -> None:
    """Re-sort imports and re-format the rewritten Python files."""
    python = [path for path in paths if path.endswith((".py", ".pyi", ".ipynb"))]
    if not python:
        return
    ruff = [sys.executable, "-m", "ruff"]
    subprocess.run(
        [*ruff, "check", "--force-exclude", "--select", "I", "--fix", "-q", *python],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [*ruff, "format", "--force-exclude", "-q", *python], cwd=REPO_ROOT, check=True
    )


def pending(old: str, new: str) -> list[str]:
    """Every ``path:line: text`` a run would still rewrite, and a pending move."""
    report = run(old, new, write=False)
    return report.moved + report.lines


def _print(report: Report) -> None:
    for line in report.moved:
        print(line)
    totals: Counter[str] = Counter()
    for path, counts in sorted(report.counts.items()):
        totals += counts
        cells = " ".join(f"{name}={counts[name]}" for name in CATEGORIES)
        print(f"{path}: {cells}")
    cells = " ".join(f"{name}={totals[name]}" for name in CATEGORIES)
    print(f"total: {len(report.counts)} files, {cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", help="the package's current import name")
    parser.add_argument("new", help="its new import name")
    parser.add_argument(
        "--check", action="store_true", help="report, write nothing, fail if pending"
    )
    args = parser.parse_args(argv)
    report = run(args.old, args.new, write=not args.check)
    _print(report)
    return 1 if args.check and (report.moved or report.counts) else 0


if __name__ == "__main__":
    sys.exit(main())
