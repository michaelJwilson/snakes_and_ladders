"""Write ``sal.<module>`` where prose says ``snakes_and_ladders.<module>`` (issue #301).

Prose only. The rule is mechanical so it can be re-run and checked:

* Markdown, LaTeX and notebook Markdown cells: every ``snakes_and_ladders.``
  followed by a module path becomes ``sal.``.
* Python: only double-backtick literals in docstrings and comments, since a
  Sphinx role (``:mod:`snakes_and_ladders.x```, ``automodule::``) must keep
  the full name to resolve, and an import statement is code.
* Never: file paths (``python/snakes_and_ladders/``), the repository URL,
  the crate ``oxi_snakes_and_ladders``, ``CHANGELOG.md`` (a release record),
  and this script.

``--check`` exits non-zero if any prose still carries the long form, which is
what the test runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LONG = re.compile(r"(?<![\w/\-])snakes_and_ladders\.(?=[a-z_]+)")
SKIP = {REPO_ROOT / "CHANGELOG.md", REPO_ROOT / "CHECKS.md", Path(__file__).resolve()}
PROSE_GLOBS = ("*.md", "docs/tex/*.tex", "changelog.d/*.md")
NOTEBOOK_GLOB = "docs/nb/*.ipynb"
PYTHON_ROOTS = ("python", "tests", "infra")


def _markdown_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in ("*.md", "*/CLAUDE.md", "python/snakes_and_ladders/*/CLAUDE.md"):
        files.update(REPO_ROOT.glob(pattern))
    files.update(REPO_ROOT.glob("docs/tex/*.tex"))
    files.update(REPO_ROOT.glob("changelog.d/*.md"))
    return sorted(path for path in files if path not in SKIP)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in PYTHON_ROOTS:
        files.extend(sorted((REPO_ROOT / root).rglob("*.py")))
    return [path for path in files if path not in SKIP]


def abbreviate_prose(text: str) -> str:
    """The long form to the short one, everywhere in prose."""
    return LONG.sub("sal.", text)


_LITERAL = re.compile(r"``snakes_and_ladders\.(?=[a-z_]+)")


def abbreviate_python(text: str) -> str:
    """Only double-backtick literals: roles and imports keep the full name."""
    return _LITERAL.sub("``sal.", text)


def abbreviate_notebook(text: str) -> str:
    """Only the Markdown cells of a notebook; code cells and outputs stay."""
    notebook = json.loads(text)
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            cell["source"] = [abbreviate_prose(line) for line in cell["source"]]
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def leftovers() -> list[str]:
    """Every prose occurrence of the long form still in the tree."""
    found: list[str] = []
    for path in _markdown_files():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if LONG.search(line):
                found.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    for path in _python_files():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _LITERAL.search(line):
                found.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    for path in sorted(REPO_ROOT.glob(NOTEBOOK_GLOB)):
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "markdown" and any(
                LONG.search(line) for line in cell["source"]
            ):
                found.append(f"{path.relative_to(REPO_ROOT)}:cell {index}")
    return found


def rewrite() -> int:
    """Apply the rule; return how many files changed."""
    changed = 0
    for path in _markdown_files():
        text = path.read_text()
        new = abbreviate_prose(text)
        if new != text:
            path.write_text(new)
            changed += 1
    for path in _python_files():
        text = path.read_text()
        new = abbreviate_python(text)
        if new != text:
            path.write_text(new)
            changed += 1
    for path in sorted(REPO_ROOT.glob(NOTEBOOK_GLOB)):
        text = path.read_text()
        new = abbreviate_notebook(text)
        if json.loads(new) != json.loads(text):
            path.write_text(new)
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="report leftovers, change nothing"
    )
    args = parser.parse_args(argv)
    if args.check:
        found = leftovers()
        for entry in found:
            print(entry)
        print(f"{len(found)} long references left in prose")
        return 1 if found else 0
    print(f"rewrote {rewrite()} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
