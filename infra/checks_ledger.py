"""Write CHECKS.md: every check the roadmap's claims rest on, read from the tests.

`STATUS.md` says what has landed and `TICKETS.md` what has not; neither is the
place for the checks themselves, and a hand-kept list of tests drifts the
moment one is renamed (issue #291). The checks significant to the roadmap are
exactly the tests marked `oracle` -- pinned to an independent exact answer --
and `simulated_truth` -- pinned to the parameters that generated the data --
under issue #237's scheme, so this reads them from the tree and writes the
ledger. The file is not committed (issue #425); `infra/ledgers.sh` writes it
at the release gate and in CI, where regenerating it must change no tracked
file.

A row is the test's file, its name, its kind, and its claim: the first line of
its docstring, else its first leading comment. Run::

    uv run python infra/checks_ledger.py --write   # regenerate
    uv run python infra/checks_ledger.py --check   # exit 1 if the tree's copy is stale
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = REPO_ROOT / "tests" / "regression"
LEDGER = REPO_ROOT / "CHECKS.md"
SIGNIFICANT = ("oracle", "simulated_truth")

PREAMBLE = """# Checks

The checks the roadmap's claims rest on, read from the regression suite by
`infra/checks_ledger.py`: a test marked `oracle` is pinned to an independent
exact answer, one marked `simulated_truth` to the parameters that generated
its data (`DEV.md`, issue #237). `STATUS.md` says what each milestone claims;
this is what pins it. Generated and not committed (issue #425) -- write it
with `infra/ledgers.sh`, and do not edit it by hand.
"""


def _markers(node: ast.FunctionDef) -> set[str]:
    found: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
        ):
            found.add(target.attr)
    return found


def _claim(node: ast.FunctionDef, lines: list[str]) -> str:
    """The docstring's first sentence, else the leading comment's first sentence."""
    docstring = ast.get_docstring(node)
    if docstring:
        text = " ".join(docstring.strip().split())
    else:
        comment: list[str] = []
        for line in lines[node.lineno : node.end_lineno or node.lineno]:
            stripped = line.strip()
            if stripped.startswith("#"):
                comment.append(stripped.lstrip("#").strip())
            elif comment or (stripped and not stripped.startswith(("def ", "@"))):
                break
        text = " ".join(comment)
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return sentence.strip()


def rows() -> list[tuple[str, str, str, str]]:
    """``(file, test, kind, claim)`` for every significant test, sorted."""
    found: list[tuple[str, str, str, str]] = []
    for path in sorted(TESTS.rglob("test_*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in tree.body:
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ):
                continue
            kinds = sorted(_markers(node) & set(SIGNIFICANT))
            if not kinds:
                continue
            found.append(
                (
                    str(path.relative_to(REPO_ROOT)),
                    node.name,
                    ", ".join(kinds),
                    _claim(node, lines).replace("|", "\\|"),
                )
            )
    return found


def render(entries: list[tuple[str, str, str, str]]) -> str:
    """The ledger as Markdown, one section per test directory."""
    out = [PREAMBLE]
    by_directory: dict[str, list[tuple[str, str, str, str]]] = {}
    for entry in entries:
        by_directory.setdefault(str(Path(entry[0]).parent), []).append(entry)
    for directory in sorted(by_directory):
        block = by_directory[directory]
        out.append(f"\n## `{directory}/` ({len(block)})\n")
        out.append("| Test | Kind | Claim |")
        out.append("| --- | --- | --- |")
        for file, name, kind, claim in block:
            out.append(f"| `{Path(file).name}::{name}` | {kind} | {claim} |")
    out.append(f"\n{len(entries)} checks.\n")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate CHECKS.md")
    mode.add_argument(
        "--check", action="store_true", help="exit 1 if CHECKS.md is stale"
    )
    arguments = parser.parse_args(argv)
    text = render(rows())
    if arguments.write:
        LEDGER.write_text(text)
        print(f"wrote {LEDGER}")
        return 0
    if not LEDGER.exists() or LEDGER.read_text() != text:
        print(
            "CHECKS.md is stale; regenerate with: uv run python infra/checks_ledger.py --write",
            file=sys.stderr,
        )
        return 1
    print("CHECKS.md is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
