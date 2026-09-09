"""The tests a branch touched carry a kind marker, and which files they are in.

`infra/review_gates.sh` calls this. One of the rows a reviewer read by hand is
that each test the branch adds or rewrites says what it is checked against
(root `CLAUDE.md`, Every Test Says What It Is Checked Against), which is a fact
about the source and is decided here.

Only the functions the diff adds or rewrites are judged. The repository-wide
guard in ``tests/regression/test_test_kinds.py`` covers the rest and runs in
the critical tier; what this adds is the verdict *before* that tier, named
against the diff, so a reviewer reads a line rather than a traceback.

``--files`` prints the changed test files instead, for running them under
``SAL_DURATION_CAP``. The gate does not: on the branch that added it those
files took 20 s of a 30 s budget, and `infra/validate.sh` had already run them
under the cap on the reference host.

Usage::

    uv run python infra/gate_changed_tests.py --base origin/main          # check
    uv run python infra/gate_changed_tests.py --base origin/main --files  # list
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests" / "regression"))
from test_kinds import EXCLUDED_DIRECTORY, KINDS  # noqa: E402


def changed_test_files(base: str) -> list[Path]:
    """The test files the branch adds or changes, excluding the benchmarks.

    Parameters
    ----------
    base : str
        The branch the pull request targets.

    Returns
    -------
    list[Path]
        Repository-relative paths that still exist, sorted.
    """
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD", "--", "tests/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted(
        Path(name)
        for name in out
        if Path(name).name.startswith("test_")
        and Path(name).suffix == ".py"
        and EXCLUDED_DIRECTORY not in Path(name).parts
        and (REPO_ROOT / name).is_file()
    )


def _changed_lines(base: str, path: Path) -> set[int]:
    """The line numbers of ``path`` the diff adds or rewrites.

    Returns
    -------
    set[int]
        Line numbers in the branch's version of the file.
    """
    diff = subprocess.run(
        ["git", "diff", "-U0", f"{base}...HEAD", "--", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines: set[int] = set()
    for header in diff.splitlines():
        if not header.startswith("@@"):
            continue
        span = header.split("+")[1].split("@@")[0].strip()
        start, _, count = span.partition(",")
        lines.update(range(int(start), int(start) + int(count or 1)))
    return lines


def _markers(node: ast.FunctionDef) -> set[str]:
    """The ``pytest.mark.<name>`` markers on one test.

    Returns
    -------
    set[str]
    """
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


def unmarked(base: str) -> list[str]:
    """The tests the branch touched that say nothing about what checks them.

    Returns
    -------
    list[str]
        ``file::name`` per offender, sorted.
    """
    offenders: list[str] = []
    for path in changed_test_files(base):
        touched = _changed_lines(base, path)
        tree = ast.parse((REPO_ROOT / path).read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith(
                "test_"
            ):
                continue
            first = min(
                [node.lineno, *(d.lineno for d in node.decorator_list)],
            )
            if not touched & set(range(first, (node.end_lineno or node.lineno) + 1)):
                continue
            if not _markers(node) & set(KINDS):
                offenders.append(f"{path}::{node.name}")
    return sorted(offenders)


def main(argv: list[str] | None = None) -> int:
    """Report the changed tests, or the ones missing a kind.

    Returns
    -------
    int
        ``0`` when every touched test carries a kind; ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument(
        "--files",
        action="store_true",
        help="print the changed test files, one per line, and exit 0",
    )
    args = parser.parse_args(argv)
    if args.files:
        for path in changed_test_files(args.base):
            print(path)
        return 0
    offenders = unmarked(args.base)
    for offender in offenders:
        print(f"  no kind marker: {offender}", file=sys.stderr)
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
