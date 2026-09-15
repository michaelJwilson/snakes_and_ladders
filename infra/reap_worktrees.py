"""Retire a worktree when the branch it holds is gone from the remote.

Issue #618. `infra/new_worktree.sh` creates worktrees and nothing retires them,
so this host reached **56** of them against **9** open pull requests. The rest
were trees whose branch had merged and been deleted upstream months earlier,
each one 5.2 GiB of apparent size and one more place for the shared `.venv` to
be pointed at.

The test is the remote branch, not the pull request API: a merged or closed
pull request has its branch deleted, so `git ls-remote` answers the same
question without a token, a rate limit, or a network round trip per tree beyond
the one `git fetch` already makes.

**Nothing is removed that could hold work.** Three conditions must all hold,
and the first is the one that matters:

1. the branch is absent from the remote --- its pull request is done;
2. no tracked file is modified, added or deleted;
3. no untracked file that is not a build artifact or a cache.

Condition 3 is deliberately conservative: an unrecognised untracked file keeps
the tree, on the same principle as `infra/select_tests.py`'s unrecognised
change selecting everything.

A stash is **not** a condition. Stashes live in the common `.git` directory and
survive worktree removal, so guarding on one would refuse every tree on the
host for a stash that belongs to none of them --- which is exactly what a first
version of this did, and why the rule is written down here.

The premise this rests on is `infra/CLAUDE.md`'s: incomplete work lives on an
origin branch behind a draft pull request, never only in a worktree. A worktree
is a scratch space with no backup and no reviewer.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

#: Untracked paths that are regenerated rather than written, so their presence
#: says nothing about whether a tree holds work.
DISPOSABLE = re.compile(
    r"(\.so$|__pycache__|\.venv|\.pytest_cache|\.mypy_cache|\.ruff_cache|egg-info)"
)


def _git(repository: Path, *arguments: str) -> str:
    """`git` in ``repository``, stdout stripped, errors raised."""
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )

    return completed.stdout.strip()


def worktrees(repository: Path) -> list[Path]:
    """Every worktree of ``repository`` except the one holding `.git`."""
    listing = _git(repository, "worktree", "list", "--porcelain")
    paths = [
        Path(line.removeprefix("worktree ").strip())
        for line in listing.splitlines()
        if line.startswith("worktree ")
    ]
    common = Path(
        _git(repository, "rev-parse", "--path-format=absolute", "--git-common-dir")
    )

    return [path for path in paths if path != common.parent]


def holds_work(tree: Path) -> str:
    """Why ``tree`` must be kept, or the empty string where nothing holds it."""
    status = _git(tree, "status", "--porcelain")
    tracked = [line for line in status.splitlines() if not line.startswith("??")]
    if tracked:
        return f"{len(tracked)} tracked change(s)"

    untracked = [
        line[3:]
        for line in status.splitlines()
        if line.startswith("??") and not DISPOSABLE.search(line[3:])
    ]
    if untracked:
        return f"untracked: {', '.join(untracked[:3])}"

    branch = _git(tree, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        return "detached HEAD"

    if _git(tree, "ls-remote", "--heads", "origin", branch):
        return f"origin/{branch} still exists"

    return ""


def main(argv: list[str] | None = None) -> int:
    """Report every worktree, and remove the retired ones unless ``--dry-run``."""
    parser = argparse.ArgumentParser(description="Retire merged worktrees.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="name what would be removed and remove nothing",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="a worktree of the repository to reap (default: the current directory)",
    )
    arguments = parser.parse_args(argv)

    retired: list[Path] = []
    for tree in worktrees(arguments.repository):
        reason = holds_work(tree)
        if reason:
            print(f"keep    {tree}  ({reason})")
        else:
            retired.append(tree)
            print(f"retire  {tree}")

    if not retired:
        print("nothing to retire")
        return 0

    if arguments.dry_run:
        print(f"{len(retired)} would be removed; --dry-run, so none were")
        return 0

    for tree in retired:
        _git(arguments.repository, "worktree", "remove", str(tree))
    _git(arguments.repository, "worktree", "prune")
    print(f"removed {len(retired)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
