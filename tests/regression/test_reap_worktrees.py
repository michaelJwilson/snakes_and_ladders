"""The reaper retires a worktree only when nothing it holds can be lost.

Issue #618. `infra/new_worktree.sh` creates worktrees and nothing retired them,
so the host reached 56 against 9 open pull requests. Removing a worktree is not
reversible from inside the repository, so each condition that keeps a tree is
asserted here against a real repository rather than reasoned about.

The repositories below are built by `git` in `tmp_path` --- a bare one standing
in for `origin` and a clone with worktrees off it --- because the behaviour
under test *is* `git`'s: what `ls-remote` reports for a deleted branch, and
what `status --porcelain` reports for each kind of change.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import reap_worktrees  # noqa: E402


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def host(tmp_path: Path) -> tuple[Path, Path]:
    """A clone with two worktrees, one branch published and one not."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(origin), str(clone)], check=True)
    _git(clone, "config", "user.email", "x@y.z")
    _git(clone, "config", "user.name", "probe")
    (clone / "README.md").write_text("main\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "first")
    _git(clone, "push", "-u", "origin", "main")

    for name in ("live", "merged"):
        tree = tmp_path / name
        _git(clone, "worktree", "add", "-b", name, str(tree), "main")
        _git(tree, "push", "-u", "origin", name)

    # `merged`'s pull request is done, so its branch is deleted upstream.
    _git(clone, "push", "origin", "--delete", "merged")

    return clone, tmp_path


@pytest.mark.infra
def test_a_published_branch_keeps_its_worktree(host: tuple[Path, Path]) -> None:
    clone, root = host

    assert reap_worktrees.holds_work(root / "live") == "origin/live still exists"
    assert reap_worktrees.holds_work(root / "merged") == ""


@pytest.mark.infra
def test_a_tracked_change_keeps_a_worktree_whose_branch_is_gone(
    host: tuple[Path, Path],
) -> None:
    """The branch being gone is not enough; an edit outranks it."""
    _, root = host
    (root / "merged" / "README.md").write_text("edited\n", encoding="utf-8")

    assert reap_worktrees.holds_work(root / "merged") == "1 tracked change(s)"


@pytest.mark.infra
def test_an_unrecognised_untracked_file_keeps_a_worktree(
    host: tuple[Path, Path],
) -> None:
    """Conservative by design, as `select_tests.py` is for an unknown change."""
    _, root = host
    (root / "merged" / "notes.md").write_text("half an idea\n", encoding="utf-8")

    assert reap_worktrees.holds_work(root / "merged").startswith("untracked: notes.md")


@pytest.mark.infra
def test_a_build_artifact_does_not_keep_a_worktree(host: tuple[Path, Path]) -> None:
    """Otherwise every tree is kept forever: each carries a `.so` and caches."""
    _, root = host
    (root / "merged" / "oxi.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    (root / "merged" / "__pycache__").mkdir()
    (root / "merged" / "__pycache__" / "x.pyc").write_bytes(b"")

    assert reap_worktrees.holds_work(root / "merged") == ""


@pytest.mark.infra
def test_a_stash_does_not_keep_a_worktree(host: tuple[Path, Path]) -> None:
    """A stash lives in the common `.git`, so it belongs to no single tree.

    Guarding on one refuses *every* worktree on the host for a stash that
    belongs to none of them, which is what a first version of the reaper did.
    The stash survives the removal, which is why it is not a reason to refuse.
    """
    clone, root = host
    (clone / "README.md").write_text("work in progress\n", encoding="utf-8")
    _git(clone, "stash", "push", "-m", "wip")

    assert _git(clone, "stash", "list") != ""
    assert reap_worktrees.holds_work(root / "merged") == ""

    reap_worktrees.main(["--repository", str(clone)])

    assert not (root / "merged").exists()
    assert (root / "live").exists()
    assert _git(clone, "stash", "list") != "", "the stash did not survive the reap"


@pytest.mark.infra
def test_a_dry_run_removes_nothing(host: tuple[Path, Path]) -> None:
    clone, root = host

    reap_worktrees.main(["--dry-run", "--repository", str(clone)])

    assert (root / "merged").exists()
    assert (root / "live").exists()
