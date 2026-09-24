"""The guard that refuses a compiled extension older than its Rust.

Issue #630. Every case is built in `tmp_path` as a tree of real files with
real timestamps, because the check reads timestamps and a mock of `stat` would
assert the mock. `tests/regression/test_uv_guard.py` and
`test_environment.py` are the precedent for guarding the environment this way.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests._extension import stale_extension


def tree(root: Path, *, extension: str | None, source: str = "src/lib.rs") -> None:
    """A minimal worktree: one Rust source, one manifest, one extension."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / source).parent.mkdir(parents=True, exist_ok=True)
    (root / source).write_text("// a kernel\n")
    (root / "Cargo.toml").write_text("[package]\nname = 'x'\n")
    (root / "Cargo.lock").write_text("# lock\n")
    if extension is not None:
        built = root / "python" / "snakes_and_ladders"
        built.mkdir(parents=True, exist_ok=True)
        (built / extension).write_bytes(b"\x7fELF")


def age(path: Path, seconds: float) -> None:
    """Back-date `path` by `seconds`, so an ordering can be asserted."""
    stamp = path.stat().st_mtime - seconds
    os.utime(path, (stamp, stamp))


@pytest.mark.critical
@pytest.mark.infra
def test_an_extension_older_than_its_rust_is_refused(tmp_path: Path) -> None:
    """The case that cost a 36-minute suite run and 67 false failures.

    A worktree merges `main`, which rewrites `src/`, and its extension stays
    as it was. The refusal must name both files, so the reader is not left to
    infer which of the two is behind.
    """
    tree(tmp_path, extension="oxisal.cpython-312-x86_64-linux-gnu.so")
    # The merge rewrote `src/`, so that is what must be named; the manifests
    # are back-dated to leave it the newest rather than whichever the fixture
    # happened to write last.
    age(tmp_path / "Cargo.toml", 300.0)
    age(tmp_path / "Cargo.lock", 300.0)
    age(
        tmp_path / "python/snakes_and_ladders/oxisal.cpython-312-x86_64-linux-gnu.so",
        60.0,
    )

    refusal = stale_extension(tmp_path)

    assert refusal, "a stale extension was accepted"
    assert "oxisal" in refusal
    assert "src/lib.rs" in refusal
    assert "infra/build_extension.sh" in refusal


@pytest.mark.critical
@pytest.mark.infra
def test_an_extension_newer_than_its_rust_is_silent(tmp_path: Path) -> None:
    """The common case, which must cost nothing and say nothing.

    Most pull requests touch no Rust, so most runs reach this branch; a guard
    that fired on them would be removed within the day.
    """
    tree(tmp_path, extension="oxisal.cpython-312-x86_64-linux-gnu.so")
    for name in ("src/lib.rs", "Cargo.toml", "Cargo.lock"):
        age(tmp_path / name, 60.0)

    assert stale_extension(tmp_path) == ""


@pytest.mark.infra
def test_a_lockfile_bump_alone_outdates_the_extension(tmp_path: Path) -> None:
    """`Cargo.lock` counts: a dependency bump changes the binary and no `.rs`.

    This is why `SOURCES` names the two manifests beside `src/`, and it is the
    case a check written against `src/**/*.rs` alone would miss.
    """
    tree(tmp_path, extension="oxisal.cpython-312-x86_64-linux-gnu.so")
    age(tmp_path / "src/lib.rs", 120.0)
    age(tmp_path / "Cargo.toml", 120.0)
    age(
        tmp_path / "python/snakes_and_ladders/oxisal.cpython-312-x86_64-linux-gnu.so",
        60.0,
    )

    refusal = stale_extension(tmp_path)

    assert refusal, "a lockfile newer than the extension was accepted"
    assert "Cargo.lock" in refusal


@pytest.mark.infra
def test_a_tree_with_no_extension_is_not_stale(tmp_path: Path) -> None:
    """A checkout that never built one: the import error says it far better.

    Reporting staleness here would name a file that does not exist, and would
    fire on every environment that installs the wheel instead of building in
    the tree.
    """
    tree(tmp_path, extension=None)

    assert stale_extension(tmp_path) == ""


@pytest.mark.infra
def test_a_tree_with_no_rust_is_not_stale(tmp_path: Path) -> None:
    """Nothing to be behind: the check is about `src/`, not about age."""
    built = tmp_path / "python" / "snakes_and_ladders"
    built.mkdir(parents=True)
    (built / "oxisal.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")

    assert stale_extension(tmp_path) == ""


@pytest.mark.infra
def test_this_worktree_passes_its_own_guard() -> None:
    """The guard holds for the tree it ships in, which `conftest` already ran.

    Asserted here as well so the claim is a test rather than a side effect of
    the session having started.
    """
    assert stale_extension(Path(__file__).resolve().parents[2]) == ""
