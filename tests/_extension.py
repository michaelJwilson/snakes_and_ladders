"""Whether the compiled extension in this tree is older than the Rust it was built from.

Issue #630. A worktree reached with ``PYTHONPATH=<worktree>/python`` carries
its own ``oxisal`` and nothing rebuilds it, so a merge that changes ``src/``
runs new Python against old Rust. One stale extension failed 67 tests over 36
minutes, read twice as a broken oracle; the rebuild costs 18 s. So the check
runs at ``pytest_configure`` and names both files and the repair. Timestamps,
not a hash: a merge that leaves ``src/`` alone stays silent and free.
"""

from __future__ import annotations

from pathlib import Path

#: Where a built extension sits in a worktree, by any platform's suffix.
EXTENSION_GLOB = "python/snakes_and_ladders/oxisal*"

#: What the extension is built from. `Cargo.lock` is here because a dependency
#: bump changes the binary without touching a line of `src/`.
SOURCES = ("src", "Cargo.toml", "Cargo.lock")

#: Suffixes a compiled extension carries. `.so` on Linux, `.pyd` on Windows,
#: `.dylib` for a `cdylib` copied by hand rather than by `maturin`.
SUFFIXES = (".so", ".pyd", ".dylib")


def _built(root: Path) -> Path | None:
    """The extension in `root`, or `None` where the tree has never built one."""
    for path in sorted(root.glob(EXTENSION_GLOB)):
        if path.suffix in SUFFIXES:
            return path
    return None


def _newest_source(root: Path) -> tuple[Path, float] | None:
    """The most recently written file the extension is built from."""
    newest: tuple[Path, float] | None = None
    for name in SOURCES:
        entry = root / name
        if not entry.exists():
            continue
        paths = entry.rglob("*.rs") if entry.is_dir() else iter((entry,))
        for path in paths:
            stamp = path.stat().st_mtime
            if newest is None or stamp > newest[1]:
                newest = (path, stamp)
    return newest


def stale_extension(root: Path) -> str:
    """The refusal for a stale extension in `root`, empty where it is current.

    Names the extension, the newer source and the rebuild command.
    """
    built = _built(root)
    if built is None:
        return ""
    newest = _newest_source(root)
    if newest is None:
        return ""
    source, stamp = newest
    if built.stat().st_mtime >= stamp:
        return ""
    return (
        f"{built.relative_to(root)} was built before {source.relative_to(root)} "
        f"was last written, so this run would exercise the previous Rust: a "
        f"signature change reads as dozens of failures in the tests that call "
        f"it. Rebuild with `infra/build_extension.sh` (issue #630)."
    )
