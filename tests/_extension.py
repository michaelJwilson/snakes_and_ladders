"""Whether the compiled extension in this tree is older than the Rust it was built from.

Issue #630. The shared environment holds one editable install, pointing at the
primary worktree, and a worktree is reached with ``PYTHONPATH=<worktree>/python``
which shadows it --- the extension included. So every worktree carries its own
``oxi_snakes_and_ladders`` and nothing rebuilds it: a merge that changes
``src/`` leaves the binary behind, and the suite then runs new Python against
old Rust.

**What that looks like is not what it is.** A worktree whose extension predated
a ``single_site_sweeps`` signature change failed 67 tests, almost all of them
in ``search/test_potts_mcmc*.py``, over 36 minutes. Read at the end of a run
those failures look exactly like the change under test breaking the oracles,
and they were diagnosed that way twice before the `TypeError` underneath was
read. The rebuild costs 18 seconds. Finding out costs a suite.

So the check runs at ``pytest_configure``, before a test is collected, and the
message names both files and the command that repairs it. Timestamps rather
than a content hash: a checkout writes the files it changes and leaves the rest
alone, so a merge that does not touch ``src/`` leaves the binary valid and this
silent, which is the common case and the one that must stay free.
"""

from __future__ import annotations

from pathlib import Path

#: Where a built extension sits in a worktree, by any platform's suffix.
EXTENSION_GLOB = "python/snakes_and_ladders/oxi_snakes_and_ladders*"

#: What the extension is built from. `Cargo.lock` is here because a dependency
#: bump changes the binary without touching a line of `src/`.
SOURCES = ("src", "Cargo.toml", "Cargo.lock")

#: Suffixes a compiled extension carries. `.so` on Linux, `.pyd` on Windows,
#: `.dylib` for a `cdylib` copied by hand rather than by `maturin`.
SUFFIXES = (".so", ".pyd", ".dylib")


def _built(root: Path) -> Path | None:
    """The extension in `root`, or `None` where the tree carries none.

    A tree with no extension is not stale: it is a checkout that has never
    built one, and every import of the package will say so far more clearly
    than a timestamp could.
    """
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

    Parameters
    ----------
    root : Path
        The worktree to check.

    Returns
    -------
    str
        A message naming the extension, the source that outdates it and the
        command that rebuilds it; empty where the extension is current, absent,
        or the tree carries no Rust.
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
