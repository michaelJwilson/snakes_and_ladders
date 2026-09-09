"""What a recorded measurement is a function of, hashed so a stale one is refused.

A baseline record beside a fixture --- an enumerated maximum, the rate at
which hill climbing reaches it --- is a number computed from that fixture,
from the modules that computed it, and by the libraries installed when it
ran. A digest over those, recorded in the record, is what
:func:`snakes_and_ladders.sim.fixtures.baseline` refuses a stale record on
(issue #401).

**It hashed committed figures and notebooks too, and no longer does** (issue
#490). A stamp beside each figure and each notebook recorded the same digest
over the renderer's import closure, and the build skipped an artifact whose
stamp matched. Over 476 decisions in two measured windows every stale call
was a false positive and no figure byte moved, at a cost of ~48 minutes in
one day; two causes are recorded in #490 rather than fixed, because the
stamps are gone. What each stamp was nominally providing is provided by
running the check instead of predicting it: ``infra/release.sh`` renders
every figure and compares bytes (issue #484), and the ``notebooks`` job
executes every notebook under ``docs/nb/`` (issue #480).

**Top level, not inside ``qa``.** It began there, where the figures are, and
moved when ``snakes_and_ladders.sim.fixtures`` came to need it for the
baseline records (issue #401): ``qa`` renders what the other modules compute
and nothing may import it, so a digest living there made ``sim`` depend on
``qa``. It stays here now that the figures have gone the other way. The
module names no model and imports nothing from this package, so it sits
beside :mod:`snakes_and_ladders.numerics` and
:mod:`snakes_and_ladders.enumeration` on the terms root ``CLAUDE.md`` states
for those: importable from anywhere, inverting no layering.

The import closure is read from the source with :mod:`ast`, as
``infra/select_tests.py`` reads the module graph: a listed dependency goes
stale silently and an import does not. A module that reaches the Rust
extension carries the Rust sources in its closure, since a kernel change
alters what it computes.

The digest is over file contents, not modification times, so a fresh clone
and the tree that recorded the digest agree.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from collections.abc import Iterable, Sequence
from importlib import metadata
from pathlib import Path

PACKAGE = "snakes_and_ladders"
EXTENSION = "oxi_snakes_and_ladders"


def _module_path(name: str, package_root: Path) -> Path | None:
    """Locate the source of a dotted module name under ``package_root``.

    Returns
    -------
    Path | None
        ``<name>.py`` or ``<name>/__init__.py``, or None when neither exists
        -- a third-party import, or a name inside a module rather than a
        module.
    """
    relative = Path(*name.split("."))
    for candidate in (
        package_root / relative.with_suffix(".py"),
        package_root / relative / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _imported_names(source: str, module: str) -> set[str]:
    """Every dotted name ``source`` imports, absolute, including ``from``-targets.

    ``from snakes_and_ladders.sim import tree`` names both the package and,
    if it is one, the submodule ``tree``; the caller resolves which exist.

    Parameters
    ----------
    source : str
        Python source.
    module : str
        The dotted name of the module the source belongs to, for resolving
        relative imports.

    Returns
    -------
    set[str]
        Dotted names, absolute.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(module.split(".")[: -node.level or None])
                base = f"{base}.{node.module}" if node.module else base
            else:
                base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def module_closure(modules: Iterable[str], root: Path) -> list[Path]:
    """The source files of ``modules`` and everything they import, transitively.

    Parameters
    ----------
    modules : Iterable[str]
        Dotted module names to start from.
    root : Path
        The repository root; sources are looked up under ``root / "python"``.

    Returns
    -------
    list[Path]
        Sorted source paths within the package, the ``__init__`` of every
        package above each of them included. When any of them names the Rust
        extension, the Rust sources under ``root / "src"`` and ``Cargo.lock``
        are appended.
    """
    package_root = root / "python"
    seen: set[str] = set()
    files: set[Path] = set()
    todo = [name for name in modules if name.split(".")[0] == PACKAGE]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        # The packages above it: importing `a.b.c` runs `a/__init__.py` and
        # `a/b/__init__.py`, so their statements are inputs too.
        parts = name.split(".")
        todo += [".".join(parts[:index]) for index in range(1, len(parts))]
        path = _module_path(name, package_root)
        if path is None:
            continue
        files.add(path)
        for imported in _imported_names(path.read_text(), name):
            if imported.split(".")[0] == PACKAGE and imported not in seen:
                todo.append(imported)
    closure = sorted(files)
    if any(EXTENSION in path.read_text() for path in closure):
        closure += sorted((root / "src").rglob("*.rs"))
        closure.append(root / "Cargo.lock")
    return closure


def library_versions(libraries: Sequence[str]) -> list[str]:
    """``name==version`` per library, ``absent`` for one not installed.

    The libraries are the caller's to name and carry no default here. The
    one caller left is a baseline record, whose numbers depend on ``numpy``,
    ``scipy`` and ``torch`` and on nothing that draws; the wider default this
    had was the *figures*' list, and it went with them (issue #490).

    Parameters
    ----------
    libraries : Sequence[str]
        Distribution names, in the order the result reports them.

    Returns
    -------
    list[str]
        One entry per library, in the given order, then the interpreter.
    """
    versions = []
    for name in libraries:
        try:
            versions.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{name}==absent")
    versions.append(f"python=={sys.version_info.major}.{sys.version_info.minor}")
    return versions


def _expanded(files: Iterable[Path]) -> list[Path]:
    """Every file among ``files``, a directory replaced by the files under it.

    Returns
    -------
    list[Path]
    """
    found: list[Path] = []
    for path in files:
        if path.is_dir():
            found += sorted(child for child in path.rglob("*") if child.is_file())
        else:
            found.append(path)
    return found


def digest(files: Iterable[Path], root: Path, *extra: str) -> str:
    """Hash file contents and extra strings into one hex digest.

    Parameters
    ----------
    files : Iterable[Path]
        Files whose bytes enter the hash, keyed by their path relative to
        ``root`` so the digest does not depend on where the checkout lives.
        A directory stands for the files under it: a fixture named as a
        problem rather than as one tier (issue #382) is a directory, and
        hashing it as a unit is what makes adding a tier to it a change the
        digest sees.
    root : Path
        The repository root.
    *extra : str
        Strings that are inputs too: the record's own algorithms, seeds and
        budgets, and the library versions.

    Returns
    -------
    str
        A SHA-256 hex digest.
    """
    hasher = hashlib.sha256()
    for path in sorted(set(_expanded(files))):
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    for item in extra:
        hasher.update(item.encode())
        hasher.update(b"\0")
    return hasher.hexdigest()
