"""What a committed artifact depends on, hashed so an unchanged one is not remade.

A committed figure, an executed notebook or a fixture's recorded baseline is a
function of its inputs: the script or the cells, every ``snakes_and_ladders``
module they reach by import, the fixture files they read, and the versions of
the libraries that draw or compute. A digest over those, recorded beside the
artifact when it was made, says whether the artifact can have changed since.
The build renders a figure only when its digest differs from the stamp, so a
pull request that changed a paragraph of the textbook renders nothing (issue
#372).

**Top level, not inside ``qa``.** It began there, where the figures are, and
moved when ``snakes_and_ladders.sim.fixtures`` came to need it for the
baseline records (issue #401): ``qa`` renders what the other modules compute
and nothing may import it, so a digest living there made ``sim`` depend on
``qa`` and a ``qa`` change select and time ``sim``. The module names no model
and imports nothing from this package, so it sits beside
:mod:`snakes_and_ladders.numerics` and :mod:`snakes_and_ladders.enumeration`
on the terms root ``CLAUDE.md`` states for those: importable from anywhere,
inverting no layering.

The import closure is read from the source with :mod:`ast`, as
``infra/select_tests.py`` reads the module graph: a listed dependency goes
stale silently and an import does not. A module that reaches the Rust
extension carries the Rust sources in its closure, since a kernel change
alters what it computes.

The digest is over file contents, not modification times, so a fresh clone
and the tree that produced the stamp agree.
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

#: The libraries whose version is part of a rendered artifact. A matplotlib
#: release moves glyph placement and a NumPy release can move a reduction's
#: last bit; both change the committed bytes with no source change.
LIBRARIES: tuple[str, ...] = ("matplotlib", "numpy", "scipy", "torch", "networkx")


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


def imported_names(source: str, module: str) -> set[str]:
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
        Sorted source paths within the package. When any of them names the
        Rust extension, the Rust sources under ``root / "src"`` and
        ``Cargo.lock`` are appended.
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
        path = _module_path(name, package_root)
        if path is None:
            continue
        files.add(path)
        for imported in imported_names(path.read_text(), name):
            if imported.split(".")[0] == PACKAGE and imported not in seen:
                todo.append(imported)
    closure = sorted(files)
    if any(EXTENSION in path.read_text() for path in closure):
        closure += sorted((root / "src").rglob("*.rs"))
        closure.append(root / "Cargo.lock")
    return closure


def library_versions(libraries: Sequence[str] = LIBRARIES) -> list[str]:
    """``name==version`` per library, ``absent`` for one not installed.

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
        stamp sees.
    root : Path
        The repository root.
    *extra : str
        Strings that are inputs too: a spec's arguments, library versions.

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


def read_stamp(path: Path) -> str | None:
    """The digest a stamp file records, or None when there is no stamp.

    Returns
    -------
    str | None
        The recorded digest.
    """
    return path.read_text().strip() if path.is_file() else None


def write_stamp(path: Path, value: str) -> None:
    """Record ``value`` as the digest the artifact beside ``path`` was made from."""
    path.write_text(f"{value}\n")
