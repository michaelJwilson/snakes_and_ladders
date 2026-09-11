"""What a recorded measurement is a function of: its code, and its libraries.

A baseline record beside a fixture --- an enumerated maximum, the rate at
which hill climbing reaches it --- is a number computed by some modules and
by the libraries installed when it ran. :func:`module_closure` says which
sources those modules reach, which is what ``infra/baselines.py --changed``
selects the records a pull request could have moved on;
:func:`library_versions` says which libraries were installed, which is what
:func:`snakes_and_ladders.sim.fixtures.baseline` refuses a record on when
they are not this machine's (issues #401, #460).

**Nothing here hashes a committed artifact any more** (issue #490). A stamp
beside each figure and each notebook recorded a digest over the renderer's
import closure, and the build skipped an artifact whose stamp matched. Over
476 decisions in two measured windows every stale call was a false positive
and no figure byte moved, at a cost of ~48 minutes in one day; two causes
are recorded in #490 rather than fixed, because the stamps are gone. What
each stamp nominally provided is provided by running the check instead of
predicting it: ``infra/release.sh`` renders every figure and compares bytes
(issue #484), and the ``notebooks`` job executes every notebook under
``docs/nb/`` (issue #480). The digest they shared went with them; only its
two terms are left, and each is read directly.

**Top level, not inside ``qa``.** It began there, where the figures are, and
moved when ``snakes_and_ladders.sim.fixtures`` came to need it for the
baseline records (issue #401): ``qa`` renders what the other modules compute
and nothing may import it, so a digest living there made ``sim`` depend on
``qa``. The baseline records are the only caller left, and this is where
they read it from. The module names no model and imports nothing from this package, so it sits
beside :mod:`snakes_and_ladders.numerics` and
:mod:`snakes_and_ladders.enumeration` on the terms root ``CLAUDE.md`` states
for those: importable from anywhere, inverting no layering.

The import closure is read from the source with :mod:`ast`, as
``infra/select_tests.py`` reads the module graph: a listed dependency goes
stale silently and an import does not. A module that reaches the Rust
extension carries the Rust sources in its closure, since a kernel change
alters what it computes. The closure is paths, relative to the root the
caller passes, so a fresh clone and the tree that wrote a record agree.
"""

from __future__ import annotations

import ast
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
