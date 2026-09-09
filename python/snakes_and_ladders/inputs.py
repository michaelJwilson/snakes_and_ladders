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
from dataclasses import dataclass
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


# --- Reachability -----------------------------------------------------------
#
# The closure above is the set of modules a renderer *imports*; what it
# *executes* is smaller, and the difference was a measured cost. `qa/runner.py`
# is imported by every figure, so a docstring reworded there, or two parameter
# constants declared there, restamped all 19 cited figures with zero PDF bytes
# changed (issue #394). Hashing the text of a whole module charges a figure for
# code it never runs.
#
# So the unit of the hash is a *definition* rather than a file, and what is
# hashed is its AST with docstrings and comments removed: reworded prose is not
# a different computation (issue #372's open question). From the entry module
# the walk follows names -- a call, a constant read, a base class, a decorator,
# a default argument -- and hashes the definitions it lands on, plus the
# module-level statements every import runs whatever it reaches.
#
# The walk over-approximates wherever it is unsure, because the two failure
# directions are not symmetric: a stamp that fires too often costs a render,
# and one that misses a change publishes a figure the code no longer produces.
# So the entry module and every included class are hashed whole; a name bound
# to a module rather than to a symbol is hashed whole; a local shadowing a
# module-level name pulls that name in; and a module whose own text defeats the
# analysis is hashed whole and *named* in `Reach.fallbacks`, which enters the
# digest, so the fallback is asserted by test rather than taken on trust.

#: Stands for every definition in a module, where the walk cannot be narrower.
WHOLE_MODULE = "*"
#: Stands for the module-level statements an import runs. Not an identifier, so
#: it cannot collide with a definition's name.
MODULE_FRAME = "<frame>"
#: Builtins that run code no name in the source points at.
DYNAMIC_BUILTINS = frozenset({"eval", "exec", "globals", "locals", "vars"})
#: Names that import a module chosen at run time. ``importlib`` itself is one
#: only when the module object is bound, since then any of its functions is
#: reachable: ``from importlib import metadata`` reads a version and is not.
DYNAMIC_IMPORTS = frozenset({"importlib", "import_module", "__import__"})
#: Attribute lookups by computed name. Flagged only on a name bound to a
#: module: `getattr(args, dest)` on an `argparse` namespace reaches an
#: attribute of that object, never a definition this walk could have missed.
DYNAMIC_ACCESSORS = frozenset({"getattr", "setattr", "delattr"})


@dataclass(frozen=True)
class Reach:
    """What one entry point executes, and how sure the walk was.

    Parameters
    ----------
    fingerprint : str
        SHA-256 over the reachable definitions' ASTs, docstrings and comments
        removed. Equal for two trees whose reachable code is the same.
    modules : tuple[Path, ...]
        Package sources any part of which entered the fingerprint, sorted.
    fallbacks : tuple[str, ...]
        ``module: reason`` per module hashed whole because its own text
        defeats the analysis. Part of the digest, so a module that starts or
        stops being analysable restamps rather than silently changing what is
        watched.
    extension : bool
        Whether any hashed definition names the compiled extension. When it
        does, the Rust sources are inputs too: a kernel change alters what the
        entry point computes and no Python source records it.
    """

    fingerprint: str
    modules: tuple[Path, ...]
    fallbacks: tuple[str, ...]
    extension: bool


@dataclass(frozen=True)
class _Binding:
    """One name an import binds, hashed as itself rather than as its line.

    Parameters
    ----------
    text : str
        Canonical form, one bound name per entry, so a name added to an import
        line does not restamp the readers of the names already on it.
    target : tuple[str, str] | None
        The ``(module, symbol)`` the walk continues to; None for a name outside
        the package, whose version :func:`library_versions` covers instead.
    """

    text: str
    target: tuple[str, str] | None


@dataclass(frozen=True)
class _Facts:
    """One module taken apart into what a name can reach.

    Parameters
    ----------
    path : Path
        The source file.
    definitions : dict[str, list[object]]
        Top-level name to what defines it: an ``ast.stmt`` for a ``def``, a
        ``class`` or an assignment to a bare name, a :class:`_Binding` for an
        imported name.
    frame : list[ast.stmt]
        Top-level statements binding no single name, which therefore run for
        every importer whatever it reaches.
    imports : tuple[str, ...]
        Package modules this one imports, whose frames its own import runs.
    dynamic : str
        Why the walk cannot narrow this module; empty when it can.
    """

    path: Path
    definitions: dict[str, list[object]]
    frame: list[ast.stmt]
    imports: tuple[str, ...]
    dynamic: str


def _strip_docstrings(tree: ast.AST) -> None:
    """Delete every module, class and function docstring in ``tree``, in place.

    A docstring is prose the renderer does not draw. Only a leading string is
    removed; a string anywhere else is a statement and stays.
    """
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del node.body[0]


def _absolute(node: ast.ImportFrom, module: str) -> str:
    """The absolute dotted module an ``ImportFrom`` reads from.

    Returns
    -------
    str
    """
    if not node.level:
        return node.module or ""
    base = ".".join(module.split(".")[: -node.level or None])
    return f"{base}.{node.module}" if node.module else base


def _referenced(node: ast.AST) -> set[str]:
    """Every bare name ``node`` mentions, at any depth.

    Parameters and locals are included: a name that shadows a module-level one
    adds an edge the walk would otherwise have to prove unnecessary. The root
    of a dotted access is a bare name and is caught; the attributes after it
    are not, which is why an included class is hashed whole.

    Returns
    -------
    set[str]
    """
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _dynamic_reason(tree: ast.Module, opaque: set[str]) -> str:
    """Why the AST cannot be trusted to name everything this module reaches.

    Parameters
    ----------
    tree : ast.Module
        The parsed module.
    opaque : set[str]
        Names bound to a module object, and names imported from within the
        package: a lookup or a decorator through one of these can reach a
        definition no name points at.

    Returns
    -------
    str
        The first reason found, or empty when every reference is a name.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "*" for alias in node.names
        ):
            return f"star import from {node.module}"
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == PACKAGE for alias in node.names
        ):
            return "plain `import snakes_and_ladders...` reaches by attribute"
        if isinstance(node, ast.Import):
            bound = {(alias.asname or alias.name).split(".")[0] for alias in node.names}
            if bound & DYNAMIC_IMPORTS:
                return f"binds the module {sorted(bound & DYNAMIC_IMPORTS)[0]}"
        if isinstance(node, ast.ImportFrom):
            named = {alias.name for alias in node.names}
            if named & DYNAMIC_IMPORTS:
                return f"imports {sorted(named & DYNAMIC_IMPORTS)[0]}"
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            return "module-level __getattr__"
        if isinstance(node, ast.Name) and node.id in DYNAMIC_BUILTINS:
            return f"{node.id}()"
        if isinstance(node, ast.Name) and node.id in DYNAMIC_IMPORTS:
            return node.id
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "modules"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            return "sys.modules"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in DYNAMIC_ACCESSORS
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in opaque
        ):
            return f"{node.func.id}() on {node.args[0].id}"
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            for decorator in node.decorator_list:
                root: ast.expr = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in opaque:
                    return f"@{root.id} may register {node.name}"
    return ""


#: Parsed modules, keyed by name, path and source text. Nineteen figures share
#: one package, so without this the gate parses the same thirty modules
#: nineteen times: 4.2 s against 0.6 s for the same nineteen digests. Keyed by
#: the text rather than by a timestamp, so a file rewritten inside one process
#: --- which is what the tests below do --- is re-analysed.
_ANALYSED: dict[tuple[str, str, str], _Facts] = {}


def _analyse(module: str, path: Path, package_root: Path) -> _Facts:
    """Take one module apart into its definitions, frame, imports and edges.

    Returns
    -------
    _Facts
    """
    source = path.read_text()
    key = (module, str(path), source)
    cached = _ANALYSED.get(key)
    if cached is not None:
        return cached
    tree = ast.parse(source)
    _strip_docstrings(tree)

    bindings: dict[str, _Binding] = {}
    imports: set[str] = set()
    opaque: set[str] = set()
    # Every import at any depth: a name imported inside a function is used
    # inside it, and registering the edge here over-approximates rather than
    # missing it.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                bindings[bound] = _Binding(f"import {alias.name} as {bound}", None)
                opaque.add(bound)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node, module)
            inside = base.split(".")[0] == PACKAGE
            if inside and _module_path(base, package_root) is not None:
                imports.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                dotted = f"{base}.{alias.name}"
                target: tuple[str, str] | None = None
                if _module_path(dotted, package_root) is not None:
                    # A submodule, reached as an object: nothing narrows it.
                    target = (dotted, WHOLE_MODULE)
                    imports.add(dotted)
                elif inside and _module_path(base, package_root) is not None:
                    target = (base, alias.name)
                if target is not None or inside:
                    opaque.add(bound)
                bindings[bound] = _Binding(f"from {base} import {alias.name}", target)

    definitions: dict[str, list[object]] = {}
    frame: list[ast.stmt] = []
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definitions.setdefault(statement.name, []).append(statement)
        elif isinstance(statement, ast.Assign) and all(
            isinstance(target, ast.Name) for target in statement.targets
        ):
            for assigned in statement.targets:
                assert isinstance(assigned, ast.Name)
                definitions.setdefault(assigned.id, []).append(statement)
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            definitions.setdefault(statement.target.id, []).append(statement)
        elif not isinstance(statement, ast.Import | ast.ImportFrom):
            frame.append(statement)
    for bound, binding in bindings.items():
        definitions.setdefault(bound, []).append(binding)

    facts = _Facts(
        path=path,
        definitions=definitions,
        frame=frame,
        imports=tuple(sorted(imports)),
        dynamic=_dynamic_reason(tree, opaque),
    )
    _ANALYSED[key] = facts
    return facts


def _ancestors(module: str) -> list[str]:
    """The packages whose ``__init__`` an import of ``module`` runs.

    Returns
    -------
    list[str]
    """
    parts = module.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts))]


def reachable(modules: Iterable[str], root: Path) -> Reach:
    """Walk from ``modules`` over the definitions their execution can reach.

    Parameters
    ----------
    modules : Iterable[str]
        Entry points, dotted. Each is hashed whole: running a module as
        ``python -m`` executes its top level, and one module's worth of
        over-approximation is cheap beside the risk of narrowing it wrongly.
    root : Path
        The repository root; sources are looked up under ``root / "python"``.

    Returns
    -------
    Reach
        The fingerprint, the sources it covers, and the modules the walk had
        to hash whole.
    """
    package_root = root / "python"
    facts: dict[str, _Facts | None] = {}
    selected: dict[str, set[str]] = {}
    fallbacks: dict[str, str] = {}
    done: set[tuple[str, str]] = set()
    todo = [(name, WHOLE_MODULE) for name in modules if name.split(".")[0] == PACKAGE]

    def load(name: str) -> _Facts | None:
        if name not in facts:
            path = _module_path(name, package_root)
            facts[name] = None if path is None else _analyse(name, path, package_root)
        return facts[name]

    while todo:
        key = todo.pop()
        if key in done:
            continue
        done.add(key)
        name, symbol = key
        fact = load(name)
        if fact is None:
            continue
        todo += [(ancestor, MODULE_FRAME) for ancestor in _ancestors(name)]
        todo.append((name, MODULE_FRAME))
        if fact.dynamic and symbol != WHOLE_MODULE:
            fallbacks[name] = fact.dynamic
            todo.append((name, WHOLE_MODULE))
            continue

        if symbol == WHOLE_MODULE:
            if fact.dynamic:
                fallbacks[name] = fact.dynamic
            selected[name] = {WHOLE_MODULE}
            entries = [item for items in fact.definitions.values() for item in items]
            todo += [(imported, MODULE_FRAME) for imported in fact.imports]
        elif symbol == MODULE_FRAME:
            selected.setdefault(name, set()).add(MODULE_FRAME)
            entries = list(fact.frame)
            todo += [(imported, MODULE_FRAME) for imported in fact.imports]
        else:
            selected.setdefault(name, set()).add(symbol)
            entries = list(fact.definitions.get(symbol, []))

        for entry in entries:
            if isinstance(entry, _Binding):
                if entry.target is not None:
                    todo.append(entry.target)
                continue
            assert isinstance(entry, ast.stmt)
            for referenced in _referenced(entry):
                binding = next(
                    (
                        item
                        for item in fact.definitions.get(referenced, [])
                        if isinstance(item, _Binding)
                    ),
                    None,
                )
                if binding is not None and binding.target is not None:
                    todo.append(binding.target)
                if referenced in fact.definitions:
                    todo.append((name, referenced))

    hasher = hashlib.sha256()
    extension = False
    for name in sorted(selected):
        fact = facts[name]
        assert fact is not None
        hasher.update(str(fact.path.relative_to(root)).encode())
        hasher.update(b"\0")
        hashed = (
            [MODULE_FRAME, *sorted(fact.definitions)]
            if WHOLE_MODULE in selected[name]
            else sorted(selected[name])
        )
        for symbol in hashed:
            hasher.update(symbol.encode())
            hasher.update(b"\0")
            items: list[object] = (
                list(fact.frame)
                if symbol == MODULE_FRAME
                else fact.definitions.get(symbol, [])
            )
            for item in items:
                text = (
                    item.text
                    if isinstance(item, _Binding)
                    else ast.dump(item, include_attributes=False)  # type: ignore[arg-type]
                )
                extension = extension or EXTENSION in text
                hasher.update(text.encode())
                hasher.update(b"\0")
    covered: list[Path] = []
    for name in selected:
        fact = facts[name]
        assert fact is not None
        covered.append(fact.path)
    return Reach(
        fingerprint=hasher.hexdigest(),
        modules=tuple(sorted(covered)),
        fallbacks=tuple(
            f"{name}: {reason}" for name, reason in sorted(fallbacks.items())
        ),
        extension=extension,
    )


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
