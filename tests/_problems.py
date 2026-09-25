"""Which problem each test module exercises, read from the fixture it loads
and from the code it imports.

Issue #614. `-k potts` under-selects, so the axis is derived: from the fixture
a module loads (`fixture(...)`, `at_fixture(...)`, ``potts_lattice/ci.yaml``)
and from the code it imports that `PROBLEMS.md`'s **Defines** column names
(issue #622: calls alone left 138 of 258 modules unmarked), unioned. Kind
markers stay author-written (`test_test_kinds.py`). The scan is cached in
`pytest`'s cache by size and mtime; a computed fixture name selects every
problem, as in `infra/select_tests.py`: a narrow guess is a test not run.
"""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path
from typing import Any

import catalogue
from sal.sim.fixtures import FIXTURES_DIR, problems

#: The repository root, from this file: `tests.` imports are resolved under it.
from tests._paths import REPO_ROOT

#: Calls whose *first* positional argument names the problem.
#: `sal.sim.fixtures.fixture` and `path_of`, and the
#: `tests/_fixtures.py` loaders.
NAMES_FIRST = (
    "fixture",
    "path_of",
    "load_fixture",
    "fixture_path",
    "simulated_alignment",
)

#: Calls whose *second* positional argument names the problem: the
#: parameterizing wrappers in `tests/_scale.py`, which pass it through to
#: `fixture` one frame down where no scan of this module would see it.
NAMES_SECOND = ("at_fixture", "at_bin")

#: A fixture named as its path, ``.../<problem>/<tier>.yaml``, anchored at the
#: end: written from the fixtures directory or from the repository root.
FIXTURE_PATH = re.compile(r"([a-z_0-9]+)/[a-z]+\.yaml$")

#: The catalogue whose **Defines** column says which code is each problem;
#: `test_problems_catalogue.py` resolves every symbol. Parsed by
#: `infra/catalogue.py`, the one reader (issue #863).
CATALOGUE = catalogue.CATALOGUE

#: Where `tests/conftest.py` keeps this between sessions, under `.pytest_cache`.
CACHE_KEY = "problems/fixtures-named"

#: ``path -> ((mtime, size, registry), names)``. Size as well as time because a
#: checkout that restores a file writes the recorded time with different bytes,
#: and the registry because `_scan` reads it as well as the file.
_CACHE: dict[str, tuple[tuple[float, int, str], frozenset[str]]] = {}


def _registry_stamp() -> str:
    """The declared problems and `PROBLEMS.md`, as one string the cache key carries.

    A stale name reaching `add_marker` is a session-wide `--strict-markers` error.
    """
    stat = CATALOGUE.stat()
    return f"{','.join(problem_names())};{stat.st_mtime};{stat.st_size}"


@cache
def problem_names(directory: Path = FIXTURES_DIR) -> tuple[str, ...]:
    """Every problem the registry declares, sorted: every marker name.

    Refuses a name that is not an identifier, which `-m` could never select.
    """
    names = problems(directory)
    unusable = [name for name in names if not name.isidentifier()]
    if unusable:
        msg = (
            f"fixture problem(s) {unusable} cannot be marker names: `-m` parses "
            "its expression as Python, so a problem directory must be named as "
            "an identifier."
        )
        raise ValueError(msg)
    return names


@cache
def _imported_constants(source: Path, mtime: float) -> dict[str, str]:  # noqa: ARG001
    """`_string_constants` of an imported module, cached by its own mtime."""
    return _string_constants(ast.parse(source.read_text()))


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"``: annotated, plain, or unpacked.

    Unpacked: ``PROBLEM, TIER = "planted_glass", Scale.CI`` is common.
    """
    found: dict[str, str] = {}
    pairs: list[tuple[ast.expr, ast.expr]] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs.append((node.target, node.value))
        elif isinstance(node, ast.Assign):
            pairs += [(target, node.value) for target in node.targets]
    while pairs:
        target, value = pairs.pop()
        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            pairs += list(zip(target.elts, value.elts, strict=False))
        elif (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            found[target.id] = value.value
    return found


def _bound_strings(path: Path, tree: ast.Module) -> dict[str, str]:
    """``name -> string`` for the module's constants and those it imports from `tests`.

    One hop; `ruff`'s unused-import rule makes a name imported a name used.
    """
    bound = _string_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        if node.module is None or node.module.split(".")[0] != "tests":
            continue
        source = REPO_ROOT / Path(*node.module.split(".")).with_suffix(".py")
        if source == path or not source.is_file():
            continue
        constants = _imported_constants(source, source.stat().st_mtime)
        for alias in node.names:
            if alias.name in constants:
                bound[alias.asname or alias.name] = constants[alias.name]
    return bound


def _called_names(tree: ast.Module, bound: dict[str, str]) -> set[str] | None:
    """The problems the module's fixture *calls* name, or None if one is computed."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        called = function.id if isinstance(function, ast.Name) else None
        if called is None and isinstance(function, ast.Attribute):
            # `pytest.fixture` shares a name with ours and is everywhere. Its
            # decorator form takes no positional argument, so without this the
            # rule below reads every `@pytest.fixture(scope=...)` as a registry
            # call it cannot resolve, and the module collects every problem.
            base = function.value
            if isinstance(base, ast.Name) and base.id == "pytest":
                continue
            called = function.attr
        index = 0 if called in NAMES_FIRST else 1 if called in NAMES_SECOND else None
        if index is None:
            continue
        if len(node.args) <= index:
            # The call is one of ours and the problem is not positional --- a
            # keyword form, or `fixture(*DECLARED)`. Unreadable, so every
            # problem, never none: a module that silently carried no marker is
            # the defect this axis exists to remove.
            return None
        argument = node.args[index]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            found.add(argument.value)
        elif isinstance(argument, ast.Name) and argument.id in bound:
            found.add(bound[argument.id])
        else:
            return None
    return found


@cache
def _defining_code() -> tuple[tuple[str, frozenset[str]], ...]:
    """`PROBLEMS.md`'s **Defines** column, as ``(dotted name, problem keys)`` pairs.

    Read from the table, not a per-module constant that goes stale (issue #622).
    """
    return tuple(catalogue.defines().items())


def _imported_code(tree: ast.Module) -> set[str]:
    """Every `sal` name the module imports, full depth, unprefixed.

    Matched by prefix: ``sim import jc`` and ``sim.jc import x`` both reach ``sim.jc``.
    """
    package = "sal"
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(f"{package}."):
                    imported.add(alias.name[len(package) + 1 :])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module != package and not module.startswith(f"{package}."):
                continue
            stem = module[len(package) + 1 :]
            for alias in node.names:
                imported.add(f"{stem}.{alias.name}" if stem else alias.name)
    return {name for name in imported if name}


def _catalogued_names(tree: ast.Module) -> set[str]:
    """The problems a module exercises, from what it imports.

    Covers sweeps: `test_alpha_expansion.py` builds fifteen lattices (issue #622).
    """
    imported = _imported_code(tree)
    found: set[str] = set()
    for name, keys in _defining_code():
        prefix = f"{name}."
        if any(one == name or one.startswith(prefix) for one in imported):
            found |= keys
    return found


def _scan(path: Path) -> frozenset[str]:
    """The problems one module names, every problem if one name is computed."""
    tree = ast.parse(path.read_text())
    bound = _bound_strings(path, tree)
    declared = set(problem_names())

    catalogued = _catalogued_names(tree)

    called = _called_names(tree, bound)
    if called is None:
        return frozenset(declared)

    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    paths = {
        match.group(1)
        for text in literals | set(bound.values())
        if (match := FIXTURE_PATH.search(text)) is not None
    }
    return frozenset((called | paths | catalogued) & declared)


def fixtures_named_in(path: Path) -> frozenset[str]:
    """The problems one module exercises, cached by size and mtime.

    Every problem if a fixture name is unreadable; none if it names no fixture.
    """
    key = str(path)
    stat = path.stat()
    # Keyed on the registry too: a renamed fixture directory would otherwise
    # reach `add_marker` and fail collection under `--strict-markers`.
    stamp = (stat.st_mtime, stat.st_size, _registry_stamp())
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    found = _scan(path)
    _CACHE[key] = (stamp, found)
    return found


def restore(entries: Any) -> None:
    """Seed the cache from a previous session's, ignoring anything malformed.

    An unreadable cache is a scan; a raising one would be a collection error.
    """
    if not isinstance(entries, dict):
        return
    for key, entry in entries.items():
        match entry:
            case [
                float() | int() as mtime,
                int() as size,
                str() as registry,
                list() as names,
            ] if all(isinstance(name, str) for name in names):
                _CACHE[str(key)] = ((float(mtime), size, registry), frozenset(names))


def snapshot() -> dict[str, list[Any]]:
    """What this session learned, for the next one, minus files since deleted."""
    return {
        key: [mtime, size, registry, sorted(names)]
        for key, ((mtime, size, registry), names) in _CACHE.items()
        if Path(key).is_file()
    }
