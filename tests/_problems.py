"""Which problem each test module exercises, read from the fixture it loads.

Issue #614. The suite cuts by *kind* (what refereed a test), by *tier* (when it
runs) and, through `infra/select_tests.py`, by *module*. None of those cuts by
problem, and a problem crosses modules: the Potts lattice is exercised from
`tests/regression/search/` and `tests/regression/likelihood/` at once. `-k potts`
under-selects, because `test_maxflow.py` carries no "potts" in its name.

This module supplies the third axis and derives it rather than asking for it. A
hand-written map of test to problem is the defect one level up --- a list that
goes stale in silence, which is what `SEAMS.md` was before #601 deleted it. A
test already names its problem by loading its instance: `fixture("potts_lattice",
tier)`, `at_fixture("instance", "tree_scale")`, or the path
``potts_lattice/ci.yaml``. Reading that call is reading a fact the test has to
keep true to run at all.

`tests/conftest.py` turns what this returns into markers at collection. The kind
markers are unaffected and are still read from the source by
`tests/regression/test_test_kinds.py`, whose rule that a marker applied by a
`conftest.py` cannot satisfy the guard is about *kind*: a kind is a claim an
author makes and a reviewer reads in the diff, and a problem is a fact about the
call, so the two are established opposite ways on purpose.

**The scan is cached across invocations, not only within one.** Collection runs
once per process and a developer types several a minute, so a cache the process
throws away saves nothing where the cost is felt. The reading of each file is
keyed on its size and modification time and kept in `pytest`'s own cache
directory; `tests/conftest.py` restores it before collection and writes it back
after. The numbers are in the pull request for issue #614.

**An unreadable name selects everything.** A `fixture(name, tier)` whose name is
computed cannot be read statically, and such a module is marked with every
problem rather than with none --- `infra/select_tests.py`'s rule for a change it
cannot attribute, for its reason: a selection that guesses narrowly is a test
that silently did not run.
"""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path
from typing import Any

from snakes_and_ladders.sim.fixtures import FIXTURES_DIR, problems

#: The repository root, from this file: `tests.` imports are resolved under it.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Calls whose *first* positional argument names the problem.
#: `snakes_and_ladders.sim.fixtures.fixture` and `path_of`.
NAMES_FIRST = ("fixture", "path_of")

#: Calls whose *second* positional argument names the problem: the
#: parameterizing wrappers in `tests/_scale.py`, which pass it through to
#: `fixture` one frame down where no scan of this module would see it.
NAMES_SECOND = ("at_fixture", "at_bin")

#: A fixture named as its path, ``.../<problem>/<tier>.yaml`` --- how
#: `load_fixture`, `fixture_path` and ``FIXTURES_DIR / ...`` spell it. Anchored
#: at the end and not at the start, because a module writes the path from the
#: fixtures directory (``"hmm/ci.yaml"``) or from the repository root
#: (``"tests/regression/fixtures/tree_search/release.yaml"``), and both name the
#: problem in the directory above the file.
FIXTURE_PATH = re.compile(r"([a-z_0-9]+)/[a-z]+\.yaml$")

#: Where `tests/conftest.py` keeps this between sessions, under `.pytest_cache`.
CACHE_KEY = "problems/fixtures-named"

#: ``path -> (mtime, size, names)``. Size as well as time because a checkout
#: that restores a file writes the recorded time with different bytes.
_CACHE: dict[str, tuple[float, int, frozenset[str]]] = {}


@cache
def problem_names(directory: Path = FIXTURES_DIR) -> tuple[str, ...]:
    """Every problem the registry declares, which is every marker name.

    Parameters
    ----------
    directory : Path
        Where the fixtures live.

    Returns
    -------
    tuple[str, ...]
        Sorted, so the registration order is the same on every host.

    Raises
    ------
    ValueError
        If a problem's name cannot be a marker --- `-m` parses its expression
        as Python, so a name that is not an identifier would register a marker
        no selection could ask for. Refused here rather than dropped, because a
        silently missing problem is the failure this axis exists to prevent.
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
    """`_string_constants` of a module imported from, cached by its own mtime.

    Around fifty modules import a fixture path constant from `tests/_fixtures.py`,
    and parsing it once per importer was most of what the scan cost.
    """
    return _string_constants(ast.parse(source.read_text()))


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"``: annotated, plain, or unpacked.

    Unpacked because the problem and its tier are often bound together ---
    ``PROBLEM, TIER = "planted_glass", Scale.CI`` --- and reading only the
    single-target form would call that module's `fixture` call computed and
    select it for all 18 problems on the strength of a comma.
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
    """``name -> string`` for the module's own constants and the ones it imports.

    One hop, and only into the `tests` package: `SMALL_SITES` and its siblings
    live in `tests/_fixtures.py` and are imported by name, so a scan that read
    the importing module alone would call every use of them computed and mark
    those modules with every problem. `ruff`'s unused-import rule is what makes
    taking the whole `import` list safe --- a name imported is a name used.
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
            called = function.attr
        index = 0 if called in NAMES_FIRST else 1 if called in NAMES_SECOND else None
        if index is None or len(node.args) <= index:
            continue
        argument = node.args[index]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            found.add(argument.value)
        elif isinstance(argument, ast.Name) and argument.id in bound:
            found.add(bound[argument.id])
        else:
            return None
    return found


def _scan(path: Path) -> frozenset[str]:
    """The problems one module names, every problem if one name is computed."""
    tree = ast.parse(path.read_text())
    bound = _bound_strings(path, tree)
    declared = set(problem_names())

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
    return frozenset((called | paths) & declared)


def fixtures_named_in(path: Path) -> frozenset[str]:
    """The problems one test module exercises, cached by size and mtime.

    Parameters
    ----------
    path : Path
        A Python source file.

    Returns
    -------
    frozenset[str]
        Registry problem names; every one of them if the module names a fixture
        this scan cannot read, and none if it names no fixture at all.
    """
    key = str(path)
    stat = path.stat()
    cached = _CACHE.get(key)
    if cached is not None and (cached[0], cached[1]) == (stat.st_mtime, stat.st_size):
        return cached[2]
    found = _scan(path)
    _CACHE[key] = (stat.st_mtime, stat.st_size, found)
    return found


def restore(entries: Any) -> None:
    """Seed the cache from a previous session's, ignoring anything malformed.

    The argument is JSON a previous session wrote and a later one may have
    edited or truncated, so every entry is checked rather than trusted: a cache
    that cannot be read is a scan, and a cache that raises is a collection
    error for a saving.
    """
    if not isinstance(entries, dict):
        return
    for key, entry in entries.items():
        match entry:
            case [float() | int() as mtime, int() as size, list() as names] if all(
                isinstance(name, str) for name in names
            ):
                _CACHE[str(key)] = (float(mtime), size, frozenset(names))


def snapshot() -> dict[str, list[Any]]:
    """What this session learned, for the next one, minus files since deleted.

    Returns
    -------
    dict[str, list[Any]]
        JSON-serializable, one entry per file read.
    """
    return {
        key: [mtime, size, sorted(names)]
        for key, (mtime, size, names) in _CACHE.items()
        if Path(key).is_file()
    }
