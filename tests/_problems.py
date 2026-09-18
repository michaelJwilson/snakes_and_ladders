"""Which problem each test module exercises, read from the fixture it loads
and from the code it imports.

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

**A call is not the only way a module names its problem** (issue #622). Reading
calls alone left 138 of 258 modules unmarked, 78 of them application code, and
among them both modules #614 gives as its motivation: `search/test_maxflow.py`
and `search/test_alpha_expansion.py` build their lattices from literals and
never load a fixture, so `-m potts_lattice` missed that problem's ground-state
tests. A module that *sweeps* an instance has none to declare --- fifteen
lattices in one module, each chosen for the property under test --- so the
second reading is over the module's **imports**, against `PROBLEMS.md`'s
**Defines** column, which the catalogue already states as the rule: *a test
module importing any of it exercises the problem*. That file is held true by
`tests/regression/test_problems_catalogue.py`, which resolves every symbol it
names, so this reading is as current as the code and not a second map to
maintain. The two readings are unioned; neither can take a marker away.

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
NAMES_FIRST = ("fixture", "path_of", "load_fixture", "fixture_path")

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

#: The catalogue whose **Defines** column says which code *is* each problem.
#: `PROBLEMS.md` states the rule this file implements --- "a test module
#: importing any of it exercises the problem" --- and
#: `tests/regression/test_problems_catalogue.py` resolves every symbol it names,
#: so a row cannot outlive its code. Reading it is reading a fact CI already
#: holds true, which is what a hand-written module-to-problem map is not.
CATALOGUE = REPO_ROOT / "PROBLEMS.md"

#: A ``| cell | cell |`` row of that table, and the ``` `name` ``` spans inside
#: a cell. The table is the only part of the file with four columns.
CATALOGUE_ROW = re.compile(r"^\|(.+)\|\s*$")
CATALOGUE_CODE = re.compile(r"`([^`]+)`")

#: Where `tests/conftest.py` keeps this between sessions, under `.pytest_cache`.
CACHE_KEY = "problems/fixtures-named"

#: ``path -> ((mtime, size, registry), names)``. Size as well as time because a
#: checkout that restores a file writes the recorded time with different bytes,
#: and the registry because `_scan` reads it as well as the file.
_CACHE: dict[str, tuple[tuple[float, int, str], frozenset[str]]] = {}


def _registry_stamp() -> str:
    """The declared problems, as one string the cache key can carry.

    A renamed or added fixture directory changes what `_scan` returns for a
    module whose own bytes have not moved: the filter narrows, and an
    unreadable module's answer is the whole set. Keying on the module alone
    left a stale name to reach `add_marker`, which `--strict-markers` raises
    inside `pytest_collection_modifyitems` --- a collection error for the whole
    session, surviving until `.pytest_cache` is deleted.

    `PROBLEMS.md` is in the key for the same reason and one more: editing a
    **Defines** cell changes which modules carry which marker without touching
    a single test file, so a cache keyed on the tests alone would answer from
    the catalogue that was current when it was written.
    """
    stat = CATALOGUE.stat()
    return f"{','.join(problem_names())};{stat.st_mtime};{stat.st_size}"


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
    """`PROBLEMS.md`'s **Defines** column, as ``(dotted name, keys)`` pairs.

    Returns
    -------
    tuple[tuple[str, frozenset[str]], ...]
        Each defining module or symbol, rooted at `snakes_and_ladders` and
        written without it, against the problem keys of every row naming it.
        A name appearing in two rows carries both, which is why the value is a
        set: `sec:phylo` is stated once and declared at two instances.

    Notes
    -----
    Read from the table rather than from a copy of it. The alternative
    considered was a module-level ``PROBLEM = "<name>"`` constant in each test,
    which is the map this ticket exists to remove, one level up: rewrite a
    module to exercise a different model, forget the constant, and the axis is
    confidently wrong --- worse than visibly empty (issue #622).
    """
    rows: dict[str, set[str]] = {}
    for line in CATALOGUE.read_text().splitlines():
        match = CATALOGUE_ROW.match(line)
        if match is None:
            continue
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if len(cells) != 4 or cells[0] in ("Problem", "---"):
            continue
        keys = frozenset(CATALOGUE_CODE.findall(cells[1]))
        for name in CATALOGUE_CODE.findall(cells[3]):
            rows.setdefault(name, set()).update(keys)
    return tuple((name, frozenset(keys)) for name, keys in sorted(rows.items()))


def _imported_code(tree: ast.Module) -> set[str]:
    """Every `snakes_and_ladders` name the module imports, without that prefix.

    Both spellings of one import are recorded at their full depth ---
    ``from snakes_and_ladders.sim import jc`` and
    ``from snakes_and_ladders.sim.jc import simulate`` both yield a name under
    ``sim.jc`` --- so the match below is a prefix test and not an equality.
    """
    package = "snakes_and_ladders"
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

    This covers what no fixture call can. A module that sweeps an instance ---
    `search/test_alpha_expansion.py` builds fifteen lattices, each chosen for
    the property under test: zero coupling, a dominant one, a negative one, a
    periodic boundary --- has no single declared instance to load, and
    declaring fifteen fixtures to carry fifteen deliberate variations would
    make the registry a list of test arguments (issue #622).
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
    # `_scan` reads the registry as well as this file: `problem_names()` is both
    # the filter and the whole answer for an unreadable module. A key over the
    # module alone survives a renamed fixture directory, and the stale name then
    # reaches `add_marker`, which `--strict-markers` turns into a collection
    # error for the session --- persisting until someone deletes `.pytest_cache`.
    stamp = (stat.st_mtime, stat.st_size, _registry_stamp())
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    found = _scan(path)
    _CACHE[key] = (stamp, found)
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
            case [
                float() | int() as mtime,
                int() as size,
                str() as registry,
                list() as names,
            ] if all(isinstance(name, str) for name in names):
                _CACHE[str(key)] = ((float(mtime), size, registry), frozenset(names))


def snapshot() -> dict[str, list[Any]]:
    """What this session learned, for the next one, minus files since deleted.

    Returns
    -------
    dict[str, list[Any]]
        JSON-serializable, one entry per file read.
    """
    return {
        key: [mtime, size, registry, sorted(names)]
        for key, ((mtime, size, registry), names) in _CACHE.items()
        if Path(key).is_file()
    }
