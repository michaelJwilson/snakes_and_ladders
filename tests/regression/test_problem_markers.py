"""The problem axis is derived, and two readings of the source say the same thing.

Issue #614. `tests/_problems.py` reads each test module's fixture calls and
`tests/conftest.py` turns them into markers at collection. Nothing here trusts
that reading on its own: the selection is checked against `grep` over the same
files --- a text search against an `ast` parse, so a disagreement is one of them
being wrong --- and against the mini-suite below, which runs `pytest -m` for real
rather than inspecting the hook.

What each direction costs is not symmetric, and the tests are written to that.
Missing a module is a test that silently did not run, so the `grep` check is
two-sided: every module the plain call spelling names must be selected, and no
module may be selected for a problem its reachable source never mentions.
Selecting a module that only mentions a fixture --- a path handed to
`select_tests.py` as data --- costs one test run, so it is allowed and the
`ast` reading is held only to naming something real.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._problems import (
    NAMES_FIRST,
    NAMES_SECOND,
    fixtures_named_in,
    problem_names,
    restore,
    snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS = REPO_ROOT / "tests"

#: Every problem, which is every marker the hook can add.
PROBLEMS = problem_names()

#: The one module where `grep` and the scan disagree, and the scan is right:
#: it hands a quoted registry call to the regex reader it tests, so the call is
#: *data*. A text search cannot tell that from a call and a parse can, which is
#: why the scan is a parse. Asserted rather than waived, below.
QUOTED_CALLS = TESTS / "regression" / "docs" / "test_problems_tables.py"

#: Spelled rather than written, so `grep` does not find this guard's own
#: generated sources and report the guard as a module exercising every problem
#: it names --- the trap `QUOTED_CALLS` fell into.
CALL = "fixture"


def _modules() -> list[Path]:
    """Every collected test module in the suite."""
    return sorted(TESTS.rglob("test_*.py"))


def _grepped(problem: str) -> set[Path]:
    """The modules ``grep -rl 'fixture("<problem>"' tests/regression`` names."""
    found = subprocess.run(
        ["grep", "-rl", f'fixture("{problem}"', "tests/regression", "--include=*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {REPO_ROOT / line for line in found.stdout.split() if line}


def _selected(problem: str) -> set[Path]:
    """The modules the scan names for one problem."""
    return {path for path in _modules() if problem in fixtures_named_in(path)}


#: A ``from snakes_and_ladders... import ...`` statement, single or parenthesized.
#: A regex where the scan uses `ast`, so the two readings share no code.
PACKAGE_IMPORT = re.compile(
    r"from\s+snakes_and_ladders(?:\.([\w.]+))?\s+import\s+(\([^)]*\)|[^\n]*)"
)


def _imports_defining(path: Path, names: tuple[str, ...]) -> bool:
    """Whether the module imports any of `names`, read by regex not by `ast`."""
    imported: set[str] = set()
    for stem, bound in PACKAGE_IMPORT.findall(path.read_text()):
        for symbol in re.findall(r"\w+", bound):
            if symbol == "as":
                continue
            imported.add(f"{stem}.{symbol}" if stem else symbol)
    return any(
        one == name or one.startswith(f"{name}.") for name in names for one in imported
    )


def _catalogue_defines() -> dict[str, tuple[str, ...]]:
    """``problem -> defining names``, re-read from `PROBLEMS.md` here.

    Deliberately a second reader rather than `tests._problems._defining_code`:
    a guard that imports the thing it checks agrees with it by construction.
    """
    defines: dict[str, list[str]] = {}
    for line in (REPO_ROOT / "PROBLEMS.md").read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("Problem", "---"):
            continue
        names = re.findall(r"`([^`]+)`", cells[3])
        for key in re.findall(r"`([^`]+)`", cells[1]):
            defines.setdefault(key, []).extend(names)
    return {key: tuple(names) for key, names in defines.items()}


def _reachable_source(path: Path) -> str:
    """One module's source, with that of every `tests` module it imports from.

    The constants naming a fixture as a path --- ``SMALL_SITES`` and its
    siblings --- live in `tests/_fixtures.py`, so the problem name a module
    reaches through one of them is nowhere in the module itself.
    """
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            source = REPO_ROOT / Path(*node.module.split(".")).with_suffix(".py")
            if node.module.split(".")[0] == "tests" and source.is_file():
                text += source.read_text()
    return text


def _computed_call(path: Path) -> bool:
    """Whether the module names a problem with something other than a literal.

    The names of the calls are shared with the scan and the reading is not: this
    asks only whether the problem argument is a string, where the scan goes on
    to follow a constant to the string it holds.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        called = node.func.id
        index = 0 if called in NAMES_FIRST else 1 if called in NAMES_SECOND else None
        if index is None:
            continue
        if len(node.args) <= index or not isinstance(node.args[index], ast.Constant):
            return True
    return False


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_module_the_call_spelling_names_is_selected(problem: str) -> None:
    """`grep`'s answer is contained in the scan's, problem by problem.

    The lower bound, and the one the ticket states: a module that writes
    the registry call in full --- the problem as a literal --- is a module that
    `-m <problem>` must collect. The scan may name more --- `at_fixture`, a path literal, a
    constant it followed --- and `test_no_module_is_selected_without_the_name`
    is what stops "more" from meaning "all of them".
    """
    missing = _grepped(problem) - _selected(problem) - {QUOTED_CALLS}
    assert not missing, (
        f"grep names {sorted(str(p.relative_to(REPO_ROOT)) for p in missing)} for "
        f"{problem!r} and the scan does not select them: `-m {problem}` would run "
        "a subset of the problem's tests while looking like it ran all of them."
    )


@pytest.mark.critical
@pytest.mark.structural
def test_a_quoted_call_is_data_and_is_not_read_as_one() -> None:
    """The single place `grep` and the scan differ, and why the parse wins.

    `test_problems_tables.py` feeds a quoted registry call naming `mixture` to
    the regex reader it tests. `grep` counts it, so a text-search implementation
    of this axis would run the whole `mixture` selection whenever that guard
    changed; the parse sees a string and not a call.
    """
    assert QUOTED_CALLS in _grepped("mixture")
    assert "mixture" not in fixtures_named_in(QUOTED_CALLS)
    calls = [
        node
        for node in ast.walk(ast.parse(QUOTED_CALLS.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fixture"
    ]
    assert not calls, "the module now calls `fixture`, so this exception is stale"


@pytest.mark.critical
@pytest.mark.structural
def test_no_module_is_selected_without_evidence_it_exercises_the_problem() -> None:
    """The upper bound: a selection must point at something written down.

    Without this the scan could return every problem for every module and pass
    every containment above. Two kinds of evidence count, because there are two
    readings: the problem's **name**, in the module or in a `tests` module it
    imports from, or an **import** of code `PROBLEMS.md` says defines it.

    The second is why this stopped being a name test (issue #622).
    `search/test_maxflow.py` exercises the Potts lattice and never writes
    "potts_lattice" anywhere --- it imports `search.maxflow` and builds its
    lattices from literals --- and that module is one of the two #614 was
    opened about. Requiring the name would refuse exactly the modules the
    catalogue reading exists to reach.

    It stays weaker than the scan, and shares no code with it: the names come
    from the source text, and the imports are re-read here from the catalogue
    rather than taken from `_defining_code`.
    """
    everything = frozenset(PROBLEMS)
    defines = _catalogue_defines()
    unevidenced = [
        f"{path.relative_to(REPO_ROOT)}: {problem}"
        for path in _modules()
        if (named := fixtures_named_in(path)) != everything
        for problem in named
        if problem not in _reachable_source(path)
        and not _imports_defining(path, defines.get(problem, ()))
    ]
    assert not unevidenced, (
        f"{len(unevidenced)} module/problem pairs were selected with neither the "
        f"problem's name in reachable source nor an import of the code "
        f"PROBLEMS.md says defines it: {unevidenced[:10]}"
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_two_modules_the_axis_was_opened_about_are_selected() -> None:
    """The check #614 states as its motivation, and #619 shipped without.

    `search/test_maxflow.py` and `search/test_alpha_expansion.py` are "that
    problem's ground-state tests" in #614's own words, and the derived axis
    gave them no marker at all: neither loads a fixture. They sweep lattices
    built from literals --- fifteen in one module, each chosen for the property
    under test, zero coupling against a dominant one against a negative one
    against a periodic boundary --- so there is no single declared instance to
    load, and declaring fifteen fixtures to carry fifteen deliberate variations
    would make the registry a list of test arguments (issue #622).

    They reach the problem the way `PROBLEMS.md` says a module does: by
    importing `search.maxflow` and `sim.graph.lattice_graph`. A green
    ``-m potts_lattice`` run over a broken solver is what this refuses.
    """
    selected = _selected("potts_lattice")
    missing = [
        name
        for name in ("test_maxflow.py", "test_alpha_expansion.py")
        if TESTS / "regression" / "search" / name not in selected
    ]
    assert not missing, f"-m potts_lattice does not select {missing}"


@pytest.mark.critical
@pytest.mark.structural
def test_a_module_selected_for_every_problem_really_has_a_computed_name() -> None:
    """Selecting every problem is the unresolvable answer, not a scan giving up.

    It costs one module's tests on every problem's selection, so it must be
    caused rather than reached: such a module names its problem with an
    expression --- a loop variable over the registry, an unpacked tuple --- and
    not with a string a reading could have had.
    """
    everything = frozenset(PROBLEMS)
    unresolved = [p for p in _modules() if fixtures_named_in(p) == everything]
    assert unresolved, "no module exercises the unresolvable branch"
    for path in unresolved:
        assert _computed_call(path), (
            f"{path.relative_to(REPO_ROOT)} is selected for every problem but "
            "names none computationally, so the scan lost a name it could read"
        )


@pytest.mark.critical
@pytest.mark.structural
def test_the_problem_crosses_module_directories() -> None:
    """The case the substring and the directory both miss.

    `-k potts` matches node ids, and `tests/regression/search/` is a directory;
    the Potts lattice is exercised from `search/`, `likelihood/` and `sim/` at
    once, which is the whole reason for a third axis.
    """
    selected = _selected("potts_lattice")
    directories = {path.parent.name for path in selected}
    assert {"search", "likelihood", "sim"} <= directories, (
        f"potts_lattice is selected only under {sorted(directories)}"
    )
    assert _grepped("potts_lattice") <= selected


@pytest.mark.critical
@pytest.mark.structural
def test_the_collected_items_carry_what_their_module_names(
    request: pytest.FixtureRequest,
) -> None:
    """The hook applied the scan to this very session's items.

    Reading `session.items` rather than a second collection: it is free, it is
    the real tree, and it covers whatever selection the invocation asked for,
    so the check widens with the run rather than being pinned to one tier.
    """
    problems = set(PROBLEMS)
    items = [item for item in request.session.items if item.path.is_relative_to(TESTS)]
    assert items, "the session collected nothing under tests/"
    for item in items:
        carried = {marker.name for marker in item.iter_markers()} & problems
        assert carried == set(fixtures_named_in(item.path)), (
            f"{item.nodeid} carries {sorted(carried)} and its module names "
            f"{sorted(fixtures_named_in(item.path))}"
        )


@pytest.mark.critical
@pytest.mark.structural
def test_the_cache_is_reread_when_the_file_changes(tmp_path: Path) -> None:
    """What makes the cross-session cache safe, rather than fast and wrong.

    A stale entry is a module selected for the problem it used to load, which
    is worse than the parse it saves. Both halves of the key are exercised, the
    other held fixed with `os.utime`: an edit within one clock tick moves the
    size and not the time, and a checkout moves the time and not necessarily
    the size.
    """

    def write(problem: str) -> None:
        module.write_text(f'def test_one() -> None:\n    {CALL}("{problem}", "ci")\n')

    module = tmp_path / "test_cached.py"
    write("hmm")
    assert fixtures_named_in(module) == {"hmm"}

    stamp = module.stat()
    write("turbo")
    os.utime(module, (stamp.st_atime, stamp.st_mtime))
    assert module.stat().st_size != stamp.st_size
    assert fixtures_named_in(module) == {"turbo"}, "the size did not invalidate"

    write("bicycle")
    assert fixtures_named_in(module) == {"bicycle"}
    stamp = module.stat()
    write("mixture")
    os.utime(module, (stamp.st_atime, stamp.st_mtime + 1))
    assert module.stat().st_size == stamp.st_size
    assert fixtures_named_in(module) == {"mixture"}, "the mtime did not invalidate"

    saved = snapshot()
    assert saved[str(module)][3] == ["mixture"]
    restore(saved)
    assert fixtures_named_in(module) == {"mixture"}


@pytest.mark.structural
@pytest.mark.edge_case
def test_a_damaged_cache_is_ignored_rather_than_raised_on() -> None:
    """A cache decides when work is redone, never whether a check runs.

    `infra/CLAUDE.md`'s rule, applied to this one: the file is JSON a previous
    session wrote and anything may have happened to it since, so a malformed
    entry is dropped and rescanned rather than ending collection.
    """
    damaged: tuple[Any, ...] = (
        None,
        [],
        {"a": "b"},
        {"a": [1, 2]},
        {"a": [1.0, 2, [3]]},
    )
    for entry in damaged:
        restore(entry)
    module = TESTS / "_problems.py"
    assert fixtures_named_in(module) == frozenset()


# --- the mini-suite: `-m` run for real, not inspected ----------------------


@pytest.fixture(scope="module")
def mini_suite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four modules naming a problem four ways, outside the real tree.

    Nothing here is executed: the runs below are ``--collect-only``, and the
    scan is static, so a call to an undefined name is exactly as readable as a
    call to the real one. The suite registers the kind and tier markers itself
    rather than borrowing `pyproject.toml`, so that `pytest` roots here and the
    node ids it prints are the three file names below.
    """
    directory = tmp_path_factory.mktemp("problem_markers")
    (directory / "conftest.py").write_text(
        f"import sys\n\nsys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'infra')!r})\n\n"
        "from test_kinds import KINDS, SCHEDULING  # noqa: E402\n"
        "from tests.conftest import (  # noqa: E402,F401\n"
        "    pytest_collection_modifyitems,\n"
        ")\n"
        "from tests.conftest import pytest_configure as _configure  # noqa: E402\n\n\n"
        "def pytest_configure(config):\n"
        "    for name in KINDS + SCHEDULING:\n"
        '        config.addinivalue_line("markers", name)\n'
        "    _configure(config)\n"
    )
    (directory / "test_literal.py").write_text(
        "import pytest\n\n\n@pytest.mark.critical\ndef test_gating() -> None:\n"
        f'    {CALL}("potts_lattice", "ci")\n\n\ndef test_plain() -> None:\n'
        f'    {CALL}("potts_lattice", "ci")\n'
    )
    (directory / "test_computed.py").write_text(
        f"def test_computed(name: str) -> None:\n    {CALL}(name, 'ci')\n"
    )
    (directory / "test_typo.py").write_text(
        "import pytest\n\n\n@pytest.mark.potts_latice\ndef test_typo() -> None:\n"
        "    pass\n"
    )
    return directory


def _collected(suite: Path, expression: str, *files: str) -> tuple[int, set[str]]:
    """Run ``pytest --collect-only -m <expression>`` over the mini-suite.

    A subprocess and not `pytester`: the claim is that the hook in
    `tests/conftest.py` runs before `pytest`'s own deselection, and only a real
    session orders hooks.
    """
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *files,
            "--collect-only",
            "-q",
            "--strict-markers",
            "-p",
            "no:cacheprovider",
            "-m",
            expression,
        ],
        cwd=suite,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = {line for line in run.stdout.splitlines() if "::" in line}
    return run.returncode, collected


@pytest.mark.structural
def test_a_computed_name_is_selected_by_every_problem(mini_suite: Path) -> None:
    """Step 3 of the plan, run rather than reasoned about.

    A module the scan cannot read is selected by `potts_lattice` *and* by
    `hmm`, and the module that names one problem in full is selected by the
    first only --- `select_tests.py`'s rule, that an unreadable input selects
    everything, applied to the axis rather than to the changed-file map.
    """
    files = ("test_literal.py", "test_computed.py")
    _, potts = _collected(mini_suite, "potts_lattice", *files)
    _, hmm = _collected(mini_suite, "hmm", *files)

    assert potts == {
        "test_literal.py::test_gating",
        "test_literal.py::test_plain",
        "test_computed.py::test_computed",
    }
    assert hmm == {"test_computed.py::test_computed"}


@pytest.mark.structural
def test_the_problem_marker_intersects_the_tier(mini_suite: Path) -> None:
    """A third axis is worth nothing if it does not compose with the other two.

    `-m "potts_lattice and critical"` is the one test that is both, not the
    three that are either --- which also pins the hook ahead of `pytest`'s own
    deselection: added afterwards, the marker would land on items already
    thrown away and both selections would be empty.
    """
    files = ("test_literal.py", "test_computed.py")
    _, both = _collected(mini_suite, "potts_lattice and critical", *files)
    _, tier = _collected(mini_suite, "critical", *files)

    assert both == {"test_literal.py::test_gating"}
    assert tier == {"test_literal.py::test_gating"}


@pytest.mark.structural
@pytest.mark.edge_case
def test_a_misspelled_problem_marker_fails_collection(mini_suite: Path) -> None:
    """Registration is what keeps `--strict-markers` able to refuse a typo.

    The hook adds the markers, so no author writes one --- but an author who
    writes ``@pytest.mark.potts_latice`` by hand must still be told, and that
    holds only while `pytest_configure` registers the real names.
    """
    status, _ = _collected(mini_suite, "potts_lattice", "test_typo.py")
    assert status != 0, "an unregistered problem marker was accepted"
