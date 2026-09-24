"""The judged-coverage recut reads a run and counts what a judging test reached.

Issue #729. `infra/coverage_recut.py` is the guard that makes the coverage
number honest about what "validated" means, so it is checked against a run it
did not produce: a `.coverage` file written here through `coverage`'s own data
API, with one context per test and the import context, over a module whose
statements are counted by hand. Every figure below is an integer the reader
can recount from the fixture, and the guard is made to fail on a floor one
statement above what the run reaches.
"""

from __future__ import annotations

from pathlib import Path

import coverage_recut
import pytest
from coverage.sqldata import CoverageData
from gates import (
    COVERAGE_GUARDS,
    JUDGED_COVERAGE,
    UNJUDGED_COVERAGE,
    JudgedCoverage,
)

from tests._paths import REPO_ROOT

#: Eight statements. Five run at import -- the two assignments and the three
#: `def` lines -- and each body is one statement a test may reach.
SOURCE = """\
X = 1
Y = 2


def judged():
    return X + Y


def smoke():
    return X - Y


def never():
    return X * Y
"""

#: The ids the contexts name, and what each carries.
MARKERS = {
    "tests/regression/test_a.py::test_oracle": frozenset({"oracle", "potts_lattice"}),
    "tests/regression/test_a.py::test_smoke[0]": frozenset({"smoke", "potts_lattice"}),
    "tests/regression/test_b.py::test_infra": frozenset({"infra"}),
}


def _run(tmp_path: Path) -> tuple[Path, Path]:
    """A package with two modules and a run over it, written through the API."""
    package = tmp_path / "package"
    (package / "search").mkdir(parents=True)
    root_module = package / "root.py"
    root_module.write_text(SOURCE)
    search_module = package / "search" / "kernel.py"
    search_module.write_text(SOURCE)
    data_file = tmp_path / ".coverage"
    data = CoverageData(basename=str(data_file))
    data.set_context("")
    data.add_lines(
        {str(root_module): [1, 2, 5, 9, 13], str(search_module): [1, 2, 5, 9, 13]}
    )
    data.set_context("tests/regression/test_a.py::test_oracle|run")
    data.add_lines({str(root_module): [6]})
    data.set_context("tests/regression/test_a.py::test_smoke[0]|run")
    data.add_lines({str(root_module): [10], str(search_module): [10]})
    data.set_context("tests/regression/test_b.py::test_infra|setup")
    data.add_lines({str(search_module): [6]})
    data.write()
    return package, data_file


@pytest.mark.critical
@pytest.mark.infra
def test_the_recut_counts_the_judged_lines_and_the_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per file: 8 statements, 5 at import; a judging test adds one, smoke one.

    `root.py`: gate 7 of 8, guard 6 of 8. `search/kernel.py`: guard 5 of 8 (infra).
    """
    package, data_file = _run(tmp_path)
    monkeypatch.setattr(coverage_recut, "PACKAGE", package)

    reach = {
        one.module: one
        for one in coverage_recut.read_reach(
            data_file, MARKERS, JUDGED_COVERAGE.counting
        )
    }

    root = reach["root.py"]
    assert root.package == coverage_recut.ROOT_PACKAGE
    assert root.statements == frozenset({1, 2, 5, 6, 9, 10, 13, 14})
    assert (root.gate, root.counted, root.deficit, root.never) == (7, 6, 1, 1)
    kernel = reach["search/kernel.py"]
    assert kernel.package == "search"
    assert (kernel.gate, kernel.counted, kernel.deficit, kernel.never) == (7, 5, 2, 1)


@pytest.mark.critical
@pytest.mark.infra
def test_the_guard_fails_one_statement_above_what_the_run_reaches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floors are compared, and an exempt package is left out of the whole.

    `search` exempt: the whole is `root.py`, 75.00%; 75 passes, 75.01 fails.
    """
    package, data_file = _run(tmp_path)
    monkeypatch.setattr(coverage_recut, "PACKAGE", package)
    reach = coverage_recut.read_reach(data_file, MARKERS, JUDGED_COVERAGE.counting)

    def guard(floor: float, search: float, exempt: tuple[str, ...]) -> JudgedCoverage:
        return JudgedCoverage(
            name="judged",
            counting=JUDGED_COVERAGE.counting,
            exempt_packages=exempt,
            floor=floor,
            package_floors={"search": search},
        )

    assert coverage_recut.shortfalls(reach, guard(75.0, 62.5, ("search",))) == []
    assert coverage_recut.shortfalls(reach, guard(75.01, 62.5, ("search",))) == [
        "judged coverage 75.00% is below the 75.01% floor"
    ]
    assert coverage_recut.shortfalls(reach, guard(75.0, 62.51, ("search",))) == [
        "search: judged coverage 62.50% is below its 62.51% floor"
    ]
    whole = coverage_recut.figure(reach)
    assert (whole.statements, whole.gate, whole.counted, whole.import_only) == (
        16,
        14,
        11,
        10,
    )


@pytest.mark.infra
def test_the_collection_names_the_markers_the_hook_adds(tmp_path: Path) -> None:
    """Markers are read from a collection, so the problem and `infra` axes are seen.

    The hook adds them; the mini-suite imports this repository's `conftest`.
    """
    (tmp_path / "conftest.py").write_text(
        f"import sys\n\nsys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from tests.conftest import (  # noqa: E402,F401\n"
        "    pytest_collection_modifyitems,\n"
        "    pytest_configure,\n"
        ")\n"
    )
    (tmp_path / "test_shared.py").write_text(
        "import pytest\n\n\n@pytest.mark.smoke\n"
        "@pytest.mark.parametrize('n', [1, 2])\n"
        "def test_shared(n) -> None:\n    assert n\n"
    )

    found = coverage_recut.collect_markers(tmp_path)

    ids = {nodeid.rsplit("::", 1)[1] for nodeid in found}
    assert ids == {"test_shared[1]", "test_shared[2]"}
    for markers in found.values():
        assert {"smoke", "infra", "parametrize"} <= markers


@pytest.mark.critical
@pytest.mark.infra
def test_the_counting_set_is_end2end_and_oracle_alone() -> None:
    """What counts is stated once in `infra/gates.py`, and this is what it says.

    The two kinds judged against something outside the implementation, nothing else.
    """
    from gates import FINDING_MARKERS, KIND_MARKERS, SCHEDULING_MARKERS

    counting = set(JUDGED_COVERAGE.counting)

    assert counting == {"end2end", "oracle"}
    assert counting < set(KIND_MARKERS)
    assert not counting & set(FINDING_MARKERS)
    assert not counting & set(SCHEDULING_MARKERS)
    assert JUDGED_COVERAGE.exempt_packages == ("qa", "validation")
    assert 0.0 < JUDGED_COVERAGE.floor <= 100.0
    assert JUDGED_COVERAGE.package_floors["search"] > JUDGED_COVERAGE.floor


@pytest.mark.critical
@pytest.mark.infra
def test_the_complement_counts_what_the_judged_guard_leaves_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third guard (issue #732): the smoke and infra lines, and the import.

    `root.py` counts 6 of 8, `search/kernel.py` 7 of 8; the sets partition kinds.
    """
    package, data_file = _run(tmp_path)
    monkeypatch.setattr(coverage_recut, "PACKAGE", package)

    reach = {
        one.module: one
        for one in coverage_recut.read_reach(
            data_file, MARKERS, UNJUDGED_COVERAGE.counting
        )
    }

    assert reach["root.py"].counted == 6
    assert reach["search/kernel.py"].counted == 7
    assert set(JUDGED_COVERAGE.counting).isdisjoint(UNJUDGED_COVERAGE.counting)
    assert COVERAGE_GUARDS == (JUDGED_COVERAGE, UNJUDGED_COVERAGE)
    # `search`'s complement floor sat below the whole's until issue #779 moved
    # three modules out of the package --- `gym.py` among them, at 28.17% judged
    # --- and it now sits above it, as its judged floor already did. Both floors
    # are the measurement rounded down; the relation is read, never chosen.
    assert UNJUDGED_COVERAGE.package_floors["search"] > UNJUDGED_COVERAGE.floor


@pytest.mark.critical
@pytest.mark.infra
def test_public_callables_are_listed_by_the_set_that_enters_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--functions`: per module, which guard's tests enter each body.

    `root.py`: `judged` judged, `smoke` complement, `never` none; kernel: complement.
    """
    package, data_file = _run(tmp_path)
    monkeypatch.setattr(coverage_recut, "PACKAGE", package)
    reaches = {
        guard.name: coverage_recut.read_reach(data_file, MARKERS, guard.counting)
        for guard in COVERAGE_GUARDS
    }

    found = {
        (one.module, one.name): (one.entered_by, one.tested)
        for one in coverage_recut.callables(reaches)
    }
    table = coverage_recut.functions_table(reaches, "judged")

    assert found[("root.py", "judged")] == (frozenset({"judged"}), True)
    assert found[("root.py", "smoke")] == (frozenset({"unjudged"}), True)
    assert found[("root.py", "never")] == (frozenset(), False)
    assert found[("search/kernel.py", "judged")] == (frozenset({"unjudged"}), True)
    rows = {line.split()[0]: line.split()[1:] for line in table.splitlines()[1:3]}
    assert rows["(root)"] == ["3", "1", "1", "1"]
    assert rows["search"] == ["3", "0", "2", "1"]
    assert "  search/kernel.py: judged, smoke, never (never)" in table
