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

import sys
from pathlib import Path

import pytest
from coverage.sqldata import CoverageData

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import coverage_recut  # noqa: E402
from gates import JUDGED_COVERAGE, JudgedCoverage  # noqa: E402

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

    `root.py`: the oracle test reaches `judged`'s body and the smoke test
    `smoke`'s, so the gate counts 7 of 8 and the guard 6 of 8, one statement
    in deficit and `never`'s body reached by nothing. `search/kernel.py`: the
    `infra` test reaches `judged`'s body and does not count, so the guard
    reads 5 of 8 there with two statements in deficit.
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

    `root.py` counts 6 of 8 and `search/kernel.py` 5 of 8; with `search`
    exempt the whole is `root.py` alone at 75.00%. A floor of 75 passes, one
    of 75.01 fails, and a package floor on `search` is judged on its own
    62.50% whether or not the package is exempt from the whole.
    """
    package, data_file = _run(tmp_path)
    monkeypatch.setattr(coverage_recut, "PACKAGE", package)
    reach = coverage_recut.read_reach(data_file, MARKERS, JUDGED_COVERAGE.counting)

    def guard(floor: float, search: float, exempt: tuple[str, ...]) -> JudgedCoverage:
        return JudgedCoverage(
            counting=JUDGED_COVERAGE.counting,
            exempt_packages=exempt,
            exempt_marker="exempt",
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

    A parse of the source would miss both: the hook adds them. The mini-suite
    imports this repository's `conftest`, as `test_problem_markers.py` does.
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

    Read against the marker tables rather than restated: the two kinds that
    judge the science against something outside the implementation, and no
    finding, no scheduling marker and no `infra`.
    """
    from gates import FINDING_MARKERS, KIND_MARKERS, SCHEDULING_MARKERS

    counting = set(JUDGED_COVERAGE.counting)

    assert counting == {"end2end", "oracle"}
    assert counting < set(KIND_MARKERS)
    assert not counting & set(FINDING_MARKERS)
    assert not counting & set(SCHEDULING_MARKERS)
    assert JUDGED_COVERAGE.exempt_packages == ("qa",)
    assert JUDGED_COVERAGE.exempt_marker == "exempt"
    assert 0.0 < JUDGED_COVERAGE.floor <= 100.0
    assert JUDGED_COVERAGE.package_floors["search"] > JUDGED_COVERAGE.floor
