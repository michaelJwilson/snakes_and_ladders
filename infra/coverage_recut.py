"""Coverage recut to the tests that judged something (issue #729).

`--cov-fail-under` counts a statement whichever test reached it. Measured on
2026-09-18 on `main` a5b6fa4, one run of the regression tier with a context
per test: the gate read **94.27%** (15,765 of 16,724 statements), **34.32%**
of which is reached by importing the package with no test at all, and a
further 4.0 points by tests that check the implementation against itself --
reachability, a shape, the absence of an exception. `CLAUDE.md` forbids
coverage theatre and the gate could not see it.

This reads the same run's `.coverage` file -- written with
``--cov-context=test``, so every line carries the test that reached it -- and
each test's markers from one collection, and counts a statement when a test
carrying one of `infra.gates.JUDGED_COVERAGE.counting` reached it, over the
packages that guard does not exempt. Import-time lines, the context with no
name, count on both bases as they do for `--cov-fail-under`: the recut asks
what a *test* added, not what the interpreter did, and stating the two on
the same basis is what makes the drop a measurement.

Run, from the repository root, after the tier::

    pytest tests/regression -n 3 -m "not release" --cov=sal \\
        --cov-context=test
    uv run python infra/coverage_recut.py               # the two tables
    uv run python infra/coverage_recut.py --fail-under  # the guard, exit 1 below a floor

The tables are the package figure on both bases with the import-alone share
beside them and, per package, the statements every test reaches, the statements the judged tests reach, the
deficit between them, and the statements no test reaches at all. The floors
live in `infra/gates.py` beside the markers that count, and CI runs the guard
on the push to `main` after `--cov-fail-under`, as `infra/release.sh` does at
the release gate.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from _paths import REPO_ROOT
from coverage.numbits import numbits_to_nums
from coverage.parser import PythonParser

PACKAGE = REPO_ROOT / "python" / "sal"

# `infra/` is on `mypy_path` and is how the guards reach their shared names.
sys.path.insert(0, str(REPO_ROOT / "infra"))

from gates import COVERAGE_GUARDS, JudgedCoverage  # noqa: E402

#: The package a module belongs to where it sits directly under the package.
ROOT_PACKAGE = "(root)"


@dataclass(frozen=True)
class Reach:
    """One file's statements, and which of them each set of tests reached."""

    #: The file, relative to the package.
    module: str
    #: Every statement `coverage` would count.
    statements: frozenset[int]
    #: Reached at import, by no test: the context with no name.
    imported: frozenset[int]
    #: Reached by any test.
    tested: frozenset[int]
    #: Reached by a test that judged something.
    judged: frozenset[int]

    @property
    def package(self) -> str:
        """The first directory of the module's path, or the root."""
        parts = self.module.split("/")
        return parts[0] if len(parts) > 1 else ROOT_PACKAGE

    @property
    def gate(self) -> int:
        """Statements `--cov-fail-under` counts: any test, or the import."""
        return len(self.statements & (self.tested | self.imported))

    @property
    def counted(self) -> int:
        """Statements the guard counts: a judged test, or the import."""
        return len(self.statements & (self.judged | self.imported))

    @property
    def import_only(self) -> int:
        """Statements the import reaches, with no test at all."""
        return len(self.statements & self.imported)

    @property
    def deficit(self) -> int:
        """Statements only an unjudged test reaches."""
        return len((self.statements & self.tested) - self.judged - self.imported)

    @property
    def never(self) -> int:
        """Statements nothing reaches, not even the import."""
        return len(self.statements - self.tested - self.imported)


def statements_of(path: Path) -> frozenset[int]:
    """The statements `coverage` counts in one source file.

    The same parser the report uses, so the denominator here is the
    denominator `--cov-fail-under` divides by, `exclude_lines` included.
    """
    parser = PythonParser(text=path.read_text(), filename=str(path))
    parser.parse_source()
    return frozenset(parser.statements)


def module_of(path: Path) -> str | None:
    """A measured file's path inside the package, whichever checkout measured it.

    The data names the file where the run happened; the source is read from
    this tree, so a run from another worktree of the same commit recuts here.
    ``None`` for a file outside the package.
    """
    parts = path.parts
    if PACKAGE.name not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index(PACKAGE.name)
    return "/".join(parts[index + 1 :])


def _test_id(context: str) -> str:
    """The test a `pytest-cov` context names: ``nodeid|phase`` without the phase."""
    return context.split("|", 1)[0]


def read_reach(
    data_file: Path,
    markers: Mapping[str, frozenset[str]],
    counting: Iterable[str],
) -> list[Reach]:
    """Every measured file's reach, read straight from the contexts table.

    Parameters
    ----------
    data_file : Path
        A `.coverage` file written with ``--cov-context=test``.
    markers : Mapping[str, frozenset[str]]
        Each test's markers, by node id.
    counting : Iterable[str]
        The markers whose tests count.

    Returns
    -------
    list[Reach]
        One per measured file, in path order.

    The tables are read directly rather than through `set_query_contexts`:
    that takes a regex list, and 2,478 node ids exceed what the regex engine
    accepts as one pattern.
    """
    counts = set(counting)
    with contextlib.closing(sqlite3.connect(data_file)) as connection:
        files = {
            int(file_id): str(path)
            for file_id, path in connection.execute("SELECT id, path FROM file")
        }
        contexts = {
            int(context_id): str(name)
            for context_id, name in connection.execute(
                "SELECT id, context FROM context"
            )
        }
        rows = [
            (int(file_id), int(context_id), bytes(numbits))
            for file_id, context_id, numbits in connection.execute(
                "SELECT file_id, context_id, numbits FROM line_bits"
            )
        ]
    imported: dict[int, set[int]] = {file_id: set() for file_id in files}
    tested: dict[int, set[int]] = {file_id: set() for file_id in files}
    judged: dict[int, set[int]] = {file_id: set() for file_id in files}
    for file_id, context_id, numbits in rows:
        name = contexts[context_id]
        lines = numbits_to_nums(numbits)
        if not name:
            imported[file_id].update(lines)
            continue
        tested[file_id].update(lines)
        if markers.get(_test_id(name), frozenset()) & counts:
            judged[file_id].update(lines)
    reach = []
    for file_id, path in sorted(files.items(), key=lambda item: item[1]):
        module = module_of(Path(path))
        source = PACKAGE / module if module else Path(path)
        if module is None or not source.is_file():
            continue
        reach.append(
            Reach(
                module=module,
                statements=statements_of(source),
                imported=frozenset(imported[file_id]),
                tested=frozenset(tested[file_id]),
                judged=frozenset(judged[file_id]),
            )
        )
    return reach


class _Markers:
    """A `pytest` plugin that records every collected item's markers."""

    def __init__(self) -> None:
        self.found: dict[str, frozenset[str]] = {}

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            self.found[item.nodeid] = frozenset(
                marker.name for marker in item.iter_markers()
            )


def collect_markers(tests: Path) -> dict[str, frozenset[str]]:
    """Every test's markers under ``tests``, from one collection.

    Collected rather than parsed: the contexts name parametrized ids, and the
    problem and `infra` markers are added by the collection hook, so the
    source alone does not say what an item carried when it ran.
    """
    plugin = _Markers()
    with contextlib.redirect_stdout(io.StringIO()):
        code = pytest.main(
            [
                str(tests),
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "-m",
                "release or not release",
            ],
            plugins=[plugin],
        )
    if code != 0:
        message = f"collecting {tests} failed with exit code {code}"
        raise RuntimeError(message)
    return plugin.found


def percent(numerator: int, denominator: int) -> float:
    """A percentage, and zero over an empty denominator."""
    return 100.0 * numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class Figure:
    """The guard's figure over a set of files."""

    statements: int
    gate: int
    counted: int
    import_only: int

    @property
    def gate_percent(self) -> float:
        return percent(self.gate, self.statements)

    @property
    def import_percent(self) -> float:
        return percent(self.import_only, self.statements)

    @property
    def counted_percent(self) -> float:
        return percent(self.counted, self.statements)


def figure(reach: Iterable[Reach]) -> Figure:
    """The figure over ``reach``."""
    files = list(reach)
    return Figure(
        statements=sum(len(one.statements) for one in files),
        gate=sum(one.gate for one in files),
        counted=sum(one.counted for one in files),
        import_only=sum(one.import_only for one in files),
    )


def by_package(reach: Iterable[Reach]) -> dict[str, list[Reach]]:
    """The files grouped by package, packages in name order."""
    groups: dict[str, list[Reach]] = {}
    for one in reach:
        groups.setdefault(one.package, []).append(one)
    return dict(sorted(groups.items()))


def package_of(module: str) -> str:
    """The first directory of a module's path, or the root."""
    return module.split("/")[0] if "/" in module else ROOT_PACKAGE


def tables(reaches: Mapping[str, list[Reach]], guards: Iterable[JudgedCoverage]) -> str:
    """The two tables, as text: one column per guard beside the gate."""
    names = [guard.name for guard in guards]
    first = reaches[names[0]]
    exempt = next(iter(guards)).exempt_packages
    head = " ".join(f"{name:>9s}" for name in names)
    lines = [f"{'set':44s} {'statements':>10s} {'import':>8s} {'gate':>8s} {head}"]
    for label, packages in (
        ("every package", None),
        ("guarded (exempt: " + ", ".join(exempt) + ")", exempt),
    ):
        keep = [
            one.package
            for one in first
            if packages is None or one.package not in packages
        ]
        whole = figure(one for one in first if one.package in keep)
        counted = " ".join(
            f"{figure(one for one in reaches[name] if one.package in keep).counted_percent:8.2f}%"
            for name in names
        )
        lines.append(
            f"{label:44s} {whole.statements:10d} {whole.import_percent:7.2f}% "
            f"{whole.gate_percent:7.2f}% {counted}"
        )
    lines += [
        "",
        f"{'package':12s} {'statements':>10s} {'gate':>8s} {head} {'deficit':>8s} {'never':>6s}",
    ]
    for package, files in by_package(first).items():
        one = figure(files)
        counted = " ".join(
            f"{figure(f for f in reaches[name] if f.package == package).counted_percent:8.2f}%"
            for name in names
        )
        lines.append(
            f"{package:12s} {one.statements:10d} {one.gate_percent:7.2f}% {counted} "
            f"{sum(f.deficit for f in files):8d} {sum(f.never for f in files):6d}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class PublicCallable:
    """One public top-level function or class, and which guards' tests enter it."""

    module: str
    name: str
    #: The guards whose tests reach a statement of the body.
    entered_by: frozenset[str]
    #: Whether any test at all reaches one.
    tested: bool

    @property
    def package(self) -> str:
        return package_of(self.module)


def callables(reaches: Mapping[str, list[Reach]]) -> list[PublicCallable]:
    """Every public top-level ``def`` and ``class`` of every measured module.

    A body is entered by a guard's set when one of its statements is in that
    set's reach; the ``def`` line itself runs at import and is left out, so a
    function nothing calls is entered by nothing. Read with `ast` from the
    source `coverage` measured, so a body's lines are the parser's.
    """
    found: list[PublicCallable] = []
    per_module: dict[str, dict[str, Reach]] = {}
    for name, reach in reaches.items():
        for one in reach:
            per_module.setdefault(one.module, {})[name] = one
    for module, per_set in sorted(per_module.items()):
        source = PACKAGE / module
        if not source.is_file() or module.endswith("__init__.py"):
            continue
        first = next(iter(per_set.values()))
        for node in ast.parse(source.read_text()).body:
            if not isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                continue
            if node.name.startswith("_"):
                continue
            body = set(range(node.lineno + 1, (node.end_lineno or node.lineno) + 1))
            body &= first.statements
            found.append(
                PublicCallable(
                    module=module,
                    name=node.name,
                    entered_by=frozenset(
                        name for name, one in per_set.items() if body & one.judged
                    ),
                    tested=bool(body & first.tested),
                )
            )
    return found


def functions_table(reaches: Mapping[str, list[Reach]], judged: str) -> str:
    """Per package, the public callables by what enters them; then the list.

    ``judged`` names the guard whose tests are the referee; a callable no test
    of that guard enters is listed by module, marked ``(never)`` where no test
    of any kind enters it either.
    """
    found = callables(reaches)
    lines = [
        f"{'package':12s} {'callables':>9s} {judged:>9s} {'others only':>12s} {'never':>6s}"
    ]
    for package in sorted({one.package for one in found}):
        ours = [one for one in found if one.package == package]
        entered = sum(1 for one in ours if judged in one.entered_by)
        others = sum(1 for one in ours if judged not in one.entered_by and one.tested)
        never = sum(1 for one in ours if not one.tested)
        lines.append(
            f"{package:12s} {len(ours):9d} {entered:9d} {others:12d} {never:6d}"
        )
    lines += ["", f"public callables no {judged} test enters, by module:"]
    by_module: dict[str, list[str]] = {}
    for one in found:
        if judged not in one.entered_by:
            by_module.setdefault(one.module, []).append(
                one.name + ("" if one.tested else " (never)")
            )
    lines += [
        f"  {module}: {', '.join(names)}" for module, names in sorted(by_module.items())
    ]
    return "\n".join(lines)


def shortfalls(reach: list[Reach], guard: JudgedCoverage) -> list[str]:
    """Every floor the run falls below, as a sentence each. Empty is a pass."""
    found = []
    guarded = figure(one for one in reach if one.package not in guard.exempt_packages)
    if guarded.counted_percent < guard.floor:
        found.append(
            f"judged coverage {guarded.counted_percent:.2f}% is below the "
            f"{guard.floor}% floor"
        )
    packages = by_package(reach)
    for package, floor in guard.package_floors.items():
        one = figure(packages.get(package, []))
        if one.counted_percent < floor:
            found.append(
                f"{package}: judged coverage {one.counted_percent:.2f}% is below "
                f"its {floor}% floor"
            )
    return found


def main(argv: list[str] | None = None) -> int:
    """Print the tables; with ``--fail-under``, exit 1 below a floor."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / ".coverage", help="the .coverage file"
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=REPO_ROOT / "tests" / "regression",
        help="the directory whose collection names each test's markers",
    )
    parser.add_argument(
        "--fail-under",
        action="store_true",
        help="exit 1 where a guard's figure is below a floor in infra/gates.py",
    )
    parser.add_argument(
        "--functions",
        action="store_true",
        help="list the public callables the judging tests never enter",
    )
    args = parser.parse_args(argv)
    markers = collect_markers(args.tests)
    reaches = {
        guard.name: read_reach(args.data, markers, guard.counting)
        for guard in COVERAGE_GUARDS
    }
    print(tables(reaches, COVERAGE_GUARDS))
    if args.functions:
        print()
        print(functions_table(reaches, COVERAGE_GUARDS[0].name))
    if not args.fail_under:
        return 0
    failed = False
    for guard in COVERAGE_GUARDS:
        short = shortfalls(reaches[guard.name], guard)
        for sentence in short:
            print(f"FAIL: {guard.name} {sentence}")
        failed |= bool(short)
        if not short:
            print(
                f"{guard.name} coverage holds the {guard.floor}% floor"
                + "".join(
                    f", {package} its {floor}%"
                    for package, floor in guard.package_floors.items()
                )
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
