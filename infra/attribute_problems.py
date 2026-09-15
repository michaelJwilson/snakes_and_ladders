"""Which problem each unmarked test module exercises, read from what it imports.

Issue #622, step 1. 121 of 232 test modules reach no problem through
`snakes_and_ladders.sim.fixtures`, so #619's derived ``-m <problem>`` axis
does not select them. Before a module is converted, the work is named: which
problem it exercises, whether the registry declares an instance at the shape
it builds, and whether a fixture must be added.

**Every reading is a fact the tree already carries.** A hand-written map of
module to problem is the defect this ticket is about, one level up. Three
readings, in order, each derived from a file CI already holds true:

1. *Registry.* A `fixture`, `path_of`, `at_fixture` or `at_bin` call, or a
   ``<problem>/<tier>.yaml`` path literal. Such a module already names its
   problem and is reported ``marked``: it needs no conversion. A call whose
   argument is computed still reaches the registry and still counts.
2. *Catalogue.* `PROBLEMS.md` gives one row per problem, naming the code that
   simulates, evaluates, fits, searches and learns on it and the fixtures it
   declares; `tests/regression/test_problems_catalogue.py` fails a row naming a
   symbol that does not resolve. A module is scored against every row by how
   many of the row's symbols it imports, and the highest-scoring row is the
   problem it exercises. Ties are not broken --- a module naming as much of two
   rows is `unattributed`, listing both.
3. *Shape.* A fixture file declares its instance as ``field: value`` pairs, and
   a module that builds one inline binds those same field names to literals as
   a module-level constant or a keyword argument. Comparing the two, field by
   field, says whether the row's declared tiers already carry the shape the
   module builds --- so whether step 2 of the plan converts it or step 3 must
   add a fixture first.

**Limits, which are why `unattributed` is a status and not a failure.**

- Reading 2 ranks; it does not prove. A module importing only symbols two rows
  share --- an oracle both are pinned to, a sampler both use --- scores both
  equally and is left `unattributed` with the candidates named, because a
  module converted to the wrong problem of a pair is a conversion that moves a
  number.
- A row declaring more than one fixture problem (the Jukes--Cantor row declares
  `tree_search` and `tree_scale`) resolves to one only where reading 3 narrows
  it.
- Reading 3 reads literals. A shape computed at run time, or built through
  ``np.array(...)``, is unread, and the fixture column says ``shape unread``
  rather than claiming no declared tier fits.
- A module importing no catalogued symbol and no `sim` or `likelihood` code
  exercises no problem and is `infra`. One that imports such code but no
  catalogued symbol is `unattributed`: it builds an instance no row names.

Run with ``--table`` for the Markdown posted to the issue.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml
from snakes_and_ladders.sim.fixtures import FIXTURES_DIR, problems

#: The repository root, from this file rather than a working directory.
REPO_ROOT: Final = Path(__file__).resolve().parents[1]

#: The problem catalogue, one row per problem.
CATALOGUE: Final = REPO_ROOT / "PROBLEMS.md"

#: A backticked cell entry in the catalogue: a dotted symbol or a path.
_CELL = re.compile(r"`([^`]+)`")

#: A fixture named as its path, as the catalogue's last column spells it.
_FIXTURE_PATH = re.compile(r"fixtures/([a-z_0-9]+)/([a-z]+)\.yaml")

#: Fixture keys stating how the instance was recorded rather than what it is.
#: A shape comparison over them would match a seed against a seed.
META_FIELDS: Final = frozenset(
    {"model", "oracle", "seed", "tolerance", "bin", "counts_digest"}
)

#: Calls whose first positional argument names a problem, and whose second
#: does --- `tests/_scale.py` wraps `fixture` and passes the name through.
NAMES_FIRST: Final = ("fixture", "path_of", "load_fixture", "fixture_path")
NAMES_SECOND: Final = ("at_fixture", "at_bin")

#: The package prefixes under which a problem instance is built or evaluated.
APPLICATION: Final = ("snakes_and_ladders.sim.", "snakes_and_ladders.likelihood.")

#: The four statuses a module lands in, exactly one each.
STATUSES: Final = ("marked", "attributed", "infra", "unattributed")


@dataclass(frozen=True)
class Row:
    """One catalogue row: a problem, its code, and the fixtures it declares.

    Parameters
    ----------
    title : str
        The row's first column, as `PROBLEMS.md` writes it.
    symbols : frozenset[str]
        Every ``snakes_and_ladders.*`` symbol the row names.
    declared : tuple[str, ...]
        The fixture problems the row's last column names, sorted. These are
        the registry directory names, and so the marker names.
    """

    title: str
    symbols: frozenset[str]
    declared: tuple[str, ...]


@dataclass(frozen=True)
class Catalogue:
    """`PROBLEMS.md` and the fixture registry, as the readings need them.

    Parameters
    ----------
    rows : tuple[Row, ...]
        One per catalogue row, in file order.
    instances : Mapping[str, Mapping[str, Mapping[str, Any]]]
        ``problem -> tier -> field -> value``, the meta fields dropped.
    problems : tuple[str, ...]
        Every problem the registry declares, sorted.
    """

    rows: tuple[Row, ...]
    instances: Mapping[str, Mapping[str, Mapping[str, Any]]]
    problems: tuple[str, ...]


@dataclass(frozen=True)
class Attribution:
    """One test module's reading.

    Parameters
    ----------
    module : str
        Path from the repository root.
    status : str
        One of :data:`STATUSES`.
    problems : tuple[str, ...]
        The problem attributed, or the candidates left where the status is
        `unattributed` and a row was ranked but not resolved.
    fixture : str
        What the registry declares at the shape the module builds.
    evidence : str
        The reading that fired, named so a reviewer can check it.
    """

    module: str
    status: str
    problems: tuple[str, ...]
    fixture: str
    evidence: str


def _instances(directory: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Every declared instance's fields, by problem and tier."""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for problem in problems(directory):
        tiers: dict[str, dict[str, Any]] = {}
        for path in sorted((directory / problem).glob("*.yaml")):
            raw = yaml.safe_load(path.read_text())
            tiers[path.stem] = {
                key: value for key, value in raw.items() if key not in META_FIELDS
            }
        found[problem] = tiers
    return found


def catalogue(path: Path = CATALOGUE, directory: Path = FIXTURES_DIR) -> Catalogue:
    """Read `PROBLEMS.md` and the registry into the indices the readings use.

    Parameters
    ----------
    path : Path
        The catalogue.
    directory : Path
        Where the fixtures live.

    Returns
    -------
    Catalogue

    Raises
    ------
    ValueError
        If a row names a fixture directory the registry does not declare. The
        two files are kept in step by `tests/regression/test_fixture_registry.py`,
        and a reading built on a broken join would attribute silently.
    """
    declared_problems = problems(directory)
    rows: list[Row] = []
    for line in path.read_text().splitlines():
        if not line.startswith("|") or line.startswith("| ---") or "`" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        entries = _CELL.findall(line)
        symbols = frozenset(
            entry for entry in entries if entry.startswith("snakes_and_ladders.")
        )
        named = sorted({match.group(1) for match in _FIXTURE_PATH.finditer(line)})
        unknown = [name for name in named if name not in declared_problems]
        if unknown:
            msg = f"PROBLEMS.md row {cells[0]!r} names undeclared fixture(s) {unknown}"
            raise ValueError(msg)
        if symbols or named:
            rows.append(Row(cells[0], symbols, tuple(named)))
    return Catalogue(
        rows=tuple(rows),
        instances=_instances(directory),
        problems=declared_problems,
    )


def _literal(node: ast.expr) -> Any:
    """A literal value, or None where the expression is not one.

    A tuple collapses to a list, because a fixture declares ``shape: [3, 3]``
    where a module writes ``SHAPE = (3, 3)`` for the same extent.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple | ast.List):
        values = [_literal(element) for element in node.elts]
        return None if any(value is None for value in values) else values
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        return -value if isinstance(value, int | float) else None
    return None


@dataclass(frozen=True)
class Reading:
    """What one module's source says, before the catalogue is consulted.

    Parameters
    ----------
    registry : bool
        Whether the module reaches `sim.fixtures` at all.
    named : frozenset[str]
        The declared problems it names there, which may be empty for a call
        whose argument is computed.
    symbols : frozenset[str]
        Dotted ``snakes_and_ladders.*`` names it imports.
    modules : frozenset[str]
        The package modules those names come from.
    bindings : Mapping[str, frozenset[str]]
        ``lowercased name -> repr of each literal bound to it``.
    """

    registry: bool
    named: frozenset[str]
    symbols: frozenset[str]
    modules: frozenset[str]
    bindings: Mapping[str, frozenset[str]]


def read(path: Path, declared: Iterable[str]) -> Reading:
    """Parse one test module: its registry calls, imports and literal bindings.

    Parameters
    ----------
    path : Path
        A test module.
    declared : Iterable[str]
        The declared problem names, against which a registry call is checked.

    Returns
    -------
    Reading
    """
    known = set(declared)
    tree = ast.parse(path.read_text())
    registry = False
    named: set[str] = set()
    symbols: set[str] = set()
    modules: set[str] = set()
    bindings: defaultdict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith("snakes_and_ladders"):
                modules.add(node.module)
                symbols |= {f"{node.module}.{alias.name}" for alias in node.names}
                registry = registry or node.module.endswith("sim.fixtures")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("snakes_and_ladders"):
                    modules.add(alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts = node.value.removesuffix(".yaml").split("/")
            if node.value.endswith(".yaml") and len(parts) > 1 and parts[-2] in known:
                registry = True
                named.add(parts[-2])
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = _literal(node.value)
            if isinstance(target, ast.Name) and value is not None:
                bindings[target.id.lower().lstrip("_")].add(repr(value))
        if isinstance(node, ast.Call):
            function = node.func
            called = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else None
            )
            index = (
                0 if called in NAMES_FIRST else 1 if called in NAMES_SECOND else None
            )
            if index is not None and len(node.args) > index:
                registry = True
                argument = node.args[index]
                if isinstance(argument, ast.Constant) and argument.value in known:
                    named.add(str(argument.value))
            for keyword in node.keywords:
                value = _literal(keyword.value)
                if keyword.arg is not None and value is not None:
                    bindings[keyword.arg.lower()].add(repr(value))
    return Reading(
        registry=registry,
        named=frozenset(named),
        symbols=frozenset(symbols),
        modules=frozenset(modules),
        bindings={key: frozenset(value) for key, value in bindings.items()},
    )


def rank(reading: Reading, rows: Sequence[Row]) -> tuple[int, tuple[Row, ...]]:
    """The catalogue rows a module names most of, and how many symbols that is.

    Parameters
    ----------
    reading : Reading
        One module's imports.
    rows : Sequence[Row]
        The catalogue.

    Returns
    -------
    tuple[int, tuple[Row, ...]]
        The top score and every row achieving it; ``(0, ())`` where the module
        names no catalogued symbol.
    """
    scores = [len(reading.symbols & row.symbols) for row in rows]
    best = max(scores, default=0)
    if best == 0:
        return 0, ()
    return best, tuple(
        row for row, score in zip(rows, scores, strict=True) if score == best
    )


def _shape(
    fields: Mapping[str, Any], bindings: Mapping[str, frozenset[str]]
) -> tuple[int, int]:
    """How many of a declared instance's fields the module binds, and matches.

    Returns
    -------
    tuple[int, int]
        ``(matched, named)``. ``named`` without ``matched`` is a shape the
        registry does not declare at this tier.
    """
    named = [key for key in fields if key in bindings]
    matched = [key for key in named if repr(fields[key]) in bindings[key]]
    return len(matched), len(named)


def fixture_column(
    candidates: Iterable[str],
    index: Catalogue,
    bindings: Mapping[str, frozenset[str]],
) -> tuple[str, tuple[str, ...]]:
    """What the registry declares at the shape the module builds.

    Parameters
    ----------
    candidates : Iterable[str]
        The fixture problems the ranked rows declare.
    index : Catalogue
        The catalogue and the registry.
    bindings : Mapping[str, frozenset[str]]
        The module's literal bindings.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        The column, and the candidates reading 3 leaves: a candidate whose
        declared fields the module names *and* matches narrows the set, which
        is the only narrowing this script performs.
    """
    exact: dict[str, str] = {}
    conflicting: list[str] = []
    unread: list[str] = []
    for problem in sorted(set(candidates)):
        best: tuple[int, int, str] | None = None
        for tier, fields in index.instances[problem].items():
            best = max(best or (-1, -1, ""), (*_shape(fields, bindings), tier))
        if best is None or best[1] == 0:
            unread.append(problem)
        elif best[0] == best[1]:
            exact[problem] = best[2]
        else:
            conflicting.append(problem)
    if exact:
        rendered = ", ".join(f"{name}/{tier}" for name, tier in sorted(exact.items()))
        return f"declared: {rendered}", tuple(sorted(exact))
    if conflicting:
        return "NEW TIER: no declared shape matches", tuple(
            sorted(conflicting + unread)
        )
    return "shape unread", tuple(unread)


def attribute(path: Path, index: Catalogue) -> Attribution:
    """Attribute one test module to a problem, to `infra`, or to neither.

    Parameters
    ----------
    path : Path
        A test module.
    index : Catalogue
        The catalogue and the registry.

    Returns
    -------
    Attribution
        Status `marked`, `attributed`, `infra` or `unattributed`; exactly one.
    """
    module = str(path.relative_to(REPO_ROOT))
    reading = read(path, index.problems)
    if reading.registry:
        return Attribution(
            module,
            "marked",
            tuple(sorted(reading.named)),
            "reaches the registry",
            "registry call",
        )

    score, ranked = rank(reading, index.rows)
    if not ranked:
        application = any(name.startswith(APPLICATION) for name in reading.modules)
        if not application:
            return Attribution(module, "infra", (), "n/a", "names no catalogued symbol")
        return Attribution(
            module,
            "unattributed",
            (),
            "NEW PROBLEM: no catalogue row names what it builds",
            "imports sim or likelihood code the catalogue does not name",
        )

    candidates = sorted({problem for row in ranked for problem in row.declared})
    fixture, narrowed = fixture_column(candidates, index, reading.bindings)
    resolved = narrowed if len(narrowed) == 1 else tuple(candidates)
    plural = "" if score == 1 else "s"
    evidence = (
        f"catalogue: {ranked[0].title}"
        if len(ranked) == 1
        else f"catalogue: {len(ranked)} rows tie at {score} symbol{plural}"
    )
    if len(resolved) == 1:
        return Attribution(module, "attributed", resolved, fixture, evidence)
    return Attribution(module, "unattributed", resolved, fixture, evidence)


def modules(root: Path = REPO_ROOT) -> tuple[Path, ...]:
    """Every test module in the tree, sorted.

    Returns
    -------
    tuple[Path, ...]
    """
    return tuple(sorted(root.glob("tests/**/test_*.py")))


def attributions(
    root: Path = REPO_ROOT, directory: Path = FIXTURES_DIR
) -> tuple[Attribution, ...]:
    """Attribute every test module in the tree.

    Parameters
    ----------
    root : Path
        The repository root.
    directory : Path
        Where the fixtures live.

    Returns
    -------
    tuple[Attribution, ...]
        One per module, in path order.
    """
    index = catalogue(root / "PROBLEMS.md", directory)
    return tuple(attribute(path, index) for path in modules(root))


def counts(rows: Iterable[Attribution]) -> dict[str, int]:
    """How many modules land in each status.

    Returns
    -------
    dict[str, int]
    """
    tally = dict.fromkeys(STATUSES, 0)
    for row in rows:
        tally[row.status] += 1
    return tally


def table(rows: Iterable[Attribution]) -> str:
    """The Markdown table posted to the issue: every module without a marker.

    Returns
    -------
    str
    """
    rows = tuple(rows)
    tally = counts(rows)
    lines = [
        f"{len(rows)} test modules: "
        + ", ".join(f"{tally[status]} {status}" for status in STATUSES),
        "",
        "| module | status | problem | fixture | read from |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.status == "marked":
            continue
        problem = " \\| ".join(row.problems) if row.problems else "--"
        lines.append(
            f"| `{row.module}` | {row.status} | {problem} | {row.fixture} "
            f"| {row.evidence} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Report the attribution, as counts or as the Markdown table.

    Returns
    -------
    int
        Process exit status; 0 always, since the script reports and the
        regression test gates.
    """
    parser = argparse.ArgumentParser(description="attribute test modules to problems")
    parser.add_argument("--table", action="store_true", help="emit Markdown")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    arguments = parser.parse_args(argv)
    rows = attributions(arguments.root)
    if arguments.table:
        print(table(rows))
        return 0
    for status, count in counts(rows).items():
        print(f"{status}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
