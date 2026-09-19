"""Count the duplications issues #230, #277 and #717 survey, by the query that found each.

The survey's numbers were an impression until they were a query. Each finding
below carries the pattern that produces it, so a reader re-runs the survey
rather than trusting a table in a pull request, and so the before-and-after
claim is a measurement rather than a recollection.

Run with ``uv run python infra/duplication_survey.py``. It prints a count per
finding and exits 0 regardless: this reports, it does not gate. What gates is
``tests/regression/test_duplication_guards.py``, which asserts the three
findings this consolidation closed and is tested to fail on a violating input.

Counts are over ``python/snakes_and_ladders`` only, its ``sandbox`` excluded
since what sits there is declined. Tests legitimately repeat structure --
a test that shares a helper with the code it checks is testing the helper --
so counting them would report the suite as the defect.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "python" / "snakes_and_ladders"
TESTS = REPO_ROOT / "tests" / "regression"

#: The suffixes a Python path above a compiled or accelerated kernel carries
#: beside its oracle: `maxflow_rust` beside `maxflow`, `pruning_torch` beside
#: `pruning` (issue #598).
TWIN_SUFFIXES = ("_rust", "_torch", "_analytic")


@dataclass(frozen=True)
class Finding:
    """One duplication, and the query that counts it.

    Parameters
    ----------
    name : str
        What is duplicated, as the survey names it.
    pattern : str
        Regular expression counted over every ``.py`` file in the package.
    at_filing : int
        The count when issue #230 was filed, for comparison.
    owner : str
        File that legitimately holds the pattern once, excluded from the
        count. A consolidated helper still matches the query that found the
        duplication, and counting its own definition would report the fix as
        the defect.
    counter : Callable[[], int] | None
        A query that is not a regular expression --- a count of files, or
        of test modules --- run in place of ``pattern`` when given.
    """

    name: str
    pattern: str
    at_filing: int
    owner: str = ""
    counter: Callable[[], int] | None = None

    def now(self) -> int:
        """The count today, by whichever query the finding carries."""
        if self.counter is not None:
            return self.counter()
        return count(self.pattern, self.owner)


def _package_modules(package: Path = PACKAGE) -> list[Path]:
    """Every module of the package, the sandbox excluded: it is declined work."""
    return [
        path
        for path in sorted(package.rglob("*.py"))
        if "sandbox" not in path.relative_to(package).parts
    ]


def _non_blank(paths: list[Path]) -> int:
    """Lines that carry something, over ``paths``.

    Blank lines are not counted because a reformat moves them and the rows
    here exist to be compared across months. A comment and a docstring line
    do count: both are read, and #813 is about what a reader must get
    through, not about statements a compiler would see.
    """
    return sum(
        1 for path in paths for line in path.read_text().splitlines() if line.strip()
    )


def package_lines(package: Path = PACKAGE) -> int:
    """Non-blank lines of the package, the sandbox excluded as everywhere here.

    The target #813 states is read against this row and the one below it.
    The sandbox is conserved declined work (`sandbox/CLAUDE.md`), so removing
    it would be a saving nobody made.
    """
    return _non_blank(_package_modules(package))


def test_modules(tests: Path = TESTS) -> int:
    """Regression test modules: the files a reader chooses between."""
    return sum(1 for path in tests.rglob("test_*.py"))


#: The module that pins these rows, excluded from the row that would count it.
#: A count including its own pin moves every time the pin is edited, so the
#: first edit after this one would have reported a duplication that is a
#: comment (#813).
PINNING_MODULE = "test_duplication_guards.py"


def test_lines(tests: Path = TESTS) -> int:
    """Non-blank lines of the regression suite, excluding the module that pins it.

    Counted beside the package's because #813's rows are meant to fall
    together: thirteen test modules pinning one twin is one duplication with
    two line counts, and a fold that moves lines from the second row into the
    first has saved nothing.
    """
    return _non_blank(
        [
            path
            for path in sorted(tests.rglob("test_*.py"))
            if path.name != PINNING_MODULE
        ]
    )


def flat_modules(package: Path = PACKAGE) -> int:
    """Modules that are not a package's ``__init__``: the entry points a reader meets."""
    return sum(1 for path in _package_modules(package) if path.name != "__init__.py")


def api_map_entries() -> int:
    """Entries of the API map, `infra/api_map.py`'s count over the package."""
    import api_map

    return sum(len(module.entries) for module in api_map.modules())


def twins(package: Path = PACKAGE) -> dict[Path, Path]:
    """Each twin module and the oracle it sits beside, where the oracle exists."""
    found: dict[Path, Path] = {}
    for path in _package_modules(package):
        for suffix in TWIN_SUFFIXES:
            if path.stem.endswith(suffix):
                oracle = path.with_name(path.stem[: -len(suffix)] + ".py")
                if oracle.exists():
                    found[path] = oracle
    return found


def twin_modules(package: Path = PACKAGE) -> int:
    """Python paths above a compiled kernel, one module each beside its oracle."""
    return len(twins(package))


def surrogate_modules(package: Path = PACKAGE) -> int:
    """Modules named for the surrogate seam: `bound` at the root and a `surrogate` per package."""
    return sum(
        1 for path in _package_modules(package) if path.stem in {"surrogate", "bound"}
    )


def undocumented_modules(package: Path = PACKAGE) -> int:
    """Modules without a module docstring."""
    return sum(
        1
        for path in _package_modules(package)
        if ast.get_docstring(ast.parse(path.read_text())) is None
    )


def root_exports(package: Path = PACKAGE) -> int:
    """Names the root ``__init__`` exports through ``__all__``."""
    tree = ast.parse((package / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return len(ast.literal_eval(node.value))
    return 0


def _imported(path: Path) -> set[str]:
    """Dotted names a module imports, with each ``from`` import's names appended."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def _dotted(path: Path, package: Path) -> str:
    return ".".join((package.name, *path.relative_to(package).with_suffix("").parts))


def twin_pins_beyond_the_first(package: Path = PACKAGE, tests: Path = TESTS) -> int:
    """Regression test modules importing a twin and its oracle, beyond one per twin.

    Two test modules that import ``maxflow_rust`` and ``maxflow`` both pin
    the kernel to its oracle, and the second pin is the overlap of intent
    issue #717 counts: one referee per seam, not one per author.
    """
    pinned = 0
    for twin, oracle in twins(package).items():
        twin_name, oracle_name = _dotted(twin, package), _dotted(oracle, package)
        modules = 0
        for path in sorted(tests.rglob("test_*.py")):
            imported = _imported(path)
            if any(
                name == twin_name or name.startswith(twin_name + ".")
                for name in imported
            ) and any(
                name == oracle_name or name.startswith(oracle_name + ".")
                for name in imported
            ):
                modules += 1
        pinned += max(modules - 1, 0)
    return pinned


FINDINGS = (
    Finding(
        "enumerate-shaped functions",
        r"^def (?:enumerate\w*|brute_force\w*|optimum)\(",
        8,
        owner="enumeration.py",
    ),
    Finding("private logsumexp copies", r"^def _logsumexp\(", 4),
    Finding(
        "open-coded edge/coupling zips",
        r"zip\(\s*\w+\.edges,\s*\w+\.coupling",
        6,
        owner="sim/graph.py",
    ),
    Finding(
        "list-of-lists adjacency",
        r"list\[list\[tuple\[int, ?float\]\]\]",
        9,
        owner="sim/graph.py",
    ),
    Finding(
        "enumeration-cap literals",
        r"^\s*MAX_ENUMERABLE\w* = \d",
        2,
        owner="enumeration.py",
    ),
    Finding("signatures taking a seed", r"seed: int", 12),
    Finding("signatures taking a generator", r"rng: np\.random\.Generator", 10),
    Finding(
        "energy-shaped definitions", r"^\s*def (?:energy|log_weights|relaxed)\(", 5
    ),
    # Issue #717's rows. `at_filing` is the count on the day the ticket was
    # written, under this survey's convention of excluding the owner; where
    # the ticket's table said otherwise the ticket was an impression and this
    # is the query.
    Finding(
        "Potts energies of a labelling",
        r"^def (?:energy|energies|cut_energy)\(",
        3,
        owner="sim/potts.py",
    ),
    Finding("site-field broadcasts", r"^def _?site_field\(", 2, owner="sim/potts.py"),
    Finding("annealers", r"^def anneal\w*\(", 4, owner="sample/gibbs.py"),
    Finding("ground-state run_ wrappers", r"^def run_\w+\(", 10),
    Finding("backend enums", r"^class \w*Backend\b", 1, owner="backend.py"),
    # The ticket counted nine; `ragged_rust` has no `ragged` beside it.
    Finding("Python paths above a compiled kernel", "", 8, counter=twin_modules),
    Finding("surrogate modules", "", 4, counter=surrogate_modules),
    Finding("modules without a docstring", "", 1, counter=undocumented_modules),
    Finding("root exports", "", 1, counter=root_exports),
    Finding(
        "test modules pinning a twin to its oracle beyond the first",
        "",
        12,
        counter=twin_pins_beyond_the_first,
    ),
    Finding("flat modules", "", 150, counter=flat_modules),
    Finding("API-map entries", "", 1576, counter=api_map_entries),
    # The three rows #813 reads its target against. `at_filing` is the count
    # on the day that ticket was filed, `main` at e795a73, so a later reader
    # can see what the folds returned rather than what was hoped for.
    Finding("package lines", "", 51516, counter=package_lines),
    Finding("test modules", "", 251, counter=test_modules),
    Finding("test lines", "", 56986, counter=test_lines),
)


def count(pattern: str, owner: str = "") -> int:
    """How many times ``pattern`` matches across the package.

    Parameters
    ----------
    pattern : str
        Regular expression, applied per file in multiline mode.
    owner : str
        Package-relative path allowed to hold the pattern, excluded from the
        count. Empty means count everywhere.

    Returns
    -------
    int
        Total matches over every ``.py`` file under ``python/snakes_and_ladders``.
    """
    compiled = re.compile(pattern, re.MULTILINE)
    return sum(
        len(compiled.findall(path.read_text()))
        for path in _package_modules()
        if not (owner and path.as_posix().endswith(owner))
    )


def main() -> None:
    """Print one row per finding: name, count at filing, count now."""
    width = max(len(finding.name) for finding in FINDINGS)
    print(f"{'finding':{width}}  {'at filing':>9}  {'now':>5}")
    for finding in FINDINGS:
        print(f"{finding.name:{width}}  {finding.at_filing:>9}  {finding.now():>5}")


if __name__ == "__main__":
    main()
