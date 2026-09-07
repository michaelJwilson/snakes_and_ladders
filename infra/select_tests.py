"""Decide which tests a change needs, and what to measure coverage against.

Continuous integration runs the whole suite on every pull request, including
one that changed only documentation, where the run cannot differ from the last
one on `main` (issue #161). This module answers two questions from the list of
changed files: which test paths to run, and which modules to measure coverage
over.

Two properties matter more than the saving.

A module's tests are not enough on their own: `snakes_and_ladders.search` imports
`snakes_and_ladders.likelihood`, so a change to the latter must run the former's tests too.
The dependents are derived from the source here rather than listed, because a
list goes stale silently and an import does not.

Anything the mapping does not recognise selects everything. A changed lockfile,
a changed shared fixture, a changed workflow -- the safe answer is the whole
suite, and the unsafe answer is the one that looks like a saving.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The submodules with their own test directory. `snakes_and_ladders.numerics` and the
# scaffolding hot path are covered by the top-level regression modules, which
# are cheap and always run.
MODULES = ("sim", "likelihood", "opt", "learn", "search", "qa")

# The modules a benchmark measures. `qa` renders figures from what these
# compute and is not itself timed, which is why issue #109's trigger excluded
# it and why this does too.
BENCHMARKED = ("sim", "likelihood", "opt", "learn", "search")

# Always run: cheap, and they cover what belongs to no single module.
ALWAYS = (
    "tests/regression/test_numerics.py",
    "tests/regression/test_pairwise_distance.py",
    "tests/regression/test_claude_md_pointers.py",
    "tests/test_run_snakes_and_ladders.py",
    "tests/test_oxi_snakes_and_ladders_bindings.py",
)

# A change to any of these could alter any result, so the whole suite runs.
EVERYTHING = (
    "pyproject.toml",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
    "src/",
    "tests/_",
    "tests/regression/fixtures/",
    ".github/workflows/",
    "python/snakes_and_ladders/__init__.py",
    "python/snakes_and_ladders/numerics.py",
    "python/snakes_and_ladders/numerics_rust.py",
    "python/snakes_and_ladders/oxi_snakes_and_ladders.pyi",
    "python/snakes_and_ladders/scripts/",
)

# Nothing here can change what a test does, so no test needs to run.
NO_TESTS_SUFFIXES = (".md", ".tex", ".bib", ".pdf", ".txt", ".rst")
NO_TESTS_PREFIXES = ("docs/", "changelog.d/", "infra/")


def _module_imports() -> dict[str, set[str]]:
    """Read which submodules each submodule imports.

    Returns
    -------
    dict[str, set[str]]
        Submodule name to the submodules it imports.
    """
    imports: dict[str, set[str]] = {module: set() for module in MODULES}
    for module in MODULES:
        for path in (REPO_ROOT / "python" / "snakes_and_ladders" / module).rglob(
            "*.py"
        ):
            for node in ast.walk(ast.parse(path.read_text())):
                names: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                for name in names:
                    parts = name.split(".")
                    if (
                        len(parts) >= 2
                        and parts[0] == "snakes_and_ladders"
                        and parts[1] in MODULES
                        and parts[1] != module
                    ):
                        imports[module].add(parts[1])
    return imports


def dependents(modules: Iterable[str]) -> set[str]:
    """Expand ``modules`` to include everything importing them, transitively.

    Parameters
    ----------
    modules : Iterable[str]
        The submodules a change touched.

    Returns
    -------
    set[str]
        Those submodules and every submodule reaching them by import.
    """
    imports = _module_imports()
    selected = set(modules)
    changed = True
    while changed:
        changed = False
        for module in MODULES:
            if module not in selected and imports[module] & selected:
                selected.add(module)
                changed = True
    return selected


def _benchmarks_for(modules: Iterable[str]) -> list[str]:
    """Find the benchmark modules pairing with ``modules``' regression tests.

    `DEV.md` requires `benchmarks/test_<name>_bench.py` to accompany
    `regression/<module>/test_<name>.py`, so the pairing is read off the
    filenames rather than listed here -- a list would go stale the next time a
    benchmark is added.

    Returns
    -------
    list[str]
        Benchmark paths, sorted.
    """
    wanted = set(modules)
    found = []
    for bench in (REPO_ROOT / "tests" / "benchmarks").glob("test_*_bench.py"):
        stem = bench.name.removesuffix("_bench.py")
        for module in MODULES:
            counterpart = REPO_ROOT / "tests" / "regression" / module / f"{stem}.py"
            if counterpart.is_file() and module in wanted:
                found.append(f"tests/benchmarks/{bench.name}")
                break
    return sorted(found)


def _touches(path: str, markers: Iterable[str]) -> bool:
    """Whether ``path`` starts with any of ``markers``.

    Returns
    -------
    bool
        True if the path is under one of them.
    """
    return any(path.startswith(marker) for marker in markers)


def select(changed: Iterable[str]) -> dict[str, list[str]]:
    """Choose test paths and coverage targets for a set of changed files.

    Parameters
    ----------
    changed : Iterable[str]
        Repository-relative paths, as `git diff --name-only` gives them.

    Returns
    -------
    dict[str, list[str]]
        ``paths`` to hand pytest and ``cov`` targets to measure, both empty
        when nothing needs running. ``paths`` is ``["tests"]`` when the whole
        suite is selected.
    """
    changed = list(changed)
    if not changed:
        return {"paths": ["tests"], "cov": ["snakes_and_ladders"]}

    relevant = [
        path
        for path in changed
        if not (path.endswith(NO_TESTS_SUFFIXES) and not path.startswith("tests/"))
        and not _touches(path, NO_TESTS_PREFIXES)
    ]
    if not relevant:
        return {"paths": [], "cov": []}

    if any(_touches(path, EVERYTHING) for path in relevant):
        return {"paths": ["tests"], "cov": ["snakes_and_ladders"]}

    touched: set[str] = set()
    for path in relevant:
        for module in MODULES:
            if path.startswith(
                (f"python/snakes_and_ladders/{module}/", f"tests/regression/{module}/")
            ):
                touched.add(module)
    if not touched:
        # Recognised as code, but not attributable to a module: run everything.
        return {"paths": ["tests"], "cov": ["snakes_and_ladders"]}

    selected = dependents(touched)
    paths = [f"tests/regression/{module}" for module in sorted(selected)]
    paths += list(ALWAYS)
    if any(path.startswith("python/snakes_and_ladders/") for path in relevant):
        paths += _benchmarks_for(selected & set(BENCHMARKED))
    return {
        "paths": paths,
        "cov": [f"snakes_and_ladders.{module}" for module in sorted(selected)],
    }


def main(argv: list[str] | None = None) -> int:
    """Print the selection for the changed files given on stdin or as arguments.

    Returns
    -------
    int
        Always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("changed", nargs="*", help="changed paths; else stdin")
    parser.add_argument("--format", choices=("json", "shell"), default="shell")
    args = parser.parse_args(argv)

    changed = args.changed or [line.strip() for line in sys.stdin if line.strip()]
    chosen = select(changed)

    if args.format == "json":
        print(json.dumps(chosen))
    else:
        print(f"paths={' '.join(chosen['paths'])}")
        print(f"cov={' '.join('--cov=' + target for target in chosen['cov'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
