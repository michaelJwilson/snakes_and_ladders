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

A change that is not code selects the guards that read it (issue #372). A
paragraph of the textbook cannot break a likelihood, but it can break a label
or a citation, and `tests/regression/test_document_labels.py` is what checks
that; the planning files, the `CLAUDE.md` files, the experiment ledger and the
notebooks each have their guard. Those run, and nothing else does.

A selection is a set of paths *and* a set of tiers, and the second is what
keeps the `key` tier affordable (issue #399). A key test is a two-minute test,
so it cannot run on every pull request; it is also the only test that runs the
declared instance end to end, so it must run on the pull requests that could
move its result. :data:`KEY_TRIGGERS` names those --- the coupled model, the
emissions, the fixtures and the Rust crate --- and every other change
deselects the tier.

The fallback is not bounded. `--budget` bounded it for a merge gate and
`UNBOUNDABLE` exempted the changes a bound was least safe for; the pair was
removed with issue #425, because a bound and a fixed timeout cannot coexist.
The merge gate's 15-minute cap fired on exactly the pull requests `UNBOUNDABLE`
refused to bound -- every `likelihood/` one -- and a cancelled job is
indistinguishable from a failing one (issue #423). The gate now runs the
selection unbounded under the job's own cap, and the whole suite still runs on
the push to `main`.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The submodules with their own test directory. `snakes_and_ladders.numerics`,
# `snakes_and_ladders.emissions` and the scaffolding hot path are covered by
# the top-level regression modules, which are cheap and always run.
#
# `sandbox` earned a row here with issue #516, and it is the row that shows
# why the dependents are derived rather than listed. Nothing imports it, so a
# change to it selects only its own tests, `qa`'s -- which render a figure
# from it -- and its benchmark. It imports `likelihood`, `search` and `sim`,
# so a change to any of those selects it, which is what
# `tests/regression/sandbox/test_pruning_burn.py` needs now that it no longer
# sits under `likelihood/`.
MODULES = ("sim", "likelihood", "opt", "learn", "search", "qa", "sandbox")

# The modules a benchmark measures. `qa` renders figures from what these
# compute and is not itself timed, which is why issue #109's trigger excluded
# it and why this does too. `sandbox` is timed: a declined route is conserved
# with the measurement that declined it.
BENCHMARKED = ("sim", "likelihood", "opt", "learn", "search", "sandbox")

# Always run: cheap, and they cover what belongs to no single module.
# `test_sandbox.py` is here because what it asserts is an *absence* -- that
# none of the five hot-path packages imports the oracle home -- and an absence
# is not an import this module can follow (issue #516). It costs 1.9 s.
ALWAYS = (
    "tests/regression/test_numerics.py",
    "tests/regression/test_emissions.py",
    "tests/regression/test_pairwise_distance.py",
    "tests/regression/test_claude_md_pointers.py",
    "tests/regression/test_sandbox.py",
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
    "python/snakes_and_ladders/emissions.py",
    "python/snakes_and_ladders/numerics.py",
    "python/snakes_and_ladders/numerics_rust.py",
    "python/snakes_and_ladders/oxi_snakes_and_ladders.pyi",
    "python/snakes_and_ladders/scripts/",
)

# What could move a key fixture's own result, and so selects the `key` tier.
# Everything else deselects it: the tier costs two minutes a test and runs
# only where it can fail (issue #399).
KEY_TRIGGERS = (
    "src/",
    "Cargo.toml",
    "Cargo.lock",
    "tests/regression/fixtures/",
    "python/snakes_and_ladders/emissions.py",
    "python/snakes_and_ladders/sim/count_pairs.py",
    "python/snakes_and_ladders/sim/spatio_sequential.py",
    "python/snakes_and_ladders/likelihood/spatio_sequential.py",
    "python/snakes_and_ladders/likelihood/spatio_sequential_rust.py",
    "python/snakes_and_ladders/search/spatio_sequential.py",
)

#: The tiers a per-pull-request selection never runs. `key` joins them unless
#: the change is one of :data:`KEY_TRIGGERS`.
ALWAYS_DESELECTED = ("release", "stress")

# Nothing here can change what a test does, so no test needs to run.
NO_TESTS_SUFFIXES = (".md", ".tex", ".bib", ".pdf", ".txt", ".rst")
NO_TESTS_PREFIXES = ("docs/", "changelog.d/", "infra/")

# The guards a non-code change selects: what a path starts with, and the tests
# that read files under it. A guard reads the repository directly, so it is
# the one test a change to that file can fail. `changelog.d/` has no guard
# here because `towncrier check` in the lint job is its guard.
GUARDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("docs/tex/",),
        (
            "tests/regression/test_document_labels.py",
            "tests/regression/docs/test_reference_taxonomy.py",
            "tests/regression/docs/test_problems_tables.py",
            "tests/regression/qa/test_qa_build.py",
        ),
    ),
    (
        ("ROADMAP.md", "STATUS.md", "TICKETS.md"),
        (
            "tests/regression/test_planning_documents_agree.py",
            "tests/regression/test_repository_links.py",
        ),
    ),
    (
        ("CLAUDE.md",),
        (
            "tests/regression/test_claude_md_pointers.py",
            "tests/regression/docs/test_reference_taxonomy.py",
        ),
    ),
    (("docs/experiments/",), ("tests/regression/test_experiments.py",)),
    (("docs/nb/",), ("tests/regression/test_check_notebooks.py",)),
    (
        ("docs/source/",),
        ("tests/regression/docs/test_docs_index_covers_every_module.py",),
    ),
    (("PROBLEMS.md",), ("tests/regression/test_problems_catalogue.py",)),
    (("DEV.md",), ("tests/regression/test_scale_tiers.py",)),
    (("README.md", "INSTALL.md"), ("tests/regression/test_repository_links.py",)),
)


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


def guards_for(changed: Iterable[str]) -> list[str]:
    """The guard tests that read any of ``changed``.

    A `CLAUDE.md` under a module directory is matched by its name, not the
    module prefix: a module's source is code and is attributed by module,
    while its `CLAUDE.md` is prose the pointer guard reads.

    Returns
    -------
    list[str]
        Test paths, sorted and without duplicates.
    """
    selected: set[str] = set()
    for path in changed:
        for markers, tests in GUARDS:
            if (path.endswith("CLAUDE.md") and "CLAUDE.md" in markers) or any(
                path.startswith(marker) for marker in markers
            ):
                selected.update(tests)
    return sorted(selected)


def deselected(changed: Iterable[str]) -> list[str]:
    """The tier markers this change does not run.

    Returns
    -------
    list[str]
        ``release`` and ``stress`` always; ``key`` unless the change touches
        one of :data:`KEY_TRIGGERS`.
    """
    if any(_touches(path, KEY_TRIGGERS) for path in changed):
        return list(ALWAYS_DESELECTED)
    return [*ALWAYS_DESELECTED, "key"]


def select(changed: Iterable[str]) -> dict[str, list[str]]:
    """Choose test paths, coverage targets and deselected tiers for a change.

    Parameters
    ----------
    changed : Iterable[str]
        Repository-relative paths, as `git diff --name-only` gives them.

    Returns
    -------
    dict[str, list[str]]
        ``paths`` to hand pytest, ``cov`` targets to measure, and the
        ``deselect`` tiers to skip. The first two are empty when nothing needs
        running; ``paths`` is ``["tests"]`` when the whole suite is selected,
        and the guards alone, with no coverage target, when nothing but prose
        changed. ``deselect`` is always populated: a selection that runs
        everything still does not run a two-minute key test unless the change
        could move its result.
    """
    changed = list(changed)
    deselect = deselected(changed)
    if not changed:
        return {
            "paths": ["tests"],
            "cov": ["snakes_and_ladders"],
            "deselect": deselect,
        }

    relevant = [
        path
        for path in changed
        if not (path.endswith(NO_TESTS_SUFFIXES) and not path.startswith("tests/"))
        and not _touches(path, NO_TESTS_PREFIXES)
    ]
    guards = guards_for(path for path in changed if path not in relevant)
    if not relevant:
        return {"paths": guards, "cov": [], "deselect": deselect}

    everything = any(_touches(path, EVERYTHING) for path in relevant)

    touched: set[str] = set()
    for path in relevant:
        for module in MODULES:
            if path.startswith(
                (f"python/snakes_and_ladders/{module}/", f"tests/regression/{module}/")
            ):
                touched.add(module)

    # Recognised as code, but not attributable to a module: the whole suite.
    if everything or not touched:
        return {
            "paths": ["tests"],
            "cov": ["snakes_and_ladders"],
            "deselect": deselect,
        }

    selected = dependents(touched)
    paths = [f"tests/regression/{module}" for module in sorted(selected)]
    paths += list(ALWAYS)
    paths += [guard for guard in guards if guard not in paths]
    if any(path.startswith("python/snakes_and_ladders/") for path in relevant):
        paths += _benchmarks_for(selected & set(BENCHMARKED))
    return {
        "paths": paths,
        "cov": [f"snakes_and_ladders.{module}" for module in sorted(selected)],
        "deselect": deselect,
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
        print(
            "markers=" + " and ".join(f"not {marker}" for marker in chosen["deselect"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
