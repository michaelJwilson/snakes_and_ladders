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

`--budget` bounds the fallback, and only the fallback (issue #409). Answering
"everything" is right for safety and wrong for a merge gate: it took 23 to 36
minutes on the two pull requests that merged on 2026-09-08, against 156 seconds
for the nine other jobs together, and with several agents opening pull requests
at once a serial merge chain forms behind it. Under `--budget` a change that
would select everything selects instead what it can attribute plus what always
runs, and the whole suite runs after the merge, on the push to `main`, where
nobody waits for it. The safety moves rather than disappearing, which is the
same trade `snakes_and_ladders.qa.build` makes for figures.

`UNBOUNDABLE` is where that trade is refused. A change touching the likelihood
kernels, the Rust sources or a lockfile keeps whatever the unbounded selection
would have given it, because those are where a regression moves a number rather
than breaking a build: nothing fails loudly, and every branch cut before the
post-merge run inherits it. Most such changes are attributable and never reach
the fallback anyway; the list matters for the ones that touch a kernel *and*
something the mapping cannot place, which is exactly when a bound would be
least safe.
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
MODULES = ("sim", "likelihood", "opt", "learn", "search", "qa")

# The modules a benchmark measures. `qa` renders figures from what these
# compute and is not itself timed, which is why issue #109's trigger excluded
# it and why this does too.
BENCHMARKED = ("sim", "likelihood", "opt", "learn", "search")

# Always run: cheap, and they cover what belongs to no single module.
ALWAYS = (
    "tests/regression/test_numerics.py",
    "tests/regression/test_emissions.py",
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
    "python/snakes_and_ladders/emissions.py",
    "python/snakes_and_ladders/numerics.py",
    "python/snakes_and_ladders/numerics_rust.py",
    "python/snakes_and_ladders/oxi_snakes_and_ladders.pyi",
    "python/snakes_and_ladders/scripts/",
)

# These refuse a bounded fallback, `--budget` or not: a regression here moves a
# number rather than breaking a build, so catching it after the merge is
# catching it too late (issue #409).
UNBOUNDABLE = (
    "python/snakes_and_ladders/likelihood/",
    "src/",
    "pyproject.toml",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
)

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


def unboundable(changed: Iterable[str]) -> list[str]:
    """The changed paths that refuse a bounded selection.

    Parameters
    ----------
    changed : Iterable[str]
        Repository-relative paths.

    Returns
    -------
    list[str]
        Those touching ``UNBOUNDABLE``, sorted. Empty when ``--budget`` may
        bound the fallback.
    """
    return sorted({path for path in changed if _touches(path, UNBOUNDABLE)})


def _bounded(
    touched: set[str], guards: list[str], relevant: list[str]
) -> dict[str, list[str]]:
    """The selection that replaces "everything" under ``--budget``.

    What a bounded fallback still runs: whatever modules the change can be
    attributed to, with their dependents; the always-run tests, which cover
    what belongs to no single module; and the guards. The critical tier runs
    before this in the workflow, unconditionally, so it is not repeated here.

    Parameters
    ----------
    touched : set[str]
        Modules the change touched; may be empty, which is the case where the
        unbounded answer would have been the whole suite for want of an
        attribution.
    guards : list[str]
        The guards the non-code part of the change selects.
    relevant : list[str]
        The code paths of the change, which decide whether benchmarks run.

    Returns
    -------
    dict[str, list[str]]
        ``paths`` and ``cov`` as :func:`select` returns them.
    """
    selected = dependents(touched) if touched else set()
    paths = [f"tests/regression/{module}" for module in sorted(selected)]
    paths += list(ALWAYS)
    paths += [guard for guard in guards if guard not in paths]
    if selected and any(
        path.startswith("python/snakes_and_ladders/") for path in relevant
    ):
        paths += _benchmarks_for(selected & set(BENCHMARKED))
    return {
        "paths": paths,
        "cov": [f"snakes_and_ladders.{module}" for module in sorted(selected)],
    }


def select(changed: Iterable[str], budget: bool = False) -> dict[str, list[str]]:
    """Choose test paths and coverage targets for a set of changed files.

    Parameters
    ----------
    changed : Iterable[str]
        Repository-relative paths, as `git diff --name-only` gives them.
    budget : bool
        Bound the fallback rather than answering with the whole suite, for a
        merge gate that runs before the post-merge run rather than instead of
        it (issue #409). A change touching ``UNBOUNDABLE`` ignores this.

    Returns
    -------
    dict[str, list[str]]
        ``paths`` to hand pytest and ``cov`` targets to measure, both empty
        when nothing needs running. ``paths`` is ``["tests"]`` when the whole
        suite is selected, and the guards alone, with no coverage target,
        when nothing but prose changed.
    """
    changed = list(changed)
    if not changed:
        return {"paths": ["tests"], "cov": ["snakes_and_ladders"]}
    budget = budget and not unboundable(changed)

    relevant = [
        path
        for path in changed
        if not (path.endswith(NO_TESTS_SUFFIXES) and not path.startswith("tests/"))
        and not _touches(path, NO_TESTS_PREFIXES)
    ]
    guards = guards_for(path for path in changed if path not in relevant)
    if not relevant:
        return {"paths": guards, "cov": []}

    everything = any(_touches(path, EVERYTHING) for path in relevant)

    touched: set[str] = set()
    for path in relevant:
        for module in MODULES:
            if path.startswith(
                (f"python/snakes_and_ladders/{module}/", f"tests/regression/{module}/")
            ):
                touched.add(module)

    # Recognised as code, but not attributable to a module: the whole suite,
    # unless a budget is asked for and the change is one it may bound.
    if everything or not touched:
        if not budget:
            return {"paths": ["tests"], "cov": ["snakes_and_ladders"]}
        return _bounded(touched, guards, relevant)

    selected = dependents(touched)
    paths = [f"tests/regression/{module}" for module in sorted(selected)]
    paths += list(ALWAYS)
    paths += [guard for guard in guards if guard not in paths]
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
    parser.add_argument(
        "--budget",
        action="store_true",
        help="bound the fallback for a merge gate; the post-merge run is unbounded",
    )
    args = parser.parse_args(argv)

    changed = args.changed or [line.strip() for line in sys.stdin if line.strip()]
    refused = unboundable(changed) if args.budget else []
    if refused:
        print(
            "budget refused, the whole suite runs: " + " ".join(refused),
            file=sys.stderr,
        )
    chosen = select(changed, budget=args.budget)
    if args.budget and not refused and chosen["paths"] != ["tests"]:
        print("selection bounded for the merge gate", file=sys.stderr)

    # Whether the selection is bounded decides how long the merge gate may
    # take, so it is an output rather than only a stderr notice: an
    # unboundable change runs the whole suite, which cannot fit the bounded
    # gate's timeout, and a job cancelled at that timeout is indistinguishable
    # from a failing test (issue #423).
    bounded = bool(args.budget) and not refused and chosen["paths"] != ["tests"]

    if args.format == "json":
        print(json.dumps({**chosen, "bounded": bounded}))
    else:
        print(f"paths={' '.join(chosen['paths'])}")
        print(f"cov={' '.join('--cov=' + target for target in chosen['cov'])}")
        print(f"bounded={'true' if bounded else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
