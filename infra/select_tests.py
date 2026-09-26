"""Decide which tests a change needs, and what to measure coverage against.

Continuous integration runs the whole suite on every pull request, including one
that changed only documentation, where the run cannot differ from the last one
on `main` (issue #161). This module answers two questions from the changed
files: which test paths to run, and which modules to measure coverage over.

Two properties matter more than the saving.

A change runs the test files that import what it changed, transitively (issue
#1086). Each test file's import closure over `sal`, the `tests` helpers and
`infra` is read with `ast`, never by importing; a package reaches its backend
twins, which `backend.twin` loads by name, and a validation
adapter reaches its script. The dependents are derived from the source rather
than listed, since a list goes stale silently and an import does not: the
subpackage list this replaced had missed `sample` and `validation` since they
were created, and so sent every change to them to the whole suite. Of the 30
merges before #1086, 13 ran the whole suite.

What no import can say still selects everything: a lockfile, `pyproject.toml`,
the root `conftest.py`, the fixtures, the workflows, and a code path the graph
does not know. Tests that read the package's source rather than import it (the
duplication guards, the API map) run on any change under `python/sal/`.

A change that is not code selects the guards that read it (issue #372). A
paragraph of the textbook cannot break a likelihood but can break a label or a
citation, which `tests/regression/test_document_labels.py` checks; the planning
files, the `CLAUDE.md` files, the experiment ledger and the notebooks each have
their guard. Those run, and nothing else does.

A selection is a set of paths *and* a set of tiers, and the second keeps the
`key` tier affordable (issue #399). A key test takes two minutes, so it cannot
run on every pull request, and it is the only test that runs the declared
instance end to end, so it must run on the pull requests that could move its
result. :data:`KEY_TRIGGERS` names those --- the coupled model, the emissions,
the fixtures and the Rust crate --- and every other change deselects the tier.

The fallback is not bounded. `--budget` bounded it for a merge gate and
`UNBOUNDABLE` exempted the changes a bound was least safe for; issue #425
removed the pair, since a bound and a fixed timeout cannot coexist. The merge
gate's 15-minute cap fired on exactly the pull requests `UNBOUNDABLE` refused to
bound -- every `likelihood/` one -- and a cancelled job is indistinguishable
from a failing one (issue #423). The gate now runs the selection unbounded under
the job's own cap, and the whole suite still runs on the push to `main`.
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import sys
from collections.abc import Iterable

from _paths import REPO_ROOT

# The submodules with their own test directory. `sal.numerics`,
# `sal.emissions` and the scaffolding hot path are covered by
# the top-level regression modules, which are cheap and always run.
#
# `sandbox` earned a row here with issue #516, and it is the row that shows
# why the dependents are derived rather than listed. Nothing imports it, so a
# change to it selects only its own tests, `qa`'s -- which render a figure
# from it -- and its benchmark. It imports `likelihood`, `search` and `sim`,
# so a change to any of those selects it, which is what
# `tests/regression/sandbox/test_pruning_burn.py` needs now that it no longer
# sits under `likelihood/`.
MODULES = (
    "sim",
    "likelihood",
    "opt",
    "learn",
    "search",
    "qa",
    "sandbox",
    "sample",
)

#: Subpackages whose tests live outside `tests/regression/<name>`.
TEST_DIRS = {"validation": "tests/validation"}

# The modules a benchmark measures. `qa` renders figures from what these
# compute and is not itself timed, as issue #109's trigger had it. `sandbox`
# is timed: a declined route is conserved with the measurement that declined it.
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
    "tests/test_oxisal_bindings.py",
)


# What could move a key fixture's own result, and so selects the `key` tier.
# Everything else deselects it: the tier costs two minutes a test (issue #399).
KEY_TRIGGERS = (
    "src/",
    "Cargo.toml",
    "Cargo.lock",
    "tests/regression/fixtures/",
    "python/sal/emissions/",
    "python/sal/sim/count_pairs/",
    "python/sal/sim/spatio_sequential.py",
    "python/sal/likelihood/spatio_sequential/",
    "python/sal/search/spatio_sequential.py",
)

#: The tiers a per-pull-request selection never runs. `key` joins them unless
#: the change is one of :data:`KEY_TRIGGERS`.
ALWAYS_DESELECTED = ("release", "stress")

# Nothing here can change what a test does, so no test needs to run.
NO_TESTS_SUFFIXES = (".md", ".tex", ".bib", ".pdf", ".txt", ".rst")
NO_TESTS_PREFIXES = ("docs/", "changelog.d/")

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
        ("ROADMAP.md", "STATUS.md"),
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


@functools.cache
def _module_imports() -> dict[str, set[str]]:
    """Read which submodules each submodule imports.

    Returns
    -------
    dict[str, set[str]]
        Submodule name to the submodules it imports.
    """
    imports: dict[str, set[str]] = {module: set() for module in MODULES}
    for module in MODULES:
        for path in (REPO_ROOT / "python" / "sal" / module).rglob("*.py"):
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
                        and parts[0] == "sal"
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


# --- File-level selection (issue #1086) -------------------------------------

#: Test files that read the package's source rather than import it, so any
#: change under `python/sal/` can fail them.
SOURCE_READERS = (
    "tests/regression/test_directory_imports.py",
    "tests/regression/test_duplication_guards.py",
    "tests/regression/test_sandbox.py",
    "tests/regression/test_validation.py",
    "tests/regression/test_environment.py",
    "tests/regression/sim/test_generator_signatures.py",
    "tests/regression/docs/test_mind_map.py",
    "tests/regression/docs/test_docs_index_covers_every_module.py",
    "tests/regression/docs/test_api_map.py",
)

#: What no import can attribute: a change to any of these runs the whole suite.
UNATTRIBUTABLE = (
    "pyproject.toml",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
    "tests/conftest.py",
    "tests/regression/fixtures/",
    ".github/workflows/",
)

#: The backend twins a gateway loads by name (`backend.twin`), which no
#: import statement shows: a package reaches its children of these names.
TWINS = frozenset({"rust", "numba", "torch", "jax", "python"})

#: The module a Rust change is a change to: every binding is reached through it.
RUST_MODULE = "sal.oxisal"


def _module_name(path: str) -> str | None:
    """The dotted name a repository path imports as, or ``None`` if it is not Python."""
    if not path.endswith((".py", ".pyi")):
        return None
    stem = path.removesuffix(".pyi").removesuffix(".py")
    if path.startswith("python/"):
        stem = stem.removeprefix("python/")
    elif path.startswith("infra/"):
        stem = stem.removeprefix("infra/")
    elif not path.startswith("tests/"):
        return None
    parts = stem.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _sources() -> dict[str, str]:
    """Every importable Python file under `python/`, `tests/` and `infra/`, by module name."""
    found: dict[str, str] = {}
    for root in ("python/sal", "tests", "infra"):
        for file in sorted((REPO_ROOT / root).rglob("*.py*")):
            relative = file.relative_to(REPO_ROOT).as_posix()
            name = _module_name(relative)
            if name is not None and "__pycache__" not in relative:
                found.setdefault(name, relative)
    return found


def _resolve(name: str, known: dict[str, str]) -> str | None:
    """The longest prefix of ``name`` that is a known module."""
    parts = name.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return None


@functools.cache
def import_graph() -> dict[str, frozenset[str]]:
    """Module name to the known modules it reaches directly (issue #1086).

    An import reaches the module named and, where ``from a import b`` names a
    submodule, that submodule; every module reaches its parent packages, which
    its import runs; a package reaches its backend twins (:data:`TWINS`),
    which a gateway loads by name; and ``sal.validation.<name>`` reaches its script.
    """
    known = _sources()
    graph: dict[str, set[str]] = {name: set() for name in known}
    for name, relative in known.items():
        package = name if relative.endswith("__init__.py") else name.rpartition(".")[0]
        edges = graph[name]
        try:
            tree = ast.parse((REPO_ROOT / relative).read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package.split(".")
                    anchor = anchor[: len(anchor) - node.level + 1]
                    base = ".".join([*anchor, base] if base else anchor)
                targets = [base] + [f"{base}.{alias.name}" for alias in node.names]
            for target in targets:
                resolved = _resolve(target, known)
                if resolved is not None and resolved != name:
                    edges.add(resolved)
        parent = name.rpartition(".")[0]
        while parent:
            if parent in known:
                edges.add(parent)
            parent = parent.rpartition(".")[0]
    for name in known:
        parent = name.rpartition(".")[0]
        if parent in graph and name.rpartition(".")[2] in TWINS:
            graph[parent].add(name)
        if name.startswith("sal.validation.") and name.count(".") == 2:
            script = f"sal.validation.scripts.{name.rpartition('.')[2]}"
            if script in known:
                graph[name].add(script)
    # Read once per process: the source does not change under a run.
    return {name: frozenset(edges) for name, edges in graph.items()}


def _closure(start: str, graph: dict[str, frozenset[str]]) -> set[str]:
    """Every module ``start`` reaches, itself included."""
    seen = {start}
    stack = [start]
    while stack:
        for target in graph.get(stack.pop(), ()):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


@functools.cache
def test_files() -> tuple[str, ...]:
    """Every collected test file: `test_*.py` under `tests/`, benchmarks aside."""
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "tests").rglob("test_*.py")
            if "benchmarks" not in path.parts
        )
    )


def importers(modules: Iterable[str]) -> list[str]:
    """The test files whose import closure contains any of ``modules``."""
    wanted = set(modules)
    if not wanted:
        return []
    closures = _test_closures()
    return [test for test in test_files() if closures[test] & wanted]


@functools.cache
def _test_closures() -> dict[str, frozenset[str]]:
    """Each test file's import closure, read once per process."""
    graph = import_graph()
    return {
        test: frozenset(_closure(_module_name(test) or "", graph))
        for test in test_files()
    }


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
            "cov": ["sal"],
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

    whole = {"paths": ["tests"], "cov": ["sal"], "deselect": deselect}
    if any(_touches(path, UNATTRIBUTABLE) for path in relevant):
        return whole

    modules: set[str] = set()
    tests: set[str] = set()
    for path in relevant:
        if path.startswith("src/"):
            modules.add(RUST_MODULE)
            continue
        name = _module_name(path)
        if name is None:
            # Recognised as code, but no import can say what reads it.
            return whole
        if path.startswith("tests/") and path.rpartition("/")[2].startswith("test_"):
            tests.add(path)
        modules.add(name)
    if not (REPO_ROOT / "tests").is_dir():
        return whole
    tests |= set(importers(modules))
    if any(path.startswith("python/sal/") for path in relevant):
        tests |= set(SOURCE_READERS)
    tests = {test for test in tests if (REPO_ROOT / test).is_file()}
    paths = sorted(tests) + [path for path in ALWAYS if path not in tests]
    paths += [guard for guard in guards if guard not in paths]
    packages = {
        name.split(".")[1]
        for name in modules
        if name.startswith("sal.") and name.count(".") >= 1
    }
    touched = {module for module in packages if module in MODULES}
    if any(path.startswith("python/sal/") for path in relevant):
        paths += _benchmarks_for(dependents(touched) & set(BENCHMARKED))
    # Coverage is measured per changed subpackage, over that subpackage's own
    # tests as well as the importers: a subpackage measured by only the files
    # importing one of its modules would read low against the push run's
    # floor, and a single module reads low where its kernels are compiled
    # (`sal.search.icm` alone is 72%, its numba twin untraced). Nothing is
    # measured when no `sal` subpackage changed, as for prose.
    measured = sorted(
        package for package in packages if package in MODULES or package in TEST_DIRS
    )
    touched_tests = [
        TEST_DIRS.get(package, f"tests/regression/{package}") for package in measured
    ]
    paths += [
        path
        for path in touched_tests
        if path not in paths and (REPO_ROOT / path).is_dir()
    ]
    return {
        "paths": paths,
        "cov": [f"sal.{package}" for package in measured],
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
