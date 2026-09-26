"""What the test selection must guarantee before it is allowed to skip anything.

Running fewer tests is safe only if the selection is right, so what is pinned
is not the saving but the three ways it could be wrong: missing a module that
imports the changed one, mistaking a change it does not understand for one that
needs nothing, and skipping a run that would have measured different coverage
(issue #161).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from select_tests import (
    ALWAYS,
    ALWAYS_DESELECTED,
    BENCHMARKED,
    GUARDS,
    KEY_TRIGGERS,
    MODULES,
    SOURCE_READERS,
    UNATTRIBUTABLE,
    _benchmarks_for,
    dependents,
    guards_for,
    import_graph,
    select,
)

from tests._paths import REPO_ROOT


def _modules_of(chosen: dict[str, list[str]]) -> set[str]:
    """The submodule names a selection runs the tests of."""
    return {
        path.split("/")[2]
        for path in chosen["paths"]
        if path.startswith("tests/regression/")
        and path.count("/") >= 3
        and path.split("/")[2] in MODULES
    }


@pytest.mark.critical
@pytest.mark.smoke
def test_a_documentation_only_change_selects_the_guards_and_measures_nothing() -> None:
    # The suite would run the same code over the same tests as the last run on
    # main, so coverage cannot have moved and no module's tests run. What runs
    # is the guards reading the changed prose -- a paragraph of the textbook
    # can break a label (issue #372).
    chosen = select(
        ["docs/tex/paper.tex", "README.md", "changelog.d/161.changed.md", "DEV.md"]
    )

    assert chosen["cov"] == []
    assert chosen["paths"]
    assert all(path.startswith("tests/regression/") for path in chosen["paths"])
    assert not any(
        path.endswith(("/opt", "/search", "/qa")) for path in chosen["paths"]
    )
    assert "tests/regression/test_document_labels.py" in chosen["paths"]


@pytest.mark.critical
@pytest.mark.smoke
def test_a_changelog_fragment_alone_selects_nothing() -> None:
    # `towncrier check` in the lint job is its guard; the suite has none.
    chosen = select(["changelog.d/372.changed.md"])

    assert chosen["paths"] == []
    assert chosen["cov"] == []


@pytest.mark.critical
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("path", "guard"),
    [
        ("docs/tex/textbook.tex", "tests/regression/test_document_labels.py"),
        ("docs/experiments/006-x.md", "tests/regression/test_experiments.py"),
        ("ROADMAP.md", "tests/regression/test_planning_documents_agree.py"),
        ("STATUS.md", "tests/regression/test_planning_documents_agree.py"),
        ("CLAUDE.md", "tests/regression/test_claude_md_pointers.py"),
        (
            "python/sal/opt/CLAUDE.md",
            "tests/regression/test_claude_md_pointers.py",
        ),
        ("docs/nb/hmm.ipynb", "tests/regression/test_check_notebooks.py"),
        (
            "docs/source/index.rst",
            "tests/regression/docs/test_docs_index_covers_every_module.py",
        ),
        ("PROBLEMS.md", "tests/regression/test_problems_catalogue.py"),
        ("DEV.md", "tests/regression/test_scale_tiers.py"),
    ],
)
def test_each_class_of_prose_selects_its_guard(path: str, guard: str) -> None:
    # One case per class: the guard that reads the file runs, no module's
    # tests do, and nothing is measured.
    chosen = select([path])

    assert guard in chosen["paths"]
    assert chosen["cov"] == []
    assert set(chosen["paths"]) == set(guards_for([path]))


@pytest.mark.critical
@pytest.mark.smoke
def test_every_guard_the_selection_names_exists() -> None:
    # A guard renamed on one side only would select a path pytest cannot
    # collect, which fails the run for the wrong reason.
    for _, tests in GUARDS:
        for test in tests:
            assert (REPO_ROOT / test).is_file(), test


@pytest.mark.critical
@pytest.mark.smoke
def test_a_code_change_beside_prose_runs_both() -> None:
    # The guards join the module's tests rather than replacing them.
    chosen = select(["python/sal/learn/reinforce.py", "docs/tex/textbook.tex"])

    assert "learn" in _modules_of(chosen)
    assert set(guards_for(["docs/tex/textbook.tex"])) <= set(chosen["paths"])


@pytest.mark.critical
@pytest.mark.smoke
def test_a_change_selects_the_modules_that_import_it() -> None:
    # `sal.search` imports `sal.likelihood`, so a likelihood change that
    # ran only likelihood's tests would let a break in search through.
    assert _modules_of(select(["python/sal/likelihood/pruning/__init__.py"])) >= {
        "likelihood",
        "search",
        "qa",
    }


@pytest.mark.critical
@pytest.mark.smoke
def test_a_module_selects_the_test_files_that_import_it_and_no_others() -> None:
    # File level (issue #1086): a module a handful of test files import
    # selects those files, not its whole subpackage's directory.
    chosen = select(["python/sal/validation/highs.py"])
    files = set(chosen["paths"])

    assert "tests/validation/test_highs.py" in files
    assert "tests/regression/search/test_trws.py" not in files
    assert set(SOURCE_READERS) <= files


@pytest.mark.critical
@pytest.mark.smoke
def test_the_dependency_expansion_is_transitive() -> None:
    # opt <- likelihood <- search: a change to opt must reach search even
    # though search does not import opt directly through that path alone.
    assert dependents({"opt"}) >= {"opt", "likelihood", "search", "qa"}


@pytest.mark.critical
@pytest.mark.smoke
def test_changing_a_test_selects_its_module() -> None:
    # A test file is as capable of lowering coverage as a source file, and a
    # changed test file runs itself.
    chosen = select(["tests/regression/opt/test_opt_fit.py"])

    assert "tests/regression/opt/test_opt_fit.py" in chosen["paths"]


@pytest.mark.critical
@pytest.mark.smoke
def test_a_likelihood_change_still_runs_the_conserved_gradient_tape() -> None:
    # `test_pruning_burn.py` moved to the sandbox directory (#516) but referees
    # the likelihood tape, so a likelihood change must still select it.
    for changed in (
        "python/sal/likelihood/pruning/torch.py",
        "python/sal/sim/topology.py",
        "python/sal/sim/tree.py",
    ):
        assert "sandbox" in _modules_of(select([changed])), changed


@pytest.mark.critical
@pytest.mark.smoke
def test_the_sandbox_import_guard_runs_on_every_package_it_guards() -> None:
    # `test_sandbox.py` asserts an absence -- that none of the five hot-path
    # packages imports the oracle home -- and an absence is not an import the
    # expansion can follow. It is in ALWAYS for that reason, so it is not
    # enough that today's import graph happens to reach it.
    assert "tests/regression/test_sandbox.py" in ALWAYS
    for package in ("sim", "likelihood", "opt", "search", "learn"):
        chosen = select([f"python/sal/{package}/__init__.py"])
        assert "tests/regression/test_sandbox.py" in chosen["paths"], package


@pytest.mark.critical
@pytest.mark.smoke
@pytest.mark.parametrize(
    "path",
    [
        "uv.lock",
        "pyproject.toml",
        "Cargo.lock",
        "tests/conftest.py",
        "tests/regression/fixtures/tree_jc/stress.yaml",
        ".github/workflows/ci.yml",
        "Makefile.custom",
    ],
)
def test_a_change_it_cannot_attribute_selects_everything(path: str) -> None:
    # The safe direction. A shared fixture or a lockfile can alter any result,
    # and a selection that guessed narrowly here would be silently wrong.
    chosen = select([path])

    assert chosen["paths"] == ["tests"]
    assert chosen["cov"] == ["sal"]


@pytest.mark.critical
@pytest.mark.smoke
def test_a_module_nothing_imports_runs_the_guards_that_read_the_source() -> None:
    # A new module no test imports can only fail a test that reads the
    # package's source, and those run on any change under `python/sal/`.
    chosen = select(["python/sal/some_new_module.py"])

    assert chosen["paths"] != ["tests"]
    assert set(SOURCE_READERS) <= set(chosen["paths"])


@pytest.mark.critical
@pytest.mark.smoke
def test_coverage_targets_match_the_selected_modules() -> None:
    # Step 3's claim is that a touched module is covered by its own tests, so
    # what is measured must be exactly what was selected -- no more, since a
    # module whose tests did not run would drag the figure down, and no less,
    # since an unmeasured module is an unmade claim.
    chosen = select(["python/sal/search/infer.py", "python/sal/sim/tree.py"])

    assert chosen["cov"] == ["sal.search", "sal.sim"]
    assert {"tests/regression/search", "tests/regression/sim"} <= set(chosen["paths"])
    assert select(["infra/select_tests.py"])["cov"] == []


def _benchmarks_of(chosen: dict[str, list[str]]) -> set[str]:
    """The benchmark filenames a selection runs."""
    return {
        path.rsplit("/", 1)[1]
        for path in chosen["paths"]
        if path.startswith("tests/benchmarks/")
    }


@pytest.mark.critical
@pytest.mark.smoke
def test_benchmarks_run_only_for_the_modules_they_measure() -> None:
    # `qa` renders figures from what the others compute and is not timed, so
    # a qa change should time nothing.
    assert _benchmarks_of(select(["python/sal/qa/build.py"])) == set()


@pytest.mark.critical
@pytest.mark.smoke
def test_a_benchmark_is_selected_with_the_module_it_pairs_with() -> None:
    # A change runs the benchmarks of its importers only: `learn` reaches
    # `search` and `likelihood`, never `sim` or `opt`; all of them cost 40 s.
    chosen = _benchmarks_of(select(["python/sal/learn/reinforce.py"]))
    expected = {
        Path(path).name for path in _benchmarks_for(sorted(dependents(["learn"])))
    }
    assert chosen == expected
    assert "test_learn_reinforce_bench.py" in chosen
    assert _benchmarks_of(select(["python/sal/qa/build.py"])) == set()


@pytest.mark.critical
@pytest.mark.smoke
def test_a_widely_imported_module_selects_its_dependents_benchmarks() -> None:
    # `opt` reaches likelihood, search and learn by import, so their
    # benchmarks are selected too -- the cost of being depended on.
    chosen = _benchmarks_of(select(["python/sal/opt/fit.py"]))

    assert "test_opt_fit_bench.py" in chosen
    assert "test_search_infer_bench.py" in chosen
    assert "test_pairwise_distance_bench.py" not in chosen


@pytest.mark.critical
@pytest.mark.smoke
def test_the_always_run_modules_are_always_run() -> None:
    # They cover what belongs to no module.
    for changed in (
        ["python/sal/learn/policy.py"],
        ["python/sal/qa/figure.py"],
    ):
        chosen = select(changed)
        assert "tests/regression/test_numerics.py" in chosen["paths"]
        assert "tests/regression/test_claude_md_pointers.py" in chosen["paths"]


@pytest.mark.critical
@pytest.mark.smoke
def test_every_module_has_a_test_directory() -> None:
    # A module absent from the tree would be selected and then run nothing,
    # which reads as a pass.
    root = Path(__file__).resolve().parents[2]
    missing = [m for m in MODULES if not (root / "tests" / "regression" / m).is_dir()]

    assert missing == []


@pytest.mark.critical
@pytest.mark.smoke
def test_the_benchmarked_modules_are_a_subset_of_the_modules() -> None:
    assert set(BENCHMARKED) < set(MODULES)


@pytest.mark.smoke
def test_every_always_run_path_names_a_file_that_exists() -> None:
    # An `ALWAYS` entry renamed on one side names a path that is gone, and a
    # test meant to run on every change quietly stops.
    root = Path(__file__).resolve().parents[2]
    missing = [path for path in ALWAYS if not (root / path).is_file()]

    assert missing == []


@pytest.mark.smoke
def test_every_whole_suite_trigger_names_something_in_the_tree() -> None:
    # `UNATTRIBUTABLE` decides when the saving is abandoned and the whole suite
    # runs. An entry that matches nothing is a trigger that never fires, so
    # a change to what it was meant to guard would select a partial suite.
    root = Path(__file__).resolve().parents[2]
    tracked = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if ".git" not in path.parts
    ]
    unmatched = [
        trigger
        for trigger in UNATTRIBUTABLE
        if not any(path.startswith(trigger) for path in tracked)
    ]

    assert unmatched == []


@pytest.mark.smoke
def test_the_key_tier_runs_only_for_what_could_move_it() -> None:
    # A key test is two minutes: deselected by default, selected by a change to
    # the coupled model, emissions, fixtures or Rust (issue #399).
    assert "key" in select(["python/sal/learn/policy.py"])["deselect"]
    for trigger in (
        "src/coupled.rs",
        "python/sal/emissions/counts.py",
        "python/sal/sim/count_pairs/__init__.py",
        "tests/regression/fixtures/spatio_sequential_counts/stress.yaml",
    ):
        assert "key" not in select([trigger])["deselect"], trigger


@pytest.mark.smoke
def test_every_key_trigger_names_something_in_the_tree() -> None:
    # `UNATTRIBUTABLE`'s check, for the second trigger list: an entry renamed on
    # one side only stops selecting the key tier and fails nothing.
    root = Path(__file__).resolve().parents[2]
    missing = [trigger for trigger in KEY_TRIGGERS if not (root / trigger).exists()]

    assert missing == []


@pytest.mark.smoke
def test_the_release_and_stress_tiers_are_never_selected_locally() -> None:
    # The two tiers `infra/validate.sh` has always deselected stay deselected
    # whatever the change: they are the release gate's and the developer's,
    # not the per-pull-request run's.
    for changed in ([], ["src/coupled.rs"], ["README.md"]):
        assert set(ALWAYS_DESELECTED) <= set(select(changed)["deselect"])


@pytest.mark.critical
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("changed", "reader"),
    [
        # The two packages the subpackage list had missed since they were
        # created (#777, #972): each now selects its own tests.
        ("python/sal/sample/chain.py", "tests/regression/sample/"),
        ("python/sal/validation/runner.py", "tests/validation/"),
        # A backend twin is loaded by name; its gateway's importers run.
        ("python/sal/search/icm/numba.py", "tests/regression/search/test_icm.py"),
        # An adapter runs its script in a subprocess.
        ("python/sal/validation/scripts/highs.py", "tests/validation/test_highs.py"),
        # A Rust change is a change to the extension's importers.
        ("src/coupled.rs", "tests/test_oxisal_bindings.py"),
        # A shared test helper selects the files that import it.
        ("tests/_rows.py", "tests/regression/search/test_trws.py"),
    ],
)
def test_what_the_subpackage_list_could_not_attribute_is_attributed(
    changed: str, reader: str
) -> None:
    chosen = select([changed])

    assert chosen["paths"] != ["tests"], changed
    assert any(path.startswith(reader) for path in chosen["paths"]), (changed, reader)


@pytest.mark.oracle
def test_python_imports_no_sal_module_outside_the_static_closure() -> None:
    # The referee is the import system itself: for each sampled test file,
    # every `sal` module importing it adds to `sys.modules` must be in the
    # closure `ast` reads. A module the static graph missed would be a test
    # the selection could skip while the change breaks it.
    sample = [
        "tests/regression/search/test_icm.py",
        "tests/regression/search/test_ground_state_compose.py",
        "tests/regression/sample/test_metropolis.py",
        "tests/regression/opt/test_opt_fit.py",
        "tests/regression/likelihood/test_forward_backward.py",
        "tests/regression/learn/test_arena.py",
        "tests/regression/sim/test_fixture_registry.py",
        "tests/validation/test_highs.py",
    ]
    sample = [path for path in sample if (REPO_ROOT / path).is_file()]
    script = (
        "import importlib, json, sys\n"
        "out = {}\n"
        "for name in sys.argv[1:]:\n"
        "    before = set(sys.modules)\n"
        "    importlib.import_module(name)\n"
        "    out[name] = sorted(m for m in set(sys.modules) - before"
        " if m == 'sal' or m.startswith('sal.'))\n"
        "print(json.dumps(out))\n"
    )
    names = [path.removesuffix(".py").replace("/", ".") for path in sample]
    result = subprocess.run(
        [sys.executable, "-c", script, *names],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'infra'}"},
        check=True,
    )
    loaded = json.loads(result.stdout.splitlines()[-1])
    graph = import_graph()
    for name, modules in loaded.items():
        static = _closure_of(name, graph)
        missed = [module for module in modules if module not in static]

        assert missed == [], (name, missed)


def _closure_of(name: str, graph: dict[str, frozenset[str]]) -> set[str]:
    seen, stack = {name}, [name]
    while stack:
        for target in graph.get(stack.pop(), ()):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen
