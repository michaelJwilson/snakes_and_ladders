"""What the test selection must guarantee before it is allowed to skip anything.

Running fewer tests is safe only if the selection is right, so what is pinned
is not the saving but the three ways it could be wrong: missing a module that
imports the changed one, mistaking a change it does not understand for one that
needs nothing, and skipping a run that would have measured different coverage
(issue #161).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))

from select_tests import (
    ALWAYS,
    ALWAYS_DESELECTED,
    BENCHMARKED,
    EVERYTHING,
    GUARDS,
    KEY_TRIGGERS,
    MODULES,
    _benchmarks_for,
    dependents,
    guards_for,
    select,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _modules_of(chosen: dict[str, list[str]]) -> set[str]:
    """Extract the submodules a selection runs the tests of.

    Returns
    -------
    set[str]
        Submodule names.
    """
    return {
        path.rsplit("/", 1)[1]
        for path in chosen["paths"]
        if path.startswith("tests/regression/") and not path.endswith(".py")
    }


@pytest.mark.critical
@pytest.mark.structural
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
@pytest.mark.structural
def test_a_changelog_fragment_alone_selects_nothing() -> None:
    # `towncrier check` in the lint job is its guard; the suite has none.
    chosen = select(["changelog.d/372.changed.md"])

    assert chosen["paths"] == []
    assert chosen["cov"] == []


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize(
    ("path", "guard"),
    [
        ("docs/tex/textbook.tex", "tests/regression/test_document_labels.py"),
        ("docs/experiments/006-x.md", "tests/regression/test_experiments.py"),
        ("ROADMAP.md", "tests/regression/test_planning_documents_agree.py"),
        ("STATUS.md", "tests/regression/test_planning_documents_agree.py"),
        ("TICKETS.md", "tests/regression/test_planning_documents_agree.py"),
        ("CLAUDE.md", "tests/regression/test_claude_md_pointers.py"),
        (
            "python/snakes_and_ladders/opt/CLAUDE.md",
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
    assert _modules_of(chosen) == set()


@pytest.mark.critical
@pytest.mark.structural
def test_every_guard_the_selection_names_exists() -> None:
    # A guard renamed on one side only would select a path pytest cannot
    # collect, which fails the run for the wrong reason.
    for _, tests in GUARDS:
        for test in tests:
            assert (REPO_ROOT / test).is_file(), test


@pytest.mark.critical
@pytest.mark.structural
def test_a_code_change_beside_prose_runs_both() -> None:
    # The guards join the module's tests rather than replacing them.
    chosen = select(
        ["python/snakes_and_ladders/learn/reinforce.py", "docs/tex/textbook.tex"]
    )

    assert "learn" in _modules_of(chosen)
    assert set(guards_for(["docs/tex/textbook.tex"])) <= set(chosen["paths"])


@pytest.mark.critical
@pytest.mark.structural
def test_a_change_selects_the_modules_that_import_it() -> None:
    # `snakes_and_ladders.search` imports `snakes_and_ladders.likelihood`, so a likelihood change that
    # ran only likelihood's tests would let a break in search through.
    assert _modules_of(select(["python/snakes_and_ladders/likelihood/pruning.py"])) >= {
        "likelihood",
        "search",
        "qa",
    }


@pytest.mark.critical
@pytest.mark.structural
def test_a_leaf_module_selects_only_itself() -> None:
    # A module nothing imports needs nothing else run. The leaf is derived
    # rather than named: `snakes_and_ladders.learn` was one until
    # `snakes_and_ladders.qa.rl_tree_policy` imported it (issue #178), and a
    # test naming a module goes stale the moment an import is added.
    leaves = [module for module in MODULES if dependents({module}) == {module}]
    assert leaves, "no submodule is a leaf; the selection can save nothing"
    for leaf in leaves:
        assert _modules_of(
            select([f"python/snakes_and_ladders/{leaf}/__init__.py"])
        ) == {leaf}


@pytest.mark.critical
@pytest.mark.structural
def test_the_dependency_expansion_is_transitive() -> None:
    # opt <- likelihood <- search: a change to opt must reach search even
    # though search does not import opt directly through that path alone.
    assert dependents({"opt"}) >= {"opt", "likelihood", "search", "qa"}


@pytest.mark.critical
@pytest.mark.structural
def test_changing_a_test_selects_its_module() -> None:
    # A test file is as capable of lowering coverage as a source file.
    assert _modules_of(select(["tests/regression/opt/test_opt_fit.py"])) >= {"opt"}


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize(
    "path",
    [
        "uv.lock",
        "pyproject.toml",
        "Cargo.lock",
        "src/lib.rs",
        "tests/_fixtures.py",
        "tests/regression/fixtures/tree_jc/stress.yaml",
        ".github/workflows/ci.yml",
        "python/snakes_and_ladders/numerics.py",
    ],
)
def test_a_change_it_cannot_attribute_selects_everything(path: str) -> None:
    # The safe direction. A shared fixture or a lockfile can alter any result,
    # and a selection that guessed narrowly here would be silently wrong.
    chosen = select([path])

    assert chosen["paths"] == ["tests"]
    assert chosen["cov"] == ["snakes_and_ladders"]


@pytest.mark.critical
@pytest.mark.structural
def test_an_unrecognised_code_path_selects_everything() -> None:
    # Not documentation, not attributable to a module: the unsafe answer is
    # the one that looks like a saving.
    chosen = select(["python/snakes_and_ladders/some_new_module.py"])

    assert chosen["paths"] == ["tests"]


@pytest.mark.critical
@pytest.mark.structural
def test_coverage_targets_match_the_selected_modules() -> None:
    # Step 3's claim is that a touched module is covered by its own tests, so
    # what is measured must be exactly what was selected -- no more, since a
    # module whose tests did not run would drag the figure down, and no less,
    # since an unmeasured module is an unmade claim.
    chosen = select(["python/snakes_and_ladders/search/infer.py"])

    assert set(chosen["cov"]) == {
        f"snakes_and_ladders.{m}" for m in _modules_of(chosen)
    }


def _benchmarks_of(chosen: dict[str, list[str]]) -> set[str]:
    """Extract the benchmark modules a selection runs.

    Returns
    -------
    set[str]
        Benchmark filenames.
    """
    return {
        path.rsplit("/", 1)[1]
        for path in chosen["paths"]
        if path.startswith("tests/benchmarks/")
    }


@pytest.mark.critical
@pytest.mark.structural
def test_benchmarks_run_only_for_the_modules_they_measure() -> None:
    # `qa` renders figures from what the others compute and is not timed, so
    # a qa change should time nothing.
    assert _benchmarks_of(select(["python/snakes_and_ladders/qa/build.py"])) == set()


@pytest.mark.critical
@pytest.mark.structural
def test_a_benchmark_is_selected_with_the_module_it_pairs_with() -> None:
    # The pairing DEV.md requires, used as the selector: a change runs the
    # benchmarks of the modules that import it and no other. A learn change
    # reaches `search` (its tree environment) and `likelihood` (the
    # surrogates) but never `sim` or `opt`; running every benchmark cost 40 s
    # against the few that measure what changed.
    chosen = _benchmarks_of(select(["python/snakes_and_ladders/learn/reinforce.py"]))
    expected = {
        Path(path).name for path in _benchmarks_for(sorted(dependents(["learn"])))
    }
    assert chosen == expected
    assert "test_learn_reinforce_bench.py" in chosen
    assert _benchmarks_of(select(["python/snakes_and_ladders/qa/build.py"])) == set()


@pytest.mark.critical
@pytest.mark.structural
def test_a_widely_imported_module_selects_its_dependents_benchmarks() -> None:
    # `opt` reaches likelihood, search and learn by import, so their
    # benchmarks are selected too -- the cost of being depended on.
    chosen = _benchmarks_of(select(["python/snakes_and_ladders/opt/fit.py"]))

    assert "test_opt_fit_bench.py" in chosen
    assert "test_search_infer_bench.py" in chosen
    assert "test_pairwise_distance_bench.py" not in chosen


@pytest.mark.critical
@pytest.mark.structural
def test_the_always_run_modules_are_always_run() -> None:
    # They cover what belongs to no module, and cost 0.5 s between them.
    for changed in (
        ["python/snakes_and_ladders/learn/policy.py"],
        ["python/snakes_and_ladders/qa/figure.py"],
    ):
        chosen = select(changed)
        assert "tests/regression/test_numerics.py" in chosen["paths"]
        assert "tests/regression/test_claude_md_pointers.py" in chosen["paths"]


@pytest.mark.critical
@pytest.mark.structural
def test_every_module_has_a_test_directory() -> None:
    # A module absent from the tree would be selected and then run nothing,
    # which reads as a pass.
    root = Path(__file__).resolve().parents[2]
    missing = [m for m in MODULES if not (root / "tests" / "regression" / m).is_dir()]

    assert missing == []


@pytest.mark.critical
@pytest.mark.structural
def test_the_benchmarked_modules_are_a_subset_of_the_modules() -> None:
    assert set(BENCHMARKED) < set(MODULES)


@pytest.mark.structural
def test_every_always_run_path_names_a_file_that_exists() -> None:
    # The failure this catches has no other symptom worth trusting: an entry
    # renamed on one side only leaves `ALWAYS` naming a path that is gone, and
    # a selection built from it either errors far from the cause or, worse,
    # quietly stops running a test that is supposed to run on every change.
    # A rename touching `tests/` is exactly when it happens.
    root = Path(__file__).resolve().parents[2]
    missing = [path for path in ALWAYS if not (root / path).is_file()]

    assert missing == []


@pytest.mark.structural
def test_every_whole_suite_trigger_names_something_in_the_tree() -> None:
    # `EVERYTHING` decides when the saving is abandoned and the whole suite
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
        for trigger in EVERYTHING
        if not any(path.startswith(trigger) for path in tracked)
    ]

    assert unmatched == []


@pytest.mark.structural
def test_the_key_tier_runs_only_for_what_could_move_it() -> None:
    # A key test is two minutes, so it is deselected by default and selected
    # by the change that could fail it: the coupled model, the emissions, the
    # fixtures or the Rust crate (issue #399). Both directions, because a
    # trigger that never fires and a tier that always runs are the two ways
    # this stops paying for itself.
    assert "key" in select(["python/snakes_and_ladders/learn/policy.py"])["deselect"]
    for trigger in (
        "src/coupled.rs",
        "python/snakes_and_ladders/emissions.py",
        "python/snakes_and_ladders/sim/count_pairs.py",
        "tests/regression/fixtures/spatio_sequential_counts/stress.yaml",
    ):
        assert "key" not in select([trigger])["deselect"], trigger


@pytest.mark.structural
def test_every_key_trigger_names_something_in_the_tree() -> None:
    # `EVERYTHING`'s check, for the second trigger list: an entry renamed on
    # one side only stops selecting the key tier and fails nothing.
    root = Path(__file__).resolve().parents[2]
    missing = [trigger for trigger in KEY_TRIGGERS if not (root / trigger).exists()]

    assert missing == []


@pytest.mark.structural
def test_the_release_and_stress_tiers_are_never_selected_locally() -> None:
    # The two tiers `infra/validate.sh` has always deselected stay deselected
    # whatever the change: they are the release gate's and the developer's,
    # not the per-pull-request run's.
    for changed in ([], ["src/coupled.rs"], ["README.md"]):
        assert set(ALWAYS_DESELECTED) <= set(select(changed)["deselect"])
