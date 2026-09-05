"""What the test selection must guarantee before it is allowed to skip anything.

Running fewer tests is only safe if the selection is right, so what is pinned
here is not the saving but the three ways it could be wrong: missing a module
that imports the changed one, mistaking a change it does not understand for a
change that needs nothing, and skipping a run that would have measured
different coverage (issue #161).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))

from select_tests import ALWAYS, BENCHMARKED, EVERYTHING, MODULES, dependents, select


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


def test_a_documentation_only_change_selects_nothing() -> None:
    # The case this exists for: the suite would run the same code over the
    # same tests as the last run on main, so coverage cannot have moved.
    chosen = select(
        ["docs/tex/paper.tex", "README.md", "changelog.d/161.changed.md", "DEV.md"]
    )

    assert chosen["paths"] == []
    assert chosen["cov"] == []


def test_a_change_selects_the_modules_that_import_it() -> None:
    # `snakes_and_ladders.search` imports `snakes_and_ladders.likelihood`, so a likelihood change that
    # ran only likelihood's tests would let a break in search through.
    assert _modules_of(select(["python/snakes_and_ladders/likelihood/pruning.py"])) >= {
        "likelihood",
        "search",
        "qa",
    }


def test_a_leaf_module_selects_only_itself() -> None:
    # Nothing imports `snakes_and_ladders.learn`, so nothing else need run. This is the
    # case the whole mechanism is worth building for.
    assert _modules_of(select(["python/snakes_and_ladders/learn/reinforce.py"])) == {
        "learn"
    }


def test_the_dependency_expansion_is_transitive() -> None:
    # opt <- likelihood <- search: a change to opt must reach search even
    # though search does not import opt directly through that path alone.
    assert dependents({"opt"}) >= {"opt", "likelihood", "search", "qa"}


def test_changing_a_test_selects_its_module() -> None:
    # A test file is as capable of lowering coverage as a source file.
    assert _modules_of(select(["tests/regression/opt/test_opt_fit.py"])) >= {"opt"}


@pytest.mark.parametrize(
    "path",
    [
        "uv.lock",
        "pyproject.toml",
        "Cargo.lock",
        "src/lib.rs",
        "tests/_fixtures.py",
        "tests/regression/fixtures/simulation_params.yaml",
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


def test_an_unrecognised_code_path_selects_everything() -> None:
    # Not documentation, not attributable to a module: the unsafe answer is
    # the one that looks like a saving.
    chosen = select(["python/snakes_and_ladders/some_new_module.py"])

    assert chosen["paths"] == ["tests"]


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


def test_benchmarks_run_only_for_the_modules_they_measure() -> None:
    # `qa` renders figures from what the others compute and is not timed, so
    # a qa change should time nothing.
    assert _benchmarks_of(select(["python/snakes_and_ladders/qa/build.py"])) == set()


def test_a_benchmark_is_selected_with_the_module_it_pairs_with() -> None:
    # The pairing DEV.md requires, used as the selector: a learn change runs
    # learn's benchmark and not the other twelve. Running all of them cost
    # 40 s against 5 s for the one that measures what changed.
    assert _benchmarks_of(select(["python/snakes_and_ladders/learn/reinforce.py"])) == {
        "test_learn_reinforce_bench.py"
    }


def test_a_widely_imported_module_selects_its_dependents_benchmarks() -> None:
    # `opt` reaches likelihood, search and learn by import, so their
    # benchmarks are selected too -- the cost of being depended on.
    chosen = _benchmarks_of(select(["python/snakes_and_ladders/opt/fit.py"]))

    assert "test_opt_fit_bench.py" in chosen
    assert "test_search_infer_bench.py" in chosen
    assert "test_pairwise_distance_bench.py" not in chosen


def test_the_always_run_modules_are_always_run() -> None:
    # They cover what belongs to no module, and cost 0.5 s between them.
    for changed in (
        ["python/snakes_and_ladders/learn/policy.py"],
        ["python/snakes_and_ladders/qa/figure.py"],
    ):
        chosen = select(changed)
        assert "tests/regression/test_numerics.py" in chosen["paths"]
        assert "tests/regression/test_claude_md_pointers.py" in chosen["paths"]


def test_every_module_has_a_test_directory() -> None:
    # A module absent from the tree would be selected and then run nothing,
    # which reads as a pass.
    root = Path(__file__).resolve().parents[2]
    missing = [m for m in MODULES if not (root / "tests" / "regression" / m).is_dir()]

    assert missing == []


def test_the_benchmarked_modules_are_a_subset_of_the_modules() -> None:
    assert set(BENCHMARKED) < set(MODULES)


def test_every_always_run_path_names_a_file_that_exists() -> None:
    # The failure this catches has no other symptom worth trusting: an entry
    # renamed on one side only leaves `ALWAYS` naming a path that is gone, and
    # a selection built from it either errors far from the cause or, worse,
    # quietly stops running a test that is supposed to run on every change.
    # A rename touching `tests/` is exactly when it happens.
    root = Path(__file__).resolve().parents[2]
    missing = [path for path in ALWAYS if not (root / path).is_file()]

    assert missing == []


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
