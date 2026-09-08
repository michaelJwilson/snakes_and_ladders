"""The test tiers, asserted structurally rather than on a clock.

`DEV.md` budgets the CI tier at 5 minutes and the stress tier at 10, and
forbids ranking performance on CI hardware. A wall-clock assertion would
therefore fail for the machine rather than for the change, so the budgets are
kept by *size* and this module checks the things that are true regardless of
how fast the machine is:

* the stress and key tiers are non-empty and reachable, because a tier
  nothing selects is a tier that rots --- the failure mode `release` avoids
  only because ``infra/release.sh`` runs it;
* every scheduling marker is registered, so a typo deselects nothing silently;
* ``at_scale`` produces exactly one CI case and one stress case, since a
  parameterization that marked both or neither would move a test between
  tiers without anyone editing it.

`infra/measure_test_budget.sh` reports the wall clock against the budgets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from snakes_and_ladders.sim.fixtures import Fixture

from tests._durations import key_over_cap, over_cap
from tests._scale import at_fixture, at_scale

REPO_ROOT = Path(__file__).resolve().parents[2]


def _collected(selector: str) -> int:
    """How many tests a marker expression selects, by asking pytest."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            selector,
            "tests/regression",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    # pytest writes "N tests collected" when nothing is deselected and
    # "N/M tests collected (K deselected)" when something is, so the count is
    # the part before any slash.
    for line in reversed(result.stdout.splitlines()):
        if "collected" in line:
            return int(line.split()[0].split("/")[0])
    msg = f"could not read a collected count from pytest: {result.stdout[-400:]}"
    raise AssertionError(msg)


@pytest.mark.edge_case
def test_the_stress_tier_is_reachable_and_not_empty() -> None:
    # A tier nothing selects is a tier that rots. This is the check that would
    # have caught `stress` being registered but never applied, or applied but
    # spelled differently in one place.
    assert _collected("stress") > 0


@pytest.mark.edge_case
def test_the_key_tier_is_reachable_and_not_empty() -> None:
    # The tier added by issue #399: exempt from `SAL_DURATION_CAP` and held to
    # `SAL_KEY_DURATION_CAP` instead. An exemption nothing selects is an
    # exemption nobody checks.
    assert _collected("key") > 0


@pytest.mark.mathematical
def test_the_ci_tier_excludes_the_stress_and_key_tiers() -> None:
    # The selections must partition, or the CI tier silently carries the sizes
    # the budget exists to keep out of it. Three tiers now: `key` is not
    # `stress`, so a selection written before it existed would have run a
    # two-minute test on every pull request.
    ci = _collected("not release and not stress and not key")
    stress = _collected("stress and not key")
    key = _collected("key")
    all_three = _collected("not release")

    assert ci + stress + key == all_three


@pytest.mark.mathematical
def test_at_scale_produces_one_case_per_tier() -> None:
    # `at_scale` is what keeps one assertion running at two sizes. If it ever
    # marked both cases or neither, tests would move tiers with no diff to
    # review.
    decorator = at_scale("size", ci=2, stress=200)
    parameters = decorator.args[1]

    assert len(parameters) == 2
    assert [parameter.values[0] for parameter in parameters] == [2, 200]
    assert parameters[0].marks == ()
    assert [mark.name for mark in parameters[1].marks] == ["stress"]


@pytest.mark.mathematical
def test_at_fixture_runs_one_case_per_declared_tier() -> None:
    # The registry-driven parameterization: the cases are the fixture files
    # a problem declares, and each carries its own tier's marker, so a
    # fixture added at a tier reaches every test written this way without
    # one of them being edited.
    parameters = at_fixture("instance", "tree_search").args[1]

    assert [parameter.id for parameter in parameters] == [
        "tree_search-ci",
        "tree_search-stress",
        "tree_search-release",
    ]
    assert [[mark.name for mark in parameter.marks] for parameter in parameters] == [
        [],
        ["stress"],
        ["release"],
    ]


@pytest.mark.simulated_truth
@at_fixture("instance", "tree_search")
def test_at_fixture_hands_the_body_a_loaded_instance(instance: Fixture) -> None:
    # Exercised end to end: the CI case runs on every pull request, and the
    # other two are deselected there by their markers.
    assert instance.params.k == 4
    assert instance.oracle == "enumeration"


@pytest.mark.edge_case
def test_an_unregistered_marker_is_an_error_not_a_silent_deselection() -> None:
    # `--strict-markers` is in `addopts`, so a typo fails at collection. Pinned
    # because the alternative is a test that selects nothing and reports as
    # passing, which is how a whole tier disappears without a red run.
    config = (REPO_ROOT / "pyproject.toml").read_text()

    assert "--strict-markers" in config
    assert "stress:" in config
    assert "key:" in config


@pytest.mark.mathematical
@at_scale("size", ci=1, stress=2)
def test_at_scale_runs_its_body_at_both_sizes(size: int) -> None:
    # The decorator exercised end to end: this test is collected twice, and
    # only the second is deselected by `-m "not stress"`.
    assert size in (1, 2)


@pytest.mark.edge_case
def test_a_key_test_over_its_own_cap_is_named() -> None:
    # The key tier's exemption is not an exemption from measurement: a key
    # fixture is by definition the largest declared instance that fits 120 s,
    # so one that does not is a fixture whose key instance is the wrong one.
    over = key_over_cap([("t::key", 130.0, frozenset({"key"}))], 120.0)
    under = key_over_cap([("t::key", 110.0, frozenset({"key"}))], 120.0)

    assert len(over) == 1
    assert "the key instance is the largest that fits it" in over[0]
    assert under == []
    assert over_cap([("t::key", 130.0, frozenset({"key"}))], 10.0) == []


@pytest.mark.simulated_truth
def test_the_budget_script_states_both_budgets() -> None:
    # The numbers `DEV.md` documents and the numbers the script measures
    # against have to be the same two, or the report is against a budget
    # nothing else knows about.
    script = (REPO_ROOT / "infra" / "measure_test_budget.sh").read_text()
    dev = (REPO_ROOT / "DEV.md").read_text()

    assert "300" in script
    assert "600" in script
    assert "**5 minutes**" in dev
    assert "**10 minutes**" in dev


@pytest.mark.structural
def test_pytest_is_importable_here() -> None:
    # `pytest` is imported for the marker types above; this keeps the import
    # used rather than removed by a linter, and costs nothing.
    assert pytest.__name__ == "pytest"
