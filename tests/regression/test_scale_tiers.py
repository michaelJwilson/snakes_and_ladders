"""The test tiers, asserted structurally rather than on a clock.

`DEV.md` budgets the CI tier at 5 minutes and the stress tier at 10, and
forbids ranking performance on CI hardware. A wall-clock assertion would
therefore fail for the machine rather than for the change, so the budgets are
kept by *size* and this module checks the things that are true regardless of
how fast the machine is:

* the stress tier is non-empty and reachable, because a tier nothing selects
  is a tier that rots --- the failure mode `release` avoids only because
  ``infra/release.sh`` runs it;
* every ``stress`` marker is registered, so a typo deselects nothing silently;
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

from tests._scale import at_scale

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


def test_the_stress_tier_is_reachable_and_not_empty() -> None:
    # A tier nothing selects is a tier that rots. This is the check that would
    # have caught `stress` being registered but never applied, or applied but
    # spelled differently in one place.
    assert _collected("stress") > 0


def test_the_ci_tier_excludes_the_stress_tier() -> None:
    # The two selections must partition, or the CI tier silently carries the
    # sizes the budget exists to keep out of it.
    ci = _collected("not release and not stress")
    stress = _collected("stress")
    both = _collected("not release")

    assert ci + stress == both


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


def test_an_unregistered_marker_is_an_error_not_a_silent_deselection() -> None:
    # `--strict-markers` is in `addopts`, so a typo fails at collection. Pinned
    # because the alternative is a test that selects nothing and reports as
    # passing, which is how a whole tier disappears without a red run.
    config = (REPO_ROOT / "pyproject.toml").read_text()

    assert "--strict-markers" in config
    assert "stress:" in config


@at_scale("size", ci=1, stress=2)
def test_at_scale_runs_its_body_at_both_sizes(size: int) -> None:
    # The decorator exercised end to end: this test is collected twice, and
    # only the second is deselected by `-m "not stress"`.
    assert size in (1, 2)


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


def test_pytest_is_importable_here() -> None:
    # `pytest` is imported for the marker types above; this keeps the import
    # used rather than removed by a linter, and costs nothing.
    assert pytest.__name__ == "pytest"
