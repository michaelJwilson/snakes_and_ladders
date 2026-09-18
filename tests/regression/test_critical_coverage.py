"""The floor the critical tier's coverage may not fall below.

Issue #635. The tier gates early, so what it *misses* is what a merge finds
out about late. Measured when #635 widened the marker: **177 tests, 13,324
statements, 8,421 missed, 36.80%** --- 149 of those tests infrastructure and
eight across the five application packages. The floor is that measurement, and
it rises as tests land.

It has since risen once, on the build-out in the same pull request: **298
tests, 14,051 statements, 7,269 missed, 48.27%**. The tier's own clock went
from 42.4 s to 48.3 s over the same 34 tests, so the eleven and a half points
cost 5.9 s.

**Why a floor here as well as the 90% on the whole tier.** The two answer
different questions. The whole tier's gate asks whether the package is tested;
this asks whether the *fast* gate tests it, and a tier that passed the first
while failing the second is the state this was written in --- every claim
refereed somewhere, none of it refereed in sixteen seconds.

**This runs where the whole-tier coverage runs, not on the early gate.**
Instrumentation costs about 4x: the tier is 15.9 s bare (`DEV.md`) and 61.56 s
under `--cov` on the same host. Paying that on every local `-m critical` would
defeat the tier's reason for existing, so the guard reads a report CI has
already produced and skips where there is none.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: The measured floor, and the only number this file states. Raise it to what
#: a pull request lands and never lower it --- the rule root `CLAUDE.md` gives
#: for `--cov-fail-under`, one tier down.
#:
#: **Rounded down, not to nearest.** The measurement is 48.267027257846415 and
#: the terminal report prints 48%; a floor set from what was printed is below
#: what was measured and gives back a point, and one rounded up is above it and
#: fails the run it was derived from --- which the 36.7 that stood here before
#: the build-out did, and this guard caught.
CRITICAL_COVERAGE_FLOOR = 48.2

#: Where CI leaves the report this reads. Absent locally, which is a skip
#: rather than a failure: a developer running `-m critical` bare has no report
#: and wants none.
COVERAGE_JSON = Path(__file__).resolve().parents[2] / "coverage-critical.json"


@pytest.mark.critical
@pytest.mark.infra
def test_the_floor_is_stated_once_and_is_a_percentage() -> None:
    """The constant is the contract, so it is checked before it is used.

    A floor above 100 or below zero would pass every run silently, which is
    the failure mode a guard reading its own constant has.
    """
    assert 0.0 < CRITICAL_COVERAGE_FLOOR <= 100.0


@pytest.mark.infra
def test_the_critical_tier_covers_at_least_the_floor() -> None:
    """The tier's coverage, against the floor #635 measured.

    Skipped where CI has left no report, which is every local run: the
    measurement costs 4x the tier and belongs where the whole-tier coverage
    is already paid for.
    """
    if not COVERAGE_JSON.exists():
        pytest.skip(
            f"no {COVERAGE_JSON.name}; run "
            "`pytest -m critical --cov=snakes_and_ladders "
            "--cov-report=json:coverage-critical.json` to produce one"
        )
    report = json.loads(COVERAGE_JSON.read_text())
    covered = float(report["totals"]["percent_covered"])

    assert covered >= CRITICAL_COVERAGE_FLOOR, (
        f"the critical tier covers {covered:.2f}%, below the "
        f"{CRITICAL_COVERAGE_FLOOR}% floor issue #635 set. The floor is raised "
        f"by landing fast tests that referee the science, and is never lowered "
        f"to pass a pull request (root CLAUDE.md, Definition of Done)."
    )


@pytest.mark.infra
def test_dev_md_names_the_file_that_holds_the_floor() -> None:
    """`DEV.md` describes the guard and points here for the number.

    It states the measurement this was set against, which is a fact of when
    #635 landed and does not move; the floor, which does move, lives here. A
    reader of either finds the other.
    """
    dev = (Path(__file__).resolve().parents[2] / "DEV.md").read_text()

    assert "tests/regression/test_critical_coverage.py" in dev
