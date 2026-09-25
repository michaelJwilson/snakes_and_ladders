"""The floor the critical tier's coverage may not fall below.

Issue #635. The tier gates early, so what it misses a merge finds late.
Measured when #635 widened the marker: 177 tests, 13,324 statements, 8,421
missed, 36.80%; after the build-out, 298 tests, 14,051 statements, 7,269
missed, 48.27%, for 5.9 s of tier clock (42.4 s to 48.3 s). The whole-tier
90% asks whether the package is tested; this, whether the fast gate tests it.
It reads the report CI produces and skips where there is none: `--cov` costs
about 4x (15.9 s bare, 61.56 s instrumented).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: The measured floor: raised with a PR, never lowered. Rounded down from
#: 48.267027257846415: rounded up it fails the run it came from, as a 36.7 once did.
CRITICAL_COVERAGE_FLOOR = 48.2

#: Where CI leaves the report this reads. Absent locally, which is a skip
#: rather than a failure: a developer running `-m critical` bare has no report
#: and wants none.
COVERAGE_JSON = Path(__file__).resolve().parents[2] / "coverage-critical.json"


@pytest.mark.critical
@pytest.mark.infra
def test_the_floor_is_stated_once_and_is_a_percentage() -> None:
    """The constant is the contract, so it is checked before it is used.

    A floor above 100 or below zero would pass every run silently.
    """
    assert 0.0 < CRITICAL_COVERAGE_FLOOR <= 100.0


@pytest.mark.infra
def test_the_critical_tier_covers_at_least_the_floor() -> None:
    """The tier's coverage, against the floor #635 measured.

    Skipped where CI has left no report, which is every local run.
    """
    if not COVERAGE_JSON.exists():
        pytest.skip(
            f"no {COVERAGE_JSON.name}; run "
            "`pytest -m critical --cov=sal "
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

    It keeps the #635 measurement, which does not move; the floor lives here.
    """
    dev = (Path(__file__).resolve().parents[2] / "DEV.md").read_text()

    assert "tests/regression/test_critical_coverage.py" in dev
