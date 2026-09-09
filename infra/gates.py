"""What each gate selects, where it runs, and what it is allowed to cost.

**Descriptive, not authoritative.** Nothing imports this module for behaviour.
Every value below is a copy of what another file says today --
`pyproject.toml`'s marker registrations, `.github/workflows/ci.yml`'s steps,
`infra/validate.sh`'s and `infra/review_gates.sh`'s exported caps,
`infra/release.sh`'s command, `DEV.md`'s tier table and its review-gate
sentence -- and `tests/regression/test_gates.py` reads each of those source
texts and asserts it still says this. A module that looks authoritative and is
not is worse than no module, so the rule is stated here rather than implied:
change the file that runs the gate, then change this, and the test names
whichever one was missed.

Issue #465 found the policy spread over twelve files that already disagreed:
`DEV.md` said `infra/review_gates.sh` had nine rows while it had eight. One
place to read the policy from is worth its copy only while a test holds the
copy true; issue #469 adds both, and issue #470 makes this module the one the
others are derived from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    """One gate: the tests it selects, the file that runs it, its budget."""

    #: How the gate is referred to in a ticket or a review.
    name: str
    #: The `-m` expression the gate passes to `pytest`. `None` where it passes
    #: none, which selects every tier.
    marker_expression: str | None
    #: The `--cov-fail-under` percentage, or `None` where the gate measures no
    #: coverage.
    coverage_floor: int | None
    #: The file that runs it, relative to the repository root.
    runs_in: str
    #: The wall clock the gate is budgeted, in seconds. `None` is unbounded.
    budget_seconds: int | None


#: The three gates a change passes, in the order it reaches them. A pull
#: request is judged by `critical` alone (issue #455): the selected tier runs
#: on the push to `main`, where nobody waits for it, at 17 to 32 minutes
#: against the 300 s budget.
GATES: tuple[Gate, ...] = (
    Gate(
        name="pr",
        marker_expression="critical",
        coverage_floor=None,
        runs_in=".github/workflows/ci.yml",
        budget_seconds=300,
    ),
    Gate(
        name="main",
        marker_expression="not release",
        coverage_floor=90,
        runs_in=".github/workflows/ci.yml",
        budget_seconds=300,
    ),
    Gate(
        name="release",
        marker_expression=None,
        coverage_floor=90,
        runs_in="infra/release.sh",
        budget_seconds=None,
    ),
)

#: The cap the `python-tests` job carries, in minutes. The job's only bound:
#: a bound on the selection and a fixed timeout cannot coexist, since the
#: timeout then cancels the pull requests the bound refused (issue #425).
WORKFLOW_TIMEOUT_MINUTES = 90


@dataclass(frozen=True)
class ReviewGate:
    """The table a reviewer runs before reading a diff."""

    #: The script, relative to the repository root.
    script: str
    #: Its rows, in the order it runs them. The count is the value that has
    #: drifted: issue #456 demoted the figure-stamp row and `DEV.md` went on
    #: saying nine.
    rows: tuple[str, ...]
    #: The whole table's budget, in seconds. Above it a check belongs in CI
    #: rather than in something a reviewer runs per branch.
    budget_seconds: int


REVIEW_GATE = ReviewGate(
    script="infra/review_gates.sh",
    rows=(
        "environment imports this checkout",
        "head carries the base",
        "PDFs are the base's (or a rebuild)",
        "changelog fragment exists",
        "critical tier passes",
        "changed tests say what checks them",
        "a new seam names its consumers",
        "generated ledgers are current",
    ),
    budget_seconds=30,
)


@dataclass(frozen=True)
class Cap:
    """A per-test duration cap, asserted on the reference host and never in CI."""

    #: The environment variable `tests/conftest.py` reads it from.
    variable: str
    #: Its default, in seconds, where a caller sets nothing.
    seconds: int
    #: The tests it applies to.
    applies_to: str


#: Exported by `infra/validate.sh`, and the first of the two also by
#: `infra/review_gates.sh`. CI sets neither, per `DEV.md`'s No CI Profiling:
#: a wall-clock assertion on a runner fails for the hardware.
CAPS: tuple[Cap, ...] = (
    Cap(
        variable="SAL_DURATION_CAP",
        seconds=10,
        applies_to="a test carrying none of release, stress, key",
    ),
    Cap(
        variable="SAL_KEY_DURATION_CAP",
        seconds=120,
        applies_to="a test carrying key",
    ),
)

#: What a test is checked against, with the text `pyproject.toml` registers.
#: `infra/test_kinds.py` holds the names alone, because the guard and the merge
#: gate need them without needing this.
KIND_MARKERS: Mapping[str, str] = {
    "oracle": (
        "checked against an independent exact answer -- enumeration, brute "
        "force, a closed form, or a published value"
    ),
    "simulated_truth": "checked against the parameters that generated the data",
    "mathematical": (
        "checked against a property the model must satisfy regardless of implementation"
    ),
    "edge_case": "checked at a boundary, or that an unusable input is refused",
    "structural": (
        "checked against an invariant this repository chose -- the import "
        "graph, a protocol, a documented contract, or a guard's own trigger"
    ),
}

#: When a test runs. The second axis, and not a kind: a test is critical *and*
#: an oracle test, never instead of one.
SCHEDULING_MARKERS: Mapping[str, str] = {
    "release": (
        "long-running scientific validity test, run on release rather than "
        "per PR (DEV.md CI & Performance Budget)"
    ),
    "critical": "gates early; fast, and its failure invalidates what runs after it",
    "stress": (
        "same claim at a size the 5-minute CI budget cannot hold; run with "
        "`-m stress` inside the 10-minute developer budget (DEV.md CI & "
        "Performance Budget)"
    ),
    "key": (
        "the key fixture's full test -- the largest declared instance whose "
        "simulate-fit-assert run fits SAL_KEY_DURATION_CAP, 120 s; run in "
        "CI's full tier and selected locally only when its inputs change "
        "(DEV.md CI & Performance Budget)"
    ),
}

#: Benchmarks measure rather than assert, so they carry no kind. The exclusion
#: is a property of the directory, not of any test in it.
KIND_EXEMPT_DIRECTORY = "benchmarks"
