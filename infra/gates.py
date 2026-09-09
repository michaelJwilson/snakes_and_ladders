"""What each gate selects, where it runs, and what it is allowed to cost.

**Authoritative for two blocks, descriptive for the rest.** `pyproject.toml`'s
marker registrations and `DEV.md`'s tier table are *written* from the tables
below, each between marker comments naming this module, by `infra/ledgers.sh`
(issue #470). There is no copy of those two left to drift: editing either block
by hand is undone on the next regeneration, and the review gate fails a branch
where regenerating rewrites a tracked file.

Every other value here is still a copy of what another file says --
`.github/workflows/ci.yml`'s steps, `infra/validate.sh`'s and
`infra/review_gates.sh`'s exported caps, `infra/release.sh`'s command,
`DEV.md`'s review-gate sentence. Each of those is read by something that cannot
import Python at the point it needs the value: a workflow condition evaluated
by GitHub, a shell export read by `tests/conftest.py`'s environment, a sentence
a reviewer reads. They stay written where they are read and
`tests/regression/test_gates.py` asserts they still say this; the file that
runs the gate wins, and the module is corrected to match it.

Issue #465 found the policy spread over twelve files that already disagreed:
`DEV.md` said `infra/review_gates.sh` had nine rows while it had eight. Issue
#469 wrote the policy down once and held the copies to it; this step removes
the two copies that are pure data by generating them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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


#: A pull request is judged by `critical` alone (issue #455).
PR_GATE = Gate(
    name="pr",
    marker_expression="critical",
    coverage_floor=None,
    runs_in=".github/workflows/ci.yml",
    budget_seconds=300,
)

#: The selected tier runs on the push to `main`, where nobody waits for it, at
#: 17 to 32 minutes against the same 300 s budget.
MAIN_GATE = Gate(
    name="main",
    marker_expression="not release",
    coverage_floor=90,
    runs_in=".github/workflows/ci.yml",
    budget_seconds=300,
)

#: Every tier, unbounded, at the release gate.
RELEASE_GATE = Gate(
    name="release",
    marker_expression=None,
    coverage_floor=90,
    runs_in="infra/release.sh",
    budget_seconds=None,
)

#: The three gates a change passes, in the order it reaches them.
GATES: tuple[Gate, ...] = (PR_GATE, MAIN_GATE, RELEASE_GATE)

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


#: The cap every test in the pull-request tier is held to.
DURATION_CAP = Cap(
    variable="SAL_DURATION_CAP",
    seconds=10,
    applies_to="a test carrying none of release, stress, key",
)

#: The ceiling that comes with the exemption from the first, or `key` becomes
#: the marker any slow test acquires.
KEY_DURATION_CAP = Cap(
    variable="SAL_KEY_DURATION_CAP",
    seconds=120,
    applies_to="a test carrying key",
)

#: Exported by `infra/validate.sh`, and the first of the two also by
#: `infra/review_gates.sh`. CI sets neither, per `DEV.md`'s No CI Profiling:
#: a wall-clock assertion on a runner fails for the hardware.
CAPS: tuple[Cap, ...] = (DURATION_CAP, KEY_DURATION_CAP)

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
    # The one marker text that states a number stated elsewhere, so it reads
    # the cap rather than repeating it: the registration and `DEV.md`'s tier
    # table then move together when the cap does.
    "key": (
        "the key fixture's full test -- the largest declared instance whose "
        f"simulate-fit-assert run fits {KEY_DURATION_CAP.variable}, "
        f"{KEY_DURATION_CAP.seconds} s; run in CI's full tier and selected "
        "locally only when its inputs change (DEV.md CI & Performance Budget)"
    ),
}

#: Benchmarks measure rather than assert, so they carry no kind. The exclusion
#: is a property of the directory, not of any test in it.
KIND_EXEMPT_DIRECTORY = "benchmarks"

#: The order `pyproject.toml` registers the markers in. It is the file's order
#: and carries no meaning -- `release` predates the kind axis and stayed first
#: -- and it is written here so the generated block reproduces what is
#: committed instead of reordering it on its first run, which would make that
#: run's diff unreadable.
MARKER_REGISTRATION_ORDER: tuple[str, ...] = (
    "release",
    *KIND_MARKERS,
    "critical",
    "stress",
    "key",
)


@dataclass(frozen=True)
class Tier:
    """One row of `DEV.md`'s tier table: a size, and what it is budgeted."""

    #: The `tier` cell: what the tier is called in a ticket and in a review.
    name: str
    #: The `marker` cell: the marker a test in it carries, or that none does.
    marker: str
    #: The `fixture` cell: where the tier's instance is declared.
    fixture: str
    #: The `contents` cell: what a size at this tier is chosen for.
    contents: str
    #: The wall clock the tier is budgeted, in seconds; `None` is unbounded.
    #: Read from the gate or the cap that carries it rather than restated here,
    #: so the table moves when the budget does and cannot disagree with it.
    budget_seconds: int | None
    #: The unit the cell states it in, `minutes` or `s`. A per-test budget is
    #: stated in seconds because the cap enforcing it is exported in seconds.
    budget_unit: str = "minutes"
    #: What the cell says inside the emphasis after the number: whom the budget
    #: applies to, where it applies to one test rather than to a run.
    budget_scope: str = ""
    #: What it says after the emphasis: the qualification, or the variable
    #: that carries the budget.
    budget_note: str = ""

    @property
    def budget(self) -> str:
        """The `budget` cell, rendered from the number the tier is held to."""
        if self.budget_seconds is None:
            return "unbounded"
        stated = (
            self.budget_seconds // 60
            if self.budget_unit == "minutes"
            else self.budget_seconds
        )
        return f"**{stated} {self.budget_unit}{self.budget_scope}**{self.budget_note}"


#: The four tiers, three budgets. A test's tier is decided by what its size is
#: for, never by how slow it happens to be, so the budget below is what the
#: tier is allowed and not what it costs.
TIERS: tuple[Tier, ...] = (
    Tier(
        name="CI",
        marker="none (the default)",
        fixture="`<problem>/ci.yaml`",
        contents="correctness at the smallest size that exercises the claim",
        budget_seconds=PR_GATE.budget_seconds,
        budget_note=", worst case",
    ),
    Tier(
        name="key",
        marker="`key`",
        fixture=(
            'the instance a file marks `key`, reached as `fixture(problem, "key")`'
        ),
        contents=(
            "one problem's declared instance run end to end — simulate, fit, "
            "assert. The one that exists: the coupled model at 5,041 vertices, "
            "bin factor 5, measured at **62 s** against 59 s at factor 10 and "
            "`release` at factor 1"
        ),
        budget_seconds=KEY_DURATION_CAP.seconds,
        budget_unit="s",
        budget_scope=" per test",
        budget_note=f" (`{KEY_DURATION_CAP.variable}`)",
    ),
    Tier(
        name="developer / stress",
        marker="`stress`",
        fixture="`<problem>/stress.yaml`",
        contents="the same claims at a size the CI budget cannot hold",
        # The one budget in this module that no gate and no cap carries: it
        # bounds what a developer waits for locally, and nothing runs it.
        budget_seconds=600,
    ),
    Tier(
        name="release",
        marker="`release`",
        fixture="`<problem>/release.yaml`",
        contents="long-running scientific validity, run by `infra/release.sh`",
        budget_seconds=RELEASE_GATE.budget_seconds,
    ),
)


def marker_registrations() -> str:
    """`pyproject.toml`'s `markers` list, in the order that file registers them."""
    described = {**KIND_MARKERS, **SCHEDULING_MARKERS}
    unplaced = [name for name in described if name not in MARKER_REGISTRATION_ORDER]
    if unplaced:
        message = f"markers with no place in the registration order: {unplaced}"
        raise ValueError(message)
    lines = [
        "markers = [",
        *(f'    "{name}: {described[name]}",' for name in MARKER_REGISTRATION_ORDER),
        "]",
    ]
    return "".join(f"{line}\n" for line in lines)


def tier_table() -> str:
    """`DEV.md`'s tier table, indented into the bullet that introduces it.

    A blank line each side: the marker comments around the block are HTML, and
    a table has to start its own block to parse as one.
    """
    rows: list[tuple[str, ...]] = [
        ("tier", "marker", "fixture", "budget", "contents"),
        ("---",) * 5,
        *(
            (tier.name, tier.marker, tier.fixture, tier.budget, tier.contents)
            for tier in TIERS
        ),
    ]
    table = "".join(f"  | {' | '.join(cells)} |\n" for cells in rows)
    return f"\n{table}\n"


@dataclass(frozen=True)
class DerivedBlock:
    """A block of another file that is written from this module."""

    #: The file, relative to the repository root.
    path: str
    #: The line that opens the block, and the line that closes it. Both stay
    #: hand-written, and only what lies between them is replaced: `DEV.md` is
    #: followed step by step, and a generator that owned the whole file would
    #: eventually delete a sentence somebody needed.
    begin: str
    end: str
    #: What to put between them.
    render: Callable[[], str]


#: The two copies of the policy that are pure data, and are therefore written
#: rather than asserted (issue #470).
DERIVED: tuple[DerivedBlock, ...] = (
    DerivedBlock(
        path="pyproject.toml",
        begin="# BEGIN GENERATED markers (infra/gates.py, via infra/ledgers.sh)",
        end="# END GENERATED markers",
        render=marker_registrations,
    ),
    DerivedBlock(
        path="DEV.md",
        begin=(
            "<!-- BEGIN GENERATED tier table (infra/gates.py, via infra/ledgers.sh) -->"
        ),
        end="<!-- END GENERATED tier table -->",
        render=tier_table,
    ),
)


def regenerated(block: DerivedBlock) -> str:
    """``block``'s whole file, with the block replaced by what this renders.

    The markers are matched stripped, so a block indented into a list keeps its
    indentation, and exactly one pair must be present: a file with two openings
    or none is a file this cannot write, which is a failure rather than a guess.
    """
    lines = (REPO_ROOT / block.path).read_text().splitlines(keepends=True)
    opens = [i for i, line in enumerate(lines) if line.strip() == block.begin]
    closes = [i for i, line in enumerate(lines) if line.strip() == block.end]
    if len(opens) != 1 or len(closes) != 1 or closes[0] <= opens[0]:
        message = (
            f"{block.path}: expected one {block.begin!r} ... {block.end!r} pair, "
            f"found {len(opens)} opening and {len(closes)} closing"
        )
        raise ValueError(message)
    return "".join([*lines[: opens[0] + 1], block.render(), *lines[closes[0] :]])


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate or check the blocks other files derive from this one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write", action="store_true", help="regenerate the derived blocks in place"
    )
    mode.add_argument(
        "--check", action="store_true", help="exit 1 if a derived block is stale"
    )
    arguments = parser.parse_args(argv)

    stale: list[str] = []
    for block in DERIVED:
        path = REPO_ROOT / block.path
        text = regenerated(block)
        if path.read_text() == text:
            continue
        stale.append(block.path)
        if arguments.write:
            path.write_text(text)
    if arguments.write:
        print(f"derived blocks rewritten: {', '.join(stale) if stale else 'none'}")
        return 0
    if stale:
        print(
            f"stale derived blocks: {', '.join(stale)}; regenerate with: "
            "uv run python infra/gates.py --write",
            file=sys.stderr,
        )
        return 1
    print("derived blocks are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
