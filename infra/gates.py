"""What each gate selects, where it runs, and what it is allowed to cost.

**Authoritative for two blocks, descriptive for the rest.** `pyproject.toml`'s
marker registrations and `DEV.md`'s tier table are *written* from the tables
below by `infra/ledgers.sh` (issue #470), each between marker comments naming
this module. Neither copy can drift: editing a block by hand is undone on the
next regeneration, and the review gate fails a branch where regenerating
rewrites a tracked file.

Every other value here is a copy of what another file says --
`.github/workflows/ci.yml`'s steps, `infra/validate.sh`'s and
`infra/review_gates.sh`'s exported caps, `infra/release.sh`'s command,
`DEV.md`'s review-gate sentence. Each is read by something that cannot import
Python at the point it needs the value: a workflow condition GitHub evaluates,
a shell export `tests/conftest.py` reads from the environment, a sentence a
reviewer reads. They stay written where they are read and
`tests/regression/test_gates.py` asserts they still say this; the file that
runs the gate wins.

Issue #465 found the policy spread over twelve files that already disagreed:
`DEV.md` said `infra/review_gates.sh` had nine rows while it had eight. Issue
#469 wrote the policy down once and held the copies to it; this step generates
the two that are pure data.
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

#: Every tier, unbounded, at the release gate. Written out as a tautology
#: rather than left unfiltered: `pyproject.toml` deselects `release` by
#: default, so an absent `-m` now means the per-PR tier and this gate has to
#: say what it selects.
RELEASE_GATE = Gate(
    name="release",
    marker_expression="release or not release",
    coverage_floor=90,
    runs_in="infra/release.sh",
    budget_seconds=None,
)

#: The three gates a change passes, in the order it reaches them.
GATES: tuple[Gate, ...] = (PR_GATE, MAIN_GATE, RELEASE_GATE)

#: The cap the `python-tests` job carries, in minutes, and its only bound: a
#: bound on the selection and a fixed timeout cannot coexist, since the timeout
#: then cancels the pull requests the bound refused (issue #425).
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
#: gate need them without needing this. Issue #729 retired `structural` -- an
#: invariant the repository chose, which is `infra` where the test is of the
#: repository's own machinery and `smoke` where it is of the science -- and
#: renamed `simulated_truth` to `end2end`, for the path such a test drives.
KIND_MARKERS: Mapping[str, str] = {
    "oracle": (
        "checked against an independent exact answer -- enumeration, brute "
        "force, a closed form, or a published value"
    ),
    "end2end": (
        "drives a real path -- simulate, fit, decode -- and checks a scientific "
        "output against the truth that generated the data, to a stated tolerance"
    ),
    "analytic": (
        "judges against a mathematical property the model must satisfy whatever "
        "the implementation does -- conservation, a limit, a symmetry, a "
        "monotonicity, a calibration"
    ),
    "smoke": (
        "checked against itself -- reachability, a shape, the absence of an "
        "exception, a boundary, a refusal, an invariant the implementation chose"
    ),
    "infra": (
        "exercises no single problem -- shared machinery, a document guard, "
        "or the build; written on a test of that machinery, added at collection "
        "where the problem scan finds none, and checked against what the module "
        "imports (issues #622, #729)"
    ),
}

#: What a finding is: the second axis issue #729 added, beside the kind and
#: never instead of one. None of these counts toward the judged coverage:
#: only `end2end` and `oracle` do, since only they judge the science against
#: something outside the implementation.
FINDING_MARKERS: Mapping[str, str] = {
    "patch": (
        "a change reproduces the sal call it replaces, bitwise or to a stated tolerance"
    ),
    "backend": (
        "the same algorithm, or the same oracle, on another backend -- Rust, "
        "Numba, Torch, a vectorized path -- reproduced bitwise or to a stated "
        "tolerance"
    ),
    "bug": (
        "pins a defect against the paper, an oracle or the code's own contract, "
        "and is written to fail when the defect is fixed"
    ),
    "warning": (
        "pins behaviour that is defensible but suspicious -- a shipped constant "
        "that asserts more than the data carries, a silent limit; nobody is "
        "wrong and somebody should know"
    ),
    "snapshot": (
        "pins current behaviour with no judgement attached, to be conserved -- "
        "a number STATUS.md records, a seed-for-seed pin"
    ),
}

#: When a test runs. The second axis, and not a kind: a test is critical *and*
#: an oracle test, never instead of one.
SCHEDULING_MARKERS: Mapping[str, str] = {
    "release": (
        "long-running scientific validity test, run on release rather than "
        "per PR (DEV.md CI & Performance Budget)"
    ),
    # Widened by issue #635. It read "its failure invalidates what runs after
    # it", which is *structural* priority, and the tier it produced was
    # 149 infrastructure tests of 177 at 37% coverage: the import graph, the
    # documentation index, the `CLAUDE.md` pointers, `select_tests` itself.
    # A broken likelihood is not structurally prior to anything and surfaced
    # after the twenty-minute tier rather than after sixteen seconds. The
    # second clause admits it, and "the rest is not worth running" is what
    # both clauses have in common.
    "critical": (
        "gates early; fast, and its failure means the rest is not worth "
        "running -- structurally, or because the science it rests on is wrong"
    ),
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

#: What a test's subject is, where it is not one of the declared problems: the
#: same `infra` marker, on a second axis. Added at collection by
#: `tests/conftest.py` where the problem scan finds nothing, so that "unmarked"
#: stops being a state a module can be in for two different reasons (issue
#: #622); since issue #729 also written by an author, as the kind a test of the
#: repository's own machinery has. One registration serves both.
SUBJECT_MARKERS: Mapping[str, str] = {"infra": KIND_MARKERS["infra"]}

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
    *(name for name in KIND_MARKERS if name not in SUBJECT_MARKERS),
    "critical",
    "stress",
    "key",
    *SUBJECT_MARKERS,
    *FINDING_MARKERS,
)


@dataclass(frozen=True)
class JudgedCoverage:
    """The coverage guard issue #729 added: what counts, what is exempt, the floors.

    `--cov-fail-under` counts a statement whichever test reached it, and 34.3
    points of the 94.3 it read on 2026-09-18 were reached by importing the
    package with no test at all. This counts a statement when an `end2end`
    or an `oracle` test reached it -- a real path judged against the truth
    that generated the data, or an exact independent answer -- and nothing
    else does: `analytic`, `patch`, `backend`, `bug`, `warning`, `snapshot`
    and `smoke` say what a test is, and `infra` is used sparingly.
    `infra/coverage_recut.py` reads the run's per-test contexts and applies it.
    """

    #: What the guard is called in a report and a floor's message.
    name: str
    #: The markers whose tests count. A test carrying any of them counts.
    counting: tuple[str, ...]
    #: Packages under `snakes_and_ladders` outside the guard: a renderer has
    #: no oracle, and `qa` is held by `snapshot` pins and stated beside the
    #: figure, never inside it.
    exempt_packages: tuple[str, ...]
    #: The floor over every package not exempt, as `--cov-fail-under` states
    #: one. Recut to the measurement and rounded down, never lowered.
    floor: float
    #: Floors a package carries above the whole: `search` is the package's
    #: product and is held higher.
    package_floors: Mapping[str, float]


JUDGED_COVERAGE = JudgedCoverage(
    name="judged",
    counting=("end2end", "oracle"),
    exempt_packages=("qa",),
    floor=85.4,
    package_floors={"search": 89.9},
)

#: The complement (issue #732): every kind and finding but the two that
#: judge, so what the self-checking tests reach is a number of its own
#: rather than the difference of two. Same exemption, its own floors.
UNJUDGED_COVERAGE = JudgedCoverage(
    name="unjudged",
    counting=tuple(
        name
        for name in (*KIND_MARKERS, *FINDING_MARKERS)
        if name not in JUDGED_COVERAGE.counting
    ),
    exempt_packages=("qa",),
    floor=89.2,
    package_floors={"search": 88.9},
)

#: The two recuts of one run, in the order the report prints them.
COVERAGE_GUARDS: tuple[JudgedCoverage, ...] = (JUDGED_COVERAGE, UNJUDGED_COVERAGE)


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
    described = {
        **KIND_MARKERS,
        **SCHEDULING_MARKERS,
        **SUBJECT_MARKERS,
        **FINDING_MARKERS,
    }
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
