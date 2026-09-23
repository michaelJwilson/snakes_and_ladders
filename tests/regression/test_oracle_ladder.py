"""The oracle ladder resolves: every rung to a callable, every pin to an `oracle` test.

Issue #734. `infra/ladder.py` is a declaration, and a declaration nothing
reads goes stale on the first rename. What is checked here is that each of
its three columns is true of the tree: the callable imports, the test
collects and carries `oracle`, and the rung below is a rung of the same
problem. A rung no test pins carries its issue number instead, and the count
of those is pinned so it moves only when a rung gains a test.

The marker check runs one collection of the whole regression suite, which is
the only way to read what an item carried: the problem and `infra` markers are
added by the collection hook (`infra/coverage_recut.py`). It is the expensive
test of the file and carries no `critical`; the rest are structural and do.
"""

from __future__ import annotations

import importlib

import coverage_recut
import ladder
import pytest
from ladder import LADDER, LADDER_UNITS, PROBLEMS, Rung
from snakes_and_ladders.cost import Cost

from tests._paths import REPO_ROOT

#: What issue #734 counts as unpinned today: none. The Potts five (step 2),
#: the tree four (step 3), the HMM four (step 4), the codes five (step 5) and
#: the mixture four (step 6) are pinned, so no ladder carries a ticket. The
#: rule below is unchanged by that: a rung without a test names an issue.
UNPINNED_TODAY = 0

#: The package every rung's callable is relative to.
PACKAGE = "snakes_and_ladders"


@pytest.fixture(scope="module")
def markers() -> dict[str, frozenset[str]]:
    """Every regression and validation test's markers, parametrization stripped.

    A rung names a test function, not one of its cases: a pin holds for every
    case or the pin is wrong. `tests/validation/` is read too, since a rung's
    external oracle runs there, in a subprocess (issue #976).
    """
    collected: dict[str, set[str]] = {}
    for directory in ("regression", "validation"):
        for nodeid, found in coverage_recut.collect_markers(
            REPO_ROOT / "tests" / directory
        ).items():
            collected.setdefault(nodeid.split("[")[0], set()).update(found)
    return {nodeid: frozenset(found) for nodeid, found in collected.items()}


def unpinned_tests(
    rungs: tuple[Rung, ...], markers: dict[str, frozenset[str]]
) -> list[str]:
    """Rungs whose named test does not collect, or collects without `oracle`.

    The checker the guard below runs, taken separately so it can be run
    against a rung that is not in the ladder.
    """
    failures = []
    for rung in rungs:
        if rung.test is None:
            continue
        found = markers.get(rung.test)
        if found is None:
            failures.append(
                f"{rung.problem} / {rung.name}: {rung.test} does not collect"
            )
        elif "oracle" not in found:
            failures.append(
                f"{rung.problem} / {rung.name}: {rung.test} carries {sorted(found)}"
            )
    return failures


@pytest.mark.infra
def test_every_named_test_collects_and_judges_against_an_oracle(
    markers: dict[str, frozenset[str]],
) -> None:
    # The pin is the whole claim: a node id that no longer collects, or one
    # that judges the implementation against itself, leaves the rung above it
    # established by nothing while the table still says it is pinned.
    assert unpinned_tests(LADDER, markers) == []


@pytest.mark.critical
@pytest.mark.infra
def test_a_misnamed_test_fails_the_check() -> None:
    # The guard above passes today, so what it would catch is asserted on a
    # rung built here rather than by editing the ladder.
    real = "tests/regression/likelihood/test_potts_exact.py::test_the_transfer_matrix_reproduces_exhaustive_enumeration"
    markers = {real: frozenset({"oracle"})}
    misnamed = Rung(
        "potts",
        "transfer matrix",
        "likelihood.potts.strip_log_partition",
        "enumeration",
        "tests/regression/likelihood/test_potts_exact.py::test_that_does_not_exist",
        cost=Cost.PASS,
    )
    unmarked = Rung(
        "potts",
        "transfer matrix",
        "likelihood.potts.strip_log_partition",
        "enumeration",
        real,
        cost=Cost.PASS,
    )

    assert unpinned_tests((misnamed,), markers) == [
        "potts / transfer matrix: "
        "tests/regression/likelihood/test_potts_exact.py::test_that_does_not_exist"
        " does not collect"
    ]
    assert unpinned_tests((unmarked,), {real: frozenset({"smoke"})}) == [
        f"potts / transfer matrix: {real} carries ['smoke']"
    ]
    assert unpinned_tests((unmarked,), markers) == []


@pytest.mark.critical
@pytest.mark.infra
def test_a_rung_carries_either_a_test_or_a_ticket() -> None:
    # The rule the ladder is declared under: a gap is a row with an issue
    # number, never a blank cell a reader has to notice.
    assert [
        rung.name for rung in LADDER if rung.test is None and rung.ticket is None
    ] == []
    assert [rung.name for rung in LADDER if rung.test is not None and rung.ticket] == []
    assert [
        rung.name for rung in ladder.unpinned() if rung.ticket != ladder.LADDER_TICKET
    ] == []


@pytest.mark.critical
@pytest.mark.infra
def test_the_rung_below_is_a_rung_of_the_same_problem_and_no_ladder_cycles() -> None:
    # A `below` naming nothing is a chain broken where the table still reads
    # continuous; a cycle is two rungs each established by the other, which
    # establishes neither.
    for problem in PROBLEMS:
        here = ladder.rungs(problem)
        names = {rung.name for rung in here}
        assert [
            rung.below for rung in here if rung.below and rung.below not in names
        ] == []

        edges: dict[str, set[str]] = {}
        for rung in here:
            edges.setdefault(rung.name, set())
            if rung.below is not None:
                edges[rung.name].add(rung.below)
        # Kahn's rule: a graph with no cycle loses every node to repeated
        # removal of the ones with nothing left below them.
        remaining = dict(edges)
        while True:
            feet = {name for name, below in remaining.items() if not below}
            if not feet:
                break
            remaining = {
                name: below - feet
                for name, below in remaining.items()
                if name not in feet
            }
        assert remaining == {}, f"{problem} ladder cycles through {sorted(remaining)}"


@pytest.mark.critical
@pytest.mark.infra
def test_the_unpinned_rungs_are_the_count_the_ticket_states() -> None:
    # Pinned so the number moves only when a rung gains a test or the survey
    # finds another gap, and never quietly.
    assert len(ladder.unpinned()) == UNPINNED_TODAY
    assert len(ladder.pinned()) + len(ladder.unpinned()) == len(LADDER)
    per_problem = {
        problem: sum(1 for rung in ladder.rungs(problem) if rung.test is None)
        for problem in PROBLEMS
    }
    assert per_problem == {"potts": 0, "tree": 0, "hmm": 0, "codes": 0, "mixture": 0}


@pytest.mark.critical
@pytest.mark.infra
def test_every_rung_names_a_callable_the_package_carries() -> None:
    # The column is a string this module resolves rather than a reference
    # `infra/` holds: the declaration names application code and imports none
    # of it (`infra/CLAUDE.md`).
    missing = []
    for rung in LADDER:
        module, _, attribute = rung.callable.rpartition(".")
        try:
            imported = importlib.import_module(f"{PACKAGE}.{module}")
        except ImportError as error:  # a framework extra, not a stale name
            missing.append(f"{rung.callable}: {error}")
            continue
        if not hasattr(imported, attribute):
            missing.append(f"{rung.callable}: {module} carries no {attribute}")
    assert missing == []


@pytest.mark.critical
@pytest.mark.infra
def test_every_rung_belongs_to_one_of_the_five_ladders() -> None:
    assert {rung.problem for rung in LADDER} == set(PROBLEMS)
    with pytest.raises(ValueError, match="unknown problem"):
        ladder.rungs("potts_lattice")


@pytest.mark.critical
@pytest.mark.infra
def test_every_rung_declares_a_unit_and_every_unit_is_spent() -> None:
    # The field is keyword-only with no default, so a rung without a cost does
    # not construct; what is asserted is the vocabulary: every rung's unit is
    # one of the eight the ladder declares, and each of the eight is spent. An
    # exact rung may sit above another exact rung (the k-means dynamic
    # programme is pinned against assignment enumeration), so exactness is not
    # a foot rule (issue #818). `Cost` is the package's vocabulary since #860
    # and carries the units `opt.budget.Budget` names too, so the ladder's
    # eight are named by `LADDER_UNITS` rather than read off the enum.
    assert [rung.name for rung in LADDER if not isinstance(rung.cost, Cost)] == []
    assert {rung.cost for rung in LADDER} == LADDER_UNITS
