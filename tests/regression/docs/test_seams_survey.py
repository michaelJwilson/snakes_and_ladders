"""The seams survey means what it says.

`SEAMS.md` is generated and not committed (issue #425), so there is no
committed copy to hold to a regeneration; what is checked is the survey
itself. The structural check the tool runs is a check: every implementer the
ledger names satisfies its protocol's members, a class missing one is not
named, and the verdict follows the consumer rule (issue #400).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import seams_survey  # noqa: E402


@pytest.fixture(scope="module")
def rows() -> list[seams_survey.Row]:
    return seams_survey.survey()


@pytest.mark.structural
def test_every_protocol_has_a_member_list_and_at_least_one_implementer(
    rows: list[seams_survey.Row],
) -> None:
    # A protocol with no members would make every class an implementer, and
    # one with no implementer is a name the package does not use.
    for row in rows:
        if row.seam.kind == "protocol":
            assert row.members, row.seam.name
            assert row.implementers, row.seam.name


@pytest.mark.structural
def test_a_class_missing_a_member_is_not_an_implementer() -> None:
    class Complete:
        def n_steps(self) -> int:
            return 1

        def __call__(self, _step: int) -> float:
            return 0.0

    class Partial:
        def __call__(self, _step: int) -> float:
            return 0.0

    members = seams_survey.members_of(
        importlib.import_module("snakes_and_ladders.opt.schedule").Schedule
    )
    assert seams_survey._satisfies(Complete, members)
    assert not seams_survey._satisfies(Partial, members)


@pytest.mark.structural
def test_the_verdict_follows_the_consumer_rule() -> None:
    seam = seams_survey.Seam("x", "X", "protocol")
    lonely = seams_survey.Row(seam, ("a",), ("m.A",), ("m",), ())
    shared = seams_survey.Row(seam, ("a",), ("m.A",), ("m", "n", "o"), ())
    argued = seams_survey.Row(
        seams_survey.Seam("x", "X", "protocol", "because"), ("a",), ("m.A",), ("m",), ()
    )
    assert (
        seams_survey.under_the_rule(lonely)
        == "one implementer, under the rule, no reason stated"
    )
    assert seams_survey.under_the_rule(shared) == "earns its place"
    assert seams_survey.under_the_rule(argued) == "under the rule; kept: because"
