"""The duration guard's trigger: a slow test in the wrong tier is named.

`DEV.md` forbids asserting wall clock in the suite, so what is pinned is the
classification, on durations handed in rather than measured: a test over the
cap without `release` or `stress` is an offender, one with either is not, and
the report is slowest first (issue #372). The same file's other guard is
pinned the same way: a test that gates early and is scale-marked out of the
per-PR tier is named, and the three cases that resemble it are not (issue
#635).
"""

from __future__ import annotations

import pytest

from tests._durations import outside_the_tier, over_cap


@pytest.mark.critical
@pytest.mark.infra
def test_a_slow_unmarked_test_is_named_and_a_gated_one_is_not() -> None:
    durations = [
        ("tests/a.py::test_fast", 0.5, frozenset({"oracle"})),
        ("tests/a.py::test_slow", 12.0, frozenset({"oracle"})),
        ("tests/a.py::test_release", 40.0, frozenset({"oracle", "release"})),
        ("tests/a.py::test_stress", 15.0, frozenset({"stress"})),
        ("tests/a.py::test_slower", 25.0, frozenset()),
    ]

    lines = over_cap(durations, cap=10.0)

    assert [line.split(":")[0] for line in lines] == [
        "tests/a.py",
        "tests/a.py",
    ]
    assert lines[0].startswith("tests/a.py::test_slower: 25.0 s")
    assert lines[1].startswith("tests/a.py::test_slow: 12.0 s")


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_test_exactly_at_the_cap_is_inside_it() -> None:
    assert over_cap([("t::x", 10.0, frozenset())], cap=10.0) == []


@pytest.mark.critical
@pytest.mark.infra
def test_a_scale_marked_critical_test_is_named_and_the_other_three_are_not() -> None:
    """The conflict is `critical` *and* a scale marker, and nothing else.

    Issue #635. The three non-offenders are the cases that look like it: a
    critical test inside the tier, a release test that does not gate, and a
    test carrying neither.
    """
    items = [
        ("tests/a.py::test_gates", frozenset({"critical", "oracle"})),
        ("tests/a.py::test_release", frozenset({"oracle", "release"})),
        ("tests/a.py::test_both[bin-1]", frozenset({"critical", "release"})),
        ("tests/a.py::test_neither", frozenset({"mathematical"})),
    ]

    lines = outside_the_tier(items)

    assert len(lines) == 1
    assert lines[0].startswith(
        "tests/a.py::test_both[bin-1]: gates early and is release"
    )


@pytest.mark.critical
@pytest.mark.edge_case
def test_every_scale_marker_is_a_conflict_and_the_message_names_them_all() -> None:
    """`key` and `stress` are the tier's other two exits, and both conflict."""
    for marker in ("release", "stress", "key"):
        assert outside_the_tier([("t::x", frozenset({"critical", marker}))])

    both = outside_the_tier([("t::x", frozenset({"critical", "key", "stress"}))])

    assert both[0].startswith("t::x: gates early and is key/stress")
