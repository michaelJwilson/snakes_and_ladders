"""The duration guard's trigger: a slow test in the wrong tier is named.

`DEV.md` forbids asserting wall clock in the suite, so what is pinned is the
classification, on durations handed in rather than measured: a test over the
cap without `release` or `stress` is an offender, one with either is not, and
the report is slowest first (issue #372).
"""

from __future__ import annotations

import pytest

from tests._durations import over_cap


@pytest.mark.critical
@pytest.mark.structural
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
