"""The marker names both the guard and the merge gate read.

`tests/regression/test_test_kinds.py` asserts every test carries a kind, and
`infra/gate_changed_tests.py` checks the same thing on a branch's changed
files before a reviewer reads it. One definition, in `infra/` rather than in
the guard, because `infra/` is already on `mypy_path` and already imported by
the tests: putting `tests/regression` on `mypy_path` instead made
`test_test_kinds.py` reachable under two module names and `mypy --strict`
refused the whole tree (issue #418).

`pyproject.toml` registers the markers, and
`test_the_registered_markers_are_these` asserts these tuples and that
registration agree, so this file cannot drift from what `pytest` accepts.
"""

from __future__ import annotations

#: What a test is checked against.
KINDS = (
    "oracle",
    "simulated_truth",
    "mathematical",
    "edge_case",
    "structural",
)

#: The second axis. Not a kind: it says when a test runs, not what it checks.
#: `stress` (#232) is the same axis as `release`: a size, not a claim.
SCHEDULING = ("critical", "release", "stress")

#: Benchmarks measure rather than assert, so they carry no kind. Excluded here
#: rather than exempted case by case, because the exclusion is a property of
#: the directory and not of any test in it.
EXCLUDED_DIRECTORY = "benchmarks"
