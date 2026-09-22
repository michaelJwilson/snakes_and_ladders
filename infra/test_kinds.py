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

The two readings of a test's source that go with the names are here for the
same reason: what a marker *is* and how it is read off a decorator are one
fact, and the three readers that each carried their own copy of it are the
gate, the ledger and the guard (issue #863).
"""

from __future__ import annotations

import ast
from pathlib import Path

#: What a test is checked against. Issue #729 retired `structural` and
#: `edge_case`, renamed `simulated_truth` to `end2end` and `mathematical` to
#: `analytic`; only `end2end` and `oracle` count toward coverage. `infra` is a kind an
#: author writes on a test of the repository's own machinery, and the subject
#: the collection hook adds where a module names no problem. `experiment`
#: (issue #902) pins a measured comparison of two runs of the package, and
#: counts toward nothing.
KINDS = (
    "oracle",
    "end2end",
    "analytic",
    "smoke",
    "experiment",
    "infra",
)

#: What a finding is: the second axis issue #729 added, carried beside a kind
#: and never instead of one. Registered and enforced here; applied by the audit
#: that issue plans, directory by directory.
FINDINGS = ("patch", "backend", "bug", "warning", "snapshot")

#: The second axis. Not a kind: it says when a test runs, not what it checks.
#: `stress` (#232) is the same axis as `release`: a size, not a claim, and so
#: is `key` (#399), which names one declared instance's own budget.
SCHEDULING = ("critical", "release", "stress", "key")

#: The third axis: what a test's subject is, where it is not one of the
#: declared problems. Added at collection like the problem markers, never by
#: an author, so that a module with no problem is a module that says so
#: (issue #622).
SUBJECTS = ("infra",)

#: Benchmarks measure rather than assert, so they carry no kind. Excluded here
#: rather than exempted case by case, because the exclusion is a property of
#: the directory and not of any test in it.
EXCLUDED_DIRECTORY = "benchmarks"


def markers(node: ast.FunctionDef) -> set[str]:
    """The ``pytest.mark.<name>`` markers decorating one test.

    Read from the source rather than from a collected item: the merge gate
    judges a diff, and the ledger reads a tree it does not run. Written three
    times byte for byte --- here, in the gate and in the ledger --- and one of
    the three would have kept a stale reading of a decorator the others had
    learned to read (issue #863).

    Returns
    -------
    set[str]
    """
    found: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
        ):
            found.add(target.attr)
    return found


def functions_of(path: Path) -> list[ast.FunctionDef]:
    """Every top-level ``test_`` function of one file.

    Named for what it returns and not ``test_functions``: a guard importing
    that name gives `pytest` a module-level ``test_`` to collect, and the
    collection fails on a fixture called ``path``.

    Returns
    -------
    list[ast.FunctionDef]
    """
    return functions_in(ast.parse(path.read_text()))


def functions_in(tree: ast.Module) -> list[ast.FunctionDef]:
    """Every top-level ``test_`` function of an already parsed module.

    Separate from :func:`functions_of` for the one caller that reads the
    source for its own second purpose --- the ledger takes a test's claim from
    the lines around it --- so that caller parses once rather than reading the
    file twice.

    Returns
    -------
    list[ast.FunctionDef]
    """
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
