"""What the three review gates added under #418 must refuse.

A gate is only worth its seconds if it fires. Each test here builds the
situation the gate exists to catch and asserts the gate's verdict on it, and
its mirror -- the situation that must pass -- because a gate that refuses
everything is removed on the first false alarm and one that refuses nothing is
never noticed at all (`DEV.md`, Review starts from `infra/review_gates.sh`).

The third gate, the generated ledgers, runs ``infra/ledgers.sh``. What that
gate protects is checked here: the three files it writes are not in the index.
A committed copy is what made a machine-written file a merge participant, and
re-adding one is the way this change is silently undone (issue #425).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import checks_ledger  # noqa: E402
import gate_changed_tests  # noqa: E402
import gate_new_seams  # noqa: E402
import problems_tables  # noqa: E402

#: What ``infra/ledgers.sh`` writes, and what must therefore never be tracked.
DERIVED = (checks_ledger.LEDGER, problems_tables.GENERATED)

MARKED = (
    "import pytest\n\n\n"
    "@pytest.mark.oracle\n"
    "def test_marked() -> None:\n    assert True\n"
)
UNMARKED = "def test_unmarked() -> None:\n    assert True\n"
#: `critical` says when a test runs, not what checks it, so it is not a kind
#: and must not satisfy the gate on its own (root `CLAUDE.md`).
SCHEDULING_ONLY = (
    "import pytest\n\n\n"
    "@pytest.mark.critical\n"
    "def test_scheduled() -> None:\n    assert True\n"
)


def _repository(root: Path, base_files: dict[str, str]) -> None:
    """Make a git repository at ``root`` whose ``main`` holds ``base_files``."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    for name, text in base_files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "checkout", "-q", "-b", "branch"], cwd=root, check=True)


def _commit(root: Path, files: dict[str, str]) -> None:
    """Add ``files`` to ``root`` and commit them on the current branch."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "change"],
        cwd=root,
        check=True,
    )


@pytest.mark.structural
def test_the_kind_gate_names_the_test_the_branch_added_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gate's own trigger: a test added with no kind is named, one added
    # with a kind is not, and a test the branch did not touch is not judged
    # whatever it carries -- that is the repository-wide guard's job.
    _repository(tmp_path, {"tests/regression/test_old.py": UNMARKED})
    _commit(
        tmp_path,
        {
            "tests/regression/test_new.py": MARKED + "\n\n" + UNMARKED,
            "tests/regression/test_tier.py": SCHEDULING_ONLY,
        },
    )
    monkeypatch.setattr(gate_changed_tests, "REPO_ROOT", tmp_path)

    assert gate_changed_tests.unmarked("main") == [
        "tests/regression/test_new.py::test_unmarked",
        "tests/regression/test_tier.py::test_scheduled",
    ]
    assert [str(path) for path in gate_changed_tests.changed_test_files("main")] == [
        "tests/regression/test_new.py",
        "tests/regression/test_tier.py",
    ]


@pytest.mark.structural
def test_the_kind_gate_passes_a_branch_whose_tests_all_say_what_checks_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path, {"tests/regression/test_old.py": UNMARKED})
    _commit(tmp_path, {"tests/regression/test_new.py": MARKED})
    monkeypatch.setattr(gate_changed_tests, "REPO_ROOT", tmp_path)

    assert gate_changed_tests.unmarked("main") == []


@pytest.mark.structural
def test_the_seam_gate_sees_only_the_protocols_the_branch_adds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `Protocol` already on the base is not a new seam, and a plain class is
    # not a seam at all.
    package = "python/snakes_and_ladders"
    existing = "from typing import Protocol\n\n\nclass Old(Protocol):\n    pass\n"
    _repository(tmp_path, {f"{package}/opt/objective.py": existing})
    _commit(
        tmp_path,
        {
            f"{package}/opt/objective.py": existing
            + "\n\nclass New(Protocol):\n    pass\n\n\nclass Plain:\n    pass\n"
        },
    )
    monkeypatch.setattr(gate_new_seams, "REPO_ROOT", tmp_path)

    assert gate_new_seams.added_protocols("main") == [("opt.objective", "New")]


def _declared(docstring: str) -> str:
    """One module's source declaring ``New``, with ``docstring`` inside it."""
    return (
        "from typing import Protocol\n\n\n"
        f'class New(Protocol):\n    """{docstring}"""\n'
    )


@pytest.mark.structural
def test_the_seam_gate_counts_the_modules_that_name_the_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The consumer count is a property of the import graph and not of the
    # class tree, which is why it lives in the gate rather than in the
    # catalogue issue #586 deleted. Three consuming modules admit the seam;
    # the module declaring it never counts as one of them.
    sources = {
        "opt.objective": _declared("An objective."),
        "a": "from snakes_and_ladders.opt.objective import New\n",
        "b": "New\n",
        "c": "Newer\n",
    }
    monkeypatch.setattr(
        gate_new_seams, "added_protocols", lambda _: [("opt.objective", "New")]
    )
    monkeypatch.setattr(gate_new_seams, "source_modules", lambda: sources)

    assert gate_new_seams.consumers("opt.objective", "New", sources) == ("a", "b")
    assert "2 consuming modules, 3 required" in gate_new_seams.verdicts("main")[0]

    sources["c"] = "New\n"

    assert gate_new_seams.verdicts("main") == []


@pytest.mark.structural
def test_the_seam_gate_takes_the_reason_from_the_declaration_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Where the reason lives is the whole of issue #586's argument: the
    # declaration is the record, so the gate reads the class's own docstring
    # and a reader reads the same words in the same place. The reason stops at
    # its paragraph, so one written above a ``Parameters`` section does not
    # swallow it.
    reason = f"{gate_new_seams.REASON} issue #418 will add two.\n\n    Parameters"
    sources = {"opt.objective": _declared(reason), "a": "New\n"}
    monkeypatch.setattr(
        gate_new_seams, "added_protocols", lambda _: [("opt.objective", "New")]
    )
    monkeypatch.setattr(gate_new_seams, "source_modules", lambda: sources)

    assert (
        gate_new_seams.reason_for("opt.objective", "New", sources)
        == "issue #418 will add two."
    )
    assert gate_new_seams.verdicts("main") == []

    sources["opt.objective"] = _declared("An objective, with no reason stated.")

    assert "states no" in gate_new_seams.verdicts("main")[0]


@pytest.mark.structural
def test_the_derived_ledgers_are_not_in_the_index() -> None:
    # The index, not the working tree: `infra/ledgers.sh` writes all three
    # into the tree on every document build and every review, so their
    # presence there says nothing. Tracking one is what the gate refuses,
    # demonstrated on a throwaway branch carrying a CHECKS.md committed with a
    # row the tests do not have: `infra/ledgers.sh --check` exited 1 on it and
    # so did this test (issue #425's pull request records the run).
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            *(str(path.relative_to(REPO_ROOT)) for path in DERIVED),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert listed == [], f"generated ledgers committed: {listed}"
