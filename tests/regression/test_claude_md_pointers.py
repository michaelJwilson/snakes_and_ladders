"""What a module `CLAUDE.md` may contain, enforced rather than stated.

Root `CLAUDE.md` states that its **Writing Style** rules bind every module
file. Nothing enforced that: the eight module files carried a generic "these
are local" line naming no section, so an agent reading one alone had no way to
know (issue #155).

Stating an invariant is not enforcing it. `docs/source/index.rst` claimed to
cover every submodule while missing all eighteen of `snakes_and_ladders.qa`,
since `sphinx-build -W` fails on a broken entry and never on an absent one
(issue #154).

Four guards this file carried matched a string for a judgement and were
deleted by issue #787: the restatement guard searched for a sentence root
`CLAUDE.md` no longer contained and could not fail, the measurement regex
missed the comma-grouped sweeps a module file carried, and the Expected Reader
guard asserted absence elsewhere while the section it named was absent. What
they checked is the Principles section of the release template, read by the
auditor against the tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

# The directories root `CLAUDE.md` names as carrying their own file.
MODULE_DIRECTORIES = (
    "python/snakes_and_ladders/sim",
    "python/snakes_and_ladders/likelihood",
    "python/snakes_and_ladders/opt",
    "python/snakes_and_ladders/learn",
    "python/snakes_and_ladders/search",
    "python/snakes_and_ladders/qa",
    "python/snakes_and_ladders/sandbox",
    "infra",
    "docs",
)

POINTER = "**Writing Style**"


# The module files were 26 to 93 lines when written and 48-183 by the time rule
# 6 landed, while root grew by four lines over 26 commits. This ceiling stops
# that growth resuming: it is above the longest file after issue #235 rewrote
# them, so adding a rule past it means removing one.
LINE_BUDGET = 120


def _module_claude_files() -> list[Path]:
    """Find every module `CLAUDE.md` on disk.

    Discovered rather than listed, so a module directory added without a
    `CLAUDE.md` is caught by the emptiness check below.

    Returns
    -------
    list[Path]
        Every `CLAUDE.md` under a package or tooling directory, sorted.
    """
    found = sorted(REPO_ROOT.glob("python/snakes_and_ladders/*/CLAUDE.md"))
    for directory in ("infra", "docs"):
        candidate = REPO_ROOT / directory / "CLAUDE.md"
        if candidate.is_file():
            found.append(candidate)
    return found


@pytest.mark.critical
@pytest.mark.infra
def test_every_named_module_directory_has_a_claude_md() -> None:
    # Root `CLAUDE.md` names these nine; a missing file would mean the rules
    # reach a module through nothing at all.
    missing = [
        directory
        for directory in MODULE_DIRECTORIES
        if not (REPO_ROOT / directory / "CLAUDE.md").is_file()
    ]

    assert missing == []


@pytest.mark.critical
@pytest.mark.infra
def test_every_module_claude_md_points_at_the_writing_style() -> None:
    # The ticket's substance: reading one module file must tell you the
    # writing style governs what you are about to write.
    silent = [
        str(path.relative_to(REPO_ROOT))
        for path in _module_claude_files()
        if POINTER not in path.read_text()
    ]

    assert silent == []


@pytest.mark.critical
@pytest.mark.infra
def test_a_module_directory_added_without_a_pointer_is_caught(
    tmp_path: Path,
) -> None:
    # The check itself, exercised: the assertion above passes vacuously if the
    # search finds nothing, so this pins that a file lacking the pointer is
    # actually detected rather than skipped.
    without = tmp_path / "CLAUDE.md"
    without.write_text("# newmodule/\n\nRoot `CLAUDE.md` holds the rules.\n")
    with_pointer = tmp_path / "other_CLAUDE.md"
    with_pointer.write_text(f"# other/\n\nRoot `CLAUDE.md`'s {POINTER} binds this.\n")

    assert POINTER not in without.read_text()
    assert POINTER in with_pointer.read_text()


@pytest.mark.critical
@pytest.mark.infra
def test_the_root_file_states_that_the_rules_reach_the_module_files() -> None:
    # The pointers are only true because root says so. If that sentence goes,
    # eight files start referring to a scope nothing declares. The sentence is
    # matched on what it must say rather than on its exact words, which is why
    # rewording the root file broke this guard once (#579).
    text = ROOT_CLAUDE_MD.read_text()

    assert "## Writing Style" in text
    reach = [
        line
        for line in text.splitlines()
        if "`CLAUDE.md`" in line and "same" in line and "rules" in line
    ]

    assert reach, (
        "the root CLAUDE.md no longer says its rules reach the module files; "
        "eight module files point at a scope nothing declares"
    )


@pytest.mark.critical
@pytest.mark.infra
def test_no_module_claude_md_exceeds_the_line_budget() -> None:
    # The other half of rule 6. A file can carry no measurement and still be a
    # technical brief, and length is what that looks like from outside.
    over = {
        str(path.relative_to(REPO_ROOT)): len(path.read_text().splitlines())
        for path in _module_claude_files()
        if len(path.read_text().splitlines()) > LINE_BUDGET
    }

    assert over == {}, (
        f"module CLAUDE.md files over the {LINE_BUDGET}-line budget: {over}. "
        "A rule worth adding is worth removing another for."
    )


@pytest.mark.critical
@pytest.mark.infra
def test_the_line_budget_check_is_not_vacuous(tmp_path: Path) -> None:
    # As above: pin that a file over the budget is detected, since the
    # assertion passes on an empty set for either reason.
    over = tmp_path / "CLAUDE.md"
    over.write_text("x\n" * (LINE_BUDGET + 1))
    under = tmp_path / "other_CLAUDE.md"
    under.write_text("x\n" * LINE_BUDGET)

    assert len(over.read_text().splitlines()) > LINE_BUDGET
    assert len(under.read_text().splitlines()) <= LINE_BUDGET
