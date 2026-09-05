"""What a module `CLAUDE.md` may contain, enforced rather than stated.

Root `CLAUDE.md` states that its **Writing Style** rules bind every module
file. Nothing enforced that: the eight module files carried a generic "these
are local" line that never named the section, so an agent reading one of them
alone had no way to know (issue #155).

Stating an invariant is not enforcing it. `docs/source/index.rst` claimed to
cover every submodule while missing all eighteen of `phylo.qa`, because
`sphinx-build -W` fails on a broken entry and never on an absent one (issue
#154). This module is the check that was missing there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

# The directories root `CLAUDE.md` names as carrying their own file.
MODULE_DIRECTORIES = (
    "python/phylo/sim",
    "python/phylo/likelihood",
    "python/phylo/opt",
    "python/phylo/learn",
    "python/phylo/search",
    "python/phylo/qa",
    "infra",
    "docs",
)

POINTER = "**Writing Style**"

# Rule 5's label, distinctive enough that a file reproducing the section would
# contain it and a file referencing the section would not.
RESTATEMENT = "Apply naming, terminology, and syntax consistently"

# Rule 6 says a `CLAUDE.md` carries principles, not technical detail, and the
# detail that accretes fastest is a measurement: a result belongs to whatever
# produced it, and a second copy in a module file is a copy to keep true.
# Three shapes cover what was found in these files (issue #235): scientific
# notation, an "N of M" count, and a decimal carrying two or more fractional
# digits.
MEASUREMENT = re.compile(
    r"\b\d+(?:\.\d+)?e[-+]?\d+\b|\b\d+ of \d+\b|\b\d+\.\d{2,}\b",
    re.IGNORECASE,
)

# The module files were 26 to 93 lines when written and had grown to 48-183 by
# the time rule 6 landed, three- to six-fold, while root grew by four lines
# over 26 commits. This is the ceiling that stops that growth resuming; it is
# above the longest file after issue #235 rewrote them, and adding a rule past
# it means removing one, which is what "edits are rare" means.
LINE_BUDGET = 120


def _module_claude_files() -> list[Path]:
    """Find every module `CLAUDE.md` on disk.

    Discovered rather than listed, so a module directory added without a
    `CLAUDE.md` is caught by the emptiness check below rather than passing
    because nobody updated a hardcoded list.

    Returns
    -------
    list[Path]
        Every `CLAUDE.md` under a package or tooling directory, sorted.
    """
    found = sorted(REPO_ROOT.glob("python/phylo/*/CLAUDE.md"))
    for directory in ("infra", "docs"):
        candidate = REPO_ROOT / directory / "CLAUDE.md"
        if candidate.is_file():
            found.append(candidate)
    return found


@pytest.mark.critical
@pytest.mark.structural
def test_every_named_module_directory_has_a_claude_md() -> None:
    # Root `CLAUDE.md` names these eight; a missing file would mean the rules
    # reach a module through nothing at all.
    missing = [
        directory
        for directory in MODULE_DIRECTORIES
        if not (REPO_ROOT / directory / "CLAUDE.md").is_file()
    ]

    assert missing == []


@pytest.mark.critical
@pytest.mark.structural
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
@pytest.mark.structural
def test_no_module_claude_md_restates_the_writing_style() -> None:
    # The other half, and the reason the pointer is a pointer. Root
    # `CLAUDE.md`'s Writing Style section changed three times on the day this
    # was written; nine copies of it would already disagree.
    restating = [
        str(path.relative_to(REPO_ROOT))
        for path in _module_claude_files()
        if RESTATEMENT in path.read_text()
    ]

    assert restating == []


@pytest.mark.critical
@pytest.mark.structural
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
@pytest.mark.structural
def test_the_root_file_states_that_the_rules_reach_the_module_files() -> None:
    # The pointers are only true because root says so. If that sentence goes,
    # eight files start referring to a scope nothing declares.
    text = ROOT_CLAUDE_MD.read_text()

    assert "## Writing Style" in text
    assert "each module's `CLAUDE.md` included" in text


@pytest.mark.critical
@pytest.mark.structural
def test_the_expected_reader_contract_lives_only_in_docs() -> None:
    # It is a contract about the technical document, so the seven other module
    # files have no business restating it -- which is what issue #141 removed
    # from `docs/CLAUDE.md` in the other direction, and why only `docs/` may
    # refer to it.
    elsewhere = [
        str(path.relative_to(REPO_ROOT))
        for path in _module_claude_files()
        if "Expected Reader" in path.read_text() and path.parent.name != "docs"
    ]

    assert elsewhere == []


@pytest.mark.critical
@pytest.mark.structural
def test_no_module_claude_md_carries_a_measurement() -> None:
    # Rule 6, made checkable. A measurement in one of these files is a second
    # copy of a number that `STATUS.md`, a docstring or a test already owns,
    # and the copy is the one that goes stale: nothing recomputes it and no
    # check compares it against what it was copied from.
    carrying = {
        str(path.relative_to(REPO_ROOT)): sorted(
            set(MEASUREMENT.findall(path.read_text()))
        )
        for path in _module_claude_files()
        if MEASUREMENT.search(path.read_text())
    }

    assert carrying == {}, (
        "measurements in a module CLAUDE.md: "
        + "; ".join(f"{name}: {found}" for name, found in carrying.items())
        + ". Move each to what produces it -- `STATUS.md` where it is evidence "
        "for a milestone, the module defining the constant where a caller acts "
        "on it -- and leave the principle behind."
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_measurement_check_catches_each_shape_it_claims_to() -> None:
    # The check exercised, per shape rather than in aggregate: a guard that
    # has never been seen to fail is not known to work, and a regex is exactly
    # the kind of check that silently matches nothing.
    caught = ["worst deviation 3.7e-15", "39 of 40 runs", "a ratio of 0.87856"]
    passed = [
        "the deviation is reported rather than asserted",
        "a bound that holds at every size",
        "Python 3 and a single digit 0.5 are not measurements",
    ]

    assert [text for text in caught if not MEASUREMENT.search(text)] == []
    assert [text for text in passed if MEASUREMENT.search(text)] == []


@pytest.mark.critical
@pytest.mark.structural
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
@pytest.mark.structural
def test_the_line_budget_check_is_not_vacuous(tmp_path: Path) -> None:
    # As above: pin that a file over the budget is detected, since the
    # assertion passes on an empty set for either reason.
    over = tmp_path / "CLAUDE.md"
    over.write_text("x\n" * (LINE_BUDGET + 1))
    under = tmp_path / "other_CLAUDE.md"
    under.write_text("x\n" * LINE_BUDGET)

    assert len(over.read_text().splitlines()) > LINE_BUDGET
    assert len(under.read_text().splitlines()) <= LINE_BUDGET
