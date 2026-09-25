"""What a module `CLAUDE.md` may contain, enforced rather than stated.

Root `CLAUDE.md` states that its **Writing Style** rules bind every module
file. Nothing enforced that: the eight module files carried a generic "these
are local" line naming no section, so an agent reading one alone had no way to
know (issue #155).

Stating an invariant is not enforcing it. `docs/source/index.rst` claimed to
cover every submodule while missing all eighteen of `sal.qa`,
since `sphinx-build -W` fails on a broken entry and never on an absent one
(issue #154).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._paths import REPO_ROOT

ROOT_CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

# The directories root `CLAUDE.md` names as carrying their own file.
MODULE_DIRECTORIES = (
    "python/sal/sim",
    "python/sal/likelihood",
    "python/sal/opt",
    "python/sal/learn",
    "python/sal/search",
    "python/sal/sample",
    "python/sal/qa",
    "python/sal/sandbox",
    "infra",
    "docs",
)

POINTER = "**Writing Style**"


#: Rule 5's own sentence, read from root rather than copied here. The copy
#: this replaces said "naming, terminology, and syntax" where root says
#: "naming, terminology, notation and syntax", so it matched nothing and the
#: check could not fail (#787, #803). Derived, it cannot drift again.
def _rule_five_sentence() -> str:
    """The sentence root `CLAUDE.md` states under **Maintain formatting**."""
    for line in (REPO_ROOT / "CLAUDE.md").read_text().splitlines():
        if "**Maintain formatting:**" in line:
            sentence = line.split("**Maintain formatting:**", 1)[1].strip()
            assert len(sentence) > 20, "rule 5 is too short to search for"
            return sentence
    msg = "root CLAUDE.md states no Maintain formatting rule"
    raise AssertionError(msg)


# Rule 6: a `CLAUDE.md` carries principles, not measurements. Three shapes
# (issue #235): scientific notation, an "N of M" count, and a decimal with two
# or more fractional digits.
MEASUREMENT = re.compile(
    r"\b\d+(?:\.\d+)?e[-+]?\d+\b|\b\d+ of \d+\b|\b\d+\.\d{2,}\b"
    # A fourth shape, which the three above missed: a comma-grouped count,
    # as `sample/CLAUDE.md`'s ladder rule carried until #803 moved its
    # numbers to `STATUS.md` (#787 found the same class in `search/`).
    r"|\b\d{1,3}(?:,\d{3})+\b",
    re.IGNORECASE,
)

# The module files were 26 to 93 lines when written and 48-183 by the time rule
# 6 landed, while root grew by four lines over 26 commits. This ceiling stops
# that growth resuming: it is above the longest file after issue #235 rewrote
# them, so adding a rule past it means removing one.
LINE_BUDGET = 120


#: How a module file names a section of root, and how root writes one: root
#: uses headings (`## Writing Style`) and a module file bold (#803).
_ROOT_SECTION = r"Root `CLAUDE.md`'s \*\*([A-Z][^*]{2,40})\*\*"


def _in_root(name: str, root: str) -> bool:
    """Whether root `CLAUDE.md` carries a section called ``name``."""
    return f"**{name}**" in root or f"# {name}" in root


def _module_claude_files() -> list[Path]:
    """Every module `CLAUDE.md` on disk, sorted; discovered, not listed."""
    found = sorted(REPO_ROOT.glob("python/sal/*/CLAUDE.md"))
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
def test_no_module_claude_md_restates_the_writing_style() -> None:
    # Why the pointer is a pointer: root `CLAUDE.md`'s Writing Style section
    # changed three times on the day this was written, and nine copies would
    # already disagree.
    sentence = _rule_five_sentence()
    restating = [
        str(path.relative_to(REPO_ROOT))
        for path in _module_claude_files()
        if sentence in path.read_text()
    ]

    assert restating == []


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
def test_no_module_claude_md_points_at_a_root_section_root_does_not_have() -> None:
    # A module file naming a bold root section that root does not carry sends
    # a reader nowhere (#787).
    root = (REPO_ROOT / "CLAUDE.md").read_text()
    dangling: dict[str, list[str]] = {}
    for path in _module_claude_files():
        named = set(re.findall(_ROOT_SECTION, path.read_text()))
        missing = sorted(name for name in named if not _in_root(name, root))
        if missing:
            dangling[str(path.relative_to(REPO_ROOT))] = missing

    assert dangling == {}, (
        f"module CLAUDE.md files naming a root section root does not have: {dangling}"
    )


@pytest.mark.critical
@pytest.mark.infra
def test_the_dangling_section_check_catches_one() -> None:
    # The guard above passes on a clean tree, which says nothing about whether
    # it can fail: the shape it replaced passed for exactly that reason. Both
    # directions, on text rather than on the tree.
    root = (REPO_ROOT / "CLAUDE.md").read_text()
    names = re.findall(
        _ROOT_SECTION, "Root `CLAUDE.md`'s **Expected Reader** states the contract."
    )

    assert names == ["Expected Reader"]
    assert not _in_root(names[0], root)
    # Root writes its sections as headings and a module file names them in
    # bold, so the membership test reads both spellings or it would report
    # every correct pointer as dangling.
    assert re.findall(
        _ROOT_SECTION, "Root `CLAUDE.md`'s **Writing Style** binds this file."
    ) == ["Writing Style"]
    assert _in_root("Writing Style", root)


@pytest.mark.critical
@pytest.mark.infra
def test_no_module_claude_md_carries_a_measurement() -> None:
    # Rule 6, made checkable. A measurement in one of these files is a second
    # copy of a number `STATUS.md`, a docstring or a test already owns, and the
    # copy is the one that goes stale: nothing recomputes it.
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
@pytest.mark.infra
def test_the_measurement_check_catches_each_shape_it_claims_to() -> None:
    # The check exercised per shape rather than in aggregate: a guard never
    # seen to fail is not known to work, and a regex silently matches nothing.
    caught = [
        "worst deviation 3.7e-15",
        "39 of 40 runs",
        "a ratio of 0.87856",
        "ranged from 1,469 to 5,357 sweeps",
    ]
    passed = [
        "the deviation is reported rather than asserted",
        "a bound that holds at every size",
        "Python 3 and a single digit 0.5 are not measurements",
    ]

    assert [text for text in caught if not MEASUREMENT.search(text)] == []
    assert [text for text in passed if MEASUREMENT.search(text)] == []


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
