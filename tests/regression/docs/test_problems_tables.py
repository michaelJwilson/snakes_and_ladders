"""The textbook's applicability tables are what ``PROBLEMS.md`` generates (issue #358).

The file is generated and not committed (issue #425), so what is checked is
what the tool writes: the textbook names no code, every catalogue row reaches
the table, and a catalogue symbol the generator cannot name fails rather than
dropping out of it.

Since issue #376 the file carries a third table, in four parts, pairing each
problem with each method family. Two things there can go wrong quietly and
are checked: a pairing nothing tests could be dropped instead of marked, so
the table would read as though the question had not been asked; and a note
could cite an experiment a later pull request renumbered or retracted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import problems_tables  # noqa: E402


@pytest.fixture(scope="module")
def generated() -> str:
    """What the generator writes, rendered once for the module."""
    return problems_tables.render()


@pytest.mark.structural
def test_the_generated_file_names_no_code(generated: str) -> None:
    # The textbook inputs this file, and `docs/CLAUDE.md` forbids a module
    # path, a filename or a function call in the textbook.
    text = generated
    assert [
        needle
        for needle in ("snakes_and_ladders.", ".py", "\\texttt{")
        if needle in text
    ] == []


@pytest.mark.structural
def test_every_catalogue_row_has_an_algorithm_and_an_oracle(generated: str) -> None:
    # A row with no oracle would be a problem nothing referees, which the
    # catalogue's own preamble forbids.
    text = generated
    # Twice for the algorithm and oracle tables, once per method family.
    appearances = 2 + len(problems_tables.METHOD_FAMILIES)
    for problem, symbols in problems_tables.rows():
        assert problems_tables.unnamed(symbols) == [], problem
        assert any(s in problems_tables.ORACLES for s in symbols), problem
        assert text.count(problems_tables._tex_text(problem)) == appearances, problem


@pytest.mark.structural
def test_every_problem_and_family_pairing_appears_once_per_family_table() -> None:
    # A pairing dropped rather than marked is the failure this table exists to
    # prevent: the reader cannot tell an untested pairing from one nobody
    # asked about.
    cells = problems_tables.method_cells()
    problems = [problem for problem, _ in problems_tables.rows()]

    assert len(cells) == len(problems) * len(problems_tables.METHOD_FAMILIES)
    for name in problems_tables.METHOD_FAMILIES:
        assert [problem for problem, family, *_ in cells if family == name] == problems


@pytest.mark.structural
def test_a_pairing_the_suite_does_not_pin_is_marked_untested(generated: str) -> None:
    # The catalogue carries a Gaussian-mixture initializer and a general
    # time-reversible start that no test of either significant kind names.
    # They are the standing examples, and each must be marked and not omitted.
    marked = {
        (problem, family)
        for problem, family, _, referee, _ in problems_tables.method_cells()
        if referee == "untested"
    }
    text = generated

    assert marked, "no untested pairing: the mark itself is then unexercised"
    assert ("Gaussian mixture", "initializers") in marked
    assert text.count("untested") >= len(marked)


@pytest.mark.structural
def test_a_pairing_with_a_method_carries_a_note() -> None:
    # The note is the one hand-written cell; a pairing that exists and says
    # nothing about when it wins is the table half-written.
    for problem, family, _, referee, note in problems_tables.method_cells():
        if referee != "--":
            assert note, f"{problem} / {family}"


@pytest.mark.structural
def test_every_untested_pairing_states_why_it_is_untested() -> None:
    # "Every compatible method is applied to every supported problem" is a
    # claim, and this is where it is checked rather than reviewed: a fixture
    # and a family the catalogue pairs, with no significant test naming both,
    # is listed with the reason it is not tested. A pairing loses its entry
    # only by gaining a test (issue #382).
    pairs = [
        f"{problem} / {name}" for problem, name in problems_tables.untested_pairs()
    ]
    stated = problems_tables.untested_notes()

    assert sorted(stated) == pairs
    assert all(reason.strip() for reason in stated.values())


@pytest.mark.structural
def test_a_fixture_is_read_from_a_call_or_from_a_path() -> None:
    # The reading behind the table above: a test names its instance either
    # through the registry or by path, and both count, or a pairing would
    # read as untested because of how the test spells the fixture.
    assert problems_tables.fixtures_named('fixture("mixture", "ci")') == {
        ("mixture", "ci")
    }
    assert problems_tables.fixtures_named('at_fixture("x", "tree_search")') == {
        ("tree_search", "")
    }
    assert problems_tables.fixtures_named('load("tree_jc/release.yaml")') == {
        ("tree_jc", "release")
    }


@pytest.mark.structural
def test_a_marked_test_keeps_its_tier_and_an_unmarked_one_takes_the_fixtures() -> None:
    # The tier column: the scheduling marker decides where there is one,
    # because that is what the selection obeys, and the fixtures the test
    # names decide where there is not.
    assert problems_tables.tier_of({"stress"}, {("tree_jc", "ci")}) == "stress"
    assert problems_tables.tier_of(set(), {("tree_search", "release")}) == "release"
    assert problems_tables.tier_of(set(), set()) == "ci"


@pytest.mark.structural
def test_every_experiment_a_note_cites_exists() -> None:
    # A renumbered or retracted experiment must break the generation rather
    # than leave the textbook pointing at a file that is not there.
    assert problems_tables.missing_experiments() == []


@pytest.mark.edge_case
def test_a_note_citing_an_absent_experiment_is_refused() -> None:
    # Guards the guard.
    fabricated = {"Made up": {"optimizers": "wins here (experiment 999)"}}

    assert problems_tables.cited_experiments("experiment 4, then experiment 999") == [
        "004",
        "999",
    ]
    assert problems_tables.missing_experiments(fabricated) == ["999"]


@pytest.mark.edge_case
def test_a_pairing_with_no_note_is_refused() -> None:
    # Guards the guard: the failure mode is a family that exists and says
    # nothing, which reads as a family that does not exist.
    with pytest.raises(problems_tables.MissingNoteError, match="initializers"):
        problems_tables.method_cells(note_map={})


@pytest.mark.edge_case
def test_a_symbol_the_generator_cannot_name_is_refused(tmp_path: Path) -> None:
    # Guards the guard: the failure mode is a table quietly narrower than the
    # catalogue, so an unnamed symbol must raise rather than be skipped.
    catalogue = tmp_path / "PROBLEMS.md"
    catalogue.write_text(
        "| Problem | Simulate |\n| --- | --- |\n"
        "| Made up | `snakes_and_ladders.no_such.symbol` |\n"
    )
    with pytest.raises(problems_tables.UnnamedSymbolError, match="no_such.symbol"):
        problems_tables.render(catalogue)
    assert problems_tables.rows(catalogue) == [("Made up", ["no_such.symbol"])]
