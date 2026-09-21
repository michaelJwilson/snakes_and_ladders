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

from collections import Counter
from pathlib import Path

import problems_tables
import pytest


@pytest.fixture(scope="module")
def generated() -> str:
    """What the generator writes, rendered once for the module."""
    return problems_tables.render()


@pytest.mark.infra
def test_the_generated_file_names_no_code(generated: str) -> None:
    # The textbook inputs this file, and `docs/CLAUDE.md` forbids a module
    # path, a filename or a function call in the textbook.
    text = generated
    assert [
        needle
        for needle in ("snakes_and_ladders.", ".py", "\\texttt{")
        if needle in text
    ] == []


#: The one row whose tests name no oracle the tables can place. Its claimed
#: oracle was a catalogue cell no test backed, which deriving the tables from
#: the suite is what exposed; issue #646 carries the gap. Named rather than
#: counted, so a second row joining it fails here.
WITHOUT_AN_ORACLE = ("Count-pair coupled spatio-sequential model",)


@pytest.mark.infra
def test_every_catalogue_row_reaches_an_algorithm_and_names_its_oracle_gap(
    generated: str,
) -> None:
    # Derived rather than claimed since issue #640: a row's methods are the
    # ones its own tests reach. A row reaching no oracle is a problem nothing
    # independent referees, so it is listed above and ticketed -- not asserted
    # away by a cell, which is how the one below went unnoticed.
    text = generated
    # Twice for the algorithm and oracle tables, once per method family, and
    # once per declared instance of every key the row names (issue #622).
    instances = Counter(
        title
        for problem, _, _ in problems_tables.fixture_rows()
        for title in problem.split("; ")
    )
    appearances = 2 + len(problems_tables.METHOD_FAMILIES)
    without = []
    for problem, symbols in problems_tables.rows():
        assert symbols, problem
        assert any(s in problems_tables.ALGORITHMS for s in symbols), problem
        if not any(s in problems_tables.ORACLES for s in symbols):
            without.append(problem)
        expected = appearances + instances[problem]
        assert text.count(problems_tables._tex_text(problem)) == expected, problem

    assert tuple(without) == WITHOUT_AN_ORACLE


@pytest.mark.infra
def test_every_problem_and_family_pairing_appears_once_per_family_table() -> None:
    # A pairing dropped rather than marked is the failure this table exists to
    # prevent: the reader cannot tell an untested pairing from one nobody
    # asked about.
    cells = problems_tables.method_cells()
    problems = [problem for problem, _ in problems_tables.rows()]

    assert len(cells) == len(problems) * len(problems_tables.METHOD_FAMILIES)
    for name in problems_tables.METHOD_FAMILIES:
        assert [problem for problem, family, *_ in cells if family == name] == problems


@pytest.mark.infra
def test_a_pairing_the_suite_does_not_pin_is_marked_untested(generated: str) -> None:
    # The catalogue carries a general time-reversible start that no test of
    # either significant kind names -- the standing example since #420 gave
    # the mixture's seeding an oracle. It must be marked and not omitted.
    marked = {
        (problem, family)
        for problem, family, _, referee, _ in problems_tables.method_cells()
        if referee == "untested"
    }
    text = generated

    assert marked, "no untested pairing: the mark itself is then unexercised"
    assert ("Phylogenetic tree, general time-reversible", "initializers") in marked
    assert text.count("untested") >= len(marked)


@pytest.mark.infra
def test_a_pairing_with_a_method_carries_a_note() -> None:
    # The note is the one hand-written cell; a pairing that exists and says
    # nothing about when it wins is the table half-written.
    for problem, family, _, referee, note in problems_tables.method_cells():
        if referee != "--":
            assert note, f"{problem} / {family}"


@pytest.mark.infra
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


@pytest.mark.infra
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


@pytest.mark.infra
def test_a_marked_test_keeps_its_tier_and_an_unmarked_one_takes_the_fixtures() -> None:
    # The tier column: the scheduling marker decides where there is one,
    # because that is what the selection obeys, and the fixtures the test
    # names decide where there is not.
    assert problems_tables.tier_of({"stress"}, {("tree_jc", "ci")}) == "stress"
    assert problems_tables.tier_of(set(), {("tree_search", "release")}) == "release"
    assert problems_tables.tier_of(set(), set()) == "ci"


@pytest.mark.infra
def test_every_experiment_a_note_cites_exists() -> None:
    # A renumbered or retracted experiment must break the generation rather
    # than leave the textbook pointing at a file that is not there.
    assert problems_tables.missing_experiments() == []


@pytest.mark.smoke
def test_a_note_citing_an_absent_experiment_is_refused() -> None:
    # Guards the guard.
    fabricated = {"Made up": {"optimizers": "wins here (experiment 999)"}}

    assert problems_tables.cited_experiments("experiment 4, then experiment 999") == [
        "004",
        "999",
    ]
    assert problems_tables.missing_experiments(fabricated) == ["999"]


@pytest.mark.smoke
def test_a_pairing_with_no_note_is_refused() -> None:
    # Guards the guard: the failure mode is a family that exists and says
    # nothing, which reads as a family that does not exist.
    with pytest.raises(problems_tables.MissingNoteError, match="initializers"):
        problems_tables.method_cells(note_map={})


@pytest.mark.smoke
def test_a_family_the_suite_runs_and_no_note_claims_is_refused() -> None:
    # Guards the guard, and the direction reversed with issue #640. The
    # catalogue no longer inventories a problem's methods, so it can no longer
    # name one the tables cannot place; what it can now do is go quiet about a
    # family that is *running*. The notes carry the claim and the suite the
    # coverage, so a family the suite reaches with no note must raise.
    silent = {
        problem: {
            name: note
            for name, note in families.items()
            if (problem, name) != ("Gaussian mixture", "optimizers")
        }
        for problem, families in problems_tables.notes().items()
    }

    with pytest.raises(problems_tables.MissingNoteError, match="Gaussian mixture"):
        problems_tables.method_cells(note_map=silent)


@pytest.mark.smoke
def test_a_row_reads_its_key_and_its_defining_code(tmp_path: Path) -> None:
    # The two hand-written columns, read from a table of one row. `Defines`
    # names code and fills no column of either table -- a simulator is not a
    # method -- and is read here so one reader parses the catalogue.
    #
    # The statement is a real label, and it has to be. An invented one reads
    # as a citation to `test_document_labels.py`, which fails any label no
    # document defines -- including one written only to explain this comment.
    catalogue = tmp_path / "PROBLEMS.md"
    catalogue.write_text(
        "| Problem | Key | Statement | Defines |\n| --- | --- | --- | --- |\n"
        "| Made up | `potts_chain`, `potts_lattice` | `sec:potts` | `sim.nothing` |\n"
    )

    assert problems_tables.catalogue_rows(catalogue) == [
        ("Made up", ["potts_chain", "potts_lattice"], ["sim.nothing"])
    ]


@pytest.mark.infra
def test_every_declared_instance_has_a_row() -> None:
    # The table cannot be narrower than the registry (issue #622, step 6). A
    # fixture added with no row would be an instance the document does not
    # know exists, which is the state `PROBLEMS.md` was written to end.
    declared = {
        (directory.name, path.stem)
        for directory in sorted(problems_tables.FIXTURES.iterdir())
        if directory.is_dir()
        for path in directory.glob("*.yaml")
    }
    rows = problems_tables.fixture_rows()
    titled = {
        key: "; ".join(
            title for title, keys, _ in problems_tables.catalogue_rows() if key in keys
        )
        for key, _ in declared
    }

    assert {(titled[key], tier) for key, tier in declared} == {
        (title, tier) for title, tier, _ in rows
    }
    assert len(rows) == len(declared), "two fixtures collapsed into one row"


@pytest.mark.infra
def test_a_declared_shape_never_collapses_to_one_number() -> None:
    # The column is a shape and not a size, which is the one thing it must not
    # become: the tiers of one problem differ in extent *and* state count, in
    # taxa *and* sites. A row naming a single field would read as a size.
    for title, tier, shape in problems_tables.fixture_rows():
        assert shape.count(";") >= 1, f"{title} / {tier} names one field: {shape}"


@pytest.mark.infra
def test_the_declared_shapes_name_no_code() -> None:
    # `docs/CLAUDE.md`: the document names no code. The field names are the
    # fixture file's own words, so a module that moves does not stale the
    # text; this asserts nothing else got in.
    for title, tier, shape in problems_tables.fixture_rows():
        for forbidden in ("snakes_and_ladders", ".py", ".yaml", "tests/"):
            assert forbidden not in shape, f"{title} / {tier} names code: {shape}"


@pytest.mark.smoke
def test_a_nested_declaration_is_left_to_the_file() -> None:
    # A transition matrix and a tree are the instance's parameters rather than
    # its extent; printing either would fill the page with numbers no reader
    # checks. Asserted on a declaration carrying both, so the rule is exercised
    # rather than described.
    shape = problems_tables.shape_cell(
        {
            "model": "hidden-markov",
            "seed": 1,
            "n_states": 3,
            "transition": [[0.5, 0.5], [0.5, 0.5]],
            "tau": {"name": "root"},
            "lengths": [15, 15, 15, 15, 15],
        }
    )

    assert "transition" not in shape
    assert "tau" not in shape
    assert "seed" not in shape
    assert "n\\_states 3" in shape
    assert "5 entries of 15" in shape
