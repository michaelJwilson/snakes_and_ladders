"""The textbook's statements and `PROBLEMS.md`'s rows are one set (issue #495).

Two hand-maintained descriptions of one set of problems, joined on the
textbook section's own ``\\label``. Before this guard nothing held them to
each other and they had drifted to eleven statements against fifteen rows,
which nobody could see because seeing it meant reading both documents and
knowing which vocabulary matched which.

The join fails in two directions and they are separate defects: a statement
no row is code for is a model the tree does not implement, and a row keyed to
no statement is code for a problem the textbook never states. A key carrying
several rows is neither --- it is a variant, and the multiplicity is what
makes the granularity a decision, since a new row must either choose an
existing key or write the statement that gives it a new one.

What the join is *not* allowed to do is close the loop the other way. The root
``CLAUDE.md`` splits the documents so the textbook can state an algorithm
without naming any code, so the key travels from the catalogue to the document
and nothing travels back; the last test here asserts that, because it is the
constraint every later part of this ticket rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import problem_join  # noqa: E402

#: The textbook's own preamble and the sketch files it inputs are LaTeX the
#: statements read, so a module path in any of them is a module path in a
#: statement.
TEXTBOOK_SOURCES = ("textbook.tex", "notation.tex", "preamble.tex")


def _catalogue(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "PROBLEMS.md"
    path.write_text("| Problem | Statement | Simulate |\n| --- | --- | --- |\n" + rows)
    return path


def _textbook(tmp_path: Path, sections: str) -> Path:
    path = tmp_path / "textbook.tex"
    path.write_text(sections)
    return path


@pytest.mark.critical
@pytest.mark.structural
def test_every_problem_statement_is_keyed_by_a_catalogue_row() -> None:
    # A statement nothing in the tree is code for. The textbook would read as
    # a description of this repository while describing something else.
    missing_rows, _ = problem_join.orphans()

    assert missing_rows == [], (
        f"problem statements no PROBLEMS.md row is keyed to: {missing_rows}"
    )


@pytest.mark.critical
@pytest.mark.structural
def test_every_catalogue_row_is_keyed_to_a_problem_statement() -> None:
    # The other direction: code for a problem the textbook never states, which
    # is how three of the four variant rows arrived unremarked.
    _, missing_statements = problem_join.orphans()

    assert missing_statements == [], (
        f"PROBLEMS.md rows keyed to no problem statement: {missing_statements}"
    )


@pytest.mark.structural
def test_the_join_covers_every_row_and_every_statement() -> None:
    # The join is only worth its runtime if it reads both in full: a parser
    # that silently dropped a row would pass both directions above.
    grouped = problem_join.variants()

    assert set(grouped) == set(problem_join.statements())
    assert sum(len(titles) for titles in grouped.values()) == len(
        problem_join.registry()
    )


@pytest.mark.edge_case
def test_an_orphaned_statement_fails_the_join_and_is_named(tmp_path: Path) -> None:
    # Guards the guard. Two statements, one row: the unkeyed statement must be
    # returned by name rather than merely making a count disagree.
    textbook = _textbook(
        tmp_path,
        "\\section{Problem Statement: A}\n\\label{sec:aaa}\n"
        "\\section{Problem Statement: B}\n\\label{sec:bbb}\n",
    )
    catalogue = _catalogue(tmp_path, "| Row A | sec:aaa | `x` |\n")

    assert problem_join.orphans(catalogue, textbook) == (["sec:bbb"], [])


@pytest.mark.edge_case
def test_an_orphaned_row_fails_the_join_and_is_named(tmp_path: Path) -> None:
    # And the reverse, including the unkeyed row: an empty Statement cell is a
    # row naming no statement, not a row exempt from the join.
    textbook = _textbook(
        tmp_path, "\\section{Problem Statement: A}\n\\label{sec:aaa}\n"
    )
    catalogue = _catalogue(
        tmp_path,
        "| Row A | sec:aaa | `x` |\n| Row B | sec:zzz | `y` |\n| Row C |  | `z` |\n",
    )

    assert problem_join.orphans(catalogue, textbook) == ([], ["Row B", "Row C"])


@pytest.mark.edge_case
def test_rows_sharing_a_key_are_variants_and_not_a_failure(tmp_path: Path) -> None:
    # The granularity decision: four keys carry two rows each in the tree, and
    # that must group rather than fail, or every variant becomes a statement.
    textbook = _textbook(
        tmp_path, "\\section{Problem Statement: A}\n\\label{sec:aaa}\n"
    )
    catalogue = _catalogue(
        tmp_path, "| Row A | sec:aaa | `x` |\n| Row A prime | sec:aaa | `y` |\n"
    )

    assert problem_join.orphans(catalogue, textbook) == ([], [])
    assert problem_join.variants(catalogue, textbook) == {
        "sec:aaa": ["Row A", "Row A prime"]
    }


@pytest.mark.structural
def test_the_join_sees_a_rendered_figure_no_document_cites() -> None:
    # The same orphaning from the other end, and the reason the report exists:
    # a figure is rendered, paid for at the release gate, and read by nobody.
    # Not asserted empty -- whether each should exist is issue #492's -- but
    # the detector is asserted to work, on a figure the manifest declares.
    rendered = problem_join.rendered_figures()
    uncited = problem_join.uncited_figures()

    assert rendered, "the manifest declares no figure"
    assert set(uncited) <= rendered
    assert set(uncited).isdisjoint(problem_join.cited_figures())


@pytest.mark.structural
def test_the_join_sees_a_statement_the_release_checklist_has_no_box_for() -> None:
    # The third orphan shape. Reported and not gated: adding the boxes is the
    # release follow-up's. What is asserted is that the checklist was read at
    # all, since an unparsed template reports every statement as unboxed.
    boxed = problem_join.checklist_labels()
    unchecked = problem_join.unchecked_statements()

    assert boxed, "the release template's problem-statement block did not parse"
    assert set(unchecked) < set(problem_join.statements().values())


@pytest.mark.critical
@pytest.mark.structural
def test_the_key_travels_one_way_and_the_textbook_names_no_code() -> None:
    # The constraint the whole design rests on (root `CLAUDE.md`, issue #249):
    # the catalogue may name the document, and the document may name no code.
    # Asserted here as well as in the citation guards because this ticket is
    # what would break it -- a join is the natural place to put a module path.
    for name in TEXTBOOK_SOURCES:
        source = (problem_join.TEX_DIR / name).read_text()
        assert "snakes_and_ladders" not in source, f"{name} names code"


@pytest.mark.edge_case
def test_the_check_exits_nonzero_on_an_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A guard whose failure path was never run is a guard nobody has checked.
    textbook = _textbook(
        tmp_path,
        "\\section{Problem Statement: A}\n\\label{sec:aaa}\n"
        "\\section{Problem Statement: B}\n\\label{sec:bbb}\n",
    )
    monkeypatch.setattr(problem_join, "TEXTBOOK", textbook)
    monkeypatch.setattr(
        problem_join, "CATALOGUE", _catalogue(tmp_path, "| Row A | sec:aaa | `x` |\n")
    )

    assert problem_join.main(["--check"]) == 1
