"""The two documents pay for things nothing reads (issues #492, #495).

An orphan here is one-sided: a figure the QA manifest renders at the release
gate and no document cites, and a problem statement the textbook carries with
no box in the release checklist. Neither is asserted empty --- whether each
should exist belongs to its own ticket --- so what is asserted is that the
detector works, on a figure the manifest declares and a checklist that
parsed. An unparsed template would otherwise report every statement as
unboxed and read as a finding.

The join these two used to sit beside is gone with ``PROBLEMS.md``'s
inventory columns (issue #640). What it protected is not: the root
``CLAUDE.md`` splits the documents so the textbook can state an algorithm
without naming any code, and the last test here asserts that of the
hand-written sources, which is where a join was always the natural place to
break it.
"""

from __future__ import annotations

import document_orphans
import pytest

#: The textbook's own preamble and the sketch files it inputs are LaTeX the
#: statements read, so a module path in any of them is a module path in a
#: statement.
TEXTBOOK_SOURCES = ("textbook.tex", "notation.tex", "preamble.tex")


@pytest.mark.infra
def test_a_rendered_figure_no_document_cites_is_seen() -> None:
    # A figure the manifest renders, paid for at the release gate, and read
    # by nobody. Not asserted empty -- whether each should exist is issue #492's -- but
    # the detector is asserted to work, on a figure the manifest declares.
    rendered = document_orphans.rendered_figures()
    uncited = document_orphans.uncited_figures()

    assert rendered, "the manifest declares no figure"
    assert set(uncited) <= rendered
    assert set(uncited).isdisjoint(document_orphans.cited_figures())


@pytest.mark.infra
def test_a_statement_the_release_checklist_has_no_box_for_is_seen() -> None:
    # The second orphan shape. Reported and not gated: adding the boxes is the
    # release follow-up's. What is asserted is that the checklist was read at
    # all, since an unparsed template reports every statement as unboxed.
    boxed = document_orphans.checklist_labels()
    unchecked = document_orphans.unchecked_statements()

    assert boxed, "the release template's problem-statement block did not parse"
    assert set(unchecked) < set(document_orphans.statements().values())


@pytest.mark.critical
@pytest.mark.infra
def test_the_key_travels_one_way_and_the_textbook_names_no_code() -> None:
    # The constraint the whole design rests on (root `CLAUDE.md`, issue #249):
    # the catalogue may name the document, and the document may name no code.
    # `test_problems_tables.py` asserts it of the *generated* table; this
    # asserts it of the hand-written sources, which is where a join would have
    # been the natural place to put a module path.
    for name in TEXTBOOK_SOURCES:
        source = (document_orphans.TEX_DIR / name).read_text()
        assert "snakes_and_ladders" not in source, f"{name} names code"
