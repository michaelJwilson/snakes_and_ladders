"""The two documents pay for things nothing reads (issues #492, #495).

An orphan is a figure the QA manifest renders and no document cites, or a
problem statement with no box in the release checklist. Neither is asserted
empty; what is asserted is that the detector works, on a declared figure and a
parsed checklist. The last test asserts the textbook's hand-written sources
name no code (root ``CLAUDE.md``; issue #640).
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
    # The textbook may name no code (root `CLAUDE.md`, issue #249); this holds
    # the hand-written sources to it, `test_problems_tables.py` the generated
    # table. Needles: a module path, a filename, a typeset identifier (#982).
    for name in TEXTBOOK_SOURCES:
        source = (document_orphans.TEX_DIR / name).read_text()
        offenders = [
            needle
            for needle in ("snakes_and_ladders", ".py", "\\texttt{")
            if needle in source
        ]
        assert offenders == [], f"{name} names code: {offenders}"
