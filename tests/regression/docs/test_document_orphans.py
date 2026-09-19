"""The documents pay for a figure nothing reads (issue #492).

An orphan here is one-sided: a figure the QA manifest renders at the release
gate and no document cites. It is not asserted empty --- whether each should
exist belongs to its own ticket --- so what is asserted is that the detector
works, on a figure the manifest declares.

The second orphan this module reported, a problem statement with no box in
the release checklist (issue #495), went with the boxes: the release template
reads the textbook's sections by name rather than from a list, since the list
was already one short (Polar Codes, 2026-09-19) when issue #787 replaced it.

The join these two used to sit beside is gone with ``PROBLEMS.md``'s
inventory columns (issue #640). What it protected is not: the root
``CLAUDE.md`` splits the documents so the textbook can state an algorithm
without naming any code, and the last test here asserts that of the
hand-written sources, which is where a join was always the natural place to
break it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import document_orphans  # noqa: E402

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
