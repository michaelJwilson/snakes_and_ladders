"""The two LaTeX escapers are one table, and the caption guard is its inverse.

Six escapers disagreed about which characters LaTeX reads as syntax, and the
generators that used the short ones would have written a document
``pdflatex`` refuses --- or, for ``_``, one it typesets as a subscript
without complaint (issue #863). `infra/tex.py` holds the table now.

`python/snakes_and_ladders/qa/` cannot import it: ``infra/`` carries no
application reference and the package depends on nothing in it
(``infra/CLAUDE.md``), so `qa.figure` keeps a copy. A copy is one definition
only while something fails on a disagreement, which is this module.

`qa.figure.check_latex_safe` reads the same alphabet from the other side: it
refuses a caption that arrives unescaped. It is not the exact inverse ---
``\\&`` is what `escape` emits and what the guard refuses, because a caption
is written by hand and ``\\_`` is the one sequence it is allowed --- so what
is asserted is the containment that makes the pair sound: every character the
guard refuses is one the escaper neutralises.
"""

from __future__ import annotations

import sys

import pytest
from snakes_and_ladders.qa.figure import (
    _LATEX_SPECIALS,
    _THOUSANDS,
    LATEX_ESCAPES,
    check_latex_safe,
    latex_escape,
)

from tests._paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "infra"))

import tex


@pytest.mark.critical
@pytest.mark.infra
def test_the_two_escape_tables_are_the_same_table() -> None:
    # Byte equality of the mapping, not of a sample: a character present in
    # one table and absent from the other is exactly the defect the six
    # escapers had, and a sample of the characters that happen to appear in
    # today's fixture names would not see it.
    assert LATEX_ESCAPES == tex.ESCAPES


@pytest.mark.critical
@pytest.mark.infra
def test_the_two_escapers_agree_character_for_character() -> None:
    # The tables read the same way as well as holding the same pairs.
    for character in tex.ESCAPES:
        assert latex_escape(character) == tex.escape(character)

    mixed = "".join(tex.ESCAPES) + "tree_search & 100% {done}"
    assert latex_escape(mixed) == tex.escape(mixed)


@pytest.mark.critical
@pytest.mark.infra
def test_the_caption_guard_refuses_only_what_the_escaper_neutralises() -> None:
    # The inverse direction: a character the guard refuses is one the escaper
    # has an entry for, so a caption built through `latex_escape` cannot carry
    # one the guard would then reject for a reason the escaper knows nothing
    # about.
    unhandled = sorted(
        character for character in _LATEX_SPECIALS if tex.escape(character) == character
    )

    assert unhandled == []

    # And the one sequence the guard permits is the one the escaper emits for
    # it, which is what lets a rendered integer through.
    assert tex.escape("_") == _THOUSANDS
    check_latex_safe(latex_escape("200_000"))
