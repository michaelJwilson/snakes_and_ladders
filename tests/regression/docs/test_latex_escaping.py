"""The two LaTeX escapers are one table, and the caption guard is its inverse.

Six escapers disagreed on LaTeX syntax characters (issue #863); `infra/tex.py`
holds the table and `qa.figure` keeps a copy, since the package imports
nothing from ``infra/`` (``infra/CLAUDE.md``). This module fails on a
disagreement. `qa.figure.check_latex_safe` is not the exact inverse ---
``\\_`` is the one sequence a hand-written caption may carry --- so the
asserted containment is: every character the guard refuses is one the
escaper neutralises.
"""

from __future__ import annotations

import pytest
import tex
from snakes_and_ladders.qa.figure import (
    _LATEX_SPECIALS,
    _THOUSANDS,
    LATEX_ESCAPES,
    check_latex_safe,
    latex_escape,
)


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
