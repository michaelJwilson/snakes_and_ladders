"""Regression tests for the documents' shared figure style.

Pins the properties the style exists to guarantee -- a fixed, never-cycled
palette; identity carried by more than colour; scoped rcParams -- rather than
that a figure renders (CLAUDE.md's no-coverage-theatre rule).
"""

from __future__ import annotations

import matplotlib as mpl
import pytest
from sal.qa.style import (
    INK,
    INK_MUTED,
    LINESTYLES,
    MARKERS,
    PALETTE,
    RULE,
    letter_style,
    series_style,
)

from tests._rows import every_value


@pytest.mark.infra
def test_palette_is_the_validated_okabe_ito_order() -> None:
    # Pinned because the order is the validation result, not a preference:
    # re-ordering changes which pairs are adjacent and so which CVD
    # separation the checker measured. See the module docstring.
    assert PALETTE == ("#0072B2", "#009E73", "#D55E00", "#CC79A7")


@pytest.mark.infra
def test_secondary_encoding_is_index_matched_to_colour() -> None:
    # Identity must survive greyscale and colour-vision deficiency, which
    # requires a distinct marker and linestyle per colour, not per figure.
    assert len(MARKERS) == len(PALETTE)
    assert len(LINESTYLES) == len(PALETTE)
    assert len(set(MARKERS)) == len(MARKERS)
    assert len(set(LINESTYLES)) == len(LINESTYLES)


@pytest.mark.infra
def test_series_style_pairs_each_colour_with_its_own_encoding() -> None:
    def check(index: int) -> None:
        style = series_style(index)
        assert style["color"] == PALETTE[index]
        assert style["marker"] == MARKERS[index]
        assert style["linestyle"] == LINESTYLES[index]

    every_value(range(len(PALETTE)), check)


@pytest.mark.smoke
def test_series_style_refuses_to_invent_a_fifth_hue() -> None:
    with pytest.raises(IndexError, match="facet or aggregate"):
        series_style(len(PALETTE))


@pytest.mark.smoke
def test_series_style_rejects_a_negative_index() -> None:
    with pytest.raises(IndexError, match="outside the"):
        series_style(-1)


@pytest.mark.infra
def test_ink_colours_are_distinct_from_every_series_colour() -> None:
    # Text wears ink, never a series hue; if an ink token collided with a
    # palette entry a label would read as a data mark.
    assert {INK, INK_MUTED, RULE}.isdisjoint(set(PALETTE))


@pytest.mark.infra
def test_letter_style_applies_serif_and_restores_on_exit() -> None:
    before = mpl.rcParams["font.family"]

    with letter_style():
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["axes.spines.top"] is False
        assert mpl.rcParams["pdf.fonttype"] == 42

    # Scoped, not global: importing the module must not change plotting
    # behaviour for anything else in the process.
    assert mpl.rcParams["font.family"] == before
