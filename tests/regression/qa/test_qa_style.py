"""Regression tests for the technical document's shared figure style.

Pins the properties the style exists to guarantee -- a fixed, never-cycled
palette; identity carried by more than colour; scoped rcParams -- rather than
that a figure renders (CLAUDE.md's no-coverage-theatre rule).
"""

from __future__ import annotations

import matplotlib as mpl
import pytest
from phylo.qa.style import (
    INK,
    INK_MUTED,
    LINESTYLES,
    MARKERS,
    PALETTE,
    RULE,
    letter_style,
    series_style,
    tolerance_note,
)


@pytest.mark.structural
def test_palette_is_the_validated_okabe_ito_order() -> None:
    # Pinned because the order is the validation result, not a preference:
    # re-ordering changes which pairs are adjacent and so which CVD
    # separation the checker measured. See the module docstring.
    assert PALETTE == ("#0072B2", "#009E73", "#D55E00", "#CC79A7")


@pytest.mark.structural
def test_secondary_encoding_is_index_matched_to_colour() -> None:
    # Identity must survive greyscale and colour-vision deficiency, which
    # requires a distinct marker and linestyle per colour, not per figure.
    assert len(MARKERS) == len(PALETTE)
    assert len(LINESTYLES) == len(PALETTE)
    assert len(set(MARKERS)) == len(MARKERS)
    assert len(set(LINESTYLES)) == len(LINESTYLES)


@pytest.mark.structural
@pytest.mark.parametrize("index", range(len(PALETTE)))
def test_series_style_pairs_each_colour_with_its_own_encoding(index: int) -> None:
    style = series_style(index)
    assert style["color"] == PALETTE[index]
    assert style["marker"] == MARKERS[index]
    assert style["linestyle"] == LINESTYLES[index]


@pytest.mark.edge_case
def test_series_style_refuses_to_invent_a_fifth_hue() -> None:
    with pytest.raises(IndexError, match="facet or aggregate"):
        series_style(len(PALETTE))


@pytest.mark.edge_case
def test_series_style_rejects_a_negative_index() -> None:
    with pytest.raises(IndexError, match="outside the"):
        series_style(-1)


@pytest.mark.structural
def test_ink_colours_are_distinct_from_every_series_colour() -> None:
    # Text wears ink, never a series hue; if an ink token collided with a
    # palette entry a label would read as a data mark.
    assert {INK, INK_MUTED, RULE}.isdisjoint(set(PALETTE))


@pytest.mark.structural
def test_letter_style_applies_serif_and_restores_on_exit() -> None:
    before = mpl.rcParams["font.family"]

    with letter_style():
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["axes.spines.top"] is False
        assert mpl.rcParams["pdf.fonttype"] == 42

    # Scoped, not global: importing the module must not change plotting
    # behaviour for anything else in the process.
    assert mpl.rcParams["font.family"] == before


@pytest.mark.structural
def test_tolerance_note_reports_the_realized_maximum() -> None:
    note = tolerance_note([1e-9, 4.2e-7, 3e-8], 1e-6)
    assert "4.20e-07" in note
    assert "1e-06" in note


@pytest.mark.edge_case
def test_tolerance_note_handles_no_deviations() -> None:
    assert "0.00e+00" in tolerance_note([], 1e-6)
