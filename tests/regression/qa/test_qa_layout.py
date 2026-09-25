"""The shared layout and the colour utilities, pinned on structure and arithmetic (issue #312).

A layout is checked for what it promises -- one legend handle per category
-- and a colour utility for the arithmetic it states; that a figure renders
is not asserted (root ``CLAUDE.md``'s no-coverage-theatre rule). The four
layouts no figure called went with their tests in issue #864.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pytest
from sal.qa.layout import (
    discrete_legend,
)
from sal.qa.style import (
    INK,
    PALETTE,
    STATE_PALETTE,
    blend_with_white,
    discrete_palette,
    letter_style,
    notebook_style,
)


@pytest.mark.analytic
def test_blend_with_white_interpolates_between_white_and_the_colour() -> None:
    assert blend_with_white("#0072B2", 0.0) == (1.0, 1.0, 1.0, 1.0)
    r, g, b, a = blend_with_white("#0072B2", 1.0)
    assert (round(r * 255), round(g * 255), round(b * 255), a) == (0, 114, 178, 1.0)
    half = blend_with_white("#000000", 0.5)
    assert half == (0.5, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        blend_with_white("#000000", 1.5)


@pytest.mark.infra
def test_state_palette_is_eight_distinct_colours_apart_from_the_series_palette() -> (
    None
):
    assert len(STATE_PALETTE) == 8
    assert len(set(STATE_PALETTE)) == 8
    assert len(PALETTE) == 4
    palette = discrete_palette(3, highlight=1)
    assert palette == {0: STATE_PALETTE[0], 1: INK, 2: STATE_PALETTE[2]}
    with pytest.raises(IndexError, match="facet"):
        discrete_palette(9)
    with pytest.raises(IndexError, match="not one of the"):
        discrete_palette(2, highlight=5)


@pytest.mark.infra
def test_notebook_style_lowers_the_dpi_and_keeps_the_letter_face() -> None:
    with letter_style():
        letter_dpi = mpl.rcParams["figure.dpi"]
    with notebook_style():
        assert mpl.rcParams["figure.dpi"] == 150
        assert mpl.rcParams["font.family"] == ["serif"]
    assert letter_dpi == 200


@pytest.mark.infra
def test_discrete_legend_names_every_category_by_its_colour() -> None:
    fig, ax = plt.subplots()
    try:
        discrete_legend(ax, ["A", "B", "C"], list(STATE_PALETTE[:3]))
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["A", "B", "C"]
        assert len(legend.legend_handles) == 3
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="categories but"):
        discrete_legend(ax, ["A"], ["#000000", "#FFFFFF"])
