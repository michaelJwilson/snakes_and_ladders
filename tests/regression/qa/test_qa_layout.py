"""The shared layouts and colour utilities, pinned on structure and arithmetic (issue #312).

A layout is checked for what it promises -- how many axes, where the gaps
fall, which axes are switched off, one legend handle per category -- and a
colour utility for the arithmetic it states; that a figure renders is not
asserted (root ``CLAUDE.md``'s no-coverage-theatre rule).
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from snakes_and_ladders.qa.layout import (
    TRACK_GAP,
    discrete_legend,
    format_track_axis,
    grouped_tracks,
    joint_distribution,
    marker_size,
    spatial_grid,
)
from snakes_and_ladders.qa.style import (
    INK,
    PALETTE,
    STATE_PALETTE,
    blend_with_white,
    discrete_palette,
    letter_style,
    notebook_style,
    with_opacity,
)


@pytest.mark.mathematical
def test_blend_with_white_interpolates_between_white_and_the_colour() -> None:
    assert blend_with_white("#0072B2", 0.0) == (1.0, 1.0, 1.0, 1.0)
    r, g, b, a = blend_with_white("#0072B2", 1.0)
    assert (round(r * 255), round(g * 255), round(b * 255), a) == (0, 114, 178, 1.0)
    half = blend_with_white("#000000", 0.5)
    assert half == (0.5, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        blend_with_white("#000000", 1.5)


@pytest.mark.mathematical
def test_with_opacity_keeps_the_colour_and_clips_the_alpha() -> None:
    rgba = with_opacity(["#000000", "#FFFFFF", "#0072B2"], [-0.5, 0.25, 7.0])
    np.testing.assert_allclose(rgba[:, 3], [0.0, 0.25, 1.0])
    np.testing.assert_allclose(rgba[0, :3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rgba[1, :3], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="colours but"):
        with_opacity(["#000000"], [0.1, 0.2])


@pytest.mark.structural
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


@pytest.mark.structural
def test_notebook_style_lowers_the_dpi_and_keeps_the_letter_face() -> None:
    with letter_style():
        letter_dpi = mpl.rcParams["figure.dpi"]
    with notebook_style():
        assert mpl.rcParams["figure.dpi"] == 150
        assert mpl.rcParams["font.family"] == ["serif"]
    assert letter_dpi == 200


@pytest.mark.mathematical
def test_marker_size_shrinks_with_density_inside_its_clamps() -> None:
    assert marker_size(1) == 25.0
    assert marker_size(1_200) == pytest.approx(10.0)
    assert marker_size(10**8) == 0.1
    with pytest.raises(ValueError, match="positive"):
        marker_size(0)


@pytest.mark.structural
def test_spatial_grid_gives_one_panel_per_feature_and_switches_the_rest_off() -> None:
    coords = np.array([[x, y] for x in range(3) for y in range(3)], dtype=float)
    features = {
        "a": np.arange(9.0),
        "b": np.zeros(9),
        "c": np.ones(9),
        "d": np.arange(9.0),
    }
    fig, axes = spatial_grid(coords, features, max_cols=3)
    try:
        assert len(axes) == 6
        assert [ax.get_title() for ax in axes[:4]] == ["a", "b", "c", "d"]
        assert not axes[4].axison
        assert not axes[5].axison
        # "b" is all zeros: hollow markers, no colourbar; every other panel has one.
        assert len(fig.axes) == 6 + 3
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="expected \\(9,\\)"):
        spatial_grid(coords, {"short": np.zeros(4)})
    with pytest.raises(ValueError, match="at least one feature"):
        spatial_grid(coords, {})


@pytest.mark.structural
def test_grouped_tracks_stacks_groups_with_a_gap_between_them() -> None:
    fig, axes = grouped_tracks(3, 2, title="tracks")
    try:
        assert len(axes) == 6
        tops = [ax.get_position().y1 for ax in axes]
        bottoms = [ax.get_position().y0 for ax in axes]
        assert tops == sorted(tops, reverse=True)
        within = bottoms[0] - tops[1]
        between = bottoms[1] - tops[2]
        assert within == pytest.approx(0.0, abs=1e-9)
        assert between == pytest.approx(TRACK_GAP * (tops[0] - bottoms[0]), rel=1e-6)
        format_track_axis(axes[0], "p", (0.0, 1.0), [0.0, 0.5, 1.0], max_x=10.0)
        assert axes[0].get_ylabel() == "p"
        assert axes[0].get_xlim() == (0.0, 10.0)
        assert list(axes[0].get_xticks()) == []
        assert len(axes[0].lines) == 3
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="positive counts"):
        grouped_tracks(0, 2)


@pytest.mark.structural
def test_joint_distribution_has_a_joint_panel_two_marginals_and_a_handle_per_group() -> (
    None
):
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 2.0 * x + rng.normal(size=200)
    groups = np.repeat([0, 1], 100)
    x[3] = np.nan
    fig, axes = joint_distribution(
        x, y, groups, discrete_palette(2), x_label="exact", y_label="surrogate"
    )
    try:
        assert set(axes) == {"joint", "top", "right"}
        assert len(fig.axes) == 3
        assert axes["joint"].get_xlabel() == "exact"
        legend = axes["joint"].get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["0", "1"]
        assert not axes["top"].axison
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="share a shape"):
        joint_distribution(x, y[:10], groups, {})
    with pytest.raises(ValueError, match="finite point"):
        joint_distribution(np.full(3, np.nan), np.zeros(3), np.zeros(3), {})


@pytest.mark.structural
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
