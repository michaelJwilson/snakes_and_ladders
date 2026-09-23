"""The gap-against-runtime figure both starts notebooks draw (`qa.starts.gap_panels`).

The referee for the band is `search.mixture_starts.gap_band`, which reads a
mixture trial: `curve_band` reads the same trials as `opt.starts.Curve`s and
must agree with it bitwise. The figure is pinned on what it draws, not on its
pixels: a panel per heading, a legend in final-gap order, and a start's
colour and line style held across panels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from snakes_and_ladders.opt.starts import Curve
from snakes_and_ladders.qa.starts import (
    RUNTIME_FLOOR,
    START_PALETTE,
    curve_band,
    gap_panels,
    start_styles,
)
from snakes_and_ladders.search.mixture_starts import GapBand, gap_band

GRID = np.geomspace(1e-3, 10.0, 50)


@pytest.mark.analytic
def test_a_curve_band_is_the_mixture_band_on_the_same_trials() -> None:
    reference = 10.0
    runs = [
        ([0.01, 0.1, 1.0], [2.0, 6.0, 9.5], 1),
        ([0.02, 0.3, 4.0], [1.0, 7.0, 10.5], 0),
        ([0.005, 0.5, 2.0], [3.0, 8.0, 9.9], 2),
    ]
    curves = [
        Curve(
            0,
            seed,
            np.asarray(times),
            np.asarray(values),
            reference - np.asarray(values),
            handover,
        )
        for seed, (times, values, handover) in enumerate(runs)
    ]
    trials = [
        _FakeTrial(tuple(zip(times, values, strict=True)), handover)
        for times, values, handover in runs
    ]
    expected = gap_band(trials, reference, GRID)  # type: ignore[arg-type]
    got = curve_band(curves, GRID)
    assert np.array_equal(got.mean, expected.mean, equal_nan=True)
    assert np.array_equal(got.std, expected.std, equal_nan=True)
    assert got.handover == expected.handover
    with pytest.raises(ValueError, match="at least one"):
        curve_band([], GRID)


class _FakeTrial:
    """What `gap_band` reads of a `Trial`: its curve and its handover index."""

    def __init__(self, curve: tuple[tuple[float, float], ...], handover: int) -> None:
        self.curve = curve
        self.handover = handover


@pytest.mark.analytic
def test_styles_take_one_colour_each_and_refuse_to_repeat_one() -> None:
    names = [f"s{i}" for i in range(len(START_PALETTE))]
    styles = start_styles(names)
    assert len(set(START_PALETTE)) == len(START_PALETTE) == 14
    assert [styles[n][0] for n in names] == list(START_PALETTE)
    assert {styles[n][1] for n in names} == {"-"}
    with pytest.raises(ValueError, match="split the figure"):
        start_styles([*names, "one more"])


@pytest.mark.smoke
def test_panels_order_their_legends_by_final_gap_and_hold_a_start_style() -> None:
    def band(level: float) -> GapBand:
        return GapBand(
            GRID, np.full(GRID.size, level), np.zeros(GRID.size), (0.1, level)
        )

    panels = {
        "left": {"a": band(5.0), "b": band(-2.0)},
        "right": {"c": band(3.0), "a": band(1.0)},
    }
    final = {"left": {"a": 5.0, "b": -2.0}, "right": {"c": 3.0, "a": 1.0}}
    styles = start_styles(["a", "b", "c"])
    figure = gap_panels(panels, final, styles, ylabel="gap")
    left, right = figure.axes
    legends = [axis.get_legend() for axis in (left, right)]
    assert legends[0] is not None
    assert legends[1] is not None
    assert [t.get_text() for t in legends[0].get_texts()] == ["b", "a"]
    assert [t.get_text() for t in legends[1].get_texts()] == ["a", "c"]
    assert legends[0].get_title().get_text() == "left"
    colour = {
        (axis, str(line.get_label())): line.get_color()
        for axis in (0, 1)
        for line in figure.axes[axis].get_lines()
        if not str(line.get_label()).startswith("_")
    }
    assert colour[(0, "a")] == colour[(1, "a")] == styles["a"][0]
    # Bands are off by default: no filled region is drawn.
    assert not left.collections
    # The runtime axis starts at the floor.
    assert left.get_xlim()[0] == RUNTIME_FLOOR
    plt.close(figure)
