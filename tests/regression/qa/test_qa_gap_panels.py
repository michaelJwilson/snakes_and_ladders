"""The gap-against-runtime figure both starts notebooks draw (`qa.starts.gap_panels`).

The band has one reader, `search.mixture_starts.curve_band`, and
`gap_band` reads a mixture trial through it (issue #926); the loop `gap_band`
ran before the fold is kept here as the referee, bitwise. The figure is pinned on what it draws, not on its
pixels: a panel per heading, a legend in final-gap order, and a start's
colour and line style held across panels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from snakes_and_ladders.qa.starts import (
    RUNTIME_FLOOR,
    START_PALETTE,
    gap_panels,
    start_styles,
)
from snakes_and_ladders.search.mixture_starts import GapBand, curve_band, gap_band

GRID = np.geomspace(1e-3, 10.0, 50)


def _held_band(
    trials: list[_FakeTrial], reference: float, seconds: np.ndarray
) -> GapBand:
    """`gap_band` as it read a mixture trial before #926 routed it through `curve_band`: the oracle."""
    held = np.full((len(trials), seconds.shape[0]), np.nan)
    for row, trial in enumerate(trials):
        times = np.asarray([point[0] for point in trial.curve])
        gaps = reference - np.asarray([point[1] for point in trial.curve])
        index = np.searchsorted(times, seconds, side="right") - 1
        known = index >= 0
        held[row, known] = gaps[index[known]]
    started = ~np.isnan(held).any(axis=0)
    mean = np.full(seconds.shape[0], np.nan)
    std = np.full(seconds.shape[0], np.nan)
    mean[started] = held[:, started].mean(axis=0)
    if len(trials) > 1:
        std[started] = held[:, started].std(axis=0, ddof=1)
    handover = (
        float(np.mean([t.curve[t.handover][0] for t in trials])),
        float(np.mean([reference - t.curve[t.handover][1] for t in trials])),
    )
    return GapBand(seconds, mean, std, handover)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(4))
def test_the_one_band_reader_is_the_mixture_band_it_replaced(seed: int) -> None:
    # Issue #926 folded `gap_band` onto `curve_band`; the loop it replaced is
    # kept above as the referee, and the band is bitwise it on random trials,
    # one to five of them, some starting after the grid does.
    rng = np.random.default_rng([926, seed])
    reference = float(rng.normal(10.0, 1.0))
    trials = []
    for _ in range(int(rng.integers(1, 6))):
        size = int(rng.integers(2, 12))
        times = np.sort(rng.uniform(1e-3, 8.0, size))
        values = reference - rng.exponential(2.0, size)
        trials.append(
            _FakeTrial(
                tuple(zip(times.tolist(), values.tolist(), strict=True)),
                int(rng.integers(0, size)),
            )
        )
    expected = _held_band(trials, reference, GRID)
    got = gap_band(trials, reference, GRID)  # type: ignore[arg-type]
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
