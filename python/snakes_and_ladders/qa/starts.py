"""The one figure and the one table a benchmark of starts renders (issue #894).

Every consumer of :class:`snakes_and_ladders.opt.starts.SolverComparison`
draws through :func:`starts_figure` and sets its table through
:func:`starts_latex`, so the notebook, the QA figures and an experiment's
figure are one renderer. They live here and not on the result: `qa/` imports
`opt/`, and the converse would be a second import cycle
(`tests/regression/test_directory_imports.py`).

**Left**, the gap below the reference in nats against wall seconds, one curve
per start, on a log axis clipped at :data:`GAP_FLOOR`; a marker at the
hand-over from seeding to polish, and the reference drawn at the floor. At one
trial every point is a tracked iteration. At more, the trials' curves are
interpolated onto :data:`GRID_POINTS` evenly spaced seconds per start, since
each trial's iterations land at other times, and the mean is drawn with a band
of one standard deviation. **Right**, the gap reached at the budget per start
as horizontal bars in the table's order, with one standard deviation at more
than one trial.

Renders what the seam computed and recomputes nothing (`qa/CLAUDE.md`).

:func:`gap_panels` is the gap-against-runtime figure both starts notebooks draw
(`emission_mixture_starts`, `potts_starts`): panels side by side on shared
axes, one colour and line style per start from :data:`START_PALETTE` held
across panels, a diamond at the mean handover, the gap on a symmetric-log axis,
spread bands off unless asked for, and each panel's legend bottom left in one
column from the lowest final gap to the highest. :func:`curve_band` reads the
trials of an :class:`~snakes_and_ladders.opt.starts.SolverComparison` onto the
:class:`~snakes_and_ladders.search.mixture_starts.GapBand` the figure takes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.opt.starts import Curve, SolverComparison
from snakes_and_ladders.qa.figure import check_latex_safe, latex_escape
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    LINESTYLES,
    STATE_PALETTE,
    blend_with_white,
    discrete_palette,
    notebook_style,
)
from snakes_and_ladders.search.mixture_starts import GapBand

#: The smallest gap drawn, in nats: a log axis has no zero, and a start that
#: reaches the reference or passes it is drawn here.
GAP_FLOOR = 1.0e-3

#: Seconds a start's trials are interpolated onto, evenly spaced from its
#: earliest tracked iteration to its latest.
GRID_POINTS = 100

#: Sized to sit whole in a notebook cell.
FIGSIZE = (10.0, 3.8)

#: Ten colours for starts traced along a runtime axis: the eight of
#: Okabe--Ito with black (`qa.style.STATE_PALETTE` and the ink it omits), and
#: wine from Tol's muted set. Past ten a colour repeats dashed, so a start is
#: its (colour, line style) pair.
START_PALETTE: tuple[str, ...] = ("#000000", *STATE_PALETTE, "#882255")

#: The runtime figure's size: two panels side by side.
GAP_FIGSIZE = (12.0, 4.8)


def _encoding(n_starts: int) -> list[tuple[str, str]]:
    """``(colour, linestyle)`` per start.

    The colours are :func:`~snakes_and_ladders.qa.style.discrete_palette`'s;
    past its eight a colour repeats with the next linestyle, so the pair
    carries identity where the hue alone does not.
    """
    palette = discrete_palette(min(n_starts, len(STATE_PALETTE)))
    return [
        (palette[index % len(palette)], LINESTYLES[(index // len(palette)) % 4])
        for index in range(n_starts)
    ]


def _finite(curve: Curve) -> Curve:
    """``curve`` without the entries whose value is not finite (a refused polish)."""
    kept = np.isfinite(curve.gaps)
    handover = int(np.count_nonzero(kept[: curve.handover + 1])) - 1
    return Curve(
        curve.instance,
        curve.seed,
        curve.seconds[kept],
        curve.values[kept],
        curve.gaps[kept],
        max(handover, 0),
    )


def _mean_curve(
    curves: tuple[Curve, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The trials on one grid: the seconds, the mean gap, and its standard deviation.

    A trial is absent before its first entry and holds its last value after
    its end.
    """
    grid = np.linspace(
        min(float(curve.seconds[0]) for curve in curves),
        max(float(curve.seconds[-1]) for curve in curves),
        GRID_POINTS,
    )
    stacked = np.stack(
        [
            np.interp(
                grid, curve.seconds, curve.gaps, left=np.nan, right=curve.gaps[-1]
            )
            for curve in curves
        ]
    )
    present = np.isfinite(stacked)
    counts = present.sum(axis=0)
    filled = np.where(present, stacked, 0.0)
    mean = filled.sum(axis=0) / np.maximum(counts, 1)
    spread = np.sqrt(
        np.where(present, (stacked - mean) ** 2, 0.0).sum(axis=0)
        / np.maximum(counts, 1)
    )
    return grid[counts > 0], mean[counts > 0], spread[counts > 0]


def _floored(values: np.ndarray) -> np.ndarray:
    """``values`` clipped at :data:`GAP_FLOOR` from below."""
    return np.asarray(np.maximum(values, GAP_FLOOR), dtype=np.float64)


def starts_figure(
    result: SolverComparison, *, reference_label: str
) -> tuple[Figure, str]:
    """The standard two-panel figure of a benchmark of starts, and its caption.

    Parameters
    ----------
    result : SolverComparison
        What :meth:`snakes_and_ladders.opt.starts.StartsBenchmark.run` returned.
    reference_label : str
        What the reference is, in plain text: ``"the best any start reached"``.

    Returns
    -------
    tuple[Figure, str]
        The figure and a caption :func:`~snakes_and_ladders.qa.figure.check_latex_safe` passes.
    """
    curves = {
        name: tuple(_finite(curve) for curve in trials if np.isfinite(curve.gaps).any())
        for name, trials in result.curves().items()
    }
    names = result.names
    encoding = _encoding(len(names))
    n_trials = max(len(result.trials(name)) for name in names)
    budget = result.comparison.budget
    with notebook_style():
        fig, (left, right) = plt.subplots(
            1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": (1.6, 1.0)}
        )
        for (colour, linestyle), name in zip(encoding, names, strict=True):
            drawn = curves[name]
            if not drawn:
                continue
            if len(drawn) == 1:
                seconds, gaps = drawn[0].seconds, _floored(drawn[0].gaps)
                left.plot(
                    seconds,
                    gaps,
                    color=colour,
                    linestyle=linestyle,
                    marker=".",
                    markersize=3,
                    label=name,
                )
                handover = drawn[0].handover
                left.plot(
                    seconds[handover],
                    gaps[handover],
                    marker="D",
                    color=colour,
                    markeredgecolor=INK_MUTED,
                    linestyle="none",
                )
                continue
            grid, mean, spread = _mean_curve(drawn)
            left.fill_between(
                grid,
                _floored(mean - spread),
                _floored(mean + spread),
                color=blend_with_white(colour, 0.3),
                linewidth=0.0,
            )
            left.plot(
                grid, _floored(mean), color=colour, linestyle=linestyle, label=name
            )
            at = float(np.mean([curve.seconds[curve.handover] for curve in drawn]))
            left.plot(
                at,
                float(np.interp(at, grid, _floored(mean))),
                marker="D",
                color=colour,
                markeredgecolor=INK_MUTED,
                linestyle="none",
            )
        left.axhline(GAP_FLOOR, color=INK_MUTED, linestyle=":", linewidth=0.8)
        left.annotate(
            reference_label,
            xy=(0.99, GAP_FLOOR),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            color=INK_MUTED,
            fontsize="small",
        )
        left.set_yscale("log")
        left.set_xlabel("seconds since the start's first call")
        left.set_ylabel("gap below the reference (nats)")
        left.set_title("(a) gap against wall clock", loc="left")
        left.legend(loc="upper right", frameon=False, fontsize="x-small", ncol=2)

        reached = [
            np.array([curve.gaps[-1] for curve in curves[name]], dtype=np.float64)
            for name in names
        ]
        means = np.array([gaps.mean() if gaps.size else np.nan for gaps in reached])
        spreads = np.array([gaps.std() if gaps.size > 1 else 0.0 for gaps in reached])
        rows = np.arange(len(names))
        right.barh(
            rows,
            _floored(means) - GAP_FLOOR,
            left=GAP_FLOOR,
            xerr=spreads if n_trials > 1 else None,
            color=[colour for colour, _ in encoding],
            ecolor=INK_MUTED,
            capsize=2.0,
        )
        right.set_yticks(rows, labels=list(names))
        right.invert_yaxis()
        right.set_xscale("log")
        right.set_xlabel(f"gap reached at {budget.size} {budget.unit} (nats)")
        right.set_title("(b) gap at the budget", loc="left")
        fig.tight_layout()

    trials_clause = (
        "One trial per start, so every point is a tracked iteration and no band "
        "or error bar is drawn."
        if n_trials == 1
        else (
            f"Over {n_trials} trials per start: each trial's curve is interpolated "
            f"onto {GRID_POINTS} evenly spaced seconds from the start's earliest "
            f"tracked iteration to its latest, absent before its first entry and "
            f"held at its last value after its end, and the mean is drawn with a "
            f"band of one standard deviation; each bar carries one standard "
            f"deviation."
        )
    )
    caption = (
        f"{len(names)} starts on {len(result.objectives)} instance(s) at "
        f"{len(result.seeds)} seed(s), each polished at {budget.size} "
        f"{budget.unit}. (a) The gap below {latex_escape(reference_label)} in "
        f"nats against wall seconds, on a log axis clipped at {GAP_FLOOR:g} "
        f"nats, where the reference is drawn; a diamond marks the hand-over "
        f"from seeding to polish. (b) The gap each start reached at the budget, "
        f"in the table's order. {trials_clause} Seconds are the host's and are "
        f"not reproduced by a rerun."
    )
    check_latex_safe(caption)
    return fig, caption


def _number(value: float) -> str:
    """A number as `infra/problems_tables.py` sets one: no trailing zero on an integer."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def starts_latex(result: SolverComparison) -> tuple[str, str]:
    """A ``tabular`` of the gap in nats and the seconds per start, and its caption.

    The body and caption :func:`~snakes_and_ladders.qa.figure.write_qa_table`
    takes, with the rules and the column and number formatting
    `infra/problems_tables.py` emits.

    Returns
    -------
    tuple[str, str]
        The ``tabular`` environment, and a caption
        :func:`~snakes_and_ladders.qa.figure.check_latex_safe` passes.
    """
    readings = {reading.start: reading for reading in result.readings()}
    lines = [
        r"\begin{tabular}{lcc}",
        r"  \toprule",
        r"  start & gap (nats) & seconds \\",
        r"  \midrule",
    ]
    for row in result.table():
        reading = readings[row.start]
        seconds = reading.seeding_seconds + reading.polish_seconds
        lines.append(
            f"  {latex_escape(row.start)} & {_number(row.gap)} & "
            f"{_number(round(seconds, 1))} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    budget = result.comparison.budget
    trials = max(row.trials for row in result.table())
    caption = (
        f"Per start, the mean gap below the reference in nats at {budget.size} "
        f"{budget.unit}, and the mean wall seconds of the seeding and the polish "
        f"together, over {trials} trial(s). Seconds are the host's and are not "
        f"reproduced by a rerun."
    )
    check_latex_safe(caption)
    return "\n".join(lines) + "\n", caption


#: The five columns of :func:`gap_table`, the order issue #898 set for the
#: emission mixture's table, with the gap's unit left to the caller.
GAP_COLUMNS = (
    "initializer",
    "init. gap",
    "init time [s]",
    "final gap",
    "final time [s]",
)


def _spread(values: list[float], plus_minus: str) -> str:
    """The mean to two decimals, and over more than one trial the sample standard deviation beside it."""
    mean = f"{np.mean(values):.2f}"
    if len(values) == 1:
        return mean
    return f"{mean} {plus_minus} {np.std(values, ddof=1):.2f}"


def _gap_cells(result: SolverComparison, plus_minus: str) -> list[list[str]]:
    """Per start, the five cells of :func:`gap_table`."""
    rows = []
    for name in result.names:
        trials = result.trials(name)
        references = [
            float(result.comparison.reference[index // len(result.seeds)])
            for index in range(len(trials))
        ]
        rows.append(
            [
                f"\\texttt{{{latex_escape(name)}}}",
                _spread(
                    [
                        one.seeded_value - ref
                        for one, ref in zip(trials, references, strict=True)
                    ],
                    plus_minus,
                ),
                _spread([one.seconds[one.handover] for one in trials], plus_minus),
                _spread(
                    [
                        one.value - ref
                        for one, ref in zip(trials, references, strict=True)
                    ],
                    plus_minus,
                ),
                _spread([one.seconds[-1] for one in trials], plus_minus),
            ]
        )
    return rows


def gap_table(result: SolverComparison, *, unit: str) -> tuple[str, str, str]:
    """The five-column table of a benchmark of starts: the gap and the seconds at the handover and at the end.

    The columns are :data:`GAP_COLUMNS`, the gap in ``unit`` above the
    reference the benchmark was read against, each cell the mean over the
    trials with the sample standard deviation beside it where there is more
    than one. Returned as the ``tabular`` a document inputs, its caption,
    and the same cells as the ``array`` MathJax sets in a notebook, where
    ``booktabs`` rules and ``$\\pm$`` are not available.

    Returns
    -------
    tuple[str, str, str]
        The ``tabular``, a caption
        :func:`~snakes_and_ladders.qa.figure.check_latex_safe` passes, and
        the ``array``.
    """
    header = [
        f"{column} [{unit}]" if "gap" in column else column for column in GAP_COLUMNS
    ]
    tabular = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"  \toprule",
            "  " + " & ".join(header) + r" \\",
            r"  \midrule",
            *(
                "  " + " & ".join(cells) + r" \\"
                for cells in _gap_cells(result, r"$\pm$")
            ),
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )
    array = "\n".join(
        [
            r"\begin{array}{lrrrr}",
            r"  \hline",
            "  " + " & ".join(rf"\text{{{name}}}" for name in header) + r" \\",
            r"  \hline",
            *(
                "  " + " & ".join(cells) + r" \\"
                for cells in _gap_cells(result, r"\pm")
            ),
            r"  \hline",
            r"\end{array}",
        ]
    )
    trials = max(len(result.trials(name)) for name in result.names)
    caption = (
        f"Per initializer, the gap in {unit} above the reference and the wall "
        f"seconds, at the handover to the polish and where the polish ended, "
        f"over {trials} trial(s); "
        + (
            "mean and sample standard deviation. "
            if trials > 1
            else "one trial, no spread. "
        )
        + "Seconds are the host's and are not reproduced by a rerun."
    )
    check_latex_safe(caption)
    return tabular, caption, array


def start_styles(names: Sequence[str]) -> dict[str, tuple[str, str]]:
    """``(colour, linestyle)`` per start, in ``names``' order: the ten colours of :data:`START_PALETTE` solid, then again dashed.

    Returns
    -------
    dict[str, tuple[str, str]]
    """
    return {
        name: (
            START_PALETTE[index % len(START_PALETTE)],
            LINESTYLES[(index // len(START_PALETTE)) % len(LINESTYLES)],
        )
        for index, name in enumerate(names)
    }


def curve_band(curves: Sequence[Curve], seconds: np.ndarray) -> GapBand:
    """Each trial's gap held from each entry to the next, read on ``seconds``, as :func:`~snakes_and_ladders.search.mixture_starts.gap_band` reads a mixture trial.

    The mean and the sample standard deviation are taken where every trial
    has an entry; the handover is the mean over trials of each one's.

    Returns
    -------
    GapBand

    Raises
    ------
    ValueError
        If ``curves`` is empty.
    """
    if not curves:
        msg = "a band needs at least one trial"
        raise ValueError(msg)
    held = np.full((len(curves), seconds.shape[0]), np.nan)
    for row, curve in enumerate(curves):
        index = np.searchsorted(curve.seconds, seconds, side="right") - 1
        known = index >= 0
        held[row, known] = curve.gaps[index[known]]
    started = ~np.isnan(held).any(axis=0)
    mean = np.full(seconds.shape[0], np.nan)
    std = np.full(seconds.shape[0], np.nan)
    mean[started] = held[:, started].mean(axis=0)
    if len(curves) > 1:
        std[started] = held[:, started].std(axis=0, ddof=1)
    handover = (
        float(np.mean([c.seconds[c.handover] for c in curves])),
        float(np.mean([c.gaps[c.handover] for c in curves])),
    )
    return GapBand(seconds, mean, std, handover)


def gap_panels(
    panels: Mapping[str, Mapping[str, GapBand]],
    final: Mapping[str, Mapping[str, float]],
    styles: Mapping[str, tuple[str, str]],
    *,
    ylabel: str,
    show_bands: bool = False,
) -> Figure:
    """The gap against runtime, one panel per entry of ``panels``, side by side on shared axes.

    Parameters
    ----------
    panels : Mapping[str, Mapping[str, GapBand]]
        Per panel heading, each start's band; the heading titles the panel's
        legend.
    final : Mapping[str, Mapping[str, float]]
        Per panel, each start's mean final gap: the legend's order, lowest first.
    styles : Mapping[str, tuple[str, str]]
        Each start's colour and line style, :func:`start_styles`, so a start
        reads the same in every panel.
    ylabel : str
        The gap's name and unit.
    show_bands : bool
        Draw one sample standard deviation either side of each mean.

    Returns
    -------
    Figure
    """
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=GAP_FIGSIZE,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (heading, bands) in zip(axes[0], panels.items(), strict=True):
        for name in sorted(bands, key=final[heading].__getitem__):
            band = bands[name]
            colour, linestyle = styles[name]
            axis.plot(
                band.seconds,
                band.mean,
                linewidth=1.2,
                alpha=0.7,
                color=colour,
                linestyle=linestyle,
                label=name,
            )
            if show_bands:
                axis.fill_between(
                    band.seconds,
                    band.mean - band.std,
                    band.mean + band.std,
                    color=colour,
                    alpha=0.25,
                    linewidth=0.0,
                )
            axis.plot(
                *band.handover,
                marker="D",
                markersize=5,
                color=colour,
                markeredgecolor="0.1",
                alpha=0.7,
                linestyle="none",
            )
        axis.axhline(0.0, color="0.2", linewidth=0.9, linestyle="--")
        axis.set_xscale("log")
        # Logarithmic both sides of zero, linear within one unit of it: a gap
        # below zero passed the reference, which a plain log axis cannot draw.
        axis.set_yscale("symlog", linthresh=1.0)
        axis.set_xlabel("runtime [s]")
        axis.legend(fontsize=6, loc="lower left", title=heading, title_fontsize=7)
    axes[0][0].set_ylabel(ylabel)
    return figure
