"""QA figure: the turbo code's bit error rate against ``E_b / N_0``.

The waterfall is the one picture a coding result is read from, and it is
also the one a reader cannot check without two things a curve alone does not
carry: what the code is being compared against, and how many blocks each
point rests on. Both are in the figure. The uncoded antipodal closed form
``Q(sqrt(2 E_b / N_0))`` is drawn as the reference every coded curve must
fall below, and each point carries the one-sigma binomial interval its frame
count supports; a point at which no frame erred is drawn at the rate one
error would have produced, with an arrow, because zero on a log axis is not
a measurement of anything.

Panel (a) is the bit error rate at four iteration counts, which is what
makes the iteration's contribution visible: the curves separate where the
code is working and converge where it is not. Panel (b) is the frame error
rate at the deepest count, on the same axis, because a bit error rate and a
frame error rate answer different questions and a reader given only the
first cannot recover the second.

Renders the instance `snakes_and_ladders.sim.convolutional` declares; it
constructs no code of its own. The textbook cites it as
``fig:turbo-waterfall``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.likelihood.turbo import (
    ErrorRates,
    measure_error_rates,
    uncoded_bit_error_rate,
)
from snakes_and_ladders.qa.figure import QAFigure, latex_integer
from snakes_and_ladders.qa.runner import TURBO_PARAMS, figure_main
from snakes_and_ladders.qa.style import INK, INK_MUTED, ONE_COLUMN_WIDE, letter_style
from snakes_and_ladders.sim.convolutional import TurboParams

#: The iteration counts panel (a) draws. Powers of two from one, so the
#: spacing on a log-ish improvement is even and four curves fit one panel.
DRAWN_ITERATIONS = (1, 2, 4, 8)


def measure(params: TurboParams) -> list[ErrorRates]:
    """Every declared point of the instance, under the fixture's own seed.

    One run per point, reporting every iteration count from the deepest, so
    the figure costs the deepest iteration and not the sum of the four it
    draws.

    Parameters
    ----------
    params : TurboParams

    Returns
    -------
    list[ErrorRates]
        One per declared ``E_b / N_0``, in the fixture's order.
    """
    code = params.code()
    return [
        measure_error_rates(
            code,
            point,
            params.frames,
            np.random.default_rng(params.seed),
            iterations=params.iterations,
        )
        for point in params.eb_n0_db
    ]


def _floor(rates: list[ErrorRates]) -> float:
    """The rate one error in one point's sample would have produced.

    Where a point saw no error, this is what is drawn: the sample bounds the
    rate from above and says nothing below it, so drawing zero would state a
    result the frame count cannot support.
    """
    return 1.0 / rates[0].bits


def _report(rate: float, floor: float) -> str:
    """A measured rate, or the bound a point that saw no error supports.

    Writing "0" would state a result the sample cannot carry: what 50 frames
    of 256 bits establish is that the rate is below one in 12,800, not that
    it is zero.
    """
    return f"{rate:.2g}" if rate > 0.0 else f"below {floor:.1g}"


def build_figure(params: TurboParams) -> tuple[Figure, str]:
    """Draw the waterfall and the frame error rate, and caption them.

    Parameters
    ----------
    params : TurboParams
        The declared instance; the interleaver is drawn under its seed.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    rates = measure(params)
    points = np.array(params.eb_n0_db)
    floor = _floor(rates)
    uncoded = uncoded_bit_error_rate(points)

    with letter_style():
        fig, axes = plt.subplots(2, 1, figsize=ONE_COLUMN_WIDE, sharex=True)

        axes[0].plot(
            points,
            uncoded,
            color=INK_MUTED,
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            label="uncoded $Q(\\sqrt{2E_b/N_0})$",
        )
        for index, iteration in enumerate(DRAWN_ITERATIONS):
            values = np.array([rate.bit_error_rate[iteration - 1] for rate in rates])
            errors = np.array(
                [rate.bit_error_interval[iteration - 1] for rate in rates]
            )
            censored = values <= 0.0
            axes[0].errorbar(
                points,
                np.where(censored, floor, values),
                yerr=np.where(censored, 0.0, errors),
                color=INK,
                alpha=0.35 + 0.65 * index / (len(DRAWN_ITERATIONS) - 1),
                marker="o",
                markersize=3,
                linewidth=1.0,
                capsize=2,
                label=f"{iteration} iteration" + ("s" if iteration > 1 else ""),
            )
            for point in points[censored]:
                axes[0].annotate(
                    "",
                    xy=(point, floor * 0.35),
                    xytext=(point, floor),
                    arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 0.8},
                )
        axes[0].set_yscale("log")
        axes[0].set_ylim(floor * 0.25, 1.0)
        axes[0].set_ylabel("bit error rate")
        axes[0].legend(loc="lower left", fontsize=5.5, frameon=False)
        axes[0].set_title(
            f"(a) message bits in error, {rates[0].bits:,} per point", loc="left"
        )

        frame_rates = np.array([rate.frame_error_rate[-1] for rate in rates])
        censored = frame_rates <= 0.0
        frame_floor = 1.0 / rates[0].frames
        axes[1].plot(
            points,
            np.where(censored, frame_floor, frame_rates),
            color=INK,
            marker="s",
            markersize=3,
            linewidth=1.0,
        )
        for point in points[censored]:
            axes[1].annotate(
                "",
                xy=(point, frame_floor * 0.35),
                xytext=(point, frame_floor),
                arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 0.8},
            )
        axes[1].set_yscale("log")
        axes[1].set_ylim(frame_floor * 0.25, 1.5)
        axes[1].set_xlabel("$E_b / N_0$ (dB)")
        axes[1].set_ylabel("frame error rate")
        axes[1].set_title(
            f"(b) frames with at least one error, {params.iterations} iterations,"
            f" {rates[0].frames} per point",
            loc="left",
        )
        fig.tight_layout()

    best = np.array([rate.bit_error_rate[-1] for rate in rates])
    # Plain text, no LaTeX: `qa.figure.check_latex_safe` allows no backslash
    # or underscore in a caption, so the closed form is written out in words
    # rather than as mathematics.
    caption = (
        f"The waterfall of the memory-{params.memory} "
        f"({params.feedback:o}, {params.feedforward:o}) turbo code at K = "
        f"{params.message_length}: rate {params.code().rate:.3f} unpunctured, "
        f"both constituent encoders terminated, a uniformly random "
        f"interleaver drawn under seed {params.seed}, on the binary-input "
        f"Gaussian channel. (a) Bit error rate at 1, 2, 4 and "
        f"{params.iterations} iterations of the extrinsic exchange, each "
        f"point {rates[0].frames} frames and {latex_integer(rates[0].bits)} "
        f"message bits, with the one-sigma binomial interval that sample "
        f"supports; the dashed line is the uncoded antipodal closed form "
        f"Q(sqrt(2 Eb/N0)), which the coded curve falls below at every "
        f"point, reaching {_report(best[-1], floor)} at {points[-1]:g} dB "
        f"against {uncoded[-1]:.2g} uncoded. (b) Frame error rate at "
        f"{params.iterations} iterations on the same axis. A point at which "
        f"no frame erred is drawn at one over the sample size with an arrow: "
        f"the sample bounds the rate from above and says nothing below it. "
        f"The interval understates the spread of the estimate, since bit "
        f"errors inside one frame are correlated; it is the bar the sample "
        f"size supports if the bits were independent, and a floor on the "
        f"true one."
    )
    return fig, caption


def main(argv: list[str] | None = None) -> QAFigure:
    """Render the figure from the command line.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    QAFigure
        Paths written, and the caption.
    """
    return figure_main(
        stem="turbo_waterfall",
        description=__doc__,
        params=(TURBO_PARAMS,),
        build=build_figure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
