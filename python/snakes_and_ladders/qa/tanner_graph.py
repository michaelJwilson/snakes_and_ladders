"""QA figure: the Tanner graph of the enumerable low-density parity-check instance.

The textbook's LDPC section carries a hand-drawn Tanner graph of six bits and
three checks. It states the structure and shows no instance: the code the
repository decodes is a draw from Gallager's regular ensemble, and what a
draw looks like -- which bits a check covers, how the bands are laid out --
is not something a sketch can say.

Panel (a) is the bipartite graph itself: one circle per bit, one square per
parity check, one edge per nonzero of the parity-check matrix. Panel (b) is
the same 36 nonzeros as the matrix, rows grouped into the bands the
construction builds them in. The first band's checks cover consecutive bits
by construction and every later band is that band under a column
permutation, so the first band is drawn heavy in both panels and the
permuted structure is what the eye is left with.

Both panels are drawn from coordinates computed here and pinned by the
suite, not from a graph-drawing library's layout: a randomized layout would
render differently on each run and the committed figure would disagree with
a re-render (``qa/CLAUDE.md``).

Renders the code `snakes_and_ladders.sim.ldpc` draws; it constructs no
matrix of its own. The textbook cites it as ``fig:ldpc-tanner``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import LDPC_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK,
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
)
from snakes_and_ladders.sim.ldpc import LdpcParams, ParityCheck

#: How far the check row sits above the bit row, in bit spacings. Tall enough
#: that an edge leaves a bit at a readable angle, short enough that the two
#: rows read as one graph.
CHECK_ROW_HEIGHT = 3.0


@dataclass(frozen=True)
class TannerLayout:
    """Where every node and every edge of the Tanner graph is drawn.

    Computed before matplotlib sees anything, so the drawing is a function of
    the code alone and two renders of one tree place every node identically.

    Parameters
    ----------
    variables : np.ndarray
        Shape ``(n_bits, 2)``: bit ``i`` at ``(i, 0)``.
    checks : np.ndarray
        Shape ``(n_checks, 2)``: the checks spread evenly across the bit row
        at height :data:`CHECK_ROW_HEIGHT`, in the order of the mean bit they
        cover, so a check sits near its own bits and the edges cross as
        little as the ordering allows.
    edges : np.ndarray
        Shape ``(n_edges, 2)``: ``(bit, check)`` per nonzero, in the order the
        matrix stores them, which is by bit and then by check.
    band : np.ndarray
        Shape ``(n_edges,)``: which Gallager band each edge's check belongs
        to, ``0`` for the band whose checks cover consecutive bits.
    """

    variables: np.ndarray
    checks: np.ndarray
    edges: np.ndarray
    band: np.ndarray


def tanner_layout(code: ParityCheck, row_weight: int) -> TannerLayout:
    """Lay the code out as two rows of nodes and one segment per nonzero.

    Parameters
    ----------
    code : ParityCheck
        The drawn instance.
    row_weight : int
        Ones per row, which is the band width: the construction stacks
        ``n_checks / (n_bits / row_weight)`` bands of ``n_bits / row_weight``
        rows each.

    Returns
    -------
    TannerLayout

    Raises
    ------
    ValueError
        If ``row_weight`` does not divide the block length, so the checks do
        not fall into equal bands and the band a check belongs to is not
        defined.
    """
    if row_weight <= 0 or code.n_bits % row_weight:
        msg = (
            f"row_weight = {row_weight} does not divide the block length "
            f"{code.n_bits}, so the checks form no bands"
        )
        raise ValueError(msg)
    rows_per_band = code.n_bits // row_weight
    variables = np.column_stack(
        [np.arange(code.n_bits, dtype=np.float64), np.zeros(code.n_bits)]
    )
    edges = np.column_stack([code.edge_variable, code.edge_check])
    # Rank the checks by the mean bit they cover and spread them evenly over
    # the bit row in that order. Evenly rather than at the mean itself
    # because two checks may share a mean and would then be drawn on top of
    # one another; by the mean rather than by index because a check drawn
    # away from its own bits crosses every edge between.
    mean_bit = np.zeros(code.n_checks)
    np.add.at(mean_bit, edges[:, 1], edges[:, 0])
    mean_bit /= code.row_weights
    rank = np.empty(code.n_checks, dtype=np.int64)
    rank[np.lexsort((np.arange(code.n_checks), mean_bit))] = np.arange(code.n_checks)
    spread = (code.n_bits - 1) * (rank + 0.5) / code.n_checks
    checks = np.column_stack([spread, np.full(code.n_checks, CHECK_ROW_HEIGHT)])
    band = code.edge_check // rows_per_band
    return TannerLayout(variables=variables, checks=checks, edges=edges, band=band)


def build_figure(params: LdpcParams) -> tuple[Figure, str]:
    """Draw the Tanner graph and the matrix it comes from, and caption them.

    Parameters
    ----------
    params : LdpcParams
        The declared instance; the code is drawn under its seed.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    code = params.code()
    layout = tanner_layout(code, params.row_weight)
    rows_per_band = code.n_bits // params.row_weight
    n_bands = code.n_checks // rows_per_band

    with letter_style():
        fig, axes = plt.subplots(2, 1, figsize=ONE_COLUMN_WIDE)

        for (bit, check), band in zip(layout.edges, layout.band, strict=True):
            start, end = layout.variables[bit], layout.checks[check]
            axes[0].plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=INK if band == 0 else INK_MUTED,
                linewidth=1.1 if band == 0 else 0.5,
                zorder=1,
            )
        axes[0].scatter(
            layout.variables[:, 0],
            layout.variables[:, 1],
            s=70,
            marker="o",
            facecolors="white",
            edgecolors=INK,
            linewidths=0.8,
            zorder=2,
        )
        axes[0].scatter(
            layout.checks[:, 0],
            layout.checks[:, 1],
            s=70,
            marker="s",
            facecolors="white",
            edgecolors=INK,
            linewidths=0.8,
            zorder=2,
        )
        # Both node rows carry their index, so panel (a) and panel (b) name
        # the same bit and the same check.
        for coordinates in (layout.variables, layout.checks):
            for index, (x, y) in enumerate(coordinates):
                axes[0].annotate(
                    str(index),
                    (x, y),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=INK,
                    zorder=3,
                )
        axes[0].set_xlim(-1.0, code.n_bits)
        axes[0].set_ylim(-0.8, CHECK_ROW_HEIGHT + 0.8)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        for spine in axes[0].spines.values():
            spine.set_visible(False)
        axes[0].set_title("(a) bits, checks, and the nonzeros between them", loc="left")

        axes[1].scatter(
            layout.edges[:, 0],
            layout.edges[:, 1],
            s=22,
            marker="s",
            color=np.where(layout.band == 0, INK, INK_MUTED),
        )
        for boundary in range(1, n_bands):
            axes[1].axhline(
                boundary * rows_per_band - 0.5, color=INK, linewidth=0.5, zorder=0
            )
        axes[1].set_xlim(-1.0, code.n_bits)
        axes[1].set_ylim(code.n_checks - 0.5, -0.5)
        axes[1].set_xlabel("bit")
        axes[1].set_ylabel("check")
        axes[1].set_xticks(range(0, code.n_bits, 2))
        axes[1].set_yticks(range(code.n_checks))
        axes[1].set_title(
            f"(b) the same nonzeros as a matrix, {n_bands} bands", loc="left"
        )
        fig.tight_layout()

    caption = (
        f"The Tanner graph of one draw from the regular "
        f"({params.column_weight}, {params.row_weight}) Gallager ensemble: "
        f"the enumerable instance, {code.n_bits} bits under seed {params.seed}, "
        f"decoded on a binary symmetric channel at crossover "
        f"{params.flip_probability:g}. (a) Bits as circles, the "
        f"{code.n_checks} parity checks as squares, and one edge per nonzero "
        f"of the parity-check matrix: {code.n_edges} of them, every bit on "
        f"{params.column_weight} checks and every check on {params.row_weight} "
        f"bits. (b) The same nonzeros as the matrix, its {n_bands} bands of "
        f"{rows_per_band} rows separated by a rule. The first band is drawn "
        f"heavy in both panels: its checks cover consecutive bits by "
        f"construction, and every later band is that band under a uniform "
        f"random permutation of the columns, which is the whole of the "
        f"randomness in the draw. The construction fixes the degrees at every "
        f"block length, so the longer instances are this structure at a size "
        f"no page holds; this one is drawn because a page holds it."
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
        stem="tanner_graph",
        description=__doc__,
        params=(LDPC_PARAMS,),
        build=build_figure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
