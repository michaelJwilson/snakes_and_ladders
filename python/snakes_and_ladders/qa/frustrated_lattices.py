"""QA figure: the two frustrated instances whose answer comes from outside.

Panel (a) is the periodic triangular Ising antiferromagnet at its smallest
periodic size, drawn with one enumerated ground state: every triangle must
keep one agreeing edge, so exactly one edge in three agrees, and the ground
state is known at every size by that double count rather than by a search.
The agreeing edges are drawn heavy so the count can be read off. Panel (b)
is the planted Viana--Bray spin glass: the energy of the planted state
against the best of a fixed number of single-site descents, as the fraction
of couplings set against the planted state rises. Where descent lands on the
planted energy the planted state is the reference; where it beats it the
planted state is only an upper bound, and the figure says which.

Renders what `snakes_and_ladders.sim.canonical`,
`snakes_and_ladders.search.max_cut` and
`snakes_and_ladders.search.alpha_expansion` computed; it reimplements no
enumeration and no descent (`qa/CLAUDE.md`). The textbook cites it as ``fig:frustrated-lattices``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import FRUSTRATED_LATTICE_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK,
    INK_MUTED,
    ONE_COLUMN_WIDE,
    discrete_palette,
    letter_style,
    series_style,
)
from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes
from snakes_and_ladders.search.max_cut import enumerate_max_cut
from snakes_and_ladders.sim.canonical import (
    FrustratedLatticeParams,
    minimum_frustrated_edges,
)
from snakes_and_ladders.sim.graph import PottsGraph

#: The lattice's three bond directions on the unit cell, as the graph lays
#: them out: along a row, down a column, and the diagonal.
_OFFSETS = ((0, 1), (1, 0), (1, 1))


def lattice_layout(graph: PottsGraph) -> np.ndarray:
    """Sheared coordinates putting every lattice bond at unit length.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, 2)``: ``x = column - row / 2``,
        ``y = row * sqrt(3) / 2``, so the row, column and diagonal bonds of
        the graph's unit cell are the three unit vectors of the triangular
        lattice.
    """
    if graph.shape is None or len(graph.shape) != 2:
        msg = "lattice_layout needs a 2-D lattice"
        raise ValueError(msg)
    _, columns = graph.shape
    rows_of = np.arange(graph.n_nodes) // columns
    columns_of = np.arange(graph.n_nodes) % columns
    return np.column_stack(
        [columns_of - rows_of / 2.0, rows_of * math.sqrt(3.0) / 2.0]
    ).astype(np.float64)


def edge_segments(graph: PottsGraph) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """One drawn segment per edge, a wrapped edge ending at a ghost of its target.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, bool]]
        ``(start, end, wraps)`` per edge, in the graph's edge order. The end
        of a wrapped edge is the target's coordinate translated by the
        lattice period so the segment is unit length.
    """
    if graph.shape is None or len(graph.shape) != 2:
        msg = "edge_segments needs a 2-D lattice"
        raise ValueError(msg)
    rows, columns = graph.shape
    layout = lattice_layout(graph)
    segments = []
    for first, second in graph.edges:
        row, column = divmod(first, columns)
        target_row, target_column = divmod(second, columns)
        for row_step, column_step in _OFFSETS:
            if ((row + row_step) % rows, (column + column_step) % columns) == (
                target_row,
                target_column,
            ):
                ghost_row, ghost_column = row + row_step, column + column_step
                break
        else:  # pragma: no cover - the graph builds only these offsets
            msg = f"edge ({first}, {second}) is not a lattice bond"
            raise ValueError(msg)
        wraps = (ghost_row, ghost_column) != (target_row, target_column)
        end = np.array(
            [ghost_column - ghost_row / 2.0, ghost_row * math.sqrt(3.0) / 2.0]
        )
        segments.append((layout[first], end, wraps))
    return segments


#: How much of a wrapped bond is drawn, from its start toward the ghost of
#: its target: a stub long enough to read as a bond and short enough not to
#: cross the lattice.
WRAP_STUB = 0.35


def ground_state(graph: PottsGraph) -> tuple[np.ndarray, int]:
    """One enumerated ground state, and how many of its edges agree.

    Returns
    -------
    tuple[np.ndarray, int]
        The two-state labelling, and the agreeing-edge count, which the
        closed form fixes at ``n_nodes`` on the periodic lattice.
    """
    labelling, cut = enumerate_max_cut(graph)
    return labelling, len(graph.edges) - int(round(cut))


@dataclass(frozen=True)
class GlassScan:
    """The planted energy against descent, per frustration.

    Parameters
    ----------
    planted : np.ndarray
        ``E`` of the planted state, one per frustration the fixture declares.
    descended : np.ndarray
        The lowest energy the declared number of single-site descents
        reached.
    """

    planted: np.ndarray
    descended: np.ndarray


def glass_scan(params: FrustratedLatticeParams, rng: np.random.Generator) -> GlassScan:
    """Build one instance per declared frustration and descend on each.

    Parameters
    ----------
    params : FrustratedLatticeParams
        The declared scan: sites, mean degree, frustrations and restarts.
    rng : np.random.Generator
        Every draw, the instances and the starts, comes from it; passed in
        rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    GlassScan
    """
    field = np.zeros(2)
    planted = np.zeros(len(params.glass_frustrations))
    descended = np.zeros(len(params.glass_frustrations))
    for index, frustration in enumerate(params.glass_frustrations):
        instance = params.glass(frustration, rng)
        planted[index] = instance.planted_energy
        descended[index] = min(
            iterated_conditional_modes(instance.graph, field, 2, rng)[1]
            for _ in range(params.glass_restarts)
        )
    return GlassScan(planted=planted, descended=descended)


def build_figure(
    params: FrustratedLatticeParams,
    graph: PottsGraph,
    labelling: np.ndarray,
    agreeing: int,
    scan: GlassScan,
) -> tuple[Figure, str]:
    """Assemble the two panels and the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    palette = discrete_palette(2)
    layout = lattice_layout(graph)
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)
        axes[0].set_aspect("equal")
        axes[0].grid(False)
        for (first, second), (start, end, wraps) in zip(
            graph.edges, edge_segments(graph), strict=True
        ):
            agrees = labelling[first] == labelling[second]
            stop = start + WRAP_STUB * (end - start) if wraps else end
            axes[0].plot(
                [start[0], stop[0]],
                [start[1], stop[1]],
                color=INK if agrees else INK_MUTED,
                linewidth=2.2 if agrees else 0.7,
                linestyle=":" if wraps else "-",
                zorder=1,
            )
        axes[0].scatter(
            layout[:, 0],
            layout[:, 1],
            c=[palette[int(state)] for state in labelling],
            s=60,
            edgecolors=INK,
            linewidths=0.6,
            zorder=2,
        )
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        for spine in axes[0].spines.values():
            spine.set_visible(False)
        axes[0].set_title("(a) a triangular ground state", loc="left", y=1.0)

        for index, (values, label) in enumerate(
            ((scan.planted, "planted state"), (scan.descended, "best descent"))
        ):
            style = series_style(index)
            axes[1].plot(
                params.glass_frustrations,
                values,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=label,
            )
        axes[1].set_xlabel("frustration")
        axes[1].set_ylabel("energy")
        axes[1].set_title("(b) the planted glass", loc="left")
        axes[1].legend(loc="upper left", frameon=False, fontsize="small")
        fig.tight_layout()

    rows, columns = params.shape
    matched = int(np.sum(scan.descended >= scan.planted - 1e-9))
    below = int(np.sum(scan.descended < scan.planted - 1e-9))
    caption = (
        f"Two frustrated instances with an answer from outside. (a) The "
        f"{rows}x{columns} periodic triangular Ising antiferromagnet, coupling "
        f"-1, {len(graph.edges)} bonds (dotted: bonds wrapping the torus), "
        f"with one of its enumerated ground states coloured by spin. Heavy "
        f"bonds join agreeing spins: {agreeing} of {len(graph.edges)}, one in "
        f"three, which the double count over triangles fixes as the minimum "
        f"at every size, so the ground-state energy is {agreeing} exactly. "
        f"(b) A planted Viana-Bray spin glass on {params.glass_nodes} sites at mean "
        f"degree {params.glass_mean_degree:g}, couplings of magnitude 1, one instance per "
        f"frustration from seed {params.seed}: the energy of the planted state "
        f"against the lowest energy reached by {params.glass_restarts} single-site "
        f"descents from random starts. Descent lands on the planted energy at "
        f"{matched} of {len(params.glass_frustrations)} frustrations and goes below it at "
        f"{below}, where the planted state is an upper bound on the ground "
        f"state and not the ground state."
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

    def build(params: FrustratedLatticeParams) -> tuple[Figure, str]:
        graph = params.lattice()
        labelling, agreeing = ground_state(graph)
        assert agreeing == minimum_frustrated_edges(graph)
        return build_figure(
            params,
            graph,
            labelling,
            agreeing,
            glass_scan(params, np.random.default_rng(params.seed)),
        )

    return figure_main(
        stem="frustrated_lattices",
        description=__doc__,
        params=(FRUSTRATED_LATTICE_PARAMS,),
        build=build,
        argv=argv,
    )


if __name__ == "__main__":
    main()
