"""QA figure: the coupled spatio-sequential model as a Forney-style factor graph.

The textbook's sketch of ``eq:joint`` is hand-drawn and shows no instance.
This draws one: the fixture's lattice, its planted labelling and its chains,
in an oblique projection whose front face is the spatial lattice and whose
depth axis is the sequential position (issue #891).

* The front face carries a label variable ``l_ij`` per lattice site, a circle
  filled with its planted class, and a pairwise Potts factor ``J`` as a small
  square on every lattice edge between two drawn sites.
* Behind it, one Markov chain per shown class: the initial-distribution node
  ``pi_m`` and a state variable ``k^m_t`` per drawn position, with a
  transition factor ``T`` as a square between each consecutive pair.
* The emission variables ``(u_gn, b_gn)`` are small filled dots at a subsample
  of sites, every ``emission_stride``-th on each lattice axis, at every drawn
  position; thin edges along the three axes tie the subsample into cuboids.
* The gated emission factor ``1(l_n = m)`` couples a site's label, its
  emission and every shown chain's state at that position. It is a triangle
  on the curved edges from the label node to the chains' states: black at
  one highlighted site, thin grey at the rest of the subsample.

At most :data:`LATTICE_CAP` sites per lattice axis and :data:`POSITION_CAP`
positions are drawn; beyond either the figure shows the leading window and an
ellipsis, and the caption states the declared sizes. Every size is read from
the parameters, and the planting is
:func:`sal.sim.count_pairs.planted_labels`, the one the
count-pair draw is conditioned on; nothing here draws data or fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from sal.qa.coupled_labelling import lattice_shape
from sal.qa.figure import latex_integer
from sal.qa.style import (
    INK,
    INK_MUTED,
    RULE,
    blend_with_white,
    discrete_palette,
    letter_style,
)
from sal.sim.count_pairs import planted_labels
from sal.sim.spatio_sequential import SpatioSequentialParams

#: Sites drawn per lattice axis. Five keeps every ``J`` square and every
#: label legible at one column width; a larger lattice is drawn as its
#: leading 5x5 window.
LATTICE_CAP = 5

#: Sequential positions drawn along the depth axis, for the same reason.
POSITION_CAP = 4

#: The oblique projection: a unit of depth moves a point this far right and
#: this far up on the page, and one position is :data:`DEPTH` units deep.
OBLIQUE = (0.62, 0.38)
DEPTH = 1.55

#: Height above the lattice's top row of the first chain, and between chains.
CHAIN_LIFT = 2.4
CHAIN_GAP = 1.1

#: The ink of a gated factor that is not highlighted, lighter than
#: ``INK_MUTED`` so the highlighted factors read above it.
GREY_EDGE = mcolors.to_hex(blend_with_white(INK_MUTED, 0.55))


@dataclass(frozen=True)
class ForneyWindow:
    """What the figure draws of an instance, computed before matplotlib sees it.

    Parameters
    ----------
    rows, columns : int
        Lattice sites drawn per axis, at most :data:`LATTICE_CAP`.
    positions : int
        Sequential positions drawn, at most :data:`POSITION_CAP`.
    classes : int
        Chains drawn, at most ``M``.
    sites : np.ndarray
        Node index of every drawn site, row-major, shape ``(rows * columns,)``.
    edges : np.ndarray
        Lattice edges with both ends drawn, shape ``(n_edges, 2)``.
    subsample : np.ndarray
        Node indices of the sites that carry emission variables.
    highlighted : int
        The subsample site whose gated factors are drawn in black.
    labels : np.ndarray
        Planted class of every node of the lattice, shape ``(n_nodes,)``.
    """

    rows: int
    columns: int
    positions: int
    classes: int
    sites: np.ndarray
    edges: np.ndarray
    subsample: np.ndarray
    highlighted: int
    labels: np.ndarray


def forney_window(
    params: SpatioSequentialParams, *, emission_stride: int, classes_shown: int
) -> ForneyWindow:
    """The sites, edges, positions and chains :func:`build_figure` draws.

    Parameters
    ----------
    params : SpatioSequentialParams
        A coupled model on a two-dimensional lattice.
    emission_stride : int
        Every ``emission_stride``-th drawn site on each axis carries emission
        variables; at least one.
    classes_shown : int
        Chains drawn, between one and ``M``.

    Returns
    -------
    ForneyWindow

    Raises
    ------
    ValueError
        If the stride is below one, or ``classes_shown`` is outside
        ``[1, M]``.
    """
    if emission_stride < 1:
        msg = f"emission_stride is at least one, got {emission_stride}"
        raise ValueError(msg)
    if not 1 <= classes_shown <= params.n_classes:
        msg = f"classes_shown is in [1, {params.n_classes}], got {classes_shown}"
        raise ValueError(msg)
    all_rows, all_columns = lattice_shape(params.graph)
    rows, columns = min(all_rows, LATTICE_CAP), min(all_columns, LATTICE_CAP)
    grid = np.arange(rows)[:, None] * all_columns + np.arange(columns)[None, :]
    sites = grid.reshape(-1)
    drawn = np.zeros(params.graph.n_nodes, dtype=bool)
    drawn[sites] = True
    edge_index = params.graph.edge_index
    edges = edge_index[drawn[edge_index[:, 0]] & drawn[edge_index[:, 1]]]
    subsample = grid[::emission_stride, ::emission_stride].reshape(-1)
    return ForneyWindow(
        rows=rows,
        columns=columns,
        positions=min(params.n_positions, POSITION_CAP),
        classes=classes_shown,
        sites=sites,
        edges=edges,
        subsample=subsample,
        highlighted=int(subsample[len(subsample) // 2]),
        labels=planted_labels(params, params.n_classes),
    )


def _project(x: float, y: float, z: float) -> tuple[float, float]:
    """The page coordinates of ``(x, y, z)`` under the oblique projection."""
    return x + z * OBLIQUE[0], y + z * OBLIQUE[1]


def _site(node: int, columns: int) -> tuple[float, float]:
    """A node's lattice coordinates: ``x`` its column, ``y`` its negated row."""
    return float(node % columns), -float(node // columns)


def _curve(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    width: float,
    bend: float,
) -> None:
    """A curved dependency edge between two page points."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-",
            connectionstyle=f"arc3,rad={bend}",
            color=color,
            linewidth=width,
            zorder=1,
        )
    )


def build_figure(
    params: SpatioSequentialParams, *, emission_stride: int = 2, classes_shown: int = 2
) -> tuple[Figure, str]:
    """Draw the coupled model's factor graph for one instance, and caption it.

    Parameters
    ----------
    params : SpatioSequentialParams
        The declared model; its lattice, ``M``, ``K`` and ``S`` set what is
        drawn, capped by :data:`LATTICE_CAP` and :data:`POSITION_CAP`.
    emission_stride : int
        Every ``emission_stride``-th drawn site on each lattice axis carries
        emission variables.
    classes_shown : int
        Chains drawn behind the lattice, classes ``0`` to
        ``classes_shown - 1``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption, which ``qa.figure.check_latex_safe``
        accepts.
    """
    window = forney_window(
        params, emission_stride=emission_stride, classes_shown=classes_shown
    )
    palette = discrete_palette(window.classes)
    with letter_style():
        fig, axis = plt.subplots(figsize=(6.4, 4.9))
        _draw(axis, params, window, emission_stride, palette)
        fig.tight_layout()
    caption = _caption(params, window, emission_stride)
    return fig, caption


def _draw(
    axis: Axes,
    params: SpatioSequentialParams,
    window: ForneyWindow,
    emission_stride: int,
    palette: dict[int, str],
) -> None:
    """Every artist of :func:`build_figure`, onto one axis."""
    all_rows, all_columns = lattice_shape(params.graph)
    columns = all_columns

    def colour(node: int) -> str:
        label = int(window.labels[node])
        return palette[label] if label < window.classes else "white"

    depth = [DEPTH * (t + 1) for t in range(window.positions)]

    # Emission cuboids first, so every node is drawn over them.
    sub = window.subsample
    sub_xy = {int(node): _site(int(node), columns) for node in sub}
    for node, (x, y) in sub_xy.items():
        for z0, z1 in pairwise(depth):
            axis.plot(
                *zip(_project(x, y, z0), _project(x, y, z1), strict=True),
                color=RULE,
                linewidth=0.5,
                zorder=0,
            )
        for other, (ox, oy) in sub_xy.items():
            step = abs(ox - x) + abs(oy - y)
            if other > node and step == emission_stride and (ox == x or oy == y):
                for z in depth:
                    axis.plot(
                        *zip(_project(x, y, z), _project(ox, oy, z), strict=True),
                        color=RULE,
                        linewidth=0.5,
                        zorder=0,
                    )

    # Front face: Potts factors on the lattice edges, then the labels.
    for first, second in window.edges:
        (x0, y0), (x1, y1) = _site(int(first), columns), _site(int(second), columns)
        axis.plot([x0, x1], [y0, y1], color=INK_MUTED, linewidth=0.6, zorder=1)
        axis.scatter(
            [(x0 + x1) / 2],
            [(y0 + y1) / 2],
            marker="s",
            s=9,
            facecolor=INK,
            edgecolor=INK,
            zorder=2,
        )
    xs, ys = zip(*(_site(int(node), columns) for node in window.sites), strict=True)
    axis.scatter(
        xs,
        ys,
        s=95,
        marker="o",
        facecolor=[colour(int(node)) for node in window.sites],
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )

    # The chains, one per shown class, lifted above the lattice's top row.
    centre = (window.columns - 1) / 2.0
    chain_nodes: dict[tuple[int, int], tuple[float, float]] = {}
    for m in range(window.classes):
        lift = CHAIN_LIFT + CHAIN_GAP * m
        points = [_project(centre, lift, 0.0)] + [
            _project(centre, lift, z) for z in depth
        ]
        for t, point in enumerate(points[1:]):
            chain_nodes[m, t] = point
        for index, (start, end) in enumerate(pairwise(points)):
            axis.plot(
                *zip(start, end, strict=True), color=palette[m], linewidth=1.0, zorder=1
            )
            if index == 0:
                continue  # pi_m is itself the factor on the first state
            middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            axis.scatter(
                *middle,
                marker="s",
                s=22,
                facecolor="white",
                edgecolor=INK,
                linewidth=0.8,
                zorder=3,
            )
        axis.scatter(
            *points[0],
            marker="D",
            s=60,
            facecolor=palette[m],
            edgecolor=INK,
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            rf"$\pi_{m}$",
            points[0],
            xytext=(-14, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            color=INK,
        )
        for t, point in enumerate(points[1:]):
            axis.scatter(
                *point,
                marker="o",
                s=80,
                facecolor="white",
                edgecolor=palette[m],
                linewidth=1.3,
                zorder=3,
            )
            axis.annotate(
                rf"$k^{{{m}}}_{{{t + 1}}}$",
                point,
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )

    # Gated emission factors, and the emission variables they gate.
    for node, (x, y) in sub_xy.items():
        highlighted = node == window.highlighted
        ink, width = (INK, 0.9) if highlighted else (GREY_EDGE, 0.3)
        for t, z in enumerate(depth):
            dot = _project(x, y, z)
            factor = _project(x + 0.3, y + 0.45, z)
            axis.plot(
                *zip(factor, dot, strict=True), color=ink, linewidth=width, zorder=1
            )
            _curve(axis, (x, y), factor, color=ink, width=width, bend=-0.25)
            # A grey factor is drawn to the chains at the first position only:
            # at every position it would bury the highlighted ones.
            for m in range(window.classes if highlighted or t == 0 else 0):
                _curve(
                    axis, factor, chain_nodes[m, t], color=ink, width=width, bend=0.18
                )
            axis.scatter(
                *factor,
                marker="^",
                s=30 if highlighted else 14,
                facecolor=ink if highlighted else "white",
                edgecolor=ink,
                linewidth=0.7,
                zorder=4,
            )
            axis.scatter(*dot, marker="o", s=12, color=INK, zorder=4)

    # What is not drawn: the rest of the lattice, and the rest of the sequence.
    last_x, last_y = window.columns - 1.0, -(window.rows - 1.0)
    if all_columns > window.columns:
        axis.annotate(
            r"$\cdots$",
            (last_x + 0.55, 0.0),
            ha="left",
            va="center",
            fontsize=10,
            color=INK_MUTED,
        )
    if all_rows > window.rows:
        axis.annotate(
            r"$\vdots$",
            (0.0, last_y - 0.5),
            ha="center",
            va="top",
            fontsize=10,
            color=INK_MUTED,
        )
    if params.n_positions > window.positions:
        tail = _project(centre, CHAIN_LIFT, depth[-1] + 0.9)
        axis.annotate(
            r"$\cdots$" + f"  S = {params.n_positions:,}",
            tail,
            ha="left",
            va="center",
            fontsize=7,
            color=INK_MUTED,
        )

    # The annotations the sketch carries.
    j_first, j_second = window.edges[0]
    jx = (_site(int(j_first), columns)[0] + _site(int(j_second), columns)[0]) / 2
    jy = (_site(int(j_first), columns)[1] + _site(int(j_second), columns)[1]) / 2
    axis.annotate(
        r"$J_{nn'}$",
        (jx, jy),
        xytext=(-2, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        color=INK,
    )
    axis.annotate(
        r"$\ell_{ij}$",
        _site(int(window.sites[-1]), columns),
        xytext=(0, -11),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        color=INK,
    )
    if window.positions > 1:
        (ax0, ay0), (ax1, ay1) = chain_nodes[0, 0], chain_nodes[0, 1]
        axis.annotate(
            r"$T_{kk'}$",
            ((ax0 + ax1) / 2, (ay0 + ay1) / 2),
            xytext=(4, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=INK,
        )
    hx, hy = _site(window.highlighted, columns)
    axis.annotate(
        r"$\mathbf{1}(\ell_n = m)$",
        _project(hx + 0.3, hy + 0.45, depth[-1]),
        xytext=(8, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    axis.annotate(
        r"$(u_{gn}, b_{gn})$",
        _project(hx, hy, depth[-1]),
        xytext=(6, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=INK,
    )

    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.margins(0.06)


def _caption(
    params: SpatioSequentialParams, window: ForneyWindow, emission_stride: int
) -> str:
    """The caption: the declared sizes, what is drawn of them, and what each mark is."""
    all_rows, all_columns = lattice_shape(params.graph)
    n_nodes = params.graph.n_nodes
    return (
        f"The coupled spatio-sequential model as a factor graph, for the "
        f"declared instance: {latex_integer(n_nodes)} label variables on a "
        f"{all_rows} by {all_columns} lattice, M = {params.n_classes} classes "
        f"of K = {params.n_states} hidden states, over S = "
        f"{latex_integer(params.n_positions)} positions. Drawn: the leading "
        f"{window.rows} by {window.columns} window of the lattice and the first "
        f"{window.positions} positions, at most {LATTICE_CAP} sites per lattice "
        f"axis and {POSITION_CAP} positions. The front face holds a label "
        f"variable per site, filled by its planted class where that class's "
        f"chain is drawn and white where it is not, and a Potts factor J as a "
        f"square on every lattice edge. Behind it, {window.classes} of the "
        f"{params.n_classes} chains: the initial-distribution node, a diamond, "
        f"and a state variable per position, with a transition factor T as a "
        f"square between consecutive ones. Emission variables, the pair of "
        f"counts, are dots at every {emission_stride}-th site on each axis and "
        f"every drawn position, tied into cuboids by thin edges. The gated "
        f"emission factor, a triangle, joins a site's label, its emission and "
        f"the state of every drawn chain at that position; black at one site, "
        f"and grey at the rest of the subsample, where it is drawn to the "
        f"chains at the first position only."
    )
