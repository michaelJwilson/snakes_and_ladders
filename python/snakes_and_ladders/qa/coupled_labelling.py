"""QA figure: the coupled model's labelling, planted and recovered, on the lattice.

The textbook's coupled section carries a hand-drawn factor graph. It states
which variables sit on which factors and shows no instance: the labelling is
what the spatial prior is for, and a labelling of a hundred nodes is a
picture rather than a structure.

Three panels over the one set of lattice coordinates. (a) The planted
labelling the observations were drawn under. (b) The labelling block ascent
recovers from the annealed start, permuted onto the planted classes so the
two panels are comparable, with any node labelled otherwise ringed. (c) The
external field at the fit, as the difference between the two classes' costs
at each node, ringed where taking the node at its cheaper class -- the
labelling the observations give with the spatial prior switched off -- misses
the planting. That count is what the third panel is for: it says how much of
the recovery in (b) the prior is doing.

The coordinates come from the lattice's own geometry -- column and row --
rather than from a graph-drawing layout, so two renders place every node
identically and the committed figure survives a re-render
(``qa/CLAUDE.md``).

Renders what `snakes_and_ladders.sim.spatio_sequential` drew and what
`snakes_and_ladders.search.spatio_sequential` fitted; it runs no annealing
and no expectation--maximization of its own. The textbook cites it as
``fig:coupled-labelling``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.opt.schedule import Exponential
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.layout import discrete_legend
from snakes_and_ladders.qa.runner import (
    SPATIO_SEQUENTIAL_PARAMS,
    Option,
    figure_main,
)
from snakes_and_ladders.qa.style import (
    INK,
    ONE_COLUMN_SHORT,
    discrete_palette,
    letter_style,
)
from snakes_and_ladders.search.spatio_sequential import (
    SpatioSequentialFit,
    fit_spatio_sequential,
    graph_burn_in,
    label_accuracy,
)
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.spatio_sequential import (
    SpatioSequentialParams,
    simulate_spatio_sequential,
)

#: The seed the observations are drawn under, and the seed the fit runs
#: under. These are the first of the six replicates the regression suite
#: averages over on this instance, so the figure draws the problem the suite
#: measures rather than one of its own.
DRAW_SEED = 50
FIT_SEED = 0

#: The annealing schedule of the start, and the blocks that polish it: the
#: suite's, for the same reason.
ANNEALING = (4.0, 1.0, 30)
POLISH_BLOCKS = 10


def lattice_shape(graph: PottsGraph) -> tuple[int, int]:
    """The rows and columns of the lattice the figure draws.

    Returns
    -------
    tuple[int, int]
        Rows and columns.

    Raises
    ------
    ValueError
        If the graph is not a two-dimensional lattice, which has no rows and
        columns to draw over.
    """
    if graph.shape is None or len(graph.shape) != 2:
        msg = f"the figure needs a 2-D lattice, got shape {graph.shape}"
        raise ValueError(msg)
    rows, columns = graph.shape
    return rows, columns


def lattice_coordinates(graph: PottsGraph) -> np.ndarray:
    """Node coordinates read off the lattice, row zero at the top.

    Parameters
    ----------
    graph : PottsGraph
        A two-dimensional lattice.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, 2)``: ``x`` is the column and ``y`` the negated row,
        so the drawing has the row order of the index.
    """
    _, columns = lattice_shape(graph)
    index = np.arange(graph.n_nodes)
    return np.column_stack([index % columns, -(index // columns)]).astype(np.float64)


def planted_labelling(params: SpatioSequentialParams) -> np.ndarray:
    """The two-stripe planting the suite and the notebook plant on this lattice.

    Half the columns carry one class and half the other, so the boundary is
    one straight cut and the recovered labelling can be read against it.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes,)``, entries in ``[0, 2)``.

    Raises
    ------
    ValueError
        If the lattice is not square, where "half the columns" is not the
        planting the suite plants.
    """
    rows, columns = lattice_shape(params.graph)
    if rows != columns:
        msg = f"the planting needs a square lattice, got {rows}x{columns}"
        raise ValueError(msg)
    return (np.arange(rows * columns) % columns < columns // 2).astype(np.int64)


def best_permutation(
    fitted: np.ndarray, planted: np.ndarray, n_classes: int
) -> np.ndarray:
    """The renaming of the fit's classes that agrees with ``planted`` most.

    A fit names the classes in whatever order it found them, so drawing the
    fitted labelling and the planted one in one palette without this compares
    colours and not labels. The agreement the permutation achieves is
    :func:`~snakes_and_ladders.search.spatio_sequential.label_accuracy`'s,
    which the suite pins this against.

    Returns
    -------
    np.ndarray
        ``mapping``, shape ``(n_classes,)``: the fit's class ``m`` is the
        planted class ``mapping[m]``.
    """
    fitted = np.asarray(fitted)
    planted = np.asarray(planted)
    best = max(
        permutations(range(n_classes)),
        key=lambda order: float((np.array(order)[fitted] == planted).mean()),
    )
    return np.array(best, dtype=np.int64)


@dataclass(frozen=True)
class Recovery:
    """One replicate: what was drawn, what was fitted, and how far apart they are.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(S, n_nodes)``, the draw under the planted labelling.
    fit : SpatioSequentialFit
        The polished fit from the annealed start.
    labels : np.ndarray
        The fitted labels permuted onto the planted classes.
    field : np.ndarray
        The external field at the fit, shape ``(n_nodes, M)``, its columns
        under the same permutation, so column ``m`` is the cost of the
        *planted* class ``m``.
    accuracy : float
        The fraction of nodes labelled as planted, up to that permutation.
    """

    observations: np.ndarray
    fit: SpatioSequentialFit
    labels: np.ndarray
    field: np.ndarray
    accuracy: float

    @property
    def margin(self) -> np.ndarray:
        """The first planted class's cost less the second's, one per node.

        Negative where the observations at a node, on their own, favour the
        first class.
        """
        return np.asarray(self.field[:, 0] - self.field[:, 1])

    @property
    def field_labelling(self) -> np.ndarray:
        """The labelling the field alone gives: each node at its cheaper class."""
        return np.asarray(np.argmin(self.field, axis=1))


def recover(
    params: SpatioSequentialParams,
    planted: np.ndarray,
    draw_seed: int,
    fit_seed: int,
) -> Recovery:
    """Draw one replicate and fit it from the annealed start.

    Parameters
    ----------
    params : SpatioSequentialParams
        The declared instance.
    planted : np.ndarray
        The labelling the observations are drawn under.
    draw_seed, fit_seed : int
        Seeds of the two generators, reported in the caption.

    Returns
    -------
    Recovery
    """
    data = simulate_spatio_sequential(
        params, np.random.default_rng(draw_seed), labels=planted
    )
    warm = graph_burn_in(
        params,
        data.observations,
        np.random.default_rng(fit_seed),
        Exponential(*ANNEALING),
    )
    fit = fit_spatio_sequential(
        warm.params,
        data.observations,
        np.random.default_rng(fit_seed),
        n_blocks=POLISH_BLOCKS,
        labels=warm.labels,
    )
    mapping = best_permutation(fit.labels, planted, params.n_classes)
    return Recovery(
        observations=data.observations,
        fit=fit,
        labels=np.asarray(mapping[fit.labels]),
        field=np.asarray(fit.field[:, np.argsort(mapping)]),
        accuracy=label_accuracy(fit.labels, planted, params.n_classes),
    )


def build_figure(
    params: SpatioSequentialParams,
    *,
    draw_seed: int,
    fit_seed: int,
) -> tuple[Figure, str]:
    """Assemble the three panels and the caption.

    Parameters
    ----------
    params : SpatioSequentialParams
        The declared instance.
    draw_seed, fit_seed : int
        Seeds of the draw and of the fit.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    planted = planted_labelling(params)
    recovery = recover(params, planted, draw_seed, fit_seed)
    coords = lattice_coordinates(params.graph)
    palette = discrete_palette(params.n_classes)
    disagreeing = np.flatnonzero(recovery.labels != planted)
    # What the observations say at a node on their own: the field is a cost,
    # so the class the node would take with the spatial prior switched off.
    field_alone = np.flatnonzero(recovery.field_labelling != planted)
    rows, columns = lattice_shape(params.graph)

    with letter_style():
        fig, axes = plt.subplots(1, 3, figsize=ONE_COLUMN_SHORT)
        for ax, labelling, title in (
            (axes[0], planted, "(a) planted"),
            (axes[1], recovery.labels, "(b) recovered"),
        ):
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=[palette[int(state)] for state in labelling],
                s=14,
                marker="s",
                edgecolors="none",
            )
            ax.set_title(title, loc="left")
        axes[1].scatter(
            coords[disagreeing, 0],
            coords[disagreeing, 1],
            s=34,
            marker="o",
            facecolors="none",
            edgecolors=INK,
            linewidths=0.8,
        )
        drawn = axes[2].scatter(
            coords[:, 0],
            coords[:, 1],
            c=recovery.margin,
            s=14,
            marker="s",
            cmap="PuOr",
            edgecolors="none",
        )
        axes[2].scatter(
            coords[field_alone, 0],
            coords[field_alone, 1],
            s=34,
            marker="o",
            facecolors="none",
            edgecolors=INK,
            linewidths=0.8,
        )
        fig.colorbar(drawn, ax=axes[2], fraction=0.046, pad=0.04).set_label(
            "cost of class 0 less class 1"
        )
        axes[2].set_title("(c) field", loc="left")
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        discrete_legend(
            axes[0],
            [f"class {state}" for state in range(params.n_classes)],
            [palette[state] for state in range(params.n_classes)],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout()

    caption = (
        f"The coupled model's labelling on the instance past enumeration: a "
        f"{rows}x{columns} open lattice, {params.n_classes} classes, "
        f"{params.n_states} chain states, {params.n_positions} positions, "
        f"spatial prior at inverse temperature {params.beta:g} and coupling "
        f"{params.graph.coupling[0]:g}, observations drawn from seed "
        f"{draw_seed} under the planted labelling and fitted from seed "
        f"{fit_seed}. (a) The planting: half the columns in each class, so "
        f"the boundary is one cut of {rows} bonds. (b) The labelling block "
        f"ascent reaches from the annealed start, permuted onto the planted "
        f"classes and ringed at every node where the two panels differ: "
        f"{len(disagreeing)} of {params.graph.n_nodes} nodes are labelled "
        f"otherwise, an agreement of {recovery.accuracy:.2f}. (c) The "
        f"external field at the fit, as the cost of the first class less that "
        f"of the second, with the sign carrying which class the observations "
        f"at that node favour on their own. Taking each node at its cheaper "
        f"class, with the spatial prior switched off, misses the planting at "
        f"{len(field_alone)} of {params.graph.n_nodes} nodes, ringed: at "
        f"{params.n_positions} positions the emissions overlap too far to "
        f"label a node one at a time, and it is the prior that recovers them."
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
        stem="coupled_labelling",
        description=__doc__,
        params=(SPATIO_SEQUENTIAL_PARAMS,),
        build=build_figure,
        options=(
            Option("draw-seed", int, DRAW_SEED),
            Option("fit-seed", int, FIT_SEED),
        ),
        argv=argv,
    )


if __name__ == "__main__":
    main()
