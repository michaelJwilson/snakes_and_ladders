"""QA figure: the assumed simulation tree, topology and branch lengths.

Renders the topology and branch lengths (expected substitutions/site) that
``snakes_and_ladders.sim.simulate`` draws an alignment over, from a
``simulation_params.yaml``-format file, as a labelled phylogram -- so the
tree a reader sees in the technical document is the one the simulator
actually used, not a schematic redrawn by hand.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import QAFigure, state_label
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node, edges, preorder

_MODEL_NAME = "Jukes-Cantor"


def tree_layout(tau: Node) -> dict[str, tuple[float, float]]:
    """Depth (x) and vertical order (y) for every node in ``tau``.

    Parameters
    ----------
    tau : Node
        Root of the topology to lay out.

    Returns
    -------
    dict[str, tuple[float, float]]
        Node name to ``(depth, y)``: ``depth`` is the sum of branch lengths
        from the root (expected substitutions/site); ``y`` is the node's
        vertical position, leaves evenly spaced in traversal order and
        internal nodes centred over their children.
    """
    depth: dict[str, float] = {}
    y: dict[str, float] = {}
    next_leaf_y = 0.0

    def _depth(node: Node, parent_depth: float) -> None:
        depth[node.name] = parent_depth + (node.branch_length or 0.0)
        for child in node.children:
            _depth(child, depth[node.name])

    def _y(node: Node) -> float:
        nonlocal next_leaf_y
        if node.is_leaf:
            y[node.name] = next_leaf_y
            next_leaf_y += 1.0
        else:
            child_ys = [_y(child) for child in node.children]
            y[node.name] = sum(child_ys) / len(child_ys)
        return y[node.name]

    _depth(tau, 0.0)
    _y(tau)
    return {name: (depth[name], y[name]) for name in depth}


# Where the sequence column starts, in axes fraction. Just outside the right
# spine, so the x axis still spans the tree alone -- its unit is expected
# substitutions per site, and extending it under a block of letters would
# put ticks on a region that has no length. Room is made by shrinking the
# axes in ``render_sim_tree_figure`` rather than by widening the data range.
_SEQUENCE_X = 1.03

# Fraction of the figure width given to the tree; the rest holds the
# sequences.
_TREE_WIDTH = 0.58

# Sites shown per leaf. Enough to read as a sequence, few enough to fit
# beside the tree at one-column width.
SITES_SHOWN = 14


def render_sim_tree(
    tau: Node,
    ax: Axes,
    alignment: Mapping[str, np.ndarray] | None = None,
    k: int = 4,
) -> dict[str, tuple[float, float]]:
    """Draw ``tau`` as a labelled phylogram on ``ax``.

    Parameters
    ----------
    tau : Node
        Root of the topology to draw.
    ax : matplotlib.axes.Axes
        Axes to draw into.

    Returns
    -------
    dict[str, tuple[float, float]]
        The layout from :func:`tree_layout`, for callers that need the
        plotted coordinates (e.g. a regression test).
    """
    layout = tree_layout(tau)
    for parent, child in edges(tau):
        parent_depth, parent_y = layout[parent.name]
        child_depth, child_y = layout[child.name]
        ax.plot(
            [parent_depth, parent_depth],
            [parent_y, child_y],
            color="black",
            linewidth=1,
        )
        ax.plot(
            [parent_depth, child_depth], [child_y, child_y], color="black", linewidth=1
        )
        midpoint = (parent_depth + child_depth) / 2
        assert child.branch_length is not None  # non-root, per Node's contract
        ax.text(
            midpoint,
            child_y,
            f"{child.branch_length:.2f}",
            fontsize=7,
            ha="center",
            va="bottom",
        )
    for node in preorder(tau):
        if node.is_leaf:
            node_depth, node_y = layout[node.name]
            ax.text(node_depth + 0.01, node_y, node.name, fontsize=9, va="center")

    if alignment is not None:
        # x in axes fraction, y in data coordinates, so the sequences form a
        # block aligned with the leaves however the tree is scaled.
        transform = ax.get_yaxis_transform()
        for node in preorder(tau):
            if not node.is_leaf:
                continue
            _, node_y = layout[node.name]
            sites = alignment[node.name][:SITES_SHOWN]
            ax.text(
                _SEQUENCE_X,
                node_y,
                "".join(state_label(int(state), k) for state in sites),
                transform=transform,
                fontsize=8,
                family="monospace",
                va="center",
                clip_on=False,
            )

    ax.set_xlabel("Expected substitutions / site")
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return layout


def build_caption(params: SimulationParams) -> str:
    """Caption text for the simulation-tree figure.

    States the seed, taxa count, and evolutionary model, per
    ``sim/CLAUDE.md``'s ground-truth-retention rule -- a plot is only
    QA-usable alongside the parameters that generated it.

    Parameters
    ----------
    params : SimulationParams
        The parameters the figure was rendered from.

    Returns
    -------
    str
        Plain-text caption, safe to ``\\input`` into LaTeX verbatim.
    """
    n_taxa = sum(1 for node in preorder(params.tau) if node.is_leaf)
    return (
        f"Simulated topology for {n_taxa} taxa under the {_MODEL_NAME} model "
        f"(seed {params.seed}), branch lengths labelled in expected "
        f"substitutions per site. The first {SITES_SHOWN} simulated sites are "
        f"shown beside each leaf, in tree order, so the alignment the "
        f"likelihood consumes can be read off against the topology that "
        f"generated it."
    )


def build_figure(params: SimulationParams) -> tuple[Figure, str]:
    """Render the simulation-tree figure and its caption from loaded params.

    Parameters
    ----------
    params : SimulationParams
        The fixture the tree and its alignment are drawn from.

    Returns
    -------
    tuple[Figure, str]
        The figure and its caption.
    """
    # Simulated here rather than passed in: the figure's whole claim is that
    # these sequences came from this tree, so they are drawn from the same
    # params the topology is.
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=max(SITES_SHOWN, 1),
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    render_sim_tree(params.tau, ax, alignment=dataset.alignment, k=params.k)
    fig.subplots_adjust(right=_TREE_WIDTH)
    return fig, build_caption(params)


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
        stem="sim_tree",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=build_figure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
