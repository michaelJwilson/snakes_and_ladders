"""QA figure: a worked simulation example -- Newick string and alignment.

Renders the Newick string and a compact table of simulated aligned
sequences for a small-taxa ``simulation_params.yaml``-format fixture, so the
technical document shows an actual generated example rather than a
schematic. Per ``sim/CLAUDE.md``'s "k-state, e.g. 4" convention, states are
shown nucleotide-coded (A/C/G/T) for ``k == 4``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from snakes_and_ladders.qa.figure import (
    QAFigure,
    latex_integer,
    state_label,
)
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, Option, figure_main
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import SimulatedDataset, simulate_alignment
from snakes_and_ladders.sim.tree import Node, preorder

_MODEL_NAME = "Jukes-Cantor"


# Greek letters for internal ancestors, in preorder. Names like
# "ancestor_CD" carry an underscore that mathtext must escape and a meaning
# ("the ancestor of C and D") already visible from the tree, so a symbol
# reads better and takes less width.
_ANCESTOR_SYMBOLS = (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\zeta",
    r"\eta",
    r"\theta",
)


def display_newick(tau: Node) -> str:
    r"""The topology as a mathtext string, for display rather than for parsing.

    ``snakes_and_ladders.sim.newick.to_newick`` is the package's serialization and stays
    the authority on what a Newick string is; this renders the same topology
    with symbols a reader can take in: ``\rho`` for the root, Greek letters
    for internal ancestors, and ``name\_length`` for leaves, since a raw
    colon and an unescaped underscore are both mathtext syntax.

    Parameters
    ----------
    tau : Node
        Root of the topology to display.

    Returns
    -------
    str
        A mathtext string, without the enclosing dollar signs.

    Raises
    ------
    ValueError
        If the tree has more internal ancestors than there are symbols for.
    """
    ancestors: dict[str, str] = {}
    for node in preorder(tau):
        if node.is_leaf or node is tau:
            continue
        if len(ancestors) >= len(_ANCESTOR_SYMBOLS):
            msg = (
                f"tree has more than {len(_ANCESTOR_SYMBOLS)} internal "
                f"ancestors; add symbols to _ANCESTOR_SYMBOLS"
            )
            raise ValueError(msg)
        ancestors[node.name] = _ANCESTOR_SYMBOLS[len(ancestors)]

    def _render(node: Node) -> str:
        label = node.name if node.is_leaf else ancestors.get(node.name, r"\rho")
        if not node.is_leaf:
            inner = ",".join(_render(child) for child in node.children)
            label = f"({inner}){label}"
        if node.branch_length is None:
            return label
        return f"{label}\\_{node.branch_length:g}"

    return _render(tau)


def render_sim_example(dataset: SimulatedDataset, n_sites_shown: int, ax: Axes) -> int:
    """Draw ``dataset``'s Newick string and a sliced alignment table on ``ax``.

    Parameters
    ----------
    dataset : SimulatedDataset
        The simulated alignment to display.
    n_sites_shown : int
        Number of leading alignment columns to tabulate.
    ax : matplotlib.axes.Axes
        Axes to draw into.

    Returns
    -------
    int
        The number of sites actually shown (``min(n_sites_shown,
        dataset.n_sites)``), for callers that need it (e.g. the caption).
    """
    n_shown = min(n_sites_shown, dataset.n_sites)
    leaves = sorted(dataset.alignment)
    rows = [
        [
            state_label(int(state), dataset.k)
            for state in dataset.alignment[name][:n_shown]
        ]
        for name in leaves
    ]

    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        f"${display_newick(dataset.tau)}$",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    table = ax.table(
        cellText=rows,
        rowLabels=leaves,
        colLabels=[str(i) for i in range(n_shown)],
        cellLoc="center",
        bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 0.6),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    return n_shown


def build_caption(params: SimulationParams, n_sites_shown: int) -> str:
    """Caption text for the worked-example figure.

    Parameters
    ----------
    params : SimulationParams
        The parameters the figure was rendered from.
    n_sites_shown : int
        Number of leading alignment columns tabulated.

    Returns
    -------
    str
        Plain-text caption, safe to ``\\input`` into LaTeX verbatim.
    """
    n_taxa = sum(1 for node in preorder(params.tau) if node.is_leaf)
    n_shown = min(n_sites_shown, params.n_sites)
    return (
        f"Worked example: {n_taxa}-taxon alignment simulated under the "
        f"{_MODEL_NAME} model (seed {params.seed}, "
        f"{latex_integer(params.n_sites)} sites total), showing the topology "
        f"with branch lengths -- rho the root, Greek letters its internal "
        f"ancestors -- and the first {n_shown} of "
        f"{latex_integer(params.n_sites)} simulated sites."
    )


def build_figure(
    params: SimulationParams, n_sites_shown: int = 10
) -> tuple[Figure, str]:
    """Render the worked-example figure and its caption from loaded params.

    Parameters
    ----------
    params : SimulationParams
        The fixture the alignment is drawn from.
    n_sites_shown : int
        Number of leading alignment columns to tabulate.

    Returns
    -------
    tuple[Figure, str]
        The figure and its caption.
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    n_shown = render_sim_example(dataset, n_sites_shown, ax)
    return fig, build_caption(params, n_shown)


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
        stem="sim_example",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=build_figure,
        options=[Option("n-sites-shown", int, 10)],
        argv=argv,
    )


if __name__ == "__main__":
    main()
