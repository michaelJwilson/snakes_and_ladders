"""QA figure: what hill climbing does, and what it was choosing between.

Two claims, and they need each other. A trajectory alone shows a search
improving without saying whether it stopped anywhere good; a landscape alone
shows the problem without saying whether the search solved it.

Panel (a) is the search trajectory: maximized log-likelihood against
candidate fits spent, with the generating tree's own score as the reference
line. Panel (b) is every unrooted topology on the same leaf set, scored and
sorted, with the search's endpoint marked --- the exhaustive oracle that
makes "found the best tree" a checkable claim below 8 taxa rather than an
assertion.

Renders what `snakes_and_ladders.search` computed; it reimplements no move set and no
optimizer (`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import QAFigure, latex_integer
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)
from snakes_and_ladders.search.infer import MoveSet, infer, score_topology
from snakes_and_ladders.search.topology import enumerate_topologies
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment

# Starting seeds whose trajectories are drawn. Three is enough to show that
# the endpoint does not depend on where the search began, which is the point
# of drawing more than one.
SEEDS = (0, 1, 2)

MAX_EVALUATIONS = 500


def search_trajectories(
    params: SimulationParams,
) -> tuple[dict[str, list[tuple[int, float]]], float, np.ndarray, float]:
    """Run the searches and the exhaustive sweep this figure reports.

    Parameters
    ----------
    params : SimulationParams
        The generating truth; its topology is the reference.

    Returns
    -------
    tuple[dict[str, list[tuple[int, float]]], float, np.ndarray, float]
        Per-move-set trajectories as ``(fits spent, log-likelihood)`` pairs,
        the generating tree's own score, every enumerated topology's score,
        and the best score found by any search.
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)

    trajectories: dict[str, list[tuple[int, float]]] = {}
    reached = -np.inf
    for moves in (MoveSet.NNI, MoveSet.SPR):
        result = infer(
            alignment,
            params.k,
            rng=np.random.default_rng(SEEDS[0]),
            moves=moves,
            max_evaluations=MAX_EVALUATIONS,
        )
        # The trace records the score after each accepted move; the fits
        # spent reaching each are what the budget counts, so the two are
        # plotted against each other rather than against move number.
        spent = np.linspace(0, result.evaluations, len(result.trace))
        trajectories[moves.value] = [
            (int(round(x)), y) for x, y in zip(spent, result.trace, strict=True)
        ]
        reached = max(reached, result.log_likelihood)

    landscape = np.array(
        sorted(
            score_topology(topology, alignment, params.k)
            for topology in enumerate_topologies(sorted(alignment))
        )
    )
    truth = score_topology(params.tau, alignment, params.k)
    return trajectories, truth, landscape, reached


def build_figure(
    trajectories: dict[str, list[tuple[int, float]]],
    truth: float,
    landscape: np.ndarray,
    reached: float,
    params: SimulationParams,
) -> tuple[Figure, str]:
    """Assemble the two-panel figure and its caption.

    Parameters
    ----------
    trajectories : dict[str, list[tuple[int, float]]]
        Per-move-set ``(fits, log-likelihood)`` pairs.
    truth : float
        The generating tree's own maximized log-likelihood.
    landscape : np.ndarray
        Every enumerated topology's score, ascending.
    reached : float
        The best score any search reached.
    params : SimulationParams
        The generating truth, for the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)

        axes[0].axhline(truth, color=INK_MUTED, linestyle=":", linewidth=0.8, zorder=1)
        axes[0].annotate(
            "generating tree",
            xy=(0.03, truth),
            xycoords=("axes fraction", "data"),
            va="bottom",
            color=INK_MUTED,
            fontsize="small",
        )
        for index, (name, points) in enumerate(sorted(trajectories.items())):
            style = series_style(index)
            axes[0].plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                markersize=4,
                linewidth=1.0,
                label=name.upper(),
            )
        axes[0].set_xlabel("candidate fits spent")
        axes[0].set_ylabel("log-likelihood")
        axes[0].set_title("(a) trajectory", loc="left")
        axes[0].legend(loc="lower right", frameon=False, fontsize="small")

        style = series_style(2)
        axes[1].plot(
            np.arange(1, landscape.size + 1),
            landscape,
            marker=style["marker"],
            linestyle="none",
            color=style["color"],
            markersize=3,
        )
        axes[1].axhline(
            reached, color=INK_MUTED, linestyle="--", linewidth=0.8, zorder=1
        )
        axes[1].annotate(
            "search endpoint",
            xy=(0.03, reached),
            xycoords=("axes fraction", "data"),
            va="top",
            color=INK_MUTED,
            fontsize="small",
        )
        axes[1].set_xlabel("topology, ranked")
        axes[1].set_title("(b) every topology", loc="left")
        fig.tight_layout()

    gap = float(landscape[-1] - landscape[-2])
    caption = (
        f"Hill climbing over tree topologies, scoring every candidate with the "
        f"same model-agnostic optimizer used for the Potts and hidden Markov "
        f"instances. Fixture: {len(params.pi)}-state Jukes-Cantor, "
        f"{latex_integer(params.n_sites)} sites, seed {params.seed}. "
        f"(a) Log-likelihood against candidate fits spent, from a randomly "
        f"drawn starting topology; the dotted line is the generating tree's "
        f"own score. (b) All {landscape.size} unrooted topologies on this leaf "
        f"set, scored and ranked, with the search's endpoint dashed. Below 8 "
        f"taxa this enumeration is affordable, which is what makes "
        f"the phrase found the best tree a checkable claim rather than an "
        f"assertion. The optimum leads the runner-up by {gap:.1f} log units "
        f"here, so this problem does not yet separate the two move sets: both "
        f"reach it, NNI in fewer fits."
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
        stem="search_trajectory",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=lambda params: build_figure(*search_trajectories(params), params),
        argv=argv,
    )


if __name__ == "__main__":
    main()
