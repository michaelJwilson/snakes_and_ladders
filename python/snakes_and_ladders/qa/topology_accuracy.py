"""QA figure: how much data the search needs to recover the true topology.

``ROADMAP.md`` states an accuracy requirement -- normalized Robinson-Foulds
distance ``<= 0.05`` against simulated truth -- and nothing in the repository
measured it. This figure does, and it measures it where the answer is
informative: against the number of sites, which is the quantity that decides
whether the alignment carries enough signal to separate one topology from
another.

Reporting it at a single site count would say little. A search that recovers
the truth on a generous alignment has shown that the pipeline works, not that
it works at the margin. Sweeping the site count finds the margin.

Renders what `snakes_and_ladders.search` computed; it reimplements no search
(`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import QAFigure, latex_integer
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN,
    letter_style,
    series_style,
)
from snakes_and_ladders.search.infer import infer
from snakes_and_ladders.search.topology import normalized_robinson_foulds
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment

# Spanning the point where the signal runs out. The low end is deliberately
# below what the fixtures use, because a sweep that only covers sizes already
# known to work locates no margin.
SITE_COUNTS = (60, 125, 250, 500, 1000, 2000)

# Independent alignments per site count. Each is a fresh draw from the same
# generating tree, so the spread is sampling noise in the data rather than in
# the search, which is seeded from the same value throughout.
REPLICATES = 8

# `ROADMAP.md`'s accuracy requirement.
REQUIREMENT = 0.05


def accuracy(params: SimulationParams) -> dict[int, list[float]]:
    """Normalized RF distance from the inferred topology to the truth.

    Parameters
    ----------
    params : SimulationParams
        The generating truth; ``params.tau`` is what the search must recover.

    Returns
    -------
    dict[int, list[float]]
        Per site count, one distance per replicate.
    """
    measured: dict[int, list[float]] = {}
    for n_sites in SITE_COUNTS:
        distances = []
        for replicate in range(REPLICATES):
            dataset = simulate_alignment(
                tau=params.tau,
                k=params.k,
                pi=params.pi,
                rng=np.random.default_rng(params.seed + replicate),
                n_sites=n_sites,
            )
            result = infer(
                dict(dataset.alignment), params.k, rng=np.random.default_rng(0)
            )
            distances.append(normalized_robinson_foulds(result.topology, params.tau))
        measured[n_sites] = distances
    return measured


def build_figure(
    measured: dict[int, list[float]], params: SimulationParams
) -> tuple[Figure, str]:
    """Assemble the figure and its caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    sizes = sorted(measured)
    means = [float(np.mean(measured[n])) for n in sizes]
    recovered = {n: sum(1 for d in measured[n] if d == 0.0) for n in sizes}
    passing = [n for n in sizes if means[sizes.index(n)] <= REQUIREMENT]

    with letter_style():
        fig, axis = plt.subplots(figsize=ONE_COLUMN)
        style = series_style(0)
        for n_sites in sizes:
            axis.plot(
                [n_sites] * len(measured[n_sites]),
                measured[n_sites],
                marker=style["marker"],
                linestyle="none",
                color=style["color"],
                markersize=3,
                alpha=0.55,
            )
        mean_style = series_style(1)
        axis.plot(
            sizes,
            means,
            marker=mean_style["marker"],
            linestyle=mean_style["linestyle"],
            color=mean_style["color"],
            markersize=4,
            linewidth=1.0,
            label="mean",
        )
        axis.axhline(
            REQUIREMENT, color=INK_MUTED, linestyle=":", linewidth=0.8, zorder=1
        )
        axis.annotate(
            f"requirement, {REQUIREMENT}",
            xy=(0.97, REQUIREMENT),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            color=INK_MUTED,
            fontsize="small",
        )
        axis.set_xscale("log")
        axis.set_xlabel("sites")
        axis.set_ylabel("normalized Robinson--Foulds distance")
        axis.legend(loc="upper right", frameon=False, fontsize="small")
        fig.tight_layout()

    threshold = (
        f"from {latex_integer(min(passing))} sites upward"
        if passing
        else "at no size swept here"
    )
    caption = (
        f"How much alignment the search needs to recover the tree that "
        f"generated it. Each marker is one independent alignment drawn from "
        f"the {len(params.pi)}-state Jukes-Cantor fixture (seed "
        f"{params.seed}, {REPLICATES} replicates per size), inferred by NNI "
        f"hill climbing and scored against the generating topology by "
        f"normalized Robinson-Foulds distance -- 0 for the same tree, 1 for "
        f"no internal split in common. The dotted line is the accuracy "
        f"requirement this project set itself, {REQUIREMENT}. It is met "
        f"{threshold}: "
        f"{recovered[max(sizes)]} of {REPLICATES} replicates recover the "
        f"topology exactly at {latex_integer(max(sizes))} sites, against "
        f"{recovered[min(sizes)]} of {REPLICATES} at "
        f"{latex_integer(min(sizes))}. The distance is normalized by internal "
        f"splits, so the bound means the same thing at any taxon count."
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
        stem="topology_accuracy",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=lambda params: build_figure(accuracy(params), params),
        argv=argv,
    )


if __name__ == "__main__":
    main()
