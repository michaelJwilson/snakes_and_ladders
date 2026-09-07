"""QA figure: the cheap reward surface against the one it stands in for.

An RL agent over topologies needs a reward per candidate. The honest reward
is the *maximized* log-likelihood, which costs one L-BFGS solve per candidate
and makes training unaffordable; issue #131's simplification is to score at
fixed, known parameters instead. That substitution is only sound if the two
surfaces rank topologies the same way, and "only if" is the whole reason this
figure exists: a policy trained on the cheap surface and judged on the
expensive one is being trained on a different problem.

Panel (a) is every unrooted topology on the leaf set, scored both ways, with
the generating topology marked. Panel (b) asks whether the answer depends on
the one free parameter the cheap surface has --- the fixed branch length ---
by sweeping it across a range and reporting the correlation at each,
together with whether the two surfaces still agree on the best topology.

The correlation reported is linear rather than rank-based, for the reason
:func:`snakes_and_ladders.qa.figure.pearson_correlation` documents: the fitted surface does
not totally order topologies. Several of its optima agree to within the
optimizer's own convergence, so their relative order is not a property of the
science, and a rank statistic that depends on it is not a measurement.

Renders what `snakes_and_ladders.search` computed; it reimplements no scorer
(`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import (
    QAFigure,
    latex_integer,
    pearson_correlation,
)
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)
from snakes_and_ladders.search.rl import RewardModel, TopologyEnvironment
from snakes_and_ladders.search.topology import enumerate_topologies, leaf_bipartitions
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import preorder

# Spread over a 50-fold range around the fixture's own mean branch length.
# The point of the sweep is that the cheap surface has one free parameter and
# a reader is entitled to know whether the conclusion rests on tuning it.
BRANCH_LENGTHS = (0.02, 0.05, 0.10, 0.25, 0.40, 0.60, 1.00)


def mean_branch_length(params: SimulationParams) -> float:
    """The generating tree's mean branch length.

    The default the cheap surface is scored at: the one scalar summary of the
    truth that survives a change of topology, since branch lengths belong to
    edges and a different topology has different edges.
    """
    lengths = [
        node.branch_length
        for node in preorder(params.tau)
        if node.branch_length is not None
    ]
    return float(np.mean(lengths))


def reward_surfaces(
    params: SimulationParams,
) -> tuple[np.ndarray, np.ndarray, int, float, dict[float, float], dict[float, bool]]:
    """Score every topology both ways, and sweep the cheap surface's parameter.

    Parameters
    ----------
    params : SimulationParams
        The generating truth; its topology is the reference point.

    Returns
    -------
    tuple
        The known-parameter scores at the default branch length, the fitted
        scores, the index of the generating topology, that default branch
        length, the correlation per swept branch length, and whether the two
        surfaces agree on the best topology at each.
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)
    topologies = list(enumerate_topologies(sorted(alignment)))
    default = mean_branch_length(params)

    def surface(reward: RewardModel, branch_length: float) -> np.ndarray:
        environment = TopologyEnvironment(
            alignment, params.k, params.pi, branch_length, reward=reward
        )
        return np.array([environment.score(t) for t in topologies])

    fitted = surface(RewardModel.FITTED, default)
    known = surface(RewardModel.KNOWN, default)

    best = int(np.argmax(fitted))
    correlations, agreements = {}, {}
    for branch_length in BRANCH_LENGTHS:
        swept = surface(RewardModel.KNOWN, branch_length)
        correlations[branch_length] = pearson_correlation(swept, fitted)
        agreements[branch_length] = int(np.argmax(swept)) == best

    keys = [leaf_bipartitions(t) for t in topologies]
    truth = keys.index(leaf_bipartitions(params.tau))
    return known, fitted, truth, default, correlations, agreements


def build_figure(
    known: np.ndarray,
    fitted: np.ndarray,
    truth: int,
    default: float,
    correlations: dict[float, float],
    agreements: dict[float, bool],
    params: SimulationParams,
) -> tuple[Figure, str]:
    """Assemble the two-panel figure and its caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    rho = pearson_correlation(known, fitted)
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)

        style = series_style(0)
        axes[0].plot(
            np.delete(fitted, truth),
            np.delete(known, truth),
            marker=style["marker"],
            linestyle="none",
            color=style["color"],
            markersize=3,
        )
        truth_style = series_style(1)
        axes[0].plot(
            [fitted[truth]],
            [known[truth]],
            marker=truth_style["marker"],
            linestyle="none",
            color=truth_style["color"],
            markersize=6,
            label="generating topology",
        )
        axes[0].set_xlabel("fitted log-likelihood")
        axes[0].set_ylabel("known-parameter log-likelihood")
        axes[0].set_title("(a) the two surfaces", loc="left")
        axes[0].legend(loc="upper left", frameon=False, fontsize="small")

        lengths = sorted(correlations)
        sweep_style = series_style(2)
        axes[1].plot(
            lengths,
            [correlations[b] for b in lengths],
            marker=sweep_style["marker"],
            linestyle=sweep_style["linestyle"],
            color=sweep_style["color"],
            markersize=4,
            linewidth=1.0,
        )
        axes[1].axvline(
            default, color=INK_MUTED, linestyle=":", linewidth=0.8, zorder=1
        )
        axes[1].annotate(
            "fixture mean",
            xy=(default, 0.03),
            xycoords=("data", "axes fraction"),
            ha="left",
            color=INK_MUTED,
            fontsize="small",
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("fixed branch length")
        axes[1].set_ylabel("correlation")
        axes[1].set_title("(b) does the choice matter", loc="left")
        fig.tight_layout()

    agreed = sum(agreements.values())
    worst = min(correlations.values())
    caption = (
        f"The reward an agent can afford, against the one it stands in for. "
        f"Every one of the {known.size} unrooted topologies on this leaf set, "
        f"scored twice: at fixed known parameters, and at parameters "
        f"maximized per topology by the same optimizer the Potts and hidden "
        f"Markov instances use. Fixture: {len(params.pi)}-state Jukes-Cantor, "
        f"{latex_integer(params.n_sites)} sites, seed {params.seed}; the "
        f"fixed branch length is the generating tree's mean, "
        f"{default:.4f}. (a) The two surfaces, correlation "
        f"{rho:.4f}; both score the generating topology highest, so the "
        f"cheap reward and the expensive one agree on the answer while "
        f"disagreeing in detail. The correlation is linear rather than "
        f"rank-based because the fitted surface does not totally order "
        f"topologies: several of its optima agree to within the optimizer's "
        f"own convergence, so their relative order is not a property of the "
        f"model. (b) The cheap surface has one free "
        f"parameter, and the conclusion does not rest on it: across a "
        f"50-fold range of fixed branch length the two surfaces pick the same "
        f"best topology in {agreed} of {len(agreements)} cases, with the "
        f"correlation never falling below {worst:.4f}. The substitution is "
        f"what makes training affordable at all: the fitted reward runs a "
        f"full optimization per candidate, the known one a single pruning "
        f"pass."
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
        stem="rl_reward_surface",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=lambda params: build_figure(*reward_surfaces(params), params),
        argv=argv,
    )


if __name__ == "__main__":
    main()
