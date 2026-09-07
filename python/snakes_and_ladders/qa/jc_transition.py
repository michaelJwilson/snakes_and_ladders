"""QA figure: analytic Jukes-Cantor P(t) against simulated frequencies.

Draws the closed-form transition probabilities of eq. (jc) as curves, and
overlays the substitution frequencies `snakes_and_ladders.sim.simulate` actually produced
along each branch of a fixture's tree. The two must agree within the
fixture's declared Monte Carlo tolerance -- the same check
`tests/regression/test_jc_simulate.py` makes numerically, shown here so a
reader can see the simulator sitting on its oracle rather than take the
assertion on trust.

Renders what `snakes_and_ladders.sim` computed; it does not reimplement the model
(`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN,
    letter_style,
    series_style,
)
from snakes_and_ladders.sim.jc import jc_transition_probabilities
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

_MODEL_NAME = "Jukes-Cantor"
_CURVE_POINTS = 200


def empirical_transitions(params: SimulationParams) -> list[tuple[float, float, float]]:
    """Per-branch empirical stay/change frequencies from one simulated dataset.

    Parameters
    ----------
    params : SimulationParams
        Fixture parameters; the simulation is run at its own seed, so the
        figure shows the same draw the regression test checks.

    Returns
    -------
    list[tuple[float, float, float]]
        One ``(branch_length, p_stay, p_change)`` per non-root branch.
        ``p_stay`` is the fraction of sites whose state is unchanged across
        the branch; ``p_change`` is the fraction reaching one specific other
        state, i.e. the off-diagonal entry, so it is comparable with
        ``P_ij(t)`` rather than with ``1 - P_ii(t)``.
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )

    points: list[tuple[float, float, float]] = []
    for parent, child in edges(dataset.tau):
        if child.branch_length is None:  # pragma: no cover
            # Unreachable: simulate_alignment above raises on a non-root node
            # without a branch length, so this narrows the type for mypy
            # rather than guarding a path a caller can reach.
            msg = f"non-root node {child.name!r} has no branch_length"
            raise ValueError(msg)
        parent_states = dataset.node_states[parent.name]
        child_states = dataset.node_states[child.name]

        stayed = float(np.mean(parent_states == child_states))
        # Each of the k-1 off-diagonal targets is equally likely under JC,
        # so one of them is (1 - p_stay) / (k - 1).
        changed = (1.0 - stayed) / (params.k - 1)
        points.append((child.branch_length, stayed, changed))
    return points


def build_figure(params: SimulationParams) -> tuple[Figure, str]:
    """Render the analytic curves with the simulated frequencies overlaid.

    Parameters
    ----------
    params : SimulationParams
        Fixture the simulated points are drawn from.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure, and its caption.
    """
    points = empirical_transitions(params)
    branch_lengths = [t for t, _, _ in points]

    t_max = max(branch_lengths) * 1.35
    grid = np.linspace(0.0, t_max, _CURVE_POINTS)
    stay_curve = np.array(
        [jc_transition_probabilities(t, k=params.k)[0, 0] for t in grid]
    )
    change_curve = np.array(
        [jc_transition_probabilities(t, k=params.k)[0, 1] for t in grid]
    )

    worst = 0.0
    with letter_style():
        fig, ax = plt.subplots(figsize=ONE_COLUMN)

        stay = series_style(0)
        change = series_style(1)

        ax.plot(grid, stay_curve, color=stay["color"], linestyle="-", zorder=2)
        ax.plot(grid, change_curve, color=change["color"], linestyle="-", zorder=2)

        for index, (t, p_stay, p_change) in enumerate(points):
            label_stay = "simulated" if index == 0 else None
            ax.plot(
                t,
                p_stay,
                marker=stay["marker"],
                color=stay["color"],
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=1.1,
                zorder=3,
                label=label_stay,
            )
            ax.plot(
                t,
                p_change,
                marker=change["marker"],
                color=change["color"],
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=1.1,
                zorder=3,
            )
            worst = max(
                worst,
                abs(p_stay - jc_transition_probabilities(t, k=params.k)[0, 0]),
                abs(p_change - jc_transition_probabilities(t, k=params.k)[0, 1]),
            )

        # Direct labels rather than a legend box: two series, and the
        # contrast check obliges visible labels.
        ax.annotate(
            r"$P_{ii}(t)$  no substitution",
            xy=(grid[-1], stay_curve[-1]),
            xytext=(-4, 17),
            textcoords="offset points",
            ha="right",
            color=stay["color"],
        )
        ax.annotate(
            r"$P_{ij}(t)$  to a given other state",
            xy=(grid[-1], change_curve[-1]),
            xytext=(-4, -17),
            textcoords="offset points",
            ha="right",
            color=change["color"],
        )
        ax.axhline(
            1.0 / params.k,
            color=INK_MUTED,
            linewidth=0.6,
            linestyle=":",
            zorder=1,
        )
        ax.annotate(
            r"stationary $1/k$",
            xy=(0.0, 1.0 / params.k),
            xytext=(2, 4),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=7.5,
        )

        ax.set_xlabel("branch length $t$ (expected substitutions/site)")
        ax.set_ylabel("transition probability")
        ax.set_xlim(0.0, t_max)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="center left")
        fig.tight_layout()

    caption = (
        f"Closed-form {_MODEL_NAME} transition probabilities for k = {params.k} "
        f"(curves) against the frequencies the simulator produced "
        f"(markers), one pair per branch of the {len(points)}-branch fixture "
        f"tree. Simulated at seed {params.seed} over {params.n_sites} sites. "
        f"Largest deviation {worst:.2e}, within the fixture's declared Monte "
        f"Carlo tolerance of {params.tolerance:.0e}. The simulator is "
        f"validated against this analytic form, never against the likelihood "
        f"code it feeds."
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
        stem="jc_transition",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=build_figure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
