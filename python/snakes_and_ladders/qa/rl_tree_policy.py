"""QA figure: what a learned policy can and cannot do on a rugged tree landscape.

Milestone 2.1's phylogenetic half asks whether a learned proposal policy beats
hill climbing. On the 6-taxon fixture the question is unaskable, because greedy
reaches the enumerated optimum from every start. Issue #177 supplied a fixture
where it does not. This figure reports what happened when a policy was trained
on it, and it is a negative result.

Panel (a) is the landscape: every unrooted topology scored and sorted, with the
states no single NNI move improves marked. Panel (b) is the answer: the
policy's success rate over independent training seeds against greedy's, both at
the same per-episode budget from the same starting topologies.

The two panels together say why the answer is what it is. Every episode --
greedy and learned alike -- ends at one of the marked states, because
`snakes_and_ladders.learn.rollout` stops when `is_terminal` holds and
`TopologyEnvironment.is_terminal` holds exactly at a state with no improving
move. So the task an agent faces here is not "escape a local optimum" but
"choose which one to walk into", and a policy scoring moves by the single
feature this environment exposes -- the improvement a move buys -- is hill
climbing up to a temperature. Neither half of that leaves much to learn.

Renders what `snakes_and_ladders.learn` and `snakes_and_ladders.search` computed; it reimplements no
estimator and no move set (`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout
from snakes_and_ladders.qa.figure import QAFigure, latex_integer
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import RewardModel, TopologyEnvironment
from snakes_and_ladders.search.topology import Topology, enumerate_topologies
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

# The budget every searcher here is held to, in moves per episode. Long enough
# that no episode is cut short: each one ends at a local optimum well inside it.
HORIZON = 30

# Starting topologies, drawn once and shared by greedy and every policy, so the
# comparison is paired rather than two samples of different problems.
STARTS = 50

# Rollouts per start for the stochastic policy. Greedy is deterministic and
# needs one. Each rollout costs the same episode as greedy's, so the per-episode
# success rate is budget-matched; this averages the policy's, it does not take
# a best-of.
ROLLOUTS_PER_START = 16

# Independent training runs. Eight because a single seed's result is a draw:
# `test_learn_reinforce.py` reports the Potts comparison over eight for the
# same reason.
TRAINING_SEEDS = 8

# 640 episodes, and the budget is a measurement rather than a habit. At 80 the
# policy leads greedy on 13 of 16 seeds and is still *worse on average*,
# because three of those seeds never leave the neighbourhood of the untrained
# policy: training has not converged, and a result read there measures the
# budget. At 640 the spread across seeds falls from 0.130 to 0.014 and the
# comparison is stable. Reporting the shorter budget's 13-of-16 while its mean
# sits below the baseline would be choosing the statistic that flatters.
ITERATIONS = 40
BATCH = 16


def _environment(
    params: SimulationParams, moves: MoveSet
) -> tuple[TopologyEnvironment, list[str]]:
    """The reward surface an agent sees, and the taxa it is over.

    Returns
    -------
    tuple[TopologyEnvironment, list[str]]
        The environment, scored at the generating tree's mean branch length,
        and the sorted taxon names -- returned rather than recovered from the
        environment, which keeps its alignment private.
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)
    environment = TopologyEnvironment(
        alignment,
        params.k,
        np.asarray(params.pi),
        branch_length=float(
            np.mean([child.branch_length for _, child in edges(params.tau)])
        ),
        reward=RewardModel.KNOWN,
        moves=moves,
    )
    return environment, sorted(alignment)


def measure(
    params: SimulationParams,
) -> tuple[np.ndarray, np.ndarray, float, list[float], int, int, float]:
    """Run the comparison this figure reports.

    Parameters
    ----------
    params : SimulationParams
        The generating truth; issue #177's fixture is the one this is for.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, list[float], int, int, float]
        Every topology's score ascending; the scores of the states no NNI move
        improves; greedy's success rate; the learned policy's success rate per
        training seed; the number of episodes each policy was trained on; the
        taxon count; and an untrained policy's success rate, the control that
        separates "learned nothing" from "learned the baseline".
    """
    environment, taxa = _environment(params, MoveSet.NNI)
    topologies = list(enumerate_topologies(taxa))
    scores = np.array(sorted(environment.score(topology) for topology in topologies))
    optima = np.array(
        sorted(
            environment.score(topology)
            for topology in topologies
            if environment.is_terminal(topology)
        )
    )
    best = float(scores[-1])

    start_rng = np.random.default_rng(params.seed + 1000)
    starts = [environment.reset(start_rng) for _ in range(STARTS)]

    def reached(endpoints: list[Topology]) -> float:
        return float(
            np.mean(
                [abs(environment.score(state) - best) < 1e-9 for state in endpoints]
            )
        )

    greedy = reached(
        [greedy_rollout(environment, start, HORIZON).states[-1] for start in starts]
    )

    untrained = LinearPolicy(environment.n_features())
    # One generator for the whole sweep. Constructing it inside the
    # comprehension would reseed it per rollout and measure a single episode
    # repeated, which reads as a rate of exactly zero.
    control_rng = np.random.default_rng(99)
    control = reached(
        [
            rollout(environment, untrained, control_rng, HORIZON, start=start).states[
                -1
            ]
            for start in starts
            for _ in range(ROLLOUTS_PER_START)
        ]
    )

    learned = []
    episodes = 0
    for seed in range(TRAINING_SEEDS):
        policy = LinearPolicy(environment.n_features())
        training = reinforce(
            environment,
            policy,
            np.random.default_rng(seed),
            iterations=ITERATIONS,
            batch=BATCH,
            max_steps=HORIZON,
        )
        episodes = training.episodes
        probe = np.random.default_rng(1000 + seed)
        learned.append(
            reached(
                [
                    rollout(environment, policy, probe, HORIZON, start=start).states[-1]
                    for start in starts
                    for _ in range(ROLLOUTS_PER_START)
                ]
            )
        )
    return scores, optima, greedy, learned, episodes, len(taxa), control


def build_figure(
    scores: np.ndarray,
    optima: np.ndarray,
    greedy: float,
    learned: list[float],
    episodes: int,
    n_taxa: int,
    control: float,
    params: SimulationParams,
) -> tuple[Figure, str]:
    """Assemble the two-panel figure and its caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    deltas = np.array(learned) - greedy
    with letter_style():
        figure, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)

        landscape = series_style(0)
        axes[0].plot(
            np.arange(1, len(scores) + 1),
            scores,
            color=landscape["color"],
            linewidth=1.0,
            label="all topologies",
        )
        for rank, value in enumerate(optima):
            axes[0].axhline(
                value,
                color=INK_MUTED,
                linestyle="-",
                linewidth=0.6,
                alpha=0.7,
                label="no improving NNI move" if rank == 0 else None,
            )
        axes[0].set_xlabel("topology, ranked by score")
        axes[0].set_ylabel("log-likelihood")
        axes[0].legend(loc="lower right")

        policy_style = series_style(1)
        axes[1].axhline(
            greedy,
            color=INK_MUTED,
            linestyle=":",
            linewidth=1.0,
            label="hill climbing",
        )
        axes[1].plot(
            np.arange(1, len(learned) + 1),
            learned,
            linestyle="none",
            marker=policy_style["marker"],
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=policy_style["color"],
            color=policy_style["color"],
            label="learned policy",
        )
        axes[1].set_xlabel("training seed")
        axes[1].set_ylabel("reached the enumerated maximum")
        axes[1].set_xticks(np.arange(1, len(learned) + 1))
        axes[1].legend(loc="lower right")
        figure.tight_layout()

    caption = (
        "A learned policy against hill climbing on a landscape where hill "
        "climbing fails, and why the answer is a null result. Fixture: "
        f"{latex_integer(len(scores))} unrooted topologies on "
        f"{latex_integer(n_taxa)} taxa, "
        f"{latex_integer(params.n_sites)} sites, 4-state Jukes-Cantor, seed "
        f"{latex_integer(params.seed)}, scored at the generating tree's mean "
        "branch length. (a) Every topology, ranked, with a rule at each of the "
        f"{latex_integer(len(optima))} states no single NNI move improves. The "
        "best of them is the generating topology; the rest are traps. (b) The "
        "fraction of episodes reaching that maximum, from "
        f"{latex_integer(STARTS)} shared starting topologies at "
        f"{latex_integer(HORIZON)} moves each -- greedy once per start, the "
        f"policy {latex_integer(ROLLOUTS_PER_START)} times per start so its "
        "rate is an average and not a best-of, which makes the budgets "
        f"comparable. Over {latex_integer(TRAINING_SEEDS)} independent "
        f"training runs of {latex_integer(episodes)} episodes each, the policy "
        f"reaches the maximum on {np.mean(learned):.3f} of episodes against "
        f"greedy's {greedy:.3f}: a difference of {deltas.mean():+.4f} with a "
        f"standard deviation of {deltas.std(ddof=1):.4f} across seeds, and "
        f"{latex_integer(int((deltas > 0).sum()))} of "
        f"{latex_integer(TRAINING_SEEDS)} seeds ahead. That is not a "
        "difference. It is not that nothing was learned: an untrained policy, "
        f"uniform over the same moves, reaches the maximum on {control:.3f} of "
        "episodes, so training moves the policy across most of the distance "
        "from chance to the baseline and then stops there. Panel (a) says why "
        "it stops there: an episode ends "
        "when no move improves, so every run terminates at one of those rules "
        "and the task is which trap to enter rather than how to leave it, "
        "while a policy scoring moves by this environment's single feature -- "
        "the improvement a move buys -- is hill climbing up to a temperature."
    )
    return figure, caption


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
        stem="rl_tree_policy",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=lambda params: build_figure(*measure(params), params),
        argv=argv,
    )


if __name__ == "__main__":
    main()
