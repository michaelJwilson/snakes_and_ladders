"""Q-learning and SARSA, and the one fixture whose pass condition is a difference.

Issue #597. On the grid both greedy policies are optimal from every start
against `canonical.value_iteration`'s ``V*``, exactly (integer returns).
Q-learning's values reach ``V*`` and SARSA's (``q_pi`` of epsilon-greedy) are
asserted away from it. On the cliff walk they disagree in the documented
direction, the one pass condition that is a difference. On the chain the
pessimistic default never sees the far prize, asserted rather than tuned away.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sal.learn.canonical import (
    ChainMdp,
    CliffWalk,
    GridWorld,
    value_iteration,
)
from sal.learn.tabular import ActionValues, q_learning, sarsa

#: What both learners are, seen from a test that runs each on the same fixture.
TabularLearner = Callable[..., ActionValues[Any, Any]]

#: Episodes every run below uses. At 2,000 Q-learning's table is 2.14e-05 from
#: ``V*`` on the 4x4 grid and its greedy policy is optimal from every start;
#: the cliff needs no more, and the module runs in 1.8 s at this count.
EPISODES = 2_000
#: Exploration rate, held fixed: annealing it to zero makes SARSA converge to
#: ``q*`` and erases the distinction the cliff walk measures.
EPSILON = 0.1


def _grid_cells(grid: GridWorld) -> list[tuple[int, int]]:
    """Every non-goal cell, which is every state an episode can start in."""
    rows, columns = grid.shape
    return [
        (row, column)
        for row in range(rows)
        for column in range(columns)
        if (row, column) != grid.goal
    ]


@pytest.mark.oracle
@pytest.mark.parametrize("learner", [q_learning, sarsa])
def test_both_reach_the_grid_optimum_from_every_start(learner: TabularLearner) -> None:
    """The greedy policy is optimal from all 15 cells, against ``V*``.

    Exact (integer returns); uniform starts via ``start=None``, so it speaks of the method.
    """
    grid = GridWorld()
    optimal = value_iteration(grid, (0, 0))
    learned = learner(
        grid, np.random.default_rng(3), episodes=EPISODES, epsilon=0.2, max_steps=100
    )

    for cell in _grid_cells(grid):
        episode = learned.greedy_episode(grid, cell, max_steps=100)
        assert episode.terminated, (
            f"the greedy policy did not reach the goal from {cell}"
        )
        assert episode.total_reward == optimal.values[cell]


@pytest.mark.analytic
@pytest.mark.oracle
def test_q_learning_matches_the_optimal_values_and_sarsa_does_not() -> None:
    """``max_a Q`` is ``V*`` for one method and is not for the other.

    SARSA at ``V*`` would mean its update was written off-policy: a lower bound.
    """
    grid = GridWorld()
    optimal = value_iteration(grid, (0, 0))
    off_policy = q_learning(
        grid,
        np.random.default_rng(3),
        episodes=EPISODES,
        epsilon=0.2,
        max_steps=100,
    )
    on_policy = sarsa(
        grid,
        np.random.default_rng(3),
        episodes=EPISODES,
        epsilon=0.2,
        max_steps=100,
    )

    # Measured: 2.14e-05 and 1.8889. Four orders separate the methods; a
    # constant step size never reaches `V*` exactly.
    assert off_policy.largest_deviation(optimal) < 1e-4
    assert on_policy.largest_deviation(optimal) > 1.0


@pytest.mark.analytic
def test_the_cliff_walk_separates_the_two_methods() -> None:
    """Q-learning hugs the cliff, SARSA keeps a row clear, and the returns order.

    7 moves beside the cliff against 9 above; greedy favours one, epsilon-greedy the other.
    """
    cliff = CliffWalk()
    optimal = value_iteration(cliff, cliff.start)
    assert optimal.values[cliff.start] == -7.0

    off_policy = q_learning(
        cliff,
        np.random.default_rng(7),
        start=cliff.start,
        episodes=2 * EPISODES,
        epsilon=EPSILON,
    )
    on_policy = sarsa(
        cliff,
        np.random.default_rng(7),
        start=cliff.start,
        episodes=2 * EPISODES,
        epsilon=EPSILON,
    )

    greedy_off = off_policy.greedy_episode(cliff, cliff.start, max_steps=200)
    greedy_on = on_policy.greedy_episode(cliff, cliff.start, max_steps=200)
    assert greedy_off.terminated
    assert greedy_on.terminated

    # Q-learning finds the optimum; SARSA gives up exactly the two moves that
    # buy the clearance. Both are integers, so both are pinned exactly.
    assert greedy_off.total_reward == -7.0
    assert greedy_on.total_reward == -9.0

    # The routes, read as rows: the cliff is the bottom row's interior, so the
    # route beside it is row `rows - 2` and the clear one is a row above that.
    cliff_row = cliff.shape[0] - 1
    off_rows = {state[0] for state in greedy_off.states}
    on_rows = {state[0] for state in greedy_on.states}
    assert cliff_row - 1 in off_rows
    assert min(on_rows) < cliff_row - 1

    # And the ordering reverses under the policy that was actually run.
    rng = np.random.default_rng(11)
    behaviour_off = float(
        np.mean(
            [
                off_policy.behaviour_episode(
                    cliff, cliff.start, rng, epsilon=EPSILON, max_steps=200
                ).total_reward
                for _ in range(200)
            ]
        )
    )
    behaviour_on = float(
        np.mean(
            [
                on_policy.behaviour_episode(
                    cliff, cliff.start, rng, epsilon=EPSILON, max_steps=200
                ).total_reward
                for _ in range(200)
            ]
        )
    )
    assert behaviour_on > behaviour_off


@pytest.mark.analytic
@pytest.mark.parametrize("learner", [q_learning, sarsa])
def test_the_chain_is_a_trap_without_optimistic_initialisation(
    learner: TabularLearner,
) -> None:
    """Pessimistic: neither learner sees the prize. Optimistic: both value it.

    Reaching the far end is ``(epsilon / 2)**7 = 7.8e-10`` per episode.
    """
    chain = ChainMdp()
    optimal = value_iteration(chain, 0)
    assert optimal.values[0] == chain.prize

    trapped = learner(chain, np.random.default_rng(5), start=0, episodes=EPISODES)
    assert trapped.value(0) == pytest.approx(chain.consolation, abs=1e-6)

    optimistic = learner(
        chain,
        np.random.default_rng(5),
        start=0,
        episodes=EPISODES,
        initial_value=chain.prize,
    )
    assert optimistic.values[0][1] > optimistic.values[0][-1]
    assert optimistic.values[0][1] == pytest.approx(chain.prize, abs=0.1)


@pytest.mark.analytic
def test_the_chain_optimum_ties_so_a_greedy_route_may_dither() -> None:
    """The oracle itself ties at every interior state, and the learner may too.

    No step cost: both moves are optimal at state 1, so ``terminated`` is reported.
    """
    chain = ChainMdp()
    optimal = value_iteration(chain, 0)
    assert optimal.optimal_actions(chain, 1) == [-1, 1]
    assert optimal.optimal_actions(chain, 0) == [1]

    learned = q_learning(
        chain,
        np.random.default_rng(5),
        start=0,
        episodes=EPISODES,
        initial_value=chain.prize,
    )
    assert learned.greedy_actions(1) != []


@pytest.mark.infra
def test_the_two_learners_share_their_loop() -> None:
    """With no exploration and one deterministic route, the two agree exactly.

    So the cliff difference is the continuation term, not two drifting loops.
    """
    grid = GridWorld(shape=(3, 3), goal=(2, 2))
    off_policy = q_learning(
        grid,
        np.random.default_rng(1),
        start=(0, 0),
        episodes=200,
        epsilon=0.0,
    )
    on_policy = sarsa(
        grid,
        np.random.default_rng(1),
        start=(0, 0),
        episodes=200,
        epsilon=0.0,
    )
    assert off_policy.values == on_policy.values
    assert off_policy.updates == on_policy.updates


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"episodes": 0}, "episodes must be >= 1"),
        ({"learning_rate": 0.0}, "learning_rate must be in"),
        ({"learning_rate": 1.5}, "learning_rate must be in"),
        ({"epsilon": -0.1}, "epsilon must be in"),
        ({"epsilon": 1.1}, "epsilon must be in"),
        ({"max_steps": 0}, "max_steps must be >= 1"),
    ],
)
def test_an_unusable_setting_is_refused(kwargs: dict[str, float], message: str) -> None:
    """Each range is refused with the reason, not clamped.

    ``learning_rate = 0`` would return the initial table as a result.
    """
    with pytest.raises(ValueError, match=message):
        q_learning(GridWorld(), np.random.default_rng(0), start=(0, 0), **kwargs)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_a_terminal_start_costs_nothing_and_teaches_nothing() -> None:
    """Starting on the goal is a legal no-op, not a division by zero.

    Reported through ``updates``: a sweep over starts hits the goal cell.
    """
    grid = GridWorld()
    learned = q_learning(grid, np.random.default_rng(0), start=grid.goal, episodes=10)
    assert learned.updates == 0
    assert learned.value(grid.goal) == 0.0
    assert learned.greedy_actions(grid.goal) == []


@pytest.mark.smoke
@pytest.mark.patch
def test_both_table_episodes_read_terminated_from_the_state_they_ended_in() -> None:
    # The other two of the four tails `Episode.from_rollout` replaced (issue
    # #862): a greedy route that reaches the goal, and one the budget cuts
    # off before it does.
    grid, corner = GridWorld(), (0, 0)
    learned = q_learning(grid, np.random.default_rng(0), episodes=200, epsilon=EPSILON)
    episodes = [
        learned.greedy_episode(grid, corner, max_steps=budget) for budget in (0, 1, 100)
    ]
    episodes += [
        learned.behaviour_episode(
            grid, corner, np.random.default_rng(1), epsilon=EPSILON, max_steps=budget
        )
        for budget in (0, 100)
    ]

    for episode in episodes:
        assert episode.terminated == grid.is_terminal(episode.states[-1])
        assert len(episode.states) == len(episode.actions) + 1
        assert len(episode.rewards) == len(episode.actions)
