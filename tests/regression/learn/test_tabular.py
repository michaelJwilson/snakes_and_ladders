"""Q-learning and SARSA, and the one fixture whose pass condition is a difference.

Issue #597, second pull request. Three kinds of claim, kept apart:

* **both reach the optimum where theory says they must.** On the grid the
  greedy policy each leaves is optimal from every start, against
  `canonical.value_iteration`'s ``V*``. Exact equality, not a tolerance: the
  rewards are integers and the transitions are deterministic, so an optimal
  route's return is an integer;
* **their values diverge, and only one of them should match ``V*``.**
  Q-learning's fixed point is ``q*`` and SARSA's is ``q_pi`` for the
  epsilon-greedy policy. Asserting agreement for both would assert something
  false, so one is asserted at ``V*`` and the other asserted *away* from it;
* **on the cliff walk they disagree in the documented direction.** Q-learning
  takes the row beside the cliff and SARSA the row above it; the greedy return
  orders one way and the epsilon-greedy return the other. An implementation
  where they agree here is wrong even though both converged, and this is the
  only test in the module whose pass condition is a difference.

The chain is the fourth claim and the sharpest about exploration: with the
default pessimistic initial value neither learner ever sees the far prize, and
that is asserted rather than tuned away --- the fixture is there to show it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from snakes_and_ladders.learn.canonical import (
    ChainMdp,
    CliffWalk,
    GridWorld,
    value_iteration,
)
from snakes_and_ladders.learn.tabular import ActionValues, q_learning, sarsa

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

    Exact equality: the step cost and the goal reward are integers and the
    transitions are deterministic, so an optimal route's undiscounted return
    is an integer and a tolerance here would only hide a wrong route.

    ``start=None`` draws a fresh cell per episode from ``GridWorld.reset``,
    which is uniform over non-goal cells. That matters for what the claim
    means: with one fixed start a deviation says which states went unvisited,
    and with uniform starts it says something about the method.
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


@pytest.mark.mathematical
def test_q_learning_matches_the_optimal_values_and_sarsa_does_not() -> None:
    """``max_a Q`` is ``V*`` for one method and is not for the other.

    Q-learning's update bootstraps from the greedy successor, so its fixed
    point is ``q*`` and its table is comparable to value iteration's directly.
    SARSA bootstraps from the action its epsilon-greedy policy actually takes,
    so its fixed point is ``q_pi``: on this grid the exploratory steps cost
    real reward and its values sit below ``V*`` by more than a unit.

    Both numbers are pinned. A SARSA table that matched ``V*`` would mean the
    on-policy update had been written as the off-policy one --- the defect this
    module is most likely to have, and the reason the second assertion is a
    lower bound rather than a comment.
    """
    grid = GridWorld()
    optimal = value_iteration(grid, (0, 0))
    off_policy = q_learning(
        grid, np.random.default_rng(3), episodes=EPISODES, epsilon=0.2, max_steps=100
    )
    on_policy = sarsa(
        grid, np.random.default_rng(3), episodes=EPISODES, epsilon=0.2, max_steps=100
    )

    # Measured on this host: 2.14e-05 and 1.8889 at these settings. The bound
    # is loose against the first and tight against the second on purpose ---
    # what separates the two methods is four orders of magnitude, not the last
    # bit of a constant-step-size fixed point, which does not reach `V*`
    # exactly at any finite episode count.
    assert off_policy.largest_deviation(optimal) < 1e-4
    assert on_policy.largest_deviation(optimal) > 1.0


@pytest.mark.mathematical
def test_the_cliff_walk_separates_the_two_methods() -> None:
    """Q-learning hugs the cliff, SARSA keeps a row clear, and the returns order.

    The optimal route leaves the start, runs along the row immediately above
    the cliff and drops onto the goal: 7 moves at -1 each. One row higher is
    9 moves. Under epsilon-greedy behaviour the first route risks a random
    step into the cliff at -101, which is what SARSA's values account for and
    Q-learning's do not.

    So the two claims point opposite ways and both are asserted:
    Q-learning's *greedy* return is the better one, and SARSA's
    *epsilon-greedy* return is. A run where either ordering fails has lost the
    distinction, whatever the tables converged to.
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


@pytest.mark.mathematical
@pytest.mark.parametrize("learner", [q_learning, sarsa])
def test_the_chain_is_a_trap_without_optimistic_initialisation(
    learner: TabularLearner,
) -> None:
    """Pessimistic: neither learner sees the prize. Optimistic: both value it.

    Reaching the far end of an 8-chain from the near end needs a run of
    exploratory steps outward, probability ``(epsilon / 2)**7 = 7.8e-10`` per
    episode, so at 2,000 episodes the near reward is absorbing and the learned
    value of the start is the consolation. That is the fixture's point and it
    is asserted, not tuned away.

    With an initial value above the prize every untried action looks better
    than every tried one, so the walk happens by construction rather than by
    luck, and ``Q(0, +1)`` reaches the prize.
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


@pytest.mark.mathematical
def test_the_chain_optimum_ties_so_a_greedy_route_may_dither() -> None:
    """The oracle itself ties at every interior state, and the learner may too.

    The chain charges nothing per step, so walking back and then out again
    pays exactly what walking out pays: ``V*`` is the prize at every interior
    state and `value_iteration`'s own `optimal_actions` returns *both* moves
    at state 1. A greedy tie-break can therefore produce a route that never
    terminates while every action it took was optimal.

    This is why :meth:`ActionValues.greedy_episode` reports ``terminated``
    rather than refusing, and why the chain's claim above is about a value and
    not about a route. It is a property of an undiscounted fixture with no step
    cost, established here against the oracle so that a future non-terminating
    greedy route on this fixture is read as the tie it is.
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


@pytest.mark.structural
def test_the_two_learners_share_their_loop() -> None:
    """With no exploration and one deterministic route, the two agree exactly.

    At ``epsilon = 0`` the behaviour policy is greedy, so the action SARSA
    bootstraps from *is* the maximising one and the two updates coincide. The
    tables must then be identical, which is what says the difference measured
    on the cliff comes from the continuation term and not from two separately
    drifting implementations.
    """
    grid = GridWorld(shape=(3, 3), goal=(2, 2))
    off_policy = q_learning(
        grid, np.random.default_rng(1), start=(0, 0), episodes=200, epsilon=0.0
    )
    on_policy = sarsa(
        grid, np.random.default_rng(1), start=(0, 0), episodes=200, epsilon=0.0
    )
    assert off_policy.values == on_policy.values
    assert off_policy.updates == on_policy.updates


@pytest.mark.edge_case
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

    ``learning_rate = 0`` is the one worth naming: it runs, it costs the
    episodes, and it returns the initial table unchanged --- a number that
    looks like a result.
    """
    with pytest.raises(ValueError, match=message):
        q_learning(GridWorld(), np.random.default_rng(0), start=(0, 0), **kwargs)  # type: ignore[arg-type]


@pytest.mark.edge_case
def test_a_terminal_start_costs_nothing_and_teaches_nothing() -> None:
    """Starting on the goal is a legal no-op, not a division by zero.

    The episode has no actions to take, so there is nothing to update and the
    table stays empty. Reported through ``updates`` rather than by raising:
    a caller sweeping starts across a grid will hit the goal cell.
    """
    grid = GridWorld()
    learned = q_learning(grid, np.random.default_rng(0), start=grid.goal, episodes=10)
    assert learned.updates == 0
    assert learned.value(grid.goal) == 0.0
    assert learned.greedy_actions(grid.goal) == []
