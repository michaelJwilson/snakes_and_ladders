"""The canonical fixtures, their two oracles, and the learners against them.

Issue #597. Every environment the learners here are measured on is a research
problem, so a tie leaves two readings open: the problem is hard, or the
learner is broken. These fixtures close that: each optimum is known from
outside, so a learner that fails here is wrong.

Three kinds of claim, kept apart:

* **the two oracles agree.** `canonical.value_iteration` sweeps the Bellman
  operator over an enumerated state set; `learn.exact.exact_optimal_value`
  recurses over trajectories to a horizon. They share no code, so agreement
  to 1e-12 is a check and not a tautology;
* **the fixtures are the problems they claim to be.** The chain's optimum is
  the far prize and its myopic choice is the near one; the cliff's optimal
  route leaves the bottom row; Hanoi's is `2^d - 1` moves, an integer with no
  tolerance;
* **the learners reach the optimum**, to a stated fraction, on the plain
  case. A learner that does not is reported rather than hidden --- that is
  the suite doing its job.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.canonical import (
    ChainMdp,
    CliffWalk,
    GridWorld,
    SweepError,
    TowersOfHanoi,
    reachable_states,
    shortest_path_length,
    value_iteration,
)
from snakes_and_ladders.learn.critic import Critic, n_state_features
from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.exact import exact_optimal_value
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.ppo import ppo
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import rollout

#: The horizon `exact_optimal_value` is given: long enough that the optimum is
#: reachable inside it on every fixture below, short enough that the
#: enumeration stays affordable.
HORIZON = 12


def _policy_reaches_every_start(
    grid: GridWorld, policy: LinearPolicy, optimal: dict[tuple[int, int], float]
) -> list[bool]:
    """Whether the policy's own greedy episode is optimal from each start.

    Under the *policy's* greedy action and not `greedy_rollout`'s, which takes
    the best immediate reward: on this grid every move costs the same, so
    reward-greedy is a coin toss between four directions and measures the
    environment rather than the learner.
    """
    rng = np.random.default_rng(0)
    reached = []
    rows, columns = grid.shape
    for row in range(rows):
        for column in range(columns):
            if (row, column) == grid.goal:
                continue
            episode = rollout(
                grid, policy, rng, 4 * (rows + columns), start=(row, column)
            )
            reached.append(episode.total_reward >= optimal[row, column] - 1e-9)
    return reached


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("environment", "start", "horizon"),
    [
        (ChainMdp(n_states=6), 0, 10),
        (GridWorld(shape=(3, 3), goal=(2, 2)), (0, 0), 8),
        (CliffWalk(shape=(3, 4)), CliffWalk(shape=(3, 4)).start, 8),
        (TowersOfHanoi(n_disks=3), TowersOfHanoi(n_disks=3).start, 7),
    ],
)
def test_the_two_oracles_agree_on_the_optimal_value(
    environment: Environment[Any, Any], start: Any, horizon: int
) -> None:
    # A sweep over an enumerated state set against a recursion over
    # trajectories: two computations of `V*` sharing no line of code. The
    # horizon is chosen per fixture to be long enough to reach the optimum,
    # since a shorter one would make the recursion's answer the smaller and
    # the disagreement a statement about the horizon.
    swept = value_iteration(environment, start)
    recursed = exact_optimal_value(environment, start, horizon)

    assert swept.values[start] == pytest.approx(recursed, abs=1e-12)


@pytest.mark.mathematical
@pytest.mark.parametrize(
    ("environment", "start"),
    [
        (ChainMdp(n_states=6), 0),
        (GridWorld(shape=(3, 3), goal=(2, 2)), (0, 0)),
        (CliffWalk(shape=(3, 4)), CliffWalk(shape=(3, 4)).start),
        (TowersOfHanoi(n_disks=3), TowersOfHanoi(n_disks=3).start),
    ],
)
def test_bellman_holds_at_every_state(
    environment: Environment[Any, Any], start: Any
) -> None:
    # What `value_iteration` returns is a fixed point of the operator, state
    # by state, and not only at the start. A sweep that stopped early or a
    # state left out of the reachable set fails here rather than silently
    # returning a smaller number.
    settled = value_iteration(environment, start)

    for state, value in settled.values.items():
        if environment.is_terminal(state):
            assert value == 0.0
            continue
        best = max(
            reward + settled.values[successor]
            for successor, reward in (
                environment.step(state, action) for action in environment.actions(state)
            )
        )
        assert value == pytest.approx(best, abs=1e-12)


@pytest.mark.oracle
def test_the_gridworld_optimum_is_its_closed_form() -> None:
    # The cheapest possible independent oracle, and the reason the gridworld
    # is the plain case: every move costs one and the goal pays ten, so the
    # optimal value is the goal reward less the Manhattan distance.
    grid = GridWorld(shape=(4, 5), goal=(3, 4))
    settled = value_iteration(grid, (0, 0))

    for (row, column), value in settled.values.items():
        distance = abs(row - grid.goal[0]) + abs(column - grid.goal[1])
        expected = 0.0 if distance == 0 else grid.goal_reward - float(distance)
        assert value == pytest.approx(expected, abs=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("disks", [1, 2, 3, 4])
def test_hanoi_costs_two_to_the_disks_minus_one(disks: int) -> None:
    # The only fixture whose oracle is an integer, so the assertion carries no
    # tolerance: the shortest solution is `2^d - 1` moves, breadth first and
    # by the closed form, and the optimal value is its cost.
    hanoi = TowersOfHanoi(n_disks=disks)
    settled = value_iteration(hanoi, hanoi.start)

    assert shortest_path_length(hanoi, hanoi.start) == hanoi.optimal_moves
    assert hanoi.optimal_moves == 2**disks - 1
    assert settled.values[hanoi.start] == pytest.approx(
        -float(hanoi.optimal_moves), abs=1e-12
    )
    assert len(reachable_states(hanoi, hanoi.start)) == 3**disks


@pytest.mark.mathematical
def test_the_chain_separates_the_far_prize_from_the_near_reward() -> None:
    # What the fixture is for. The optimum walks to the far end and collects
    # the prize; the myopic choice -- the action with the largest immediate
    # reward -- collects the consolation and ends the episode, so a method
    # that never explores cannot reach the optimum here.
    chain = ChainMdp(n_states=6, prize=10.0, consolation=1.0)
    settled = value_iteration(chain, 0)

    assert settled.values[0] == pytest.approx(chain.prize, abs=1e-12)
    assert settled.optimal_actions(chain, 0) == [1]

    myopic = max(chain.actions(0), key=lambda action: chain.step(0, action)[1])
    successor, reward = chain.step(0, myopic)
    assert reward == chain.consolation
    assert chain.is_terminal(successor)


@pytest.mark.mathematical
def test_the_cliff_walk_optimum_leaves_the_bottom_row() -> None:
    # The route along the cliff's edge is the shortest and every step of it
    # would fall in, so the optimum goes up first. That is the structure
    # SARSA and Q-learning part on (#597's second pull request); here it is
    # asserted of the optimum itself, which neither learner is needed for.
    cliff = CliffWalk(shape=(4, 6))
    settled = value_iteration(cliff, cliff.start)

    detour = shortest_path_length(cliff, cliff.start, goal=cliff.goal)
    assert settled.values[cliff.start] == pytest.approx(-float(detour), abs=1e-12)
    assert detour == (cliff.shape[1] - 1) + 2, "one up, across, one down"
    assert settled.optimal_actions(cliff, cliff.start) == [(-1, 0)]

    stepping_right, reward = cliff.step(cliff.start, (0, 1))
    assert cliff.is_cliff(stepping_right)
    assert reward == cliff.step_cost + cliff.cliff_cost


@pytest.mark.simulated_truth
def test_reinforce_reaches_the_optimum_on_the_plain_case() -> None:
    # The learner against a known optimum rather than against greedy: on the
    # gridworld the optimal return from every start is the closed form above,
    # so "learned" is checkable. The fraction is stated rather than tuned --
    # the greedy policy after training must reach the optimum from every
    # start, which is what an exactly-expressive feature set should buy.
    grid = GridWorld(shape=(4, 4), goal=(3, 3))
    settled = value_iteration(grid, (0, 0))
    policy = LinearPolicy(grid.n_features())

    reinforce(
        grid,
        policy,
        np.random.default_rng(597),
        iterations=300,
        batch=16,
        max_steps=16,
    )

    reached = _policy_reaches_every_start(grid, policy, settled.values)
    assert all(reached), f"{sum(reached)} of {len(reached)} starts optimal"


@pytest.mark.simulated_truth
def test_ppo_reaches_the_optimum_on_the_plain_case() -> None:
    # The same claim for the second learner, on the same fixture and the same
    # oracle, so the two are readable side by side. A learner that failed here
    # would be a defect in code this repository has published results from,
    # and the suite is where that surfaces.
    grid = GridWorld(shape=(4, 4), goal=(3, 3))
    settled = value_iteration(grid, (0, 0))
    policy = LinearPolicy(grid.n_features())

    ppo(
        grid,
        policy,
        Critic(
            n_state_features(grid),
            hidden=None,
            generator=torch.Generator().manual_seed(597),
        ),
        np.random.default_rng(597),
        iterations=60,
        batch=16,
        max_steps=16,
    )

    reached = _policy_reaches_every_start(grid, policy, settled.values)
    assert all(reached), f"{sum(reached)} of {len(reached)} starts optimal"


@pytest.mark.edge_case
def test_a_positive_cycle_is_refused_rather_than_returned() -> None:
    # An undiscounted optimum is unbounded where a cycle pays, and this is
    # where that shows. A value function read off unfinished sweeps is not an
    # optimum and a caller cannot tell it from one that is.
    cycle = ChainMdp(n_states=3, prize=0.0, consolation=0.0)
    with pytest.raises(SweepError, match="did not settle"):
        value_iteration(_Rewarding(cycle), 0, residual=1e-13, max_sweeps=50)


@pytest.mark.edge_case
def test_the_fixtures_refuse_a_shape_they_cannot_be() -> None:
    with pytest.raises(ValueError, match="at least two states"):
        ChainMdp(n_states=1)
    with pytest.raises(ValueError, match="at least one cell"):
        GridWorld(shape=(0, 3))
    with pytest.raises(ValueError, match="lies outside"):
        GridWorld(shape=(2, 2), goal=(5, 5))
    with pytest.raises(ValueError, match=r"at least \(2, 3\)"):
        CliffWalk(shape=(1, 4))
    with pytest.raises(ValueError, match="at least one disk"):
        TowersOfHanoi(n_disks=0)


class _Rewarding(Environment[int, int]):
    """A chain whose every move pays, so its undiscounted optimum diverges.

    Written here rather than in the package: it exists to trip the refusal and
    is not a problem anything solves.
    """

    def __init__(self, inner: ChainMdp) -> None:
        self._inner = inner

    def reset(self, rng: np.random.Generator) -> int:
        return self._inner.reset(rng)

    def actions(self, state: int) -> Sequence[int]:
        return (-1, 1) if 0 < state < self._inner.n_states - 1 else (1,)

    def step(self, state: int, action: int) -> tuple[int, float]:
        return min(max(state + action, 0), self._inner.n_states - 2), 1.0

    def features(self, state: int, actions: Sequence[int]) -> torch.Tensor:
        return self._inner.features(state, actions)

    def n_features(self) -> int:
        return self._inner.n_features()

    def is_terminal(self, state: int) -> bool:
        del state
        return False
