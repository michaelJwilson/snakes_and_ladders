"""The only like-for-like timing axis the learners have.

Correctness is pinned in `tests/regression/learn/test_learn_canonical.py`.

Everywhere else in this package the problem differs with the method --- a
topology search against a Potts landscape --- so a wall clock compares two
things at once. On one canonical fixture it compares one: the same
environment, the same feature width, the same episode budget, and the
learner is what changes. `STATUS.md` carries the episode counts each needs to
reach the optimum, which is the other half of the comparison and not a time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.learn.canonical import GridWorld, value_iteration
from sal.learn.critic import Critic, n_state_features
from sal.learn.policy import LinearPolicy
from sal.learn.ppo import ppo
from sal.learn.reinforce import reinforce
from sal.learn.rollout import rollout
from sal.learn.tabular import ActionValues, q_learning, sarsa

#: What both tabular learners are, seen from here: same signature, same return.
TabularLearner = Callable[..., ActionValues[tuple[int, int], tuple[int, int]]]

GRID = GridWorld(shape=(4, 4), goal=(3, 3))
ITERATIONS = 20
BATCH = 8
STEPS = 16
#: Episodes the tabular learners are billed over. Fewer than the suite's 2,000
#: so the bench stays inside the per-PR tier; the cost is linear in it, so a
#: reading here converts to any count by multiplying.
TABULAR_EPISODES = 200


def test_reinforce_on_the_grid_benchmark(benchmark: BenchmarkFixture) -> None:
    def train() -> LinearPolicy:
        policy = LinearPolicy(GRID.n_features())
        reinforce(
            GRID,
            policy,
            np.random.default_rng(597),
            iterations=ITERATIONS,
            batch=BATCH,
            max_steps=STEPS,
        )
        return policy

    policy = benchmark(train)

    assert policy.weights.shape == (GRID.n_features(),)


def test_ppo_on_the_grid_benchmark(benchmark: BenchmarkFixture) -> None:
    def train() -> LinearPolicy:
        policy = LinearPolicy(GRID.n_features())
        ppo(
            GRID,
            policy,
            Critic(
                n_state_features(GRID),
                hidden=None,
                generator=torch.Generator().manual_seed(597),
            ),
            np.random.default_rng(597),
            iterations=ITERATIONS,
            batch=BATCH,
            max_steps=STEPS,
        )
        return policy

    policy = benchmark(train)

    assert policy.weights.shape == (GRID.n_features(),)


@pytest.mark.parametrize("shape", [(4, 4), (8, 8), (16, 16)])
def test_value_iteration_benchmark(
    benchmark: BenchmarkFixture, shape: tuple[int, int]
) -> None:
    # The oracle's own cost, which grows with the state count and is what
    # decides how large a canonical fixture can stay checkable.
    grid = GridWorld(shape=shape, goal=(shape[0] - 1, shape[1] - 1))

    settled = benchmark(value_iteration, grid, (0, 0))

    assert settled.residual <= 1e-13


def test_one_episode_benchmark(benchmark: BenchmarkFixture) -> None:
    # The unit both learners are billed in, so an episode count in `STATUS.md`
    # converts to a wall clock without a second measurement.
    policy = LinearPolicy(GRID.n_features())
    rng = np.random.default_rng(0)

    episode = benchmark(rollout, GRID, policy, rng, STEPS)

    assert len(episode.states) >= 1


@pytest.mark.parametrize("learner", [q_learning, sarsa], ids=["q_learning", "sarsa"])
def test_tabular_control_benchmark(
    benchmark: BenchmarkFixture, learner: TabularLearner
) -> None:
    # The two tabular methods on the same environment as the policy-gradient
    # learners above, which is the only like-for-like axis this module has:
    # everywhere else the problem differs with the method. Both run the same
    # loop and differ in one expression, so a gap between these two readings is
    # the cost of the `max` and nothing else.
    def train() -> ActionValues[tuple[int, int], tuple[int, int]]:
        return learner(
            GRID,
            np.random.default_rng(597),
            episodes=TABULAR_EPISODES,
            max_steps=STEPS,
        )

    learned = benchmark(train)

    assert learned.episodes == TABULAR_EPISODES
    assert learned.updates > 0
