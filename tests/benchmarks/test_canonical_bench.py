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

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.learn.canonical import GridWorld, value_iteration
from snakes_and_ladders.learn.critic import Critic, n_state_features
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.ppo import ppo
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import rollout

GRID = GridWorld(shape=(4, 4), goal=(3, 3))
ITERATIONS = 20
BATCH = 8
STEPS = 16


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
