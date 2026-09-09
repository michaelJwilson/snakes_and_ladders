"""The batched rollout against the sequential one it was meant to beat (issue #392).

See tests/regression/search/test_search_gym_vector.py for correctness: at one
copy `rollout_batch` is `rollout` draw for draw, so these two benchmarks time
the same episodes through two loops and nothing else varies.

The unit is one episode, because that is what a training run buys. The batch
sizes are 1, 4 and 16, and the number the ticket wants is whether the cost per
episode falls as the batch grows. It does not:
`docs/experiments/007-batched-rollout-and-torchrl-ppo.md` carries the table.

Both sides are timed although only one is on a hot path. The batched rollout
was declined and lives in `snakes_and_ladders.sandbox.gym_vector`
(`sandbox/CLAUDE.md`); the ratio that declined it is a property of a
Gymnasium version, so it is re-measured here rather than quoted from the
pull request that took the decision.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.learn.environment import Episode
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.rollout import rollout

pytest.importorskip("gymnasium")
from snakes_and_ladders.sandbox.gym_vector import rollout_batch

_COUPLING = 0.75
_FIELD = np.array([0.4, -0.1, -0.3])
_CHAIN_LENGTH = 4
_HORIZON = 6
_EPISODES = 32
_WEIGHTS = torch.tensor([0.3, -0.6], dtype=torch.float64)


def _landscape() -> PottsLandscape:
    return PottsLandscape(_COUPLING, _FIELD, _CHAIN_LENGTH)


def _policy() -> LinearPolicy:
    policy = LinearPolicy(2)
    policy.set_weights(_WEIGHTS)
    return policy


def _n_max(landscape: PottsLandscape) -> int:
    return _CHAIN_LENGTH * (landscape.n_states - 1)


def test_a_batch_of_episodes_rolled_one_at_a_time_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The reference: `_EPISODES` episodes through `learn.rollout.rollout`."""
    landscape, policy = _landscape(), _policy()

    def collect() -> list[Episode[tuple[int, ...], tuple[int, int]]]:
        rng = np.random.default_rng(0)
        return [rollout(landscape, policy, rng, _HORIZON) for _ in range(_EPISODES)]

    episodes = benchmark(collect)

    assert len(episodes) == _EPISODES


@pytest.mark.parametrize("n", [1, 4, 16])
def test_a_batch_of_episodes_rolled_through_gymnasium_vector_benchmark(
    benchmark: BenchmarkFixture, n: int
) -> None:
    """The same episodes through `n` copies stepped together."""
    landscape, policy = _landscape(), _policy()

    def collect() -> list[Episode[tuple[int, ...], tuple[int, int]]]:
        return rollout_batch(
            landscape,
            policy,
            list(np.random.default_rng(0).spawn(n)),
            n_max=_n_max(landscape),
            max_steps=_HORIZON,
            episodes=_EPISODES,
        )

    episodes = benchmark(collect)

    assert len(episodes) == _EPISODES
