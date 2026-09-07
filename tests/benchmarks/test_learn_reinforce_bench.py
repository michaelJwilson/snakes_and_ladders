"""Benchmarks for the RL machinery.

See tests/regression/test_learn_reinforce.py for correctness. Two units
matter and they are measured against each other. An **episode** is what a
training run spends its time in, and a **gradient update** is one batch of
them plus the backward pass; if the update ever costs much more than its
episodes, the estimator has become the bottleneck rather than the
environment, which would be worth knowing before the environment gets
expensive.

The enumerated oracle is timed too, for a different reason: it costs
``|A| ** horizon``, so its number is the constraint on how far the exact
checks can be pushed rather than a performance target.
"""

from __future__ import annotations

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.learn.exact import exact_policy_gradient
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import rollout

_COUPLING = 0.75
_FIELD = np.array([0.4, -0.1, -0.3])
_CHAIN_LENGTH = 4
_HORIZON = 6
_BATCH = 32


def _landscape() -> PottsLandscape:
    return PottsLandscape(_COUPLING, _FIELD, _CHAIN_LENGTH)


def test_one_episode_benchmark(benchmark: BenchmarkFixture) -> None:
    """What a training run spends its time in."""
    landscape = _landscape()
    policy = LinearPolicy(2)
    rng = np.random.default_rng(0)

    episode = benchmark(rollout, landscape, policy, rng, _HORIZON)

    # Benchmarks assert shape only; correctness is pinned in the regression
    # counterpart.
    assert len(episode.actions) <= _HORIZON


def test_one_gradient_update_benchmark(benchmark: BenchmarkFixture) -> None:
    """One batch of episodes and the backward pass over them."""
    landscape = _landscape()

    def update() -> None:
        reinforce(
            landscape,
            LinearPolicy(2),
            np.random.default_rng(0),
            iterations=1,
            batch=_BATCH,
            max_steps=_HORIZON,
        )

    benchmark(update)


def test_enumerated_gradient_benchmark(benchmark: BenchmarkFixture) -> None:
    """The oracle, whose cost bounds how far the exact checks can go."""
    landscape = _landscape()
    policy = LinearPolicy(2)

    gradient = benchmark(exact_policy_gradient, landscape, policy, (2, 1, 1, 0), 3)

    assert gradient.shape == (2,)
