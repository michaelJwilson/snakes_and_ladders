"""What a PPO update costs, split into the terms issue #391 asked about.

See tests/regression/learn/test_learn_ppo.py for correctness. Three units,
and the reason for three is that the ticket proposed replacing the smallest of
them: an **iteration** is a batch collected, a critic refit and `epochs`
clipped steps on it; **scoring the batch's neighbourhoods** is what those
epochs share and is hoisted out of them; and the **clipped surrogate** is the
one equation TorchRL's `ClipPPOLoss` computes. Timing the third against the
first is what says whether a port of it could pay --- it is 0.6% of an
iteration, so it cannot
(`docs/experiments/007-batched-rollout-and-torchrl-ppo.md`).
"""

from __future__ import annotations

import numpy as np
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.learn.critic import Critic, n_state_features
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.ppo import (
    _log_probabilities,
    _neighbourhoods,
    episode_advantages,
    ppo,
    ppo_loss,
)
from snakes_and_ladders.learn.rollout import rollout

_COUPLING = 0.75
_FIELD = np.array([0.4, -0.1, -0.3])
_CHAIN_LENGTH = 4
_HORIZON = 6
_BATCH = 32
_CLIP = 0.2
_LAM = 0.95


def _landscape() -> PottsLandscape:
    return PottsLandscape(_COUPLING, _FIELD, _CHAIN_LENGTH)


def _critic(landscape: PottsLandscape) -> Critic:
    return Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )


def test_one_ppo_iteration_benchmark(benchmark: BenchmarkFixture) -> None:
    """A batch collected, the critic refit, and four clipped steps on it."""
    landscape = _landscape()

    def update() -> None:
        ppo(
            landscape,
            LinearPolicy(2),
            _critic(landscape),
            np.random.default_rng(0),
            iterations=1,
            batch=_BATCH,
            max_steps=_HORIZON,
        )

    benchmark(update)


def test_scoring_a_batch_s_neighbourhoods_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """What every epoch of an iteration used to repeat, now done once."""
    landscape = _landscape()
    policy = LinearPolicy(2)
    rng = np.random.default_rng(0)
    episodes = [rollout(landscape, policy, rng, _HORIZON) for _ in range(_BATCH)]

    scored = benchmark(_neighbourhoods, landscape, episodes)

    assert len(scored) == _BATCH


def test_the_clipped_surrogate_benchmark(benchmark: BenchmarkFixture) -> None:
    """The one equation `ClipPPOLoss` would replace, on a scored batch."""
    landscape = _landscape()
    policy = LinearPolicy(2)
    critic = _critic(landscape)
    rng = np.random.default_rng(0)
    episodes = [rollout(landscape, policy, rng, _HORIZON) for _ in range(_BATCH)]
    neighbourhoods = _neighbourhoods(landscape, episodes)
    advantages = episode_advantages(landscape, episodes, critic, lam=_LAM)
    old = [t.detach() for t in _log_probabilities(policy, neighbourhoods)]

    loss, fraction = benchmark(ppo_loss, old, old, advantages, clip=_CLIP)

    # At the collecting policy every ratio is one, so nothing clips; the
    # regression counterpart pins the value.
    assert fraction == 0.0
