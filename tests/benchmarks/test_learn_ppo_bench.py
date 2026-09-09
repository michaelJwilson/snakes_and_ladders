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

Each of the two equations #391 proposed replacing is timed on both sides: the
declined TorchRL fronts live in `snakes_and_ladders.sandbox`
(`sandbox/CLAUDE.md`) and are timed here beside ours on the same batch. The
ratio is a property of a TorchRL version, so it is re-measured rather than
quoted from the pull request that took the decision; the pins that say the
two sides compute the same number are in
tests/regression/learn/test_learn_ppo_torchrl.py. Only the two TorchRL
benchmarks skip without the `frameworks` extra, so the reference numbers are
taken wherever the suite runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.learn.critic import Critic, n_state_features, state_features
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.ppo import (
    _log_probabilities,
    _neighbourhoods,
    episode_advantages,
    generalized_advantages,
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


def test_the_clipped_surrogate_through_torchrl_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The declined `ClipPPOLoss` front, on the batch above.

    The front takes the policy and the scored neighbourhoods where
    `ppo_loss` takes log-probabilities, because `ClipPPOLoss` calls an actor
    to compute them; `snakes_and_ladders.sandbox.torchrl_clip` says why that
    is the level the substitution has to be stated at. The extra work that
    difference implies is part of what an adoption would have cost, so it is
    inside what is timed.
    """
    torchrl_clip = pytest.importorskip("snakes_and_ladders.sandbox.torchrl_clip")
    landscape = _landscape()
    policy = LinearPolicy(2)
    critic = _critic(landscape)
    rng = np.random.default_rng(0)
    episodes = [rollout(landscape, policy, rng, _HORIZON) for _ in range(_BATCH)]
    neighbourhoods = _neighbourhoods(landscape, episodes)
    advantages = episode_advantages(landscape, episodes, critic, lam=_LAM)
    old = [t.detach() for t in _log_probabilities(policy, neighbourhoods)]

    loss = benchmark(
        torchrl_clip.clipped_objective,
        policy,
        neighbourhoods,
        old,
        advantages,
        clip=_CLIP,
    )

    assert loss.requires_grad


def _prepared(
    landscape: PottsLandscape, critic: Critic
) -> tuple[list[list[float]], list[list[float]], list[bool]]:
    """One batch's rewards, critic values and termination flags, per episode."""
    policy = LinearPolicy(2)
    rng = np.random.default_rng(0)
    episodes = [rollout(landscape, policy, rng, _HORIZON) for _ in range(_BATCH)]
    with torch.no_grad():
        values = [
            [
                float(critic(state_features(landscape, state)[None, :])[0])
                for state in episode.states
            ]
            for episode in episodes
        ]
    return (
        [list(episode.rewards) for episode in episodes],
        values,
        [episode.terminated for episode in episodes],
    )


def test_the_batch_s_advantages_benchmark(benchmark: BenchmarkFixture) -> None:
    """`eq:gae` over a whole batch, the critic already read.

    The critic is outside the timing on both sides: it is the same call for
    either implementation of the recursion, and leaving it in would divide
    the ratio #391 asked for by a term neither owns.
    """
    landscape = _landscape()
    rewards, values, terminated = _prepared(landscape, _critic(landscape))

    def estimate() -> list[list[float]]:
        return [
            generalized_advantages(r, v, lam=_LAM, terminated=t)
            for r, v, t in zip(rewards, values, terminated, strict=True)
        ]

    advantages = benchmark(estimate)

    assert len(advantages) == _BATCH


def test_the_batch_s_advantages_through_torchrl_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """The declined `GAE` front, on the same batch and the same critic values.

    The tensordict and the estimator are built inside the timing, because a
    front pays for them on every call. The experiment that declined `GAE`
    reports the prebuilt figure as well, and the two differ by that
    construction rather than by the recursion.
    """
    torchrl_advantage = pytest.importorskip(
        "snakes_and_ladders.sandbox.torchrl_advantage"
    )
    landscape = _landscape()
    rewards, values, terminated = _prepared(landscape, _critic(landscape))

    advantages = benchmark(
        torchrl_advantage.episode_advantages, rewards, values, terminated, lam=_LAM
    )

    assert len(advantages) == _BATCH
