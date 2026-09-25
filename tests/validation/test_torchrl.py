"""`learn.ppo` and `learn.reinforce` against TorchRL's estimators, in a subprocess (issues #322, #376, #977).

To ``1e-10``: ``generalized_advantages`` against ``GAE`` at ``gamma = 1``,
four ``lam``, both end flags; ``ppo_loss`` and its gradient against
``ClipPPOLoss`` at three clips, one active; ``surrogate_loss`` against
``ReinforceLoss`` at three baselines on greedy episodes. Mapped conventions:
truncation read from ``done``; means over decisions against ours per
episode, scaled by ``n_decisions / n_episodes``; the advantage supplied, with
the constant critic the constructor requires.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.learn import ppo
from sal.learn.policy import LinearPolicy
from sal.learn.potts import PottsEnvironment
from sal.learn.reinforce import surrogate_loss
from sal.learn.rollout import rollout
from sal.validation import torchrl

from tests._frameworks import requires
from tests.regression.learn.conftest import potts_environment
from tests.validation._rl import greedy_episodes, reinforce_decisions

pytestmark = [
    pytest.mark.validation,
    requires("torchrl"),
]

# One fixed rollout each, in the shape `generalized_advantages` takes: rewards
# per decision and one value per state, the last being the final state's.
ROLLOUTS = (
    ([1.0, -0.5, 2.0], [1.0, 2.0, 0.5, 4.0]),
    ([0.3], [0.2, -0.7]),
    ([-1.0, 0.25, 0.5, 0.125, -2.0], [0.0, 1.0, -1.0, 0.5, 0.25, 3.0]),
)
BASELINES = (0.0, 0.5, -1.25)


@pytest.mark.oracle
def test_the_advantages_are_torchrl_s_gae_at_gamma_one() -> None:
    cases = [
        torchrl.Rollout(rewards, values, lam, terminated)
        for rewards, values in ROLLOUTS
        for terminated in (True, False)
        for lam in (0.0, 0.5, 0.95, 1.0)
    ]
    theirs = torchrl.gae(cases)
    for case, advantages in zip(cases, theirs.advantages, strict=True):
        ours = ppo.generalized_advantages(
            case.rewards, case.values, lam=case.lam, terminated=case.terminated
        )
        np.testing.assert_allclose(advantages, ours, rtol=0.0, atol=1e-10)


def _log_probabilities(
    policy: LinearPolicy, features: torch.Tensor, taken: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [
            policy.log_probabilities(rows)[index]
            for rows, index in zip(features, taken, strict=True)
        ]
    )


def _sampled_decisions(
    environment: PottsEnvironment, policy: LinearPolicy, seed: int, episodes: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Every decision's neighbourhood features, the index taken, and its episode."""
    rng = np.random.default_rng(seed)
    rolled = [rollout(environment, policy, rng, 3) for _ in range(episodes)]
    features, taken, owner = [], [], []
    for index, episode in enumerate(rolled):
        for state, action in zip(episode.states, episode.actions, strict=False):
            available_actions = environment.actions(state)
            features.append(environment.features(state, available_actions))
            taken.append(available_actions.index(action))
            owner.append(index)
    # Every neighbourhood on the chain has the same width: no padding, no mask.
    return torch.as_tensor(np.stack(features)), torch.tensor(taken), owner


@pytest.mark.oracle
def test_the_clipped_objective_and_its_gradient_are_torchrl_s() -> None:
    environment = potts_environment()
    collector = LinearPolicy(2)
    collector.set_weights(torch.tensor([0.1, 0.2], dtype=torch.float64))
    features, taken, owner = _sampled_decisions(environment, collector, 0, 12)
    old = _log_probabilities(collector, features, taken).detach()
    n_episodes = max(owner) + 1
    advantages = torch.tensor(np.random.default_rng(1).normal(size=len(taken)))

    def by_episode(values: torch.Tensor) -> list[torch.Tensor]:
        return [
            torch.stack([v for v, o in zip(values, owner, strict=True) if o == e])
            for e in range(n_episodes)
        ]

    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    current = _log_probabilities(policy, features, taken)
    per_episode = [[float(a) for a in part] for part in by_episode(advantages)]
    clips = (0.1, 0.2, 0.5)
    theirs = torchrl.clip_ppo_loss(
        features.numpy(),
        taken.numpy(),
        old.numpy(),
        advantages.numpy(),
        policy.weights.detach().numpy(),
        clips=clips,
        n_episodes=n_episodes,
    )
    for clip, value, gradient in zip(
        clips, theirs.values, theirs.gradients, strict=True
    ):
        ours, _ = ppo.ppo_loss(
            by_episode(current), by_episode(old), per_episode, clip=clip
        )
        (our_gradient,) = torch.autograd.grad(ours, policy.weights, retain_graph=True)
        np.testing.assert_allclose(value, float(ours.detach()), rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(gradient, our_gradient.numpy(), rtol=0.0, atol=1e-10)
        # Clipping is active somewhere, or the pin says nothing about the clip.
        ratio = torch.exp(current - old)
        assert bool(((ratio < 1 - clip) | (ratio > 1 + clip)).any())


@pytest.mark.oracle
def test_the_surrogate_loss_and_its_gradient_are_torchrl_s_reinforce_loss() -> None:
    environment: PottsEnvironment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    episodes = greedy_episodes(environment, policy.weights, 12, 3, 0)
    features, taken, returns = reinforce_decisions(environment, episodes)
    # The decisions-to-episodes factor is a claim only where the counts differ.
    assert len(taken) > len(episodes) > 1
    theirs = torchrl.reinforce_loss(
        features.numpy(),
        taken,
        returns[None, :] - np.asarray(BASELINES)[:, None],
        policy.weights.detach().numpy(),
        n_episodes=len(episodes),
    )
    # The actor resamples; the greedy episodes make that draw ours.
    assert theirs.resampled is not None
    np.testing.assert_array_equal(
        theirs.resampled, np.broadcast_to(taken, theirs.resampled.shape)
    )
    for baseline, value, gradient in zip(
        BASELINES, theirs.values, theirs.gradients, strict=True
    ):
        ours = surrogate_loss(environment, policy, episodes, baseline)
        (our_gradient,) = torch.autograd.grad(ours, policy.weights)
        np.testing.assert_allclose(value, float(ours.detach()), rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(gradient, our_gradient.numpy(), rtol=0.0, atol=1e-10)
