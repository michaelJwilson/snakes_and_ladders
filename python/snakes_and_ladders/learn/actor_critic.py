"""Actor-critic: REINFORCE with the critic as its baseline (issue #313).

``eq:reinforce`` holds for any baseline that does not depend on the action
sampled at that step, and a state-value critic is such a baseline. The
estimator is therefore the same score function weighted by an advantage
``G_t - V(s_t)`` instead of ``G_t - b``; when the critic is exact this is
the lowest-variance member of that family, and when the critic is a
constant it is REINFORCE, which is the identity the tests pin. The critic
is refitted between policy updates on the episodes just collected, from
Monte Carlo returns by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.learn.critic import (
    Critic,
    fit_critic,
    monte_carlo_targets,
    state_features,
)
from snakes_and_ladders.learn.environment import Environment, Episode
from snakes_and_ladders.learn.policy import TrainablePolicy
from snakes_and_ladders.learn.rollout import rollout


@dataclass(frozen=True)
class ActorCriticTraining:
    """The outcome of an actor-critic run.

    Parameters
    ----------
    mean_returns : tuple[float, ...]
        Mean sampled return per iteration; a diagnostic, as in ``reinforce.py``.
    critic_losses : tuple[float, ...]
        The critic's final squared error per iteration.
    episodes : int
        Episodes sampled in total, the budget's unit.
    """

    mean_returns: tuple[float, ...]
    critic_losses: tuple[float, ...]
    episodes: int


def advantage_surrogate_loss[S, A](
    environment: Environment[S, A],
    policy: TrainablePolicy,
    episodes: Sequence[Episode[S, A]],
    advantages: Sequence[Sequence[float]],
) -> torch.Tensor:
    """``-(1/N) sum_episodes sum_t log pi(a_t | s_t) A_t``: minus the score-function estimator with the given advantages."""
    if not episodes:
        msg = "need at least one episode to form an estimate"
        raise ValueError(msg)
    total = torch.zeros((), dtype=policy.dtype)
    for episode, episode_advantages in zip(episodes, advantages, strict=True):
        for step, action in enumerate(episode.actions):
            state = episode.states[step]
            available = environment.actions(state)
            log_probabilities = policy.log_probabilities(
                environment.features(state, available)
            )
            total = (
                total
                + log_probabilities[available.index(action)] * episode_advantages[step]
            )
    return -total / len(episodes)


def critic_advantages[S, A](
    environment: Environment[S, A], episodes: Sequence[Episode[S, A]], critic: Critic
) -> list[list[float]]:
    """``G_t - V(s_t)`` per step per episode, the critic read without gradient."""
    advantages = []
    with torch.no_grad():
        for episode in episodes:
            returns = episode.returns_to_go()
            values = [
                float(critic(state_features(environment, state)[None, :])[0])
                for state in episode.states[:-1]
            ]
            advantages.append([g - v for g, v in zip(returns, values, strict=True)])
    return advantages


def actor_critic[S, A](
    environment: Environment[S, A],
    policy: TrainablePolicy,
    critic: Critic,
    rng: np.random.Generator,
    *,
    iterations: int,
    batch: int,
    max_steps: int,
    learning_rate: float = 0.05,
    critic_steps: int = 50,
) -> ActorCriticTraining:
    """Alternate a critic refit and one advantage-weighted policy step, ``iterations`` times.

    The critic is fitted on the batch *before* the advantages are read from
    it, so the baseline at step ``t`` depends on the episode's own return
    through the fit. That is a small bias the exact-critic identity does not
    have, and it is why the identity is pinned with a fixed critic and the
    training curve is reported as a diagnostic.
    """
    if iterations < 1 or batch < 1:
        msg = f"iterations and batch must be >= 1, got {iterations}, {batch}"
        raise ValueError(msg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    mean_returns, critic_losses = [], []
    for _ in range(iterations):
        episodes = [rollout(environment, policy, rng, max_steps) for _ in range(batch)]
        features, targets = monte_carlo_targets(environment, episodes)
        critic_losses.append(
            fit_critic(critic, features, targets, steps=critic_steps).losses[-1]
        )
        advantages = critic_advantages(environment, episodes, critic)
        optimizer.zero_grad()
        advantage_surrogate_loss(environment, policy, episodes, advantages).backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        mean_returns.append(
            float(np.mean([episode.total_reward for episode in episodes]))
        )
    return ActorCriticTraining(
        tuple(mean_returns), tuple(critic_losses), iterations * batch
    )


__all__ = [
    "ActorCriticTraining",
    "actor_critic",
    "advantage_surrogate_loss",
    "critic_advantages",
]
