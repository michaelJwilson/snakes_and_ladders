"""The Potts-chain decisions the TorchRL pins, benchmarks and goals share (issue #977)."""

from __future__ import annotations

import numpy as np
import torch
from snakes_and_ladders.learn.environment import Episode
from snakes_and_ladders.learn.potts import PottsEnvironment

from tests.regression.learn.conftest import potts_environment

#: The scored policy's weights.
WEIGHTS = (0.3, -0.6)

PottsEpisode = Episode[tuple[int, ...], tuple[int, int]]


def potts_decisions(steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``steps`` Potts decisions' features and a uniform index, seed 977, 64 starts."""
    environment = potts_environment()
    rng = np.random.default_rng(977)
    states = [environment.reset(rng) for _ in range(64)]
    table = torch.as_tensor(
        np.stack([environment.features(s, environment.actions(s)) for s in states])
    )
    pick = torch.as_tensor(rng.integers(0, len(states), steps))
    taken = torch.as_tensor(rng.integers(0, table.shape[1], steps))
    return table[pick], taken


def greedy_episodes(
    environment: PottsEnvironment,
    weights: torch.Tensor,
    count: int,
    length: int,
    seed: int,
) -> list[PottsEpisode]:
    """``count`` rollouts of ``length`` decisions, each the highest-scoring action.

    `ReinforceLoss` resamples the action: deterministic, that is the argmax.
    """
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(count):
        state = environment.reset(rng)
        states, actions, rewards = [state], [], []
        for _ in range(length):
            available = environment.actions(state)
            scores = torch.as_tensor(environment.features(state, available)) @ weights
            action = available[int(torch.argmax(scores.detach()))]
            state, reward = environment.step(state, action)
            actions.append(action)
            rewards.append(reward)
            states.append(state)
        episodes.append(
            Episode(tuple(states), tuple(actions), tuple(rewards), terminated=False)
        )
    return episodes


def reinforce_decisions(
    environment: PottsEnvironment, episodes: list[PottsEpisode]
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Every decision's neighbourhood features, the index taken, and its return-to-go."""
    features, taken, returns = [], [], []
    for episode in episodes:
        to_go = episode.returns_to_go()
        for step, action in enumerate(episode.actions):
            available = environment.actions(episode.states[step])
            features.append(environment.features(episode.states[step], available))
            taken.append(available.index(action))
            returns.append(to_go[step])
    return torch.as_tensor(np.stack(features)), np.asarray(taken), np.asarray(returns)


def ppo_loss_and_gradient(
    features: torch.Tensor,
    taken: torch.Tensor,
    old: torch.Tensor,
    advantages: torch.Tensor,
    episode_length: int,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``learn.ppo.ppo_loss`` under ``WEIGHTS`` and its gradient, episodes of ``episode_length``."""
    from snakes_and_ladders.learn import ppo

    weights = torch.tensor(WEIGHTS, dtype=torch.float64, requires_grad=True)
    n_episodes = taken.shape[0] // episode_length
    # The policy's log-probability of each decision, vectorized over them.
    log_probability = torch.log_softmax(features @ weights, dim=1)
    current = log_probability.gather(1, taken[:, None])[:, 0]
    value, _ = ppo.ppo_loss(
        list(current.reshape(n_episodes, episode_length)),
        list(old.reshape(n_episodes, episode_length)),
        # Views, as TorchRL is handed its advantages: a tensor, not floats.
        list(advantages.reshape(n_episodes, episode_length)),
        clip=clip,
    )
    (gradient,) = torch.autograd.grad(value, weights)
    return value, gradient
