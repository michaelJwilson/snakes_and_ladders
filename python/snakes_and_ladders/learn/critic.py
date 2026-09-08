"""The state-value critic, and the oracle that referees it (issue #313).

A critic estimates ``V^pi(s)``, the return a policy still expects from
``s``; with a telescoping reward that is the improvement the search will
still find. It replaces the constant baseline of ``reinforce.py`` with a
state-dependent one, which leaves the policy gradient unbiased and lowers
its variance, and it is the leaf evaluator a planner reads. It is held to
enumeration: on an instance small enough, ``exact.exact_expected_return``
is ``V^pi`` exactly, and a fitted critic is scored against it rather than
against its own training loss (``learn/CLAUDE.md``).

The critic reads **state features**, which the environment protocol does
not supply; :func:`state_features` derives them from the action features
the protocol does supply -- their mean and their maximum over the
neighbourhood, plus the log of its size -- so every instance is covered
without a change to the protocol. A feature constant across states is not
unidentifiable here as it is for the policy, since a value is not a
softmax: the critic carries a bias.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.learn.environment import Environment, Episode


def state_features[S, A](environment: Environment[S, A], state: S) -> torch.Tensor:
    """Mean and maximum of the available actions' features, and ``log(1 + n_actions)``.

    Shape ``(2 * environment.n_features() + 1,)``. A terminal state, with no
    actions, gets zeros for the aggregates and ``0`` for the count.
    """
    width = environment.n_features()
    actions = environment.actions(state)
    if not actions:
        return torch.zeros(2 * width + 1, dtype=torch.float64)
    features = environment.features(state, actions).to(torch.float64)
    return torch.cat(
        [
            features.mean(dim=0),
            features.max(dim=0).values,
            torch.tensor([float(np.log1p(len(actions)))], dtype=torch.float64),
        ]
    )


def n_state_features[S, A](environment: Environment[S, A]) -> int:
    """Width of :func:`state_features`."""
    return 2 * environment.n_features() + 1


class Critic(torch.nn.Module):
    """``V(s)`` from state features: linear when ``hidden`` is ``None``, a two-layer MLP otherwise."""

    def __init__(
        self, n_features: int, *, hidden: int | None, generator: torch.Generator
    ) -> None:
        super().__init__()
        if n_features < 1:
            msg = f"n_features must be >= 1, got {n_features}"
            raise ValueError(msg)
        if hidden is None:
            self.net: torch.nn.Module = torch.nn.Linear(n_features, 1)
        else:
            if hidden < 1:
                msg = f"hidden must be >= 1, got {hidden}"
                raise ValueError(msg)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, 1),
            )
        self.net.to(torch.float64)
        for parameter in self.net.parameters():
            if parameter.dim() > 1:
                torch.nn.init.xavier_uniform_(parameter, generator=generator)
            else:
                torch.nn.init.zeros_(parameter)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Values for a batch of state features, shape ``(n,)``."""
        out: torch.Tensor = self.net(features.to(torch.float64))[..., 0]
        return out


@dataclass(frozen=True)
class CriticFit:
    """What fitting a critic recorded: the loss per step and the targets' count."""

    losses: tuple[float, ...]
    n_targets: int


def monte_carlo_targets[S, A](
    environment: Environment[S, A], episodes: Sequence[Episode[S, A]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """State features and returns-to-go for every non-final state in ``episodes``.

    The return-to-go is an unbiased sample of ``V^pi(s_t)`` under the policy
    that produced the episodes; a truncated episode's tail is biased low by
    what the budget cut off, and ``Episode.terminated`` says which.
    """
    features, targets = [], []
    for episode in episodes:
        returns = episode.returns_to_go()
        for step, state in enumerate(episode.states[:-1]):
            features.append(state_features(environment, state))
            targets.append(returns[step])
    if not features:
        msg = "no decisions in the episodes to fit a critic to"
        raise ValueError(msg)
    return torch.stack(features), torch.tensor(targets, dtype=torch.float64)


def temporal_difference_targets[S, A](
    environment: Environment[S, A], episodes: Sequence[Episode[S, A]], critic: Critic
) -> tuple[torch.Tensor, torch.Tensor]:
    """State features and one-step bootstrapped targets ``r_t + V(s_{t+1})``, zero past a terminal state.

    Lower variance than the Monte Carlo target and biased by the critic's
    own error, which is the trade GAE interpolates in ``ppo.py``.
    """
    features, targets = [], []
    with torch.no_grad():
        for episode in episodes:
            for step, state in enumerate(episode.states[:-1]):
                successor = episode.states[step + 1]
                bootstrap = (
                    0.0
                    if environment.is_terminal(successor)
                    else float(
                        critic(state_features(environment, successor)[None, :])[0]
                    )
                )
                features.append(state_features(environment, state))
                targets.append(episode.rewards[step] + bootstrap)
    if not features:
        msg = "no decisions in the episodes to fit a critic to"
        raise ValueError(msg)
    return torch.stack(features), torch.tensor(targets, dtype=torch.float64)


def fit_critic(
    critic: Critic,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int = 200,
    learning_rate: float = 1e-2,
) -> CriticFit:
    """Adam on the squared error over the whole batch, ``steps`` times."""
    if features.shape[0] != targets.shape[0]:
        msg = f"{features.shape[0]} feature rows but {targets.shape[0]} targets"
        raise ValueError(msg)
    if steps < 1:
        msg = f"steps must be >= 1, got {steps}"
        raise ValueError(msg)
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.mean((critic(features) - targets) ** 2)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))
    return CriticFit(tuple(losses), int(targets.shape[0]))


__all__ = [
    "Critic",
    "CriticFit",
    "fit_critic",
    "monte_carlo_targets",
    "n_state_features",
    "state_features",
    "temporal_difference_targets",
]
