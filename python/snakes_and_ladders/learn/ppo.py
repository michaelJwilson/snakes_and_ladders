"""Proximal policy optimization with generalized advantage estimation (issue #313).

REINFORCE and the actor-critic take one gradient step per batch, because the
score-function estimator is only valid for the policy that collected the
data. PPO reuses a batch for several epochs by importance-weighting each
step with the ratio ``rho_t = pi(a_t | s_t) / pi_old(a_t | s_t)`` and
clipping the ratio's effect on the objective (``eq:ppo-clip``), so an
update cannot move the policy far from the one the advantages were
estimated under (Schulman et al., 2017). The advantages are the exponentially
weighted temporal differences of ``eq:gae`` at ``gamma = 1``, whose
``lambda`` interpolates between the one-step bootstrap (``lambda = 0``,
low variance, the critic's bias) and the Monte Carlo return less the
value (``lambda = 1``, unbiased, high variance) (Schulman et al., 2016).

Two limits are pinned. With ``lambda = 1``, one epoch and no clipping the
PPO gradient at the collecting policy is the actor-critic's, since every
ratio is one; and the enumerated expected return rises across iterations on
an instance where that return is exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.learn.critic import Critic, fit_critic, state_features
from snakes_and_ladders.learn.environment import Environment, Episode
from snakes_and_ladders.learn.policy import EpsilonGreedyPolicy, TrainablePolicy
from snakes_and_ladders.learn.rollout import rollout
from snakes_and_ladders.opt.schedule import Schedule


@dataclass(frozen=True)
class PPOTraining:
    """The outcome of a PPO run: the same diagnostics as the actor-critic, plus the clipped fraction."""

    mean_returns: tuple[float, ...]
    critic_losses: tuple[float, ...]
    clipped_fraction: tuple[float, ...]
    episodes: int


def generalized_advantages(
    rewards: Sequence[float], values: Sequence[float], *, lam: float, terminated: bool
) -> list[float]:
    """``eq:gae`` at ``gamma = 1``: ``A_t = sum_l lam^l delta_{t+l}``, ``delta_t = r_t + V(s_{t+1}) - V(s_t)``.

    ``values`` has one entry per state, one more than ``rewards``; the last
    is the value of the final state, taken as zero when the episode
    ``terminated`` (nothing improves from a local optimum) and kept when
    the step budget truncated it (the search would have gone on).
    """
    if len(values) != len(rewards) + 1:
        msg = (
            f"{len(rewards)} rewards need {len(rewards) + 1} values, got {len(values)}"
        )
        raise ValueError(msg)
    if not 0.0 <= lam <= 1.0:
        msg = f"lam must be in [0, 1], got {lam}"
        raise ValueError(msg)
    tail = 0.0 if terminated else values[-1]
    advantages = [0.0] * len(rewards)
    running = 0.0
    for step in reversed(range(len(rewards))):
        next_value = tail if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + next_value - values[step]
        running = delta + lam * running
        advantages[step] = running
    return advantages


def episode_advantages[S, A](
    environment: Environment[S, A],
    episodes: Sequence[Episode[S, A]],
    critic: Critic,
    *,
    lam: float,
) -> list[list[float]]:
    """GAE per episode, the critic read without gradient."""
    advantages = []
    with torch.no_grad():
        for episode in episodes:
            values = [
                float(critic(state_features(environment, state)[None, :])[0])
                for state in episode.states
            ]
            advantages.append(
                generalized_advantages(
                    episode.rewards, values, lam=lam, terminated=episode.terminated
                )
            )
    return advantages


type Neighbourhoods = list[list[tuple[torch.Tensor, int]]]


def _neighbourhoods[S, A](
    environment: Environment[S, A], episodes: Sequence[Episode[S, A]]
) -> Neighbourhoods:
    """Every decision's neighbourhood features and the index taken, per episode.

    Scored once for the whole batch, because PPO reuses a batch for
    ``epochs`` gradient steps and the neighbourhood a decision was taken
    from does not move while the policy is updated on it. Scoring it inside
    the epoch loop instead spent the batch's ``environment.actions`` and
    ``environment.features`` four times over: on the Potts chain at the CI
    budget of 1,920 episodes that was 8.06 s against 7.14 s, with the
    training curve identical iteration for iteration.
    """
    out: Neighbourhoods = []
    for episode in episodes:
        steps = []
        for step, action in enumerate(episode.actions):
            state = episode.states[step]
            available = environment.actions(state)
            steps.append(
                (environment.features(state, available), available.index(action))
            )
        out.append(steps)
    return out


def _log_probabilities(
    policy: TrainablePolicy | EpsilonGreedyPolicy,
    neighbourhoods: Neighbourhoods,
) -> list[torch.Tensor]:
    """``log pi(a_t | s_t)`` for every decision, one tensor per episode, on the current graph."""
    return [
        torch.stack(
            [policy.log_probabilities(features)[index] for features, index in steps]
        )
        if steps
        else torch.zeros(0, dtype=torch.float64)
        for steps in neighbourhoods
    ]


def ppo_loss(
    log_probabilities: Sequence[torch.Tensor],
    old_log_probabilities: Sequence[torch.Tensor],
    advantages: Sequence[Sequence[float]],
    *,
    clip: float,
) -> tuple[torch.Tensor, float]:
    """``eq:ppo-clip``: minus the mean over episodes of ``sum_t min(rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t)``.

    Returns the loss and the fraction of steps at which the clipped term was
    the active one, a diagnostic of how far the policy has moved.

    The episodes are concatenated and clipped in one pass. The objective is a
    sum over every decision divided by the number of *episodes*, so the
    episode boundaries carry no term of their own and a loop over them was
    Python interpreting what one vectorized clip does: at the CI budget of
    1,920 episodes the loop cost 6.84 s against 6.25 s, and the sole
    difference is the order the sum is accumulated in.
    """
    if clip <= 0.0:
        msg = f"clip must be positive (use math.inf for no clipping), got {clip}"
        raise ValueError(msg)
    dtype = log_probabilities[0].dtype if log_probabilities else torch.float64
    current = torch.cat(
        [t for t in log_probabilities if t.shape[0]] or [torch.zeros(0, dtype=dtype)]
    )
    old = torch.cat(
        [
            t.detach()
            for t, c in zip(old_log_probabilities, log_probabilities, strict=True)
            if c.shape[0]
        ]
        or [torch.zeros(0, dtype=dtype)]
    )
    weights = torch.as_tensor(
        [
            a
            for episode, c in zip(advantages, log_probabilities, strict=True)
            if c.shape[0]
            for a in episode
        ],
        dtype=dtype,
    )
    steps = int(current.shape[0])
    if steps == 0:
        return torch.zeros((), dtype=dtype), 0.0
    ratio = torch.exp(current - old)
    unclipped = ratio * weights
    limited = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * weights
    clipped = int((limited < unclipped).sum())
    return (
        -torch.minimum(unclipped, limited).sum() / len(log_probabilities),
        clipped / steps,
    )


def ppo[S, A](
    environment: Environment[S, A],
    policy: TrainablePolicy,
    critic: Critic,
    rng: np.random.Generator,
    *,
    iterations: int,
    batch: int,
    max_steps: int,
    epochs: int = 4,
    clip: float = 0.2,
    lam: float = 0.95,
    learning_rate: float = 0.05,
    critic_steps: int = 50,
    entropy_coefficient: float = 0.0,
    behaviour: EpsilonGreedyPolicy | None = None,
    epsilon_schedule: Schedule | None = None,
) -> PPOTraining:
    """Collect a batch, refit the critic, then ``epochs`` clipped policy steps on that batch.

    ``entropy_coefficient`` adds ``-c H(pi(. | s_t))`` per decision to the
    loss, an exploration bonus that is zero by default so the pinned limits
    hold exactly. ``behaviour``, when given, collects the episodes instead of
    the policy being trained -- an epsilon-greedy wrapper around it, so the
    search can leave a local optimum the softmax would settle into -- and the
    ratio is then ``pi / beta`` with ``beta`` the wrapper's own distribution,
    which is what makes the update valid off-policy. ``epsilon_schedule``
    sets the wrapper's epsilon per iteration from ``opt.schedule`` (issue
    #194 measured a constant one) and must span ``iterations`` steps.
    """
    if iterations < 1 or batch < 1 or epochs < 1:
        msg = f"iterations, batch and epochs must be >= 1, got {iterations}, {batch}, {epochs}"
        raise ValueError(msg)
    if epsilon_schedule is not None and behaviour is None:
        msg = "an epsilon schedule needs a behaviour policy to set epsilon on"
        raise ValueError(msg)
    if epsilon_schedule is not None and epsilon_schedule.n_steps < iterations:
        msg = (
            f"the epsilon schedule spans {epsilon_schedule.n_steps} steps, "
            f"fewer than {iterations} iterations"
        )
        raise ValueError(msg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    mean_returns, critic_losses, clipped_fraction = [], [], []
    collector: TrainablePolicy | EpsilonGreedyPolicy = (
        policy if behaviour is None else behaviour
    )
    for iteration in range(iterations):
        if behaviour is not None and epsilon_schedule is not None:
            behaviour.epsilon = epsilon_schedule(iteration)
        episodes = [
            rollout(environment, collector, rng, max_steps) for _ in range(batch)
        ]
        features = torch.stack(
            [state_features(environment, s) for e in episodes for s in e.states[:-1]]
        )
        targets = torch.tensor(
            [g for e in episodes for g in e.returns_to_go()], dtype=torch.float64
        )
        critic_losses.append(
            fit_critic(critic, features, targets, steps=critic_steps).losses[-1]
        )
        advantages = episode_advantages(environment, episodes, critic, lam=lam)
        neighbourhoods = _neighbourhoods(environment, episodes)
        with torch.no_grad():
            old = [t.detach() for t in _log_probabilities(collector, neighbourhoods)]
        fraction = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            current = _log_probabilities(policy, neighbourhoods)
            loss, fraction = ppo_loss(current, old, advantages, clip=clip)
            if entropy_coefficient > 0.0:
                loss = loss - entropy_coefficient * _entropy(policy, neighbourhoods)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        clipped_fraction.append(fraction)
        mean_returns.append(
            float(np.mean([episode.total_reward for episode in episodes]))
        )
    return PPOTraining(
        tuple(mean_returns),
        tuple(critic_losses),
        tuple(clipped_fraction),
        iterations * batch,
    )


def _entropy(policy: TrainablePolicy, neighbourhoods: Neighbourhoods) -> torch.Tensor:
    """Mean policy entropy over the visited decisions, on the scored neighbourhoods."""
    total = torch.zeros((), dtype=policy.dtype)
    count = 0
    for steps in neighbourhoods:
        for features, _ in steps:
            log_probabilities = policy.log_probabilities(features)
            total = total - (torch.exp(log_probabilities) * log_probabilities).sum()
            count += 1
    return total / max(count, 1)


__all__ = [
    "PPOTraining",
    "episode_advantages",
    "generalized_advantages",
    "ppo",
    "ppo_loss",
]
