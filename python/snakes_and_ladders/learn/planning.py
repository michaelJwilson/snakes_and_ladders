"""Planning with a learned prior and value: PUCT search and expert iteration (issue #313).

A policy answers "which move" from what it has seen; a planner answers it by
looking ahead. The search here is the single-agent form of the AlphaZero
loop: from the current state, ``n_simulations`` descents through a tree of
successors, each choosing the child that maximizes the PUCT score
(``eq:puct``) -- the mean return seen through that child plus an exploration
bonus that is the prior probability the policy gives the move, scaled by
how rarely the child has been tried -- expanding one leaf per descent and
scoring it by the critic, then backing the return up the path. The root's
visit counts are a distribution over moves that improves on the prior it
started from, and **expert iteration** trains the policy toward that
distribution (cross-entropy) and the critic toward the achieved return
(squared error), so the planner's answers become the policy's.

Every expansion pays for the successor's reward, which is an evaluation of
the objective, so the search is budget-counted in the unit the rest of the
package counts: a planner that reaches the optimum by evaluating more
candidates than greedy has not won, and the comparison states both. Two
things are pinned on an enumerable instance: with the exact optimal value as
leaf evaluator and enough simulations the root's most-visited action is an
optimal one, and the visit distribution's exact expected return is no less
than the prior's.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from snakes_and_ladders.learn.critic import Critic, fit_critic, state_features
from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.policy import TrainablePolicy

LeafValue = Callable[[object, int], float]


@dataclass
class _Node[S, A]:
    state: S
    remaining: int
    prior: np.ndarray
    actions: Sequence[A]
    visits: np.ndarray
    total_return: np.ndarray
    children: dict[int, _Node[S, A]] = field(default_factory=dict)
    rewards: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult[A]:
    """What one search from a root recorded.

    Parameters
    ----------
    actions : tuple[A, ...]
        The root's available moves, in the environment's order.
    visits : np.ndarray
        How often each was descended through; the improved policy is
        ``visits / visits.sum()``.
    values : np.ndarray
        Mean backed-up return through each, ``0`` where never visited.
    evaluations : int
        Successor rewards computed, the budget the search spent.
    """

    actions: tuple[A, ...]
    visits: np.ndarray
    values: np.ndarray
    evaluations: int

    @property
    def distribution(self) -> np.ndarray:
        """The visit counts normalized: the search-improved policy at the root."""
        total = float(self.visits.sum())
        if total == 0.0:
            return np.full(len(self.actions), 1.0 / len(self.actions))
        return self.visits / total

    def best(self) -> int:
        """Index of the most visited move, ties to the higher mean value."""
        top = np.flatnonzero(self.visits == self.visits.max())
        return int(top[np.argmax(self.values[top])])


def critic_leaf_value[S, A](
    environment: Environment[S, A], critic: Critic
) -> LeafValue:
    """A leaf evaluator reading the critic, zero at a terminal state or with no decisions left."""

    def value(state: object, remaining: int) -> float:
        typed: S = state  # type: ignore[assignment]
        if remaining <= 0 or environment.is_terminal(typed):
            return 0.0
        with torch.no_grad():
            return float(critic(state_features(environment, typed)[None, :])[0])

    return value


def puct_search[S, A](
    environment: Environment[S, A],
    policy: TrainablePolicy,
    state: S,
    *,
    horizon: int,
    n_simulations: int,
    leaf_value: LeafValue,
    c_puct: float = 1.5,
) -> SearchResult[A]:
    """``n_simulations`` PUCT descents from ``state``, ``horizon`` decisions deep.

    Each descent selects, at every expanded node, the child maximizing
    ``Q(s, a) + c_puct * P(a | s) * sqrt(N(s)) / (1 + N(s, a))``
    (``eq:puct``), expands the first unexpanded child it reaches, scores the
    new leaf by ``leaf_value(state, remaining)``, and backs the return -- the
    rewards along the path plus the leaf value -- up to the root. Priors come
    from the policy's softmax without gradient.
    """
    if n_simulations < 1 or horizon < 1:
        msg = f"n_simulations and horizon must be >= 1, got {n_simulations}, {horizon}"
        raise ValueError(msg)
    evaluations = 0

    def expand(node_state: S, remaining: int) -> _Node[S, A]:
        actions = list(environment.actions(node_state))
        with torch.no_grad():
            prior = (
                torch.exp(
                    policy.log_probabilities(environment.features(node_state, actions))
                )
                .numpy()
                .astype(float)
                if actions
                else np.zeros(0)
            )
        return _Node(
            node_state,
            remaining,
            prior,
            actions,
            np.zeros(len(actions)),
            np.zeros(len(actions)),
        )

    root = expand(state, horizon)
    if not root.actions:
        return SearchResult((), np.zeros(0), np.zeros(0), 0)
    for _ in range(n_simulations):
        node = root
        path: list[tuple[_Node[S, A], int]] = []
        leaf = 0.0
        while True:
            if (
                not node.actions
                or node.remaining == 0
                or environment.is_terminal(node.state)
            ):
                break
            total_visits = float(node.visits.sum())
            means = np.where(
                node.visits > 0, node.total_return / np.maximum(node.visits, 1), 0.0
            )
            scores = means + c_puct * node.prior * math.sqrt(total_visits + 1.0) / (
                1.0 + node.visits
            )
            index = int(np.argmax(scores))
            path.append((node, index))
            if index not in node.children:
                successor, reward = environment.step(node.state, node.actions[index])
                evaluations += 1
                node.rewards[index] = reward
                child = expand(successor, node.remaining - 1)
                node.children[index] = child
                leaf = leaf_value(successor, node.remaining - 1)
                break
            node = node.children[index]
        value = leaf
        for parent, index in reversed(path):
            value = parent.rewards[index] + value
            parent.visits[index] += 1
            parent.total_return[index] += value
    means = np.where(
        root.visits > 0, root.total_return / np.maximum(root.visits, 1), 0.0
    )
    return SearchResult(tuple(root.actions), root.visits.copy(), means, evaluations)


@dataclass(frozen=True)
class PlannedEpisode[S, A]:
    """One episode played by the planner: states, chosen actions, rewards, and the root distributions."""

    states: tuple[S, ...]
    actions: tuple[A, ...]
    rewards: tuple[float, ...]
    distributions: tuple[np.ndarray, ...]
    evaluations: int

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))


def plan_episode[S, A](
    environment: Environment[S, A],
    policy: TrainablePolicy,
    start: S,
    rng: np.random.Generator,
    *,
    horizon: int,
    n_simulations: int,
    leaf_value: LeafValue,
    c_puct: float = 1.5,
    sample: bool = True,
) -> PlannedEpisode[S, A]:
    """Play from ``start`` choosing each move by a search, sampled from the visit distribution or its argmax."""
    states, actions, rewards, distributions = [start], [], [], []
    evaluations = 0
    state = start
    for step in range(horizon):
        if environment.is_terminal(state):
            break
        result = puct_search(
            environment,
            policy,
            state,
            horizon=horizon - step,
            n_simulations=n_simulations,
            leaf_value=leaf_value,
            c_puct=c_puct,
        )
        evaluations += result.evaluations
        if not result.actions:
            break
        index = (
            int(rng.choice(len(result.actions), p=result.distribution))
            if sample
            else result.best()
        )
        successor, reward = environment.step(state, result.actions[index])
        evaluations += 1
        states.append(successor)
        actions.append(result.actions[index])
        rewards.append(reward)
        distributions.append(result.distribution)
        state = successor
    return PlannedEpisode(
        tuple(states), tuple(actions), tuple(rewards), tuple(distributions), evaluations
    )


@dataclass(frozen=True)
class ExpertIterationTraining:
    """The outcome of expert iteration: mean planned return per iteration, the losses, and the evaluations spent."""

    mean_returns: tuple[float, ...]
    policy_losses: tuple[float, ...]
    critic_losses: tuple[float, ...]
    evaluations: int


def expert_iteration[S, A](
    environment: Environment[S, A],
    policy: TrainablePolicy,
    critic: Critic,
    rng: np.random.Generator,
    *,
    iterations: int,
    batch: int,
    horizon: int,
    n_simulations: int,
    learning_rate: float = 0.05,
    critic_steps: int = 50,
    c_puct: float = 1.5,
) -> ExpertIterationTraining:
    """Alternate planned episodes with fits of the policy to the visit distributions and of the critic to the returns."""
    if iterations < 1 or batch < 1:
        msg = f"iterations and batch must be >= 1, got {iterations}, {batch}"
        raise ValueError(msg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    mean_returns, policy_losses, critic_losses = [], [], []
    evaluations = 0
    for _ in range(iterations):
        episodes = [
            plan_episode(
                environment,
                policy,
                environment.reset(rng),
                rng,
                horizon=horizon,
                n_simulations=n_simulations,
                leaf_value=critic_leaf_value(environment, critic),
                c_puct=c_puct,
            )
            for _ in range(batch)
        ]
        evaluations += sum(e.evaluations for e in episodes)
        features, targets = [], []
        for episode in episodes:
            returns = (
                np.cumsum(episode.rewards[::-1])[::-1]
                if episode.rewards
                else np.zeros(0)
            )
            for step, state in enumerate(episode.states[:-1]):
                features.append(state_features(environment, state))
                targets.append(float(returns[step]))
        if features:
            critic_losses.append(
                fit_critic(
                    critic,
                    torch.stack(features),
                    torch.tensor(targets, dtype=torch.float64),
                    steps=critic_steps,
                ).losses[-1]
            )
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=policy.dtype)
        count = 0
        for episode in episodes:
            for step, state in enumerate(episode.states[:-1]):
                available = environment.actions(state)
                log_probabilities = policy.log_probabilities(
                    environment.features(state, available)
                )
                target = torch.as_tensor(
                    episode.distributions[step], dtype=policy.dtype
                )
                loss = loss - (target * log_probabilities).sum()
                count += 1
        if count:
            (loss / count).backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            policy_losses.append(float(loss.detach()) / count)
        mean_returns.append(float(np.mean([e.total_reward for e in episodes])))
    return ExpertIterationTraining(
        tuple(mean_returns), tuple(policy_losses), tuple(critic_losses), evaluations
    )


__all__ = [
    "ExpertIterationTraining",
    "LeafValue",
    "PlannedEpisode",
    "SearchResult",
    "critic_leaf_value",
    "expert_iteration",
    "plan_episode",
    "puct_search",
]
