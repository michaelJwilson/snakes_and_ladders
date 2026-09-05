"""Generating episodes: by a policy, and by the greedy baseline it must beat.

Both agents live here because the comparison between them is the point, and
it is only meaningful if they are driven through the *same* loop. ``docs/tex``
puts it as a budget: the measure is not the objective reached but the
objective reached per evaluation, so every method has to be run at a matched
one.

**What a step costs, and why decisions are the matched unit.** A greedy
searcher scores every action in the neighbourhood and takes the best. A
policy scores every action in the neighbourhood and samples one. Both
therefore evaluate ``len(actions(state))`` candidates per decision, so a
decision -- not a reward evaluation -- is the unit at which the two are
comparable, and ``max_steps`` counts decisions for both. This mirrors
``snakes_and_ladders.search.infer``, which counts a budget in candidate fits for the same
reason: a wall-clock budget would make a result depend on the machine that
produced it.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.learn.environment import Environment, Episode
from snakes_and_ladders.learn.policy import LinearPolicy


def rollout[S, A](
    environment: Environment[S, A],
    policy: LinearPolicy,
    rng: np.random.Generator,
    max_steps: int,
    start: S | None = None,
) -> Episode[S, A]:
    """Run one episode under ``policy``.

    Parameters
    ----------
    environment : Environment[S, A]
        The problem to search.
    policy : LinearPolicy
        Scores the available actions; sampled from, not maximized over.
    rng : np.random.Generator
        The only source of randomness, covering both the starting state and
        every action drawn.
    max_steps : int
        Decision budget. Reaching it truncates the episode, which is
        recorded: ``docs/tex`` notes that a truncation landing between a
        sacrifice and its payoff teaches the opposite of the truth, so a
        consumer needs to know it happened.
    start : S | None
        Starting state; ``None`` draws one from ``environment.reset``.

    Returns
    -------
    Episode[S, A]
        The trajectory, including its rewards and whether it terminated.

    Raises
    ------
    ValueError
        If ``max_steps`` is negative.
    """
    if max_steps < 0:
        msg = f"max_steps must be >= 0, got {max_steps}"
        raise ValueError(msg)

    state = environment.reset(rng) if start is None else start
    states: list[S] = [state]
    actions: list[A] = []
    rewards: list[float] = []
    terminated = environment.is_terminal(state)
    while not terminated and len(actions) < max_steps:
        available = environment.actions(state)
        index = policy.sample(environment.features(state, available), rng)
        action = available[index]
        state, reward = environment.step(state, action)
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        terminated = environment.is_terminal(state)

    return Episode(
        states=tuple(states),
        actions=tuple(actions),
        rewards=tuple(rewards),
        terminated=terminated,
    )


def greedy_rollout[S, A](
    environment: Environment[S, A],
    start: S,
    max_steps: int,
) -> Episode[S, A]:
    """Run one episode taking the best-rewarded action at every step.

    This is hill climbing, and it is the baseline ``ROADMAP.md``'s
    Milestone 8 requires a learned policy to beat --- the same rule
    ``snakes_and_ladders.search.infer`` applies to topologies, on a landscape small enough
    that the answer can be checked against exhaustive enumeration.

    It takes no ``rng``: given a start it is deterministic, with ties broken
    towards the first action the environment lists. A seeded tie-break would
    make the baseline depend on a second seed nobody declared.

    Parameters
    ----------
    environment : Environment[S, A]
        The problem to search.
    start : S
        Starting state. Required rather than drawn, because a comparison
        against a policy is only fair from the same start.
    max_steps : int
        Decision budget, counted as for :func:`rollout`.

    Returns
    -------
    Episode[S, A]
        The trajectory. ``terminated`` is ``True`` when it stopped at a local
        maximum rather than on the budget.

    Raises
    ------
    ValueError
        If ``max_steps`` is negative.
    """
    if max_steps < 0:
        msg = f"max_steps must be >= 0, got {max_steps}"
        raise ValueError(msg)

    state = start
    states: list[S] = [state]
    actions: list[A] = []
    rewards: list[float] = []
    terminated = environment.is_terminal(state)
    while not terminated and len(actions) < max_steps:
        available = environment.actions(state)
        scored = [environment.step(state, action) for action in available]
        index = max(range(len(scored)), key=lambda i: scored[i][1])
        state, reward = scored[index]
        actions.append(available[index])
        rewards.append(reward)
        states.append(state)
        terminated = environment.is_terminal(state)

    return Episode(
        states=tuple(states),
        actions=tuple(actions),
        rewards=tuple(rewards),
        terminated=terminated,
    )
