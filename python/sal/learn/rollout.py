"""Generating episodes: by a policy, and by the greedy baseline it must beat.

Both agents live here because the comparison between them is the point, and
it is only meaningful if they are driven through the *same* loop.
``sec:policy-gradient`` of ``docs/tex/textbook.tex`` puts it as a budget: the measure is not the objective reached but the
objective reached per evaluation, so every method has to be run at a matched
one.

**What a step costs, and why decisions are the matched unit.** A greedy
searcher scores every action in the neighbourhood and takes the best. A
policy scores every action in the neighbourhood and samples one. Both
therefore evaluate ``len(actions(state))`` candidates per decision, so a
decision -- not a reward evaluation -- is the unit at which the two are
comparable, and ``max_steps`` counts decisions for both. This mirrors
``sal.search.infer``, which counts a budget in candidate fits for the same
reason: a wall-clock budget would make a result depend on the machine that
produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sal.learn.environment import Environment, Episode
from sal.learn.policy import EpsilonGreedyPolicy, Policy, TrainablePolicy


def rollout[S, A](
    environment: Environment[S, A],
    policy: Policy,
    rng: np.random.Generator,
    *,
    max_steps: int,
    start: S | None = None,
    stop_at_local_optimum: bool = True,
) -> Episode[S, A]:
    """Run one episode under ``policy``.

    Parameters
    ----------
    environment : Environment[S, A]
        The problem to search.
    policy : Policy
        Chooses among the available actions. :class:`LinearPolicy` samples
        from its softmax; :class:`EpsilonGreedyPolicy` takes the greedy action
        except a declared fraction of the time.
    rng : np.random.Generator
        The only source of randomness, covering both the starting state and
        every action drawn.
    max_steps : int
        Decision budget. Reaching it truncates the episode, which is
        recorded: ``sec:policy-gradient`` notes that a truncation landing between a
        sacrifice and its payoff teaches the opposite of the truth, so a
        consumer needs to know it happened.
    start : S | None
        Starting state; ``None`` draws one from ``environment.reset``.
    stop_at_local_optimum : bool
        Whether reaching a state ``environment.is_terminal`` accepts ends the
        episode. ``True`` is the historical behaviour and stays the default,
        because it is what makes an episode's end mean "nothing here improves".

        ``False`` lets the episode run to ``max_steps`` regardless, which is
        the only way an agent can leave a local optimum at all (issue #194).
        Two things follow and are the caller's to handle. The episode's
        outcome is then the best state it *visited*, not the state it ended
        in, since a wandering searcher keeps its best. And the budget is no
        longer bounded by reaching a local optimum, so a comparison against
        hill climbing has to match budgets explicitly --- restarting greedy
        until it has spent the same number of decisions is the honest
        baseline, not a single greedy run.

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
    while (not terminated or not stop_at_local_optimum) and len(actions) < max_steps:
        available = environment.actions(state)
        if not available:
            # An absorbing state with no actions ends the episode whatever
            # the stopping rule says: there is nothing to sample from, and
            # asking a policy to choose among no actions is not a decision.
            break
        index = policy.sample(environment.features(state, available), rng)
        action = available[index]
        state, reward = environment.step(state, action)
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        terminated = environment.is_terminal(state)

    return Episode.from_rollout(states, actions, rewards, environment)


def greedy_rollout[S, A](
    environment: Environment[S, A],
    *,
    max_steps: int,
    start: S,
) -> Episode[S, A]:
    """Run one episode taking the best-rewarded action at every step.

    This is hill climbing, and it is the baseline ``ROADMAP.md``'s
    Milestone 8 requires a learned policy to beat --- the same rule
    ``sal.search.infer`` applies to topologies, on a environment small enough
    that the answer can be checked against exhaustive enumeration.

    It takes no ``rng``: given a start it is deterministic, with ties broken
    towards the first action the environment lists. A seeded tie-break would
    make the baseline depend on a second seed nobody declared.

    Parameters
    ----------
    environment : Environment[S, A]
        The problem to search.
    max_steps : int
        Decision budget, counted as for :func:`rollout`.
    start : S
        Starting state. Required rather than drawn, because a comparison
        against a policy is only fair from the same start.

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

    return Episode.from_rollout(states, actions, rewards, environment)


def greedy_restarts[S, A](
    environment: Environment[S, A],
    rng: np.random.Generator,
    *,
    max_steps: int,
    start: S,
) -> tuple[Episode[S, A], ...]:
    """Hill climbing restarted until ``max_steps`` decisions are spent.

    The baseline a *wandering* searcher is read against (`learn/CLAUDE.md`,
    issue #194): once an episode is no longer bounded by a local optimum, one
    greedy run stops after a few decisions and leaves the budget unspent, so
    the comparison at equal budget is greedy restarted from a fresh state
    until the budget is gone. The first run starts from ``start``, so the two
    sides of a comparison share their start; every later run starts from
    ``environment.reset(rng)``. A run that spends nothing --- a start that is
    already a local optimum --- still costs one decision, or a budget could
    never be spent.

    Parameters
    ----------
    environment : Environment[S, A]
        The problem to search.
    rng : np.random.Generator
        Draws every restart's state, and nothing else.
    max_steps : int
        Decision budget over every run together, counted as for
        :func:`rollout`.
    start : S
        Starting state of the first run.

    Returns
    -------
    tuple[Episode[S, A], ...]
        One :class:`~sal.learn.environment.Episode` per run, in
        order, so a reader scores the best state over all of them and sums
        what they cost; a single record would have to invent an action for a
        restart.

    Raises
    ------
    ValueError
        If ``max_steps`` is negative.
    """
    if max_steps < 0:
        msg = f"max_steps must be >= 0, got {max_steps}"
        raise ValueError(msg)
    runs: list[Episode[S, A]] = []
    state, spent = start, 0
    while spent < max_steps:
        episode = greedy_rollout(environment, start=state, max_steps=max_steps - spent)
        runs.append(episode)
        spent += max(len(episode.actions), 1)
        state = environment.reset(rng)
    return tuple(runs)


@dataclass(frozen=True)
class Decision:
    """One recorded decision, scored again under a policy.

    Parameters
    ----------
    log_probabilities : torch.Tensor
        ``log pi(. | s_t)`` over the neighbourhood of the state the decision
        was taken in, shape ``(len(actions(s_t)),)``, on the current graph.
    chosen : int
        Index of the action the episode took, into ``log_probabilities``.
    """

    log_probabilities: torch.Tensor
    chosen: int

    @property
    def taken(self) -> torch.Tensor:
        """``log pi(a_t | s_t)``: the entry the episode's action sits at."""
        return self.log_probabilities[self.chosen]


def log_probabilities_of[S, A](
    policy: TrainablePolicy | EpsilonGreedyPolicy,
    environment: Environment[S, A],
    episodes: Sequence[Episode[S, A]],
) -> list[list[Decision]]:
    """Score every recorded decision under ``policy``, one list per episode.

    The five estimators walked this loop apiece --- episode, step, the
    neighbourhood the state offers, the policy's distribution over it, the
    index the taken action sits at --- and three of them wanted the whole
    distribution rather than the entry: an entropy bonus and a
    cross-entropy against a planner's visit counts read all of it. So a
    :class:`Decision` carries the vector and the index, and
    :attr:`Decision.taken` is the score-function term.

    **Recomputed, not cached from the rollout.** The rollout samples under
    ``no_grad``, and a graph cached there would tie an estimator to the
    policy that *collected* the data rather than the one being updated ---
    which is the difference between REINFORCE and the importance-weighted
    objective :mod:`sal.learn.ppo` forms.

    Parameters
    ----------
    policy : TrainablePolicy | EpsilonGreedyPolicy
        The policy to score under; the behaviour policy where the ratio's
        denominator is wanted.
    environment : Environment[S, A]
        The environment the episodes were collected in.
    episodes : Sequence[Episode[S, A]]
        The trajectories to replay.

    Returns
    -------
    list[list[Decision]]
        One list per episode, one :class:`Decision` per action it took, in
        order.
    """
    out = []
    scored = _Scored(policy, environment)
    for episode in episodes:
        out.append(
            [
                Decision(*scored(episode.states[step], action))
                for step, action in enumerate(episode.actions)
            ]
        )
    return out


def taken_log_probabilities[S, A](
    policy: TrainablePolicy | EpsilonGreedyPolicy,
    environment: Environment[S, A],
    episodes: Sequence[Episode[S, A]],
) -> torch.Tensor:
    """``log pi(a_t | s_t)`` of every recorded decision, episodes end to end, as one tensor.

    :func:`log_probabilities_of` reduced to the taken entries, gathered in
    one indexing operation rather than one per decision: the autograd graph
    then has one node for every distinct state and one for the gather,
    where a sum over :attr:`Decision.taken` had one per decision (issue
    #986: 10^5 decisions of the Potts chain, 81 states among them).
    """
    scored = _Scored(policy, environment)
    # `(state, action)` as `zip` builds it is the key: one dictionary probe
    # per decision, and a miss only on the first visit of a pair.
    seen: dict[Any, int] = {}
    flat: list[int] = []
    for episode in episodes:
        for pair in zip(episode.states, episode.actions, strict=False):
            try:
                at = seen.get(pair)
            except TypeError:
                at = scored.position(*pair)
            else:
                if at is None:
                    at = seen[pair] = scored.position(*pair)
            flat.append(at)
    return scored.table()[torch.as_tensor(flat, dtype=torch.long)]


class _Scored[S, A]:
    """Each distinct state's distribution under ``policy``, formed once and kept.

    An :class:`~sal.learn.environment.Environment` is
    stateless in the episode, so a state's neighbourhood and features are
    its own whatever episode reaches it, and a distribution scored once is
    the one every later visit would score (issue #986). A state that is not
    hashable is scored at every visit, as before.
    """

    def __init__(
        self,
        policy: TrainablePolicy | EpsilonGreedyPolicy,
        environment: Environment[S, A],
    ) -> None:
        self._policy = policy
        self._environment = environment
        self._states: dict[Any, tuple[torch.Tensor, Sequence[A], int]] = {}
        self._rows: list[torch.Tensor] = []
        self._width = 0

    def _score(self, state: S) -> tuple[torch.Tensor, Sequence[A], int]:
        available = self._environment.actions(state)
        log_probabilities = self._policy.log_probabilities(
            self._environment.features(state, available)
        )
        offset = self._width
        self._rows.append(log_probabilities)
        self._width += len(available)
        return log_probabilities, available, offset

    def _entry(self, state: S) -> tuple[torch.Tensor, Sequence[A], int]:
        try:
            entry = self._states.get(state)
        except TypeError:
            return self._score(state)
        if entry is None:
            entry = self._states[state] = self._score(state)
        return entry

    def __call__(self, state: S, action: A) -> tuple[torch.Tensor, int]:
        log_probabilities, available, _ = self._entry(state)
        return log_probabilities, available.index(action)

    def position(self, state: S, action: A) -> int:
        """Where ``(state, action)``'s entry sits in :meth:`table`."""
        _, available, offset = self._entry(state)
        return offset + available.index(action)

    def table(self) -> torch.Tensor:
        """Every scored distribution, end to end."""
        return torch.cat(self._rows) if self._rows else torch.zeros(0)
