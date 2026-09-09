"""``gymnasium.vector`` fronting the sequential rollout, measured and declined (issues #322, #392).

The batched rollout that lost. :func:`vector_environment` puts ``n`` copies
of :class:`~snakes_and_ladders.search.gym.GymnasiumEnvironment` behind one
``gymnasium.vector.SyncVectorEnv``, and :func:`rollout_batch` steps them
together to collect the batch
:func:`~snakes_and_ladders.learn.rollout.rollout` collects one episode at a
time. The budget comes from ``gymnasium.wrappers.TimeLimit`` and the episode
bookkeeping from ``gymnasium.wrappers.vector.RecordEpisodeStatistics``,
rather than from counters of our own. One generator per copy is what makes
this a change in throughput and in nothing else: copy ``i`` reads ``rngs[i]``
for every reset and every action, so its episodes are the ones
:func:`~snakes_and_ladders.learn.rollout.rollout` produces from that generator,
and at one copy the two are equal draw for draw.

**Declined on its numbers**, recorded in
`docs/experiments/007-batched-rollout-and-torchrl-ppo.md`: 1.18x to 1.75x
slower per episode than the sequential rollout, at batch sizes 1, 4 and 16,
on the Potts chain and on the 5- and 7-taxon tree fixtures, and the penalty
does not fall as the batch grows. ``SyncVectorEnv`` is a serial loop in one
process, so there is no parallelism to amortize the vector API's observation
stacking, autoreset bookkeeping and pad-to-``n_max`` round trip through NumPy
against. All on one thread of a shared four-core Linux x86-64 host under the
exclusive lock. `tests/benchmarks/test_search_gym_bench.py` re-measures both
sides and `tests/regression/search/test_search_gym_vector.py` pins the
batched draws against the sequential ones, so a later Gymnasium that moves
the ratio is visible rather than assumed.

The adapter it is built from was *not* declined and stays in
:mod:`snakes_and_ladders.search.gym`, on the live path: a driver that lost is
the second implementation, and the interface it drives is not.

What is not claimed here is anything about a collector with real
parallelism. ``AsyncVectorEnv`` and a process pool were not run, and only one
of those could deliver a budget at the sizes ``ROADMAP.md``'s Milestone 2.2
asks for; each is its own measurement.

Imports ``gymnasium`` at module scope: it is the ``frameworks`` extra, and a
caller without it gets an ``ImportError`` here rather than a silent fallback
to the rollout this exists to referee.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import gymnasium
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from snakes_and_ladders.learn.environment import Environment, Episode
from snakes_and_ladders.learn.policy import Policy
from snakes_and_ladders.search.gym import GymnasiumEnvironment, Observation


def vector_environment[S, A](
    environment: Environment[S, A],
    *,
    n_max: int,
    max_steps: int,
    n: int,
) -> RecordEpisodeStatistics:
    """``n`` copies of the adapter as one ``gymnasium.vector.SyncVectorEnv`` (issue #392).

    The copies share the ``environment`` object, as
    :func:`~snakes_and_ladders.learn.rollout.rollout` shares it across successive
    episodes: an ``Environment`` is stateless in the episode, so the only
    per-copy state is the adapter's, and a memoizing environment keeps one
    cache rather than ``n``.

    Each copy is wrapped in ``gymnasium.wrappers.TimeLimit`` at ``max_steps``
    and the vector in ``gymnasium.wrappers.vector.RecordEpisodeStatistics``,
    which is where the batched rollout reads its budget and its per-episode
    return and length rather than counting them itself. ``TimeLimit``
    truncates on the same decision the adapter's own counter does --- both at
    ``max_steps`` --- and a test asserts the two flags agree, so the wrapper
    is the batched path's authority without being a second rule.

    ``SyncVectorEnv`` rather than ``AsyncVectorEnv``: one process, because
    the batched path is measured against the sequential one on the same core
    and a subprocess per copy would measure the host instead.

    Parameters
    ----------
    environment : Environment[S, A]
        The search problem, shared by every copy.
    n_max : int
        The widest neighbourhood, as :class:`GymnasiumEnvironment` takes it.
    max_steps : int
        The decision budget per episode.
    n : int
        Number of copies.

    Returns
    -------
    RecordEpisodeStatistics
        The vector environment, autoresetting on the step after an episode
        ends (Gymnasium's ``NEXT_STEP`` default).

    Raises
    ------
    ValueError
        If ``n`` is below 1.
    """
    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)

    def make() -> gymnasium.Env[Observation, int]:
        return TimeLimit(
            GymnasiumEnvironment(environment, n_max=n_max, max_steps=max_steps),
            max_episode_steps=max_steps,
        )

    return RecordEpisodeStatistics(SyncVectorEnv([make] * n))


def _adapters[S, A](
    vector: RecordEpisodeStatistics,
) -> list[GymnasiumEnvironment[S, A]]:
    """The unwrapped adapters, in env order."""
    inner = vector.env
    assert isinstance(inner, SyncVectorEnv)
    return [cast("GymnasiumEnvironment[S, A]", env.unwrapped) for env in inner.envs]


def rollout_batch[S, A](
    environment: Environment[S, A],
    policy: Policy,
    rngs: Sequence[np.random.Generator],
    *,
    n_max: int,
    max_steps: int,
    episodes: int,
) -> list[Episode[S, A]]:
    """Roll ``episodes`` episodes through ``len(rngs)`` copies stepped together (issue #392).

    Copy ``i`` draws every one of its resets and every one of its actions from
    ``rngs[i]`` and from nothing else, so the episodes it produces are exactly
    the sequence :func:`~snakes_and_ladders.learn.rollout.rollout` produces from
    that generator, called until the batch is full. That is what makes the
    batching a change in throughput and in nothing else, and at ``len(rngs)
    == 1`` it is the pin: ``rollout_batch`` with one generator equals repeated
    :func:`~snakes_and_ladders.learn.rollout.rollout` under the same generator,
    draw for draw.

    Episodes are returned in the order they complete. At one copy that is the
    order they were rolled in; at ``n`` copies it is a reordering of ``n``
    independent sequences, which is why the pin is stated at one.

    The stopping rule is the adapter's, which is
    :func:`~snakes_and_ladders.learn.rollout.rollout` at
    ``stop_at_local_optimum=True``. The wandering variant has no Gymnasium
    expression --- a ``gymnasium.Env`` that reports ``terminated`` and keeps
    stepping is not one --- so a caller that wants it rolls sequentially.

    A start state that is already terminal ends its episode before any
    decision, exactly as :func:`~snakes_and_ladders.learn.rollout.rollout` records
    it. Gymnasium's autoreset fires on a ``done`` returned by ``step`` and
    there is none here, so the copy is reset again directly; its recorded
    return and length are both zero at that point, so the statistics wrapper
    stays in step.

    Parameters
    ----------
    environment : Environment[S, A]
        The search problem, shared by every copy.
    policy : Policy
        Chooses among the scored actions, read the same way for every copy.
    rngs : Sequence[np.random.Generator]
        One generator per copy, and the only source of randomness. Distinct
        generators, or the copies are correlated; ``rng.spawn(n)`` supplies
        them.
    n_max : int
        The widest neighbourhood, as :class:`GymnasiumEnvironment` takes it.
    max_steps : int
        Decision budget per episode, counted in decisions.
    episodes : int
        How many episodes to collect.

    Returns
    -------
    list[Episode[S, A]]
        ``episodes`` trajectories, in completion order.

    Raises
    ------
    ValueError
        If ``rngs`` is empty or ``episodes`` is below 1.
    """
    if not rngs:
        msg = "rollout_batch needs at least one generator"
        raise ValueError(msg)
    if episodes < 1:
        msg = f"episodes must be >= 1, got {episodes}"
        raise ValueError(msg)

    n = len(rngs)
    vector = vector_environment(environment, n_max=n_max, max_steps=max_steps, n=n)
    inner = vector.env
    assert isinstance(inner, SyncVectorEnv)
    adapters: list[GymnasiumEnvironment[S, A]] = _adapters(vector)
    for adapter, rng in zip(adapters, rngs, strict=True):
        adapter.np_random = rng

    collected: list[Episode[S, A]] = []
    states: list[list[S]] = [[] for _ in range(n)]
    actions: list[list[A]] = [[] for _ in range(n)]
    rewards: list[list[float]] = [[] for _ in range(n)]
    observations, _ = vector.reset()
    current = [observations[i] for i in range(n)]
    restarting = np.zeros(n, dtype=np.bool_)

    def begin(index: int) -> None:
        """Open an episode on copy ``index``, closing empty ones at a terminal start."""
        while len(collected) < episodes:
            adapter = adapters[index]
            states[index], actions[index], rewards[index] = [adapter.state], [], []
            if not adapter.environment.is_terminal(adapter.state):
                return
            collected.append(
                Episode(
                    states=(adapter.state,), actions=(), rewards=(), terminated=True
                )
            )
            current[index], _ = inner.envs[index].reset()

    for index in range(n):
        begin(index)
        if len(collected) >= episodes:
            return collected[:episodes]

    chosen = np.zeros(n, dtype=np.int64)
    taken: list[A | None] = [None] * n
    while len(collected) < episodes:
        for index in range(n):
            if restarting[index]:
                continue
            adapter = adapters[index]
            rows = torch.from_numpy(current[index][: len(adapter.available)])
            chosen[index] = policy.sample(rows, rngs[index])
            taken[index] = adapter.available[int(chosen[index])]
        observations, step_rewards, terminations, _, infos = vector.step(chosen)
        # `RecordEpisodeStatistics` marks the copies whose episode ended on
        # this step, whether at a local optimum or on `TimeLimit`'s budget;
        # it is read rather than recomputed from the two flags.
        finished = infos.get("_episode", np.zeros(n, dtype=np.bool_))
        for index in range(n):
            current[index] = observations[index]
            if restarting[index]:
                # The copy was autoreset by this step; its action was ignored.
                restarting[index] = False
                begin(index)
                continue
            action = taken[index]
            assert action is not None
            actions[index].append(action)
            rewards[index].append(float(step_rewards[index]))
            states[index].append(adapters[index].state)
            if finished[index]:
                collected.append(
                    Episode(
                        states=tuple(states[index]),
                        actions=tuple(actions[index]),
                        rewards=tuple(rewards[index]),
                        terminated=bool(terminations[index]),
                    )
                )
                restarting[index] = True
            if len(collected) >= episodes:
                break
    return collected[:episodes]


__all__ = [
    "rollout_batch",
    "vector_environment",
]
