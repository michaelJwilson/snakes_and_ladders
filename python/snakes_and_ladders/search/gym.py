"""Gymnasium compliance as an adapter over ``learn.Environment``, not a rewrite of it (issue #322).

:class:`GymnasiumEnvironment` wraps any
:class:`~snakes_and_ladders.learn.environment.Environment` as a
``gymnasium.Env``, so a Farama-API agent, collector or checker can drive the
same search the in-house :func:`~snakes_and_ladders.learn.rollout.rollout` drives.
The protocol underneath does not change, and the reasons are what its
oracles rest on: an ``Environment`` is stateless in the episode, which is
what lets :mod:`snakes_and_ladders.learn.exact` enumerate it, and it scores the
whole neighbourhood at once, which is what a policy over a state-dependent
action set consumes. Gymnasium's stateful ``step`` expresses neither, so the
statefulness lives here, in the adapter, and nowhere below it.

**What the two vocabularies map to.** The observation is the neighbourhood's
feature matrix, padded to ``(n_max, n_features)``; the action is an index
into that neighbourhood, from ``Discrete(n_max)``, with ``info["action_mask"]``
marking the rows that exist (the MaskablePPO convention). ``terminated`` is
the environment's own local optimum, ``truncated`` the decision budget,
counted in decisions as :func:`~snakes_and_ladders.learn.rollout.rollout` counts
it (``learn/CLAUDE.md``: a budget is counted in decisions, never in seconds).
``info["evaluations"]`` carries the count that budget accounting rests on and
Gymnasium has no slot for: the candidates scored so far, one neighbourhood
per observed state, which is what both a policy and the greedy searcher pay
per decision.

**A masked action is refused, not repaired.** An index at or past the
neighbourhood's end leaves the state where it is, earns a reward of zero and
sets ``info["action_valid"]`` false; it still spends a decision, so an agent
that ignores the mask reaches the budget rather than looping. Repairing it --
folding it onto a valid row -- would be a silent behaviour change root
``CLAUDE.md`` forbids, and raising would leave ``gymnasium``'s own checker,
which samples the action space unmasked, unable to step at all.

**The batched rollout (issue #392).** :func:`vector_environment` puts ``n``
copies of the adapter behind one ``gymnasium.vector.SyncVectorEnv``, and
:func:`rollout_batch` steps them together to collect a batch of episodes.
The budget comes from ``gymnasium.wrappers.TimeLimit`` and the episode
bookkeeping from ``gymnasium.wrappers.vector.RecordEpisodeStatistics``,
rather than from counters of our own. One generator per copy is what makes
this a change in throughput and in nothing else: copy ``i`` reads
``rngs[i]`` for every reset and every action, so its episodes are the ones
:func:`~snakes_and_ladders.learn.rollout.rollout` produces from that generator,
and at one copy the two are equal draw for draw.

It lives in ``search/`` because it may import both halves; ``learn/`` may
not import ``gymnasium`` on the core install, and ``gymnasium`` is the
``frameworks`` extra rather than a core dependency. That is why the batched
rollout is here and not beside
:func:`~snakes_and_ladders.learn.rollout.rollout`, which issue #392 proposed:
``learn.rollout`` may not reach a Gymnasium wrapper, and neither may
``learn.reinforce`` or ``learn.ppo``, so wiring the batch into a training
loop needs a seam those modules do not yet have.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from snakes_and_ladders.learn.environment import Environment, Episode
from snakes_and_ladders.learn.policy import Policy

Observation = np.ndarray[Any, np.dtype[np.float64]]


class GymnasiumEnvironment[S, A](gymnasium.Env[Observation, int]):
    """A ``learn.Environment`` as a ``gymnasium.Env``.

    Parameters
    ----------
    environment : Environment[S, A]
        The search problem. It is not copied; a memoizing environment such as
        :class:`~snakes_and_ladders.search.rl.TopologyEnvironment` keeps its cache
        across episodes, as it does under :func:`~snakes_and_ladders.learn.rollout.rollout`.
    n_max : int
        The widest neighbourhood any state may have. The observation is
        padded to it and the action space indexes it; a state whose
        neighbourhood exceeds it is refused when observed, since a silently
        dropped move would be a different search.
    max_steps : int
        The decision budget. Reaching it without terminating truncates the
        episode.

    Raises
    ------
    ValueError
        If ``n_max`` or ``max_steps`` is below 1.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self, environment: Environment[S, A], n_max: int, max_steps: int
    ) -> None:
        if n_max < 1:
            msg = f"n_max must be >= 1, got {n_max}"
            raise ValueError(msg)
        if max_steps < 1:
            msg = f"max_steps must be >= 1, got {max_steps}"
            raise ValueError(msg)
        self._environment = environment
        self._n_max = n_max
        self._max_steps = max_steps
        self._state: S | None = None
        self._available: Sequence[A] = ()
        self._steps = 0
        self._evaluations = 0
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(n_max, environment.n_features()),
            dtype=np.float64,
        )
        self.action_space = spaces.Discrete(n_max)

    @property
    def state(self) -> S:
        """The current state, in the wrapped environment's own type.

        Raises
        ------
        RuntimeError
            Before the first :meth:`reset`.
        """
        if self._state is None:
            msg = "no episode: call reset() first"
            raise RuntimeError(msg)
        return self._state

    @property
    def environment(self) -> Environment[S, A]:
        """The wrapped environment."""
        return self._environment

    @property
    def available(self) -> Sequence[A]:
        """The current neighbourhood, in the wrapped environment's own action type.

        The observation is an index into it, and a caller assembling an
        :class:`~snakes_and_ladders.learn.environment.Episode` needs the action
        itself. It is the neighbourhood the last :meth:`reset` or :meth:`step`
        scored, so reading it costs nothing; recomputing it would score the
        neighbourhood a second time and double what a decision costs.
        """
        return self._available

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Start an episode from ``environment.reset`` drawn under ``seed``.

        ``self.np_random`` is the generator handed to the environment, so a
        caller sampling actions from it reproduces
        :func:`~snakes_and_ladders.learn.rollout.rollout` under
        ``np.random.default_rng(seed)`` exactly, which a test pins.
        ``options={"start": state}`` starts from ``state`` instead of drawing
        one, as :func:`~snakes_and_ladders.learn.rollout.rollout`'s ``start`` does;
        any other key is refused.

        Raises
        ------
        ValueError
            If ``options`` carries a key other than ``"start"``.
        """
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"start"}:
            msg = f"reset options are {{'start'}}, got {sorted(options)}"
            raise ValueError(msg)
        self._state = (
            options["start"]
            if "start" in options
            else self._environment.reset(self.np_random)
        )
        self._steps = 0
        self._evaluations = 0
        return self._observe(), self._info(action_valid=True)

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Take the ``action``-th move of the current neighbourhood."""
        index = int(action)
        valid = 0 <= index < len(self._available)
        if valid:
            self._state, reward = self._environment.step(
                self.state, self._available[index]
            )
        else:
            reward = 0.0
        self._steps += 1
        observation = self._observe()
        terminated = self._environment.is_terminal(self.state)
        truncated = not terminated and self._steps >= self._max_steps
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            self._info(action_valid=valid),
        )

    def _observe(self) -> Observation:
        """Score the current neighbourhood and pad its features to ``n_max`` rows."""
        self._available = self._environment.actions(self.state)
        if len(self._available) > self._n_max:
            msg = (
                f"a neighbourhood of {len(self._available)} exceeds n_max = "
                f"{self._n_max}; the action space cannot index it"
            )
            raise ValueError(msg)
        self._evaluations += len(self._available)
        features = self._environment.features(self.state, self._available)
        observation = np.zeros(
            (self._n_max, self._environment.n_features()), dtype=np.float64
        )
        observation[: len(self._available)] = features.detach().numpy()
        return observation

    def _info(self, *, action_valid: bool) -> dict[str, Any]:
        mask = np.zeros(self._n_max, dtype=np.bool_)
        mask[: len(self._available)] = True
        return {
            "action_mask": mask,
            "evaluations": self._evaluations,
            "action_valid": action_valid,
            "terminal": self._environment.is_terminal(self.state),
        }


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
    return [env.unwrapped for env in inner.envs]  # type: ignore[misc]


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
                Episode(states=(adapter.state,), actions=(), rewards=(), terminated=True)
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
    "GymnasiumEnvironment",
    "Observation",
    "rollout_batch",
    "vector_environment",
]
