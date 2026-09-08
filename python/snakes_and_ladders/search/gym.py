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

It lives in ``search/`` because it may import both halves; ``learn/`` may
not import ``gymnasium`` on the core install, and ``gymnasium`` is the
``frameworks`` extra rather than a core dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from snakes_and_ladders.learn.environment import Environment

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


__all__ = ["GymnasiumEnvironment", "Observation"]
