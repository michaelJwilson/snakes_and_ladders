"""Gymnasium's checker and ``reset``/``step`` over a ``learn.Environment`` the adapter named (issues #322, #977).

``mode`` selects one run over the environment
:func:`snakes_and_ladders.validation.gymnasium.build` makes from the spec:

- ``"check"``: ``gymnasium.utils.env_checker.check_env`` on the adapter;
  outputs ``passed``.
- ``"episodes"``: one episode per entry of ``seeds``, actions drawn by a
  ``LinearPolicy`` of ``weights`` from the adapter's own generator, from
  ``start`` where it is given; outputs each episode's encoded ``states`` and
  ``rewards`` (split by ``state_offset`` and ``reward_offset``),
  ``evaluations``, ``terminated``, ``truncated``, and the ``observation`` and
  ``mask`` its reset returned.
- ``"throughput"``: ``n_steps`` decisions under the policy, resetting at each
  episode's end; the measured seconds are the loop alone.

The adapter, formerly ``snakes_and_ladders.learn.gym.GymnasiumEnvironment``,
is defined inside :func:`adapter` so this module imports without Gymnasium.
The protocol underneath does not change: an ``Environment`` is stateless in
the episode, which is what lets :mod:`snakes_and_ladders.learn.exact`
enumerate it, and it scores the whole neighbourhood at once. Gymnasium's
stateful ``step`` expresses neither, so the statefulness lives in the
adapter and nowhere below it.

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
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.validation.gymnasium import Spec, build, encode
from snakes_and_ladders.validation.protocol import dump, received, timed

Observation = np.ndarray[Any, np.dtype[np.float64]]


def adapter(gymnasium: Any) -> Any:
    """The ``gymnasium.Env`` over a ``learn.Environment``, built on the imported framework."""
    spaces = gymnasium.spaces

    class GymnasiumEnvironment[S, A](gymnasium.Env):
        """A ``learn.Environment`` as a ``gymnasium.Env``.

        ``n_max`` is the widest neighbourhood any state may have: the
        observation is padded to it and the action space indexes it, and a
        wider neighbourhood is refused when observed. ``max_steps`` is the
        decision budget; reaching it without terminating truncates.
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

    return GymnasiumEnvironment


def _episode(
    env: Any, policy: LinearPolicy, seed: int, start: Sequence[int] | None
) -> tuple[list[str], list[float], int, bool, bool, np.ndarray, np.ndarray]:
    options = None if start is None else {"start": tuple(start)}
    observation, info = env.reset(seed=seed, options=options)
    first, mask = observation, info["action_mask"]
    states, rewards = [encode(env.state)], []
    terminated, truncated = bool(info["terminal"]), False
    while not terminated and not truncated:
        features = torch.from_numpy(observation[info["action_mask"]])
        index = policy.sample(features, env.np_random)
        observation, reward, terminated, truncated, info = env.step(index)
        if not info["action_valid"]:
            message = "the policy chose a masked action"
            raise SystemExit(message)
        states.append(encode(env.state))
        rewards.append(reward)
    return (
        states,
        rewards,
        int(info["evaluations"]),
        bool(terminated),
        bool(truncated),
        first,
        mask,
    )


def main() -> None:
    """Run the requested mode and write its answer back."""
    import gymnasium  # the framework, imported only in this interpreter
    from gymnasium.utils.env_checker import check_env

    inputs, returned = received()
    mode = str(inputs["mode"])
    environment, n_max = build(Spec.of(inputs))
    wrap = adapter(gymnasium)
    max_steps = int(inputs["max_steps"])
    outputs: dict[str, np.ndarray] = {}
    seconds = 0.0
    if mode == "check":
        with warnings.catch_warnings():
            # The checker warns that the environment has no `spec` and so no
            # registered render modes; neither is a claim this adapter makes.
            warnings.simplefilter("ignore", UserWarning)
            check_env(wrap(environment, n_max=n_max, max_steps=max_steps))
        outputs["passed"] = np.asarray(True)
    elif mode in {"episodes", "throughput"}:
        policy = LinearPolicy(inputs["weights"].size)
        policy.set_weights(torch.as_tensor(inputs["weights"], dtype=torch.float64))
        width = n_max + int(inputs.get("extra_width", 0))
        env = wrap(environment, n_max=width, max_steps=max_steps)
        if mode == "episodes":
            start = inputs["start"].tolist() or None
            runs = [_episode(env, policy, int(s), start) for s in inputs["seeds"]]
            outputs["states"] = np.asarray([s for r in runs for s in r[0]])
            outputs["state_offset"] = np.cumsum([0, *(len(r[0]) for r in runs)])
            outputs["rewards"] = np.asarray(
                [x for r in runs for x in r[1]], dtype=np.float64
            )
            outputs["reward_offset"] = np.cumsum([0, *(len(r[1]) for r in runs)])
            outputs["evaluations"] = np.asarray([r[2] for r in runs])
            outputs["terminated"] = np.asarray([r[3] for r in runs])
            outputs["truncated"] = np.asarray([r[4] for r in runs])
            outputs["observation"] = np.stack([r[5] for r in runs])
            outputs["mask"] = np.stack([r[6] for r in runs])
        else:
            n_steps = int(inputs["n_steps"])

            def loop() -> None:
                observation, info = env.reset(seed=0)
                for _ in range(n_steps):
                    features = torch.from_numpy(observation[info["action_mask"]])
                    index = policy.sample(features, env.np_random)
                    observation, _, terminated, truncated, info = env.step(index)
                    if terminated or truncated or info["terminal"]:
                        observation, info = env.reset()

            _, seconds = timed(loop)
    else:
        message = f"unknown mode {mode!r}"
        raise SystemExit(message)
    dump(returned, outputs, seconds)


if __name__ == "__main__":
    main()
