"""TorchRL as an oracle for the advantage estimate and the policy losses (issue #977).

TorchRL's ``GAE``, ``ClipPPOLoss`` and ``ReinforceLoss`` are a second
implementation of ``eq:gae``, the clipped surrogate and ``eq:reinforce``,
written by people who never saw :mod:`sal.learn.ppo` or
:mod:`sal.learn.reinforce`. It runs only in
``scripts/torchrl.py``, in a subprocess (issue #322 ran it in-process).

Each call takes several cases, since importing TorchRL costs about 4 s per
interpreter. Every loss comes back as a float64 scalar per case on the
package's scale, the mean over episodes of the sum over decisions, with its
gradient in the policy's weights.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sal.validation.runner import run

#: The script this adapter runs.
SCRIPT = "torchrl"


@dataclass(frozen=True)
class Advantages:
    """TorchRL's generalized advantages, one array per case, and what each cost."""

    advantages: list[np.ndarray]
    #: Wall seconds of each case's ``GAE`` call alone.
    seconds: np.ndarray


@dataclass(frozen=True)
class Losses:
    """A TorchRL policy loss per case, its gradient in the weights, and what each cost."""

    values: np.ndarray
    #: One row per case.
    gradients: np.ndarray
    #: Wall seconds of each case's loss and gradient.
    seconds: np.ndarray
    #: The action TorchRL's actor drew per decision, one row per case;
    #: ``ReinforceLoss`` only.
    resampled: np.ndarray | None


@dataclass(frozen=True)
class Rollout:
    """One GAE case: ``values`` has one entry per state, one more than ``rewards``."""

    rewards: Sequence[float]
    values: Sequence[float]
    lam: float
    terminated: bool


def gae(episodes: Sequence[Rollout]) -> Advantages:
    """``GAE`` at ``gamma = 1`` on each episode, at its ``lam`` and end flag."""
    rewards = [np.asarray(e.rewards, dtype=np.float64) for e in episodes]
    values = [np.asarray(e.values, dtype=np.float64) for e in episodes]
    reward_offset = np.cumsum([0, *(r.size for r in rewards)])
    result = run(
        SCRIPT,
        {
            "call": np.asarray("gae"),
            "rewards": np.concatenate(rewards),
            "values": np.concatenate(values),
            "reward_offset": reward_offset,
            "value_offset": np.cumsum([0, *(v.size for v in values)]),
            "lam": np.asarray([e.lam for e in episodes], dtype=np.float64),
            "terminated": np.asarray([e.terminated for e in episodes]),
        },
    )
    joined = result.outputs["advantages"]
    return Advantages(
        [joined[a:b] for a, b in itertools.pairwise(reward_offset)],
        result.outputs["case_seconds"],
    )


def _losses(call: str, inputs: dict[str, np.ndarray]) -> Losses:
    out = run(SCRIPT, {"call": np.asarray(call), **inputs}).outputs
    return Losses(
        values=out["value"],
        gradients=out["gradient"],
        seconds=out["case_seconds"],
        resampled=out.get("resampled"),
    )


def _decisions(
    features: np.ndarray,
    taken: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
    n_episodes: int,
) -> dict[str, np.ndarray]:
    return {
        "features": np.ascontiguousarray(features, dtype=np.float64),
        "taken": np.ascontiguousarray(taken, dtype=np.int64),
        "advantages": np.ascontiguousarray(advantages, dtype=np.float64),
        "weights": np.ascontiguousarray(weights, dtype=np.float64),
        "n_episodes": np.asarray(n_episodes, dtype=np.int64),
    }


def clip_ppo_loss(
    features: np.ndarray,
    taken: np.ndarray,
    old: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
    *,
    clips: Sequence[float],
    n_episodes: int,
) -> Losses:
    """``ClipPPOLoss`` at each of ``clips``: ``features`` is ``(n, width, n_features)``."""
    inputs = _decisions(features, taken, advantages, weights, n_episodes)
    inputs["old"] = np.ascontiguousarray(old, dtype=np.float64)
    inputs["clip"] = np.asarray(clips, dtype=np.float64)
    return _losses("clip_ppo", inputs)


def reinforce_loss(
    features: np.ndarray,
    taken: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
    *,
    n_episodes: int,
) -> Losses:
    """``ReinforceLoss`` for each row of ``advantages``, the actor deterministic."""
    inputs = _decisions(features, taken, np.atleast_2d(advantages), weights, n_episodes)
    return _losses("reinforce", inputs)
