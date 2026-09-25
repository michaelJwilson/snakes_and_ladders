"""Gymnasium as a referee for the ``learn.Environment`` protocol (issues #322, #977).

Farama's ``check_env`` states what a ``gymnasium.Env`` must do --- seeded
resets that agree, observations inside their space, a deterministic step ---
and an episode driven through its ``reset``/``step`` must be the episode
:func:`sal.learn.rollout.rollout` draws. The adapter that
makes a ``learn.Environment`` a ``gymnasium.Env`` lived in the package as
``learn.gym.GymnasiumEnvironment`` with no consumer but its tests; it now
lives in ``scripts/gymnasium.py``, and Gymnasium runs only there, in a
subprocess.

An environment crosses the boundary as a :class:`Spec`, rebuilt on each side
by :func:`build`; a state crosses as :func:`encode`'s string.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sal.sim.simulator import simulate_tree
from sal.validation.runner import run

#: The script this adapter runs.
SCRIPT = "gymnasium"


@dataclass(frozen=True)
class Spec:
    """A ``learn.Environment`` by construction: the Potts chain or the NNI tree search."""

    #: ``"potts"`` or ``"tree"``.
    kind: str
    #: The Potts chain's coupling, field and length.
    coupling: float = 0.0
    field: tuple[float, ...] = ()
    chain_length: int = 0
    #: The tree search's simulation fixture and fixed branch length.
    fixture: str = ""
    branch_length: float = 0.0

    def arrays(self) -> dict[str, np.ndarray]:
        """The spec as ``.npz`` entries."""
        return {
            "kind": np.asarray(self.kind),
            "coupling": np.asarray(self.coupling),
            "field": np.asarray(self.field, dtype=np.float64),
            "chain_length": np.asarray(self.chain_length),
            "fixture": np.asarray(self.fixture),
            "branch_length": np.asarray(self.branch_length),
        }

    @classmethod
    def of(cls, arrays: dict[str, np.ndarray]) -> Spec:
        """The spec :meth:`arrays` wrote."""
        return cls(
            kind=str(arrays["kind"]),
            coupling=float(arrays["coupling"]),
            field=tuple(arrays["field"].tolist()),
            chain_length=int(arrays["chain_length"]),
            fixture=str(arrays["fixture"]),
            branch_length=float(arrays["branch_length"]),
        )


def build(spec: Spec) -> tuple[Any, int]:
    """The environment ``spec`` names, and the widest neighbourhood it can have."""
    if spec.kind == "potts":
        from sal.learn.potts import PottsEnvironment

        potts = PottsEnvironment(
            coupling=spec.coupling,
            field=np.asarray(spec.field),
            chain_length=spec.chain_length,
        )
        return potts, spec.chain_length * (potts.n_states - 1)
    from sal.fixtures import load_params
    from sal.learn.tree import RewardModel, TreeEnvironment
    from sal.sim.params import SimulationParams
    from sal.sim.topology import MoveSet

    params = load_params(Path(spec.fixture), SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed))
    tree = TreeEnvironment(
        dict(dataset.alignment),
        params.k,
        params.pi,
        spec.branch_length,
        reward=RewardModel.KNOWN,
        moves=MoveSet.NNI,
    )
    # An unrooted tree on n taxa has 2 (n - 3) NNI neighbours before
    # deduplication, so no state can exceed this.
    return tree, 2 * (len(dataset.alignment) - 3)


def encode(state: Any) -> str:
    """A state as a string equal across the boundary exactly where the states are.

    A Potts labelling is its tuple; a topology is its sorted leaf
    bipartitions, since the tree's own object carries arbitrary node ids.
    """
    if isinstance(state, tuple):
        return ",".join(str(int(s)) for s in state)
    from sal.sim.topology import leaf_bipartitions

    return ";".join(sorted("|".join(sorted(side)) for side in leaf_bipartitions(state)))


@dataclass(frozen=True)
class RoundTrip:
    """One episode through ``reset``/``step``, actions drawn from the adapter's generator."""

    states: list[str]
    rewards: np.ndarray
    evaluations: int
    terminated: bool
    truncated: bool
    #: The observation and the action mask ``reset`` returned.
    observation: np.ndarray
    mask: np.ndarray


def check(spec: Spec, *, max_steps: int) -> bool:
    """Whether Farama's ``check_env`` passes the adapter over ``spec``'s environment."""
    inputs = {"mode": np.asarray("check"), "max_steps": np.asarray(max_steps)}
    return bool(run(SCRIPT, {**spec.arrays(), **inputs}).outputs["passed"])


def episodes(
    spec: Spec,
    weights: Sequence[float],
    seeds: Sequence[int],
    *,
    max_steps: int,
    extra_width: int = 0,
    start: Sequence[int] | None = None,
) -> list[RoundTrip]:
    """One episode per seed under a ``LinearPolicy`` of ``weights``.

    The observation is padded to the widest neighbourhood plus
    ``extra_width``; ``start`` replaces the drawn start state.
    """
    inputs = {
        "mode": np.asarray("episodes"),
        "weights": np.asarray(weights, dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "max_steps": np.asarray(max_steps),
        "extra_width": np.asarray(extra_width),
        "start": np.asarray([] if start is None else start, dtype=np.int64),
    }
    out = run(SCRIPT, {**spec.arrays(), **inputs}).outputs
    state_offset, reward_offset = out["state_offset"], out["reward_offset"]
    return [
        RoundTrip(
            states=out["states"][state_offset[k] : state_offset[k + 1]].tolist(),
            rewards=out["rewards"][reward_offset[k] : reward_offset[k + 1]],
            evaluations=int(out["evaluations"][k]),
            terminated=bool(out["terminated"][k]),
            truncated=bool(out["truncated"][k]),
            observation=out["observation"][k],
            mask=out["mask"][k],
        )
        for k in range(len(seeds))
    ]


def throughput(
    spec: Spec, weights: Sequence[float], *, n_steps: int, max_steps: int
) -> float:
    """Wall seconds for ``n_steps`` decisions through the adapter, resetting at each episode's end."""
    inputs = {
        "mode": np.asarray("throughput"),
        "weights": np.asarray(weights, dtype=np.float64),
        "n_steps": np.asarray(n_steps),
        "max_steps": np.asarray(max_steps),
    }
    return run(SCRIPT, {**spec.arrays(), **inputs}).seconds
