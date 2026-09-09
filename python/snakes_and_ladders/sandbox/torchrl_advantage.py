"""TorchRL's ``GAE`` fronting :func:`~snakes_and_ladders.learn.ppo.generalized_advantages` (issues #322, #391).

The front that was measured and declined. ``eq:gae`` at ``gamma = 1`` is
twelve lines of Python over two lists; ``torchrl.objectives.value.GAE``
computes the same recursion over a ``TensorDict`` built per episode, and
issue #391 asked whether the library should own the equation.

**It should not, and this module is why the answer is re-checkable.** The
declining measurement is `docs/experiments/007-batched-rollout-and-torchrl-ppo.md`:
per 32-episode batch on the Potts chain fixture, one thread on a shared
four-core Linux x86-64 host under the exclusive lock, TorchRL took 6.384 ms
against 0.029 ms for the recursion, 220x, with the tensordicts prebuilt and
so with the marshalling excluded. `tests/benchmarks/test_learn_ppo_bench.py`
re-measures both sides, and `tests/regression/learn/test_learn_ppo_torchrl.py`
pins the values against each other, so a later TorchRL that moves the ratio
shows up as a number rather than as a claim nobody can run.

Two conventions are mapped rather than hidden, and they are the reason the
front is a front and not a call. TorchRL reads a truncated episode from
``done`` without ``terminated``, which is the flag
:func:`~snakes_and_ladders.learn.ppo.generalized_advantages` takes; and
``gamma`` and ``lmbda`` given as Python floats are kept in ``float32``, so
they are passed as ``float64`` tensors and the disagreement that remains is
of formula rather than of precision.

Imports ``torchrl`` at module scope: it is the ``frameworks`` extra, and a
caller without it gets an ``ImportError`` here rather than a silent fallback
to the implementation this exists to referee.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDict
from torchrl.objectives.value import GAE


def generalized_advantages(
    rewards: Sequence[float], values: Sequence[float], *, lam: float, terminated: bool
) -> list[float]:
    """``eq:gae`` at ``gamma = 1`` through ``torchrl.objectives.value.GAE``.

    Signature for signature what
    :func:`~snakes_and_ladders.learn.ppo.generalized_advantages` takes and
    returns, so the two can be pinned against each other and timed against
    each other without an adapter in between.

    Parameters
    ----------
    rewards : Sequence[float]
        One reward per decision.
    values : Sequence[float]
        The critic at every visited state, one entry longer than ``rewards``.
    lam : float
        The exponential weight of ``eq:gae``, in ``[0, 1]``.
    terminated : bool
        Whether the episode ended at a local optimum rather than on the
        decision budget. TorchRL takes it as ``next.terminated`` and zeroes
        the bootstrap on it, which is what the hand-rolled recursion does
        with its ``tail``.

    Returns
    -------
    list[float]
        One advantage per decision.

    Raises
    ------
    ValueError
        If ``values`` is not one longer than ``rewards``, if ``lam`` is
        outside ``[0, 1]``, or if ``rewards`` is empty. The first two are the
        refusals the hand-rolled implementation makes and are made here so
        the front declines the same inputs; the third is TorchRL's own
        limit, which has no zero-length trajectory.
    """
    if len(values) != len(rewards) + 1:
        msg = (
            f"{len(rewards)} rewards need {len(rewards) + 1} values, got {len(values)}"
        )
        raise ValueError(msg)
    if not 0.0 <= lam <= 1.0:
        msg = f"lam must be in [0, 1], got {lam}"
        raise ValueError(msg)
    if not rewards:
        msg = "GAE has no expression for an episode of no decisions"
        raise ValueError(msg)
    steps = len(rewards)
    done = torch.zeros(steps, 1, dtype=torch.bool)
    done[-1] = True
    flags = torch.zeros(steps, 1, dtype=torch.bool)
    flags[-1] = terminated
    value = torch.tensor(values, dtype=torch.float64)[:, None]
    trajectory = TensorDict(
        {
            "state_value": value[:-1],
            "next": TensorDict(
                {
                    "state_value": value[1:],
                    "reward": torch.tensor(rewards, dtype=torch.float64)[:, None],
                    "done": done,
                    "terminated": flags,
                },
                [steps],
            ),
        },
        [steps],
    )
    estimator = GAE(
        gamma=torch.tensor(1.0, dtype=torch.float64),
        lmbda=torch.tensor(lam, dtype=torch.float64),
        value_network=None,
    )
    return [float(a) for a in estimator(trajectory)["advantage"][:, 0]]


def episode_advantages(
    rewards: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    terminated: Sequence[bool],
    *,
    lam: float,
) -> list[list[float]]:
    """:func:`generalized_advantages` over a whole batch, one call per episode.

    What a front pays over a batch, which is the unit the decline was
    measured in: TorchRL's estimator takes one trajectory, so a batch is a
    Python loop over it either way and the term that differs is the
    tensordict.

    Parameters
    ----------
    rewards : Sequence[Sequence[float]]
        Per episode, one reward per decision.
    values : Sequence[Sequence[float]]
        Per episode, the critic at every visited state.
    terminated : Sequence[bool]
        Per episode, whether it ended at a local optimum.
    lam : float
        The exponential weight of ``eq:gae``.

    Returns
    -------
    list[list[float]]
        Per episode, one advantage per decision.
    """
    return [
        generalized_advantages(r, v, lam=lam, terminated=t)
        for r, v, t in zip(rewards, values, terminated, strict=True)
    ]


__all__ = ["episode_advantages", "generalized_advantages"]
