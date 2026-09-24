"""TorchRL's advantage estimate and policy losses on rollouts the adapter wrote (issue #977).

``call`` selects one:

- ``"gae"``: ``GAE`` at ``gamma = 1`` on each case: an episode's ``rewards``
  per decision and ``values`` per state, split by ``reward_offset`` and
  ``value_offset``, at its ``lam``, its last step flagged ``terminated`` or
  truncated; outputs the ``advantages`` concatenated in the rewards' order.
- ``"clip_ppo"``: ``ClipPPOLoss`` with no critic and no entropy bonus, at
  ``clip``, over decisions whose neighbourhood ``features`` score under
  ``weights`` as a ``Categorical``, the index ``taken``, the collector's
  ``old`` log-probability and the ``advantages``, once per entry of ``clip``.
- ``"reinforce"``: ``ReinforceLoss`` with a constant critic, under the
  deterministic interaction type, once per row of ``advantages``; also
  outputs ``resampled``, the action TorchRL's actor drew per decision, per row.

Both losses output ``value`` per case, TorchRL's objective scaled from the mean over
decisions to the mean over episodes by ``n_decisions / n_episodes``, and
``gradient`` in ``weights``, one row per case. ``gamma``, ``lmbda`` and ``clip_epsilon`` are set
as float64 tensors: given as floats TorchRL keeps them in float32, a 1e-8
difference of precision and not of formula. ``case_seconds`` times each
case's framework call alone; the measured seconds are their sum.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from snakes_and_ladders.validation.protocol import dump, load, paths, timed


def _actor(weights: torch.Tensor) -> Any:
    """A softmax over ``features @ weights`` as a TorchRL probabilistic actor."""
    from tensordict.nn import (
        ProbabilisticTensorDictModule,
        ProbabilisticTensorDictSequential,
        TensorDictModule,
    )

    def scores(rows: torch.Tensor) -> torch.Tensor:
        return rows @ weights

    return ProbabilisticTensorDictSequential(
        TensorDictModule(scores, in_keys=["features"], out_keys=["logits"]),
        ProbabilisticTensorDictModule(
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        ),
    )


def _gae(inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[float]]:
    from tensordict import TensorDict
    from torchrl.objectives.value import GAE

    all_rewards = torch.as_tensor(inputs["rewards"], dtype=torch.float64)
    all_values = torch.as_tensor(inputs["values"], dtype=torch.float64)
    reward_offset, value_offset = inputs["reward_offset"], inputs["value_offset"]
    advantages, case_seconds = [], []
    for case in range(inputs["lam"].size):
        rewards = all_rewards[reward_offset[case] : reward_offset[case + 1], None]
        value = all_values[value_offset[case] : value_offset[case + 1], None]
        steps = rewards.shape[0]
        done = torch.zeros(steps, 1, dtype=torch.bool)
        done[-1] = True
        flags = torch.zeros(steps, 1, dtype=torch.bool)
        flags[-1] = bool(inputs["terminated"][case])
        rollout = TensorDict(
            {
                "state_value": value[:-1],
                "next": TensorDict(
                    {
                        "state_value": value[1:],
                        "reward": rewards,
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
            lmbda=torch.tensor(float(inputs["lam"][case]), dtype=torch.float64),
            value_network=None,
        )
        out, seconds = timed(lambda: estimator(rollout))  # noqa: B023
        advantages.append(out["advantage"][:, 0].numpy())
        case_seconds.append(seconds)
    return {"advantages": np.concatenate(advantages)}, case_seconds


def _loss(
    call: str, inputs: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[float]]:
    from tensordict import TensorDict
    from tensordict.nn import InteractionType, TensorDictModule, set_interaction_type
    from torchrl.objectives import ClipPPOLoss, ReinforceLoss

    weights = torch.tensor(inputs["weights"], dtype=torch.float64, requires_grad=True)
    features = torch.as_tensor(inputs["features"], dtype=torch.float64)
    taken = torch.as_tensor(inputs["taken"], dtype=torch.int64)
    n = taken.shape[0]
    scale = n / int(inputs["n_episodes"])
    advantages = torch.as_tensor(inputs["advantages"], dtype=torch.float64)
    values, gradients, resampled, case_seconds = [], [], [], []

    def columns(row: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "features": features,
            "action": taken.clone(),
            "advantage": row[:, None],
        }

    if call == "clip_ppo":
        old = torch.as_tensor(inputs["old"], dtype=torch.float64)
        for clip in inputs["clip"].tolist():
            loss = ClipPPOLoss(
                _actor(weights),
                critic_network=None,
                clip_epsilon=clip,
                entropy_bonus=False,
                critic_coeff=None,
            )
            loss.clip_epsilon = torch.tensor(clip, dtype=torch.float64)
            batch = TensorDict({**columns(advantages), "action_log_prob": old}, [n])

            def evaluate() -> tuple[torch.Tensor, torch.Tensor]:
                value = loss(batch)["loss_objective"] * scale  # noqa: B023
                (gradient,) = torch.autograd.grad(value, weights)
                return value, gradient

            (value, gradient), seconds = timed(evaluate)
            values.append(float(value.detach()))
            gradients.append(gradient.numpy())
            case_seconds.append(seconds)
    else:
        critic = TensorDictModule(
            lambda rows: rows.new_zeros((rows.shape[0], 1)),
            in_keys=["features"],
            out_keys=["state_value"],
        )
        loss = ReinforceLoss(_actor(weights), critic_network=critic, functional=False)
        target = torch.zeros(n, 1, dtype=torch.float64)
        for row in advantages:
            batch = TensorDict({**columns(row), "value_target": target}, [n])

            def evaluate() -> tuple[torch.Tensor, torch.Tensor]:
                with set_interaction_type(InteractionType.DETERMINISTIC):
                    value = loss(batch)["loss_actor"] * scale  # noqa: B023
                (gradient,) = torch.autograd.grad(value, weights)
                return value, gradient

            (value, gradient), seconds = timed(evaluate)
            values.append(float(value.detach()))
            gradients.append(gradient.numpy())
            resampled.append(batch["action"].numpy())
            case_seconds.append(seconds)
    outputs = {"value": np.asarray(values), "gradient": np.stack(gradients)}
    if resampled:
        outputs["resampled"] = np.stack(resampled)
    return outputs, case_seconds


def main() -> None:
    """Run the requested TorchRL call and write its answer back."""
    given, returned = paths()
    inputs = load(given)
    call = str(inputs["call"])
    if call == "gae":
        outputs, case_seconds = _gae(inputs)
    elif call in {"clip_ppo", "reinforce"}:
        outputs, case_seconds = _loss(call, inputs)
    else:
        message = f"unknown call {call!r}"
        raise SystemExit(message)
    outputs["case_seconds"] = np.asarray(case_seconds)
    dump(returned, outputs, float(sum(case_seconds)))


if __name__ == "__main__":
    main()
