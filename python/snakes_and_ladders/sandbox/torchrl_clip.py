"""TorchRL's ``ClipPPOLoss`` fronting the clipped surrogate, one level above ``ppo_loss`` (issues #322, #391).

**``ClipPPOLoss`` cannot front
:func:`~snakes_and_ladders.learn.ppo.ppo_loss` at its signature, and no
wrapper here pretends otherwise.** ``ppo_loss`` is handed the current
log-probabilities, already computed: ``learn.ppo`` scores a batch's
neighbourhoods once per iteration and reads the policy off that scoring in
every epoch, so the log-probability is an argument rather than something the
loss goes and fetches. ``ClipPPOLoss`` is handed an *actor* and calls it, to
build the distribution and take ``log_prob`` of the action itself; the only
argument of ours it can take is ``old_log_probabilities``. A front at
``ppo_loss``'s signature would have to invent an actor whose distribution
reproduces log-probabilities it was given, which is a fiction, not a
substitution.

So the front is stated one level up, at the batch ``ppo_loss`` is called on:
:func:`clipped_objective` takes the policy and the scored neighbourhoods
instead of the log-probabilities, and returns the same number ``ppo_loss``
returns. That level is what would have had to change to adopt the framework,
which is the honest measure of what adoption costs.

**Declined on its numbers**, recorded in
`docs/experiments/007-batched-rollout-and-torchrl-ppo.md`. The clipped
surrogate is 1.467 ms of a 175 ms PPO iteration on the Potts chain fixture,
0.8%, so no port of it can pay under the profile-first rule root
``CLAUDE.md`` states (Gorelick & Ozsvald ch. 2); and the whole loop rebuilt
on TorchRL ran 6.31 s against 8.06 s over 1,920 episodes, of which 1.13x was
a hoist in ``learn.ppo`` itself and landed there instead. All on one thread
of a shared four-core Linux x86-64 host under the exclusive lock.
`tests/benchmarks/test_learn_ppo_bench.py` re-measures both sides and
`tests/regression/learn/test_learn_ppo_torchrl.py` pins them against each
other, so a later TorchRL that moves the ratio is visible.

**What the front does not return.** ``ppo_loss`` also reports the fraction of
decisions the clip was active at, a diagnostic ``PPOTraining`` carries;
``ClipPPOLoss`` computes it internally and exposes nothing, so an adoption
would have lost the diagnostic or recomputed the ratios beside the loss.

Imports ``torchrl`` at module scope: it is the ``frameworks`` extra, and a
caller without it gets an ``ImportError`` here rather than a silent fallback
to the implementation this exists to referee.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torchrl.objectives import ClipPPOLoss

from snakes_and_ladders.learn.policy import TrainablePolicy

type Neighbourhoods = Sequence[Sequence[tuple[torch.Tensor, int]]]


def _padded_logits(
    policy: TrainablePolicy, neighbourhoods: Neighbourhoods
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every decision's policy log-probabilities, padded to the widest neighbourhood.

    The padding is on the *logits* and not on the features, because the
    policy scores a neighbourhood of the width it has and a padded feature
    row would be a move that does not exist scored as though it did. A
    ``-inf`` logit leaves the softmax over the real rows exactly the
    distribution the policy states, and carries no gradient.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The ``(n_decisions, n_max)`` logits and the ``(n_decisions,)`` index
        taken at each decision.
    """
    steps = [step for episode in neighbourhoods for step in episode]
    width = max(int(features.shape[0]) for features, _ in steps)
    rows = []
    for features, _ in steps:
        scored = policy.log_probabilities(features)
        pad = torch.full(
            (width - int(scored.shape[0]),), -torch.inf, dtype=scored.dtype
        )
        rows.append(torch.cat([scored, pad]))
    return torch.stack(rows), torch.tensor([index for _, index in steps])


def clipped_objective(
    policy: TrainablePolicy,
    neighbourhoods: Neighbourhoods,
    old_log_probabilities: Sequence[torch.Tensor],
    advantages: Sequence[Sequence[float]],
    *,
    clip: float,
) -> torch.Tensor:
    """``eq:ppo-clip`` through ``torchrl.objectives.ClipPPOLoss``, in ``ppo_loss``'s units.

    TorchRL's objective is the mean over decisions; ours is the mean over
    episodes of the sum over decisions, so the returned value is theirs
    scaled by ``n_decisions / n_episodes``. That factor is the whole of the
    difference: the two agree on the value and on its gradient.

    The actor handed to ``ClipPPOLoss`` is the policy's own distribution,
    written as a ``Categorical`` over the neighbourhood whose logits are the
    policy's log-probabilities. A softmax of a normalized log-probability
    vector is that same distribution, and it is differentiable in the same
    parameters, so the gradient TorchRL takes is a gradient of ours.

    Parameters
    ----------
    policy : TrainablePolicy
        The policy being updated. Read on the current graph, so the returned
        loss is differentiable in its parameters.
    neighbourhoods : Neighbourhoods
        Per episode, every decision's neighbourhood features and the index
        taken, as :func:`snakes_and_ladders.learn.ppo._neighbourhoods` builds
        them.
    old_log_probabilities : Sequence[torch.Tensor]
        Per episode, the collecting policy's log-probability of each decision.
    advantages : Sequence[Sequence[float]]
        Per episode, one advantage per decision.
    clip : float
        The clip half-width ``eps`` of ``eq:ppo-clip``.

    Returns
    -------
    torch.Tensor
        The loss :func:`~snakes_and_ladders.learn.ppo.ppo_loss` returns first,
        without the clipped fraction it returns second.

    Raises
    ------
    ValueError
        If ``clip`` is not positive, or if the batch has no decisions.
        ``ppo_loss`` returns zero for an empty batch; ``ClipPPOLoss`` has no
        expression for one, which is a limit of the front and is refused
        rather than papered over.
    """
    if clip <= 0.0:
        msg = f"clip must be positive, got {clip}"
        raise ValueError(msg)
    decisions = sum(len(episode) for episode in neighbourhoods)
    if decisions == 0:
        msg = "ClipPPOLoss has no expression for a batch of no decisions"
        raise ValueError(msg)
    logits, taken = _padded_logits(policy, neighbourhoods)
    old = torch.cat([t.detach() for t in old_log_probabilities if t.shape[0]])
    weights = torch.as_tensor(
        [a for episode in advantages for a in episode], dtype=logits.dtype
    )
    # The actor is the identity on the logits assembled above: `ClipPPOLoss`
    # calls it to build the distribution and take `log_prob` itself, which is
    # exactly the step `ppo_loss`'s signature has already done by the time it
    # is called.
    actor = ProbabilisticTensorDictSequential(
        TensorDictModule(lambda rows: rows, in_keys=["logits"], out_keys=["logits"]),
        ProbabilisticTensorDictModule(
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        ),
    )
    loss = ClipPPOLoss(
        actor,
        critic_network=None,
        clip_epsilon=clip,
        entropy_bonus=False,
        critic_coeff=None,
    )
    # `clip_epsilon` is a float32 buffer by default and the bounds are
    # `log1p` of it, which is a difference of precision and not of formula.
    loss.clip_epsilon = torch.tensor(clip, dtype=logits.dtype)
    batch = TensorDict(
        {
            "logits": logits,
            "action": taken,
            "action_log_prob": old,
            "advantage": weights[:, None],
        },
        [decisions],
    )
    # `ClipPPOLoss` returns a `TensorDict`, which mypy reads as `Any`; the
    # entry is a tensor and the cast is where that is asserted.
    objective = cast("torch.Tensor", loss(batch)["loss_objective"])
    return objective * decisions / len(neighbourhoods)


__all__ = ["clipped_objective"]
