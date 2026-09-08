"""`learn.ppo`'s advantage estimate and clipped loss against TorchRL's, on one fixed rollout (issue #322).

TorchRL's ``GAE`` and ``ClipPPOLoss`` are a second implementation of the
two equations ``learn.ppo`` states -- the generalized advantage and the
clipped surrogate -- written by people who never saw ours: the validation
use issue #315 scoped, needing no code to move. Same rollout tensors in, same advantages and the same
objective out, to ``1e-10``.

Two conventions differ and are mapped rather than hidden. TorchRL reads a
truncated episode from ``done`` without ``terminated``, which is exactly the
flag ``generalized_advantages`` takes; and its objective is the mean over
decisions where ours is the mean over episodes of the sum over decisions,
so the two agree up to the factor ``n_decisions / n_episodes``.

Skips without the ``frameworks`` extra, and until ``learn.ppo`` lands
(issue #313): a pin against a module that is not there pins nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.rollout import rollout

torchrl = pytest.importorskip("torchrl")
tensordict = pytest.importorskip("tensordict")
ppo = pytest.importorskip(
    "snakes_and_ladders.learn.ppo",
    reason="learn.ppo lands with issue #313; nothing to pin until it does",
)

from tensordict import TensorDict  # noqa: E402
from tensordict.nn import (  # noqa: E402
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torchrl.objectives import ClipPPOLoss  # noqa: E402
from torchrl.objectives.value import GAE  # noqa: E402

FIELD = np.array([0.4, -0.1, -0.3])

# One fixed rollout, in the shape `generalized_advantages` takes: rewards per
# decision and one value per state, the last being the final state's.
ROLLOUTS = (
    ([1.0, -0.5, 2.0], [1.0, 2.0, 0.5, 4.0]),
    ([0.3], [0.2, -0.7]),
    ([-1.0, 0.25, 0.5, 0.125, -2.0], [0.0, 1.0, -1.0, 0.5, 0.25, 3.0]),
)


def _landscape() -> PottsLandscape:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=4)


@pytest.mark.oracle
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize(("rewards", "values"), ROLLOUTS)
def test_the_advantages_are_torchrl_s_gae_at_gamma_one(
    rewards: list[float], values: list[float], lam: float, terminated: bool
) -> None:
    ours = ppo.generalized_advantages(rewards, values, lam=lam, terminated=terminated)
    steps = len(rewards)
    done = torch.zeros(steps, 1, dtype=torch.bool)
    done[-1] = True
    flags = torch.zeros(steps, 1, dtype=torch.bool)
    flags[-1] = terminated
    value = torch.tensor(values, dtype=torch.float64)[:, None]
    rollout_data = TensorDict(
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
    # `gamma` and `lmbda` as float64 tensors: given as floats, GAE keeps
    # them in float32 and 0.95 becomes 0.949999988, a 2e-8 disagreement
    # that is TorchRL's default precision and not a difference of formula.
    estimator = GAE(
        gamma=torch.tensor(1.0, dtype=torch.float64),
        lmbda=torch.tensor(lam, dtype=torch.float64),
        value_network=None,
    )
    theirs = estimator(rollout_data)
    torch.testing.assert_close(
        theirs["advantage"][:, 0],
        torch.tensor(ours, dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )


def _decisions(
    landscape: PottsLandscape, policy: LinearPolicy, seed: int, episodes: int
) -> tuple[list[tuple[tuple[int, ...], ...]], torch.Tensor, torch.Tensor, list[int]]:
    """Episodes under ``policy``, with every decision's neighbourhood features and the index taken."""
    rng = np.random.default_rng(seed)
    rolled = [rollout(landscape, policy, rng, 3) for _ in range(episodes)]
    features, taken, owner = [], [], []
    for index, episode in enumerate(rolled):
        for state, action in zip(episode.states, episode.actions, strict=False):
            available = landscape.actions(state)
            features.append(landscape.features(state, available))
            taken.append(available.index(action))
            owner.append(index)
    # Every neighbourhood on the chain has the same width, so no padding and
    # an all-true mask; the tree case would pad to `n_max` and mask.
    return (
        [episode.states for episode in rolled],
        torch.stack(features),
        torch.tensor(taken),
        owner,
    )


def _log_probabilities(
    policy: LinearPolicy, features: torch.Tensor, taken: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [
            policy.log_probabilities(rows)[index]
            for rows, index in zip(features, taken, strict=True)
        ]
    )


@pytest.mark.oracle
@pytest.mark.parametrize("clip", [0.1, 0.2, 0.5])
def test_the_clipped_objective_and_its_gradient_are_torchrl_s(clip: float) -> None:
    landscape = _landscape()
    collector = LinearPolicy(2)
    collector.set_weights(torch.tensor([0.1, 0.2], dtype=torch.float64))
    _, features, taken, owner = _decisions(landscape, collector, seed=0, episodes=12)
    old = _log_probabilities(collector, features, taken).detach()
    n_episodes = max(owner) + 1
    rng = np.random.default_rng(1)
    advantages = torch.tensor(rng.normal(size=len(taken)))
    per_episode = [
        [float(a) for a, o in zip(advantages, owner, strict=True) if o == episode]
        for episode in range(n_episodes)
    ]

    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    current = _log_probabilities(policy, features, taken)
    current_by_episode = [
        torch.stack([c for c, o in zip(current, owner, strict=True) if o == episode])
        for episode in range(n_episodes)
    ]
    old_by_episode = [
        torch.stack([c for c, o in zip(old, owner, strict=True) if o == episode])
        for episode in range(n_episodes)
    ]
    ours, _ = ppo.ppo_loss(current_by_episode, old_by_episode, per_episode, clip=clip)
    (our_gradient,) = torch.autograd.grad(ours, policy.weights)

    # TorchRL's actor: the same softmax over the same scores, as a
    # Categorical over the neighbourhood, differentiable in the same weights.
    def scores(rows: torch.Tensor) -> torch.Tensor:
        return rows @ policy.weights

    actor = ProbabilisticTensorDictSequential(
        TensorDictModule(scores, in_keys=["features"], out_keys=["logits"]),
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
    # The same float32 default as GAE's: the clip bounds are `log1p` of a
    # float32 buffer, 1e-8 off the float64 bounds ours clamps at.
    loss.clip_epsilon = torch.tensor(clip, dtype=torch.float64)
    batch = TensorDict(
        {
            "features": features,
            "action": taken,
            "action_log_prob": old,
            "advantage": advantages[:, None],
        },
        [len(taken)],
    )
    theirs = loss(batch)["loss_objective"] * len(taken) / n_episodes
    (their_gradient,) = torch.autograd.grad(theirs, policy.weights)
    torch.testing.assert_close(theirs, ours, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(their_gradient, our_gradient, rtol=0.0, atol=1e-10)
    # Clipping is active somewhere, or the pin says nothing about the clip.
    ratio = torch.exp(current - old)
    assert bool(((ratio < 1 - clip) | (ratio > 1 + clip)).any())
