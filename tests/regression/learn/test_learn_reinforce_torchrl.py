"""`learn.reinforce`'s surrogate loss against TorchRL's ``ReinforceLoss`` (issue #376).

The score-function estimator ``eq:reinforce`` states is a formula, and
TorchRL's ``ReinforceLoss`` is a second implementation of it written by
people who never saw ours: the validation use ``infra/CLAUDE.md`` scopes,
needing no code to move. Same episodes, same advantages and the same policy
in, the same scalar and the same gradient out.

Two conventions differ and are mapped rather than hidden. TorchRL's objective
is the mean over decisions where ours is the mean over episodes of the sum
over decisions, so the two agree up to ``n_decisions / n_episodes``; and
TorchRL forms the advantage itself when the key is absent, which would
compute a different estimator, so the return-to-go minus the baseline is
supplied. Its constructor requires a critic and its forward pass scores one,
so a constant critic and a matching target are passed and only
``loss_actor`` is read; REINFORCE with a baseline that is a constant, which
is what ``eq:reinforce`` states, has no critic to compare against.

One further difference is structural rather than a convention. TorchRL's
loss re-runs the actor and *resamples* the action rather than scoring the one
in the batch, so the two agree on arbitrary data only up to a draw. The
episodes here are therefore greedy rollouts under the policy being scored,
taken under the deterministic interaction type, where the action TorchRL
draws is the action the episode recorded. The estimator is a function of
(state, action, return) triples and does not care that they are greedy.

Skips without the ``frameworks`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.environment import Environment, Episode
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.reinforce import surrogate_loss

torchrl = pytest.importorskip("torchrl")
tensordict = pytest.importorskip("tensordict")

from tensordict import TensorDict  # noqa: E402
from tensordict.nn import (  # noqa: E402
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    set_interaction_type,
)
from torchrl.objectives import ReinforceLoss  # noqa: E402

FIELD = np.array([0.4, -0.1, -0.3])
BASELINES = (0.0, 0.5, -1.25)


def _landscape() -> Environment[tuple[int, ...], tuple[int, int]]:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=4)


def _greedy_episode(
    landscape: Environment[tuple[int, ...], tuple[int, int]],
    policy: LinearPolicy,
    start: tuple[int, ...],
    steps: int,
) -> Episode[tuple[int, ...], tuple[int, int]]:
    """One rollout taking the highest-scoring action at every decision."""
    states, actions, rewards = [start], [], []
    state = start
    for _ in range(steps):
        available = landscape.actions(state)
        scores = landscape.features(state, available) @ policy.weights
        action = available[int(torch.argmax(scores.detach()))]
        state, reward = landscape.step(state, action)
        actions.append(action)
        rewards.append(reward)
        states.append(state)
    return Episode(
        states=tuple(states),
        actions=tuple(actions),
        rewards=tuple(rewards),
        terminated=False,
    )


def _episodes(
    landscape: Environment[tuple[int, ...], tuple[int, int]],
    policy: LinearPolicy,
    seed: int,
    count: int,
) -> list[Episode[tuple[int, ...], tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    return [
        _greedy_episode(landscape, policy, landscape.reset(rng), 3)
        for _ in range(count)
    ]


def _decisions(
    landscape: Environment[tuple[int, ...], tuple[int, int]],
    episodes: list[Episode[tuple[int, ...], tuple[int, int]]],
    baseline: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every decision's neighbourhood features, the index taken, and its advantage."""
    features, taken, advantages = [], [], []
    for episode in episodes:
        returns = episode.returns_to_go()
        for step, action in enumerate(episode.actions):
            available = landscape.actions(episode.states[step])
            features.append(landscape.features(episode.states[step], available))
            taken.append(available.index(action))
            advantages.append(returns[step] - baseline)
    # Every neighbourhood on the chain has the same width, so no padding and
    # no mask; the tree case would pad to `n_max`.
    return (
        torch.stack(features),
        torch.tensor(taken),
        torch.tensor(advantages, dtype=torch.float64),
    )


def _actor(policy: LinearPolicy) -> ProbabilisticTensorDictSequential:
    """TorchRL's actor: the same softmax over the same scores, same weights."""

    def scores(rows: torch.Tensor) -> torch.Tensor:
        return rows @ policy.weights

    return ProbabilisticTensorDictSequential(
        TensorDictModule(scores, in_keys=["features"], out_keys=["logits"]),
        ProbabilisticTensorDictModule(
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        ),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("baseline", BASELINES)
def test_the_surrogate_loss_and_its_gradient_are_torchrl_s_reinforce_loss(
    baseline: float,
) -> None:
    landscape = _landscape()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    episodes = _episodes(landscape, policy, seed=0, count=12)
    ours = surrogate_loss(landscape, policy, episodes, baseline)
    (our_gradient,) = torch.autograd.grad(ours, policy.weights, retain_graph=True)

    features, taken, advantages = _decisions(landscape, episodes, baseline)
    # The mean-over-decisions to mean-over-episodes factor is exercised only
    # if the two counts differ, which is what makes the mapping a claim.
    assert len(taken) > len(episodes) > 1
    zeros = torch.zeros(len(taken), 1, dtype=torch.float64)
    critic = TensorDictModule(
        lambda rows: rows.new_zeros((rows.shape[0], 1)),
        in_keys=["features"],
        out_keys=["state_value"],
    )
    loss = ReinforceLoss(_actor(policy), critic_network=critic, functional=False)
    batch = TensorDict(
        {
            "features": features,
            "action": taken,
            "advantage": advantages[:, None],
            "value_target": zeros,
        },
        [len(taken)],
    )
    with set_interaction_type(InteractionType.DETERMINISTIC):
        theirs = loss(batch)["loss_actor"] * len(taken) / len(episodes)
    # The actor resamples; the greedy episodes make that draw ours, and this
    # is the assertion that says so rather than leaving it to the numbers.
    assert torch.equal(batch["action"], taken)
    (their_gradient,) = torch.autograd.grad(theirs, policy.weights)

    torch.testing.assert_close(theirs, ours, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(their_gradient, our_gradient, rtol=0.0, atol=1e-10)
