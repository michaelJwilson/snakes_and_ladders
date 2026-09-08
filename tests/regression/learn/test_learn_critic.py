"""The critic and the actor-critic against enumeration (issue #313).

``exact_expected_return`` is ``V^pi`` exactly on the Potts chain, so a fitted
critic is scored against it and not against its own loss; the action values
must satisfy Bellman's equation against it; the optimal value must dominate
it for every policy; and the score-function estimator with the exact critic
as baseline must have the exact policy gradient as its expectation, which
is checked by sampling to a stated Monte Carlo tolerance as #135 checked
REINFORCE.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.learn.actor_critic import (
    actor_critic,
    advantage_surrogate_loss,
    critic_advantages,
)
from snakes_and_ladders.learn.critic import (
    Critic,
    fit_critic,
    monte_carlo_targets,
    n_state_features,
    state_features,
    temporal_difference_targets,
)
from snakes_and_ladders.learn.exact import (
    exact_action_values,
    exact_expected_return,
    exact_optimal_value,
    exact_policy_gradient,
)
from snakes_and_ladders.learn.policy import LinearPolicy, MLPPolicy
from snakes_and_ladders.learn.potts import (
    PottsLandscape,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.rollout import rollout

FIELD = np.array([0.4, -0.1, -0.3])
HORIZON = 3


def _landscape() -> PottsLandscape:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=4)


def _policy(weights: list[float]) -> LinearPolicy:
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(weights, dtype=torch.float64))
    return policy


def _states(landscape: PottsLandscape) -> list[tuple[int, ...]]:
    return list(enumerate_configurations(landscape.n_states, landscape.chain_length))


@pytest.mark.mathematical
def test_action_values_satisfy_bellmans_equation() -> None:
    # V^pi(s) = sum_a pi(a | s) Q^pi(s, a), the two sides computed by different
    # recursions; and the optimal value dominates every policy's value.
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    for state in _states(landscape)[::7]:
        value = float(exact_expected_return(landscape, policy, state, HORIZON))
        q = exact_action_values(landscape, policy, state, HORIZON)
        if landscape.is_terminal(state):
            # A local optimum ends the episode: nothing is expected from it.
            assert q.shape == (0,)
            assert value == 0.0
            assert exact_optimal_value(landscape, state, HORIZON) == 0.0
            continue
        probabilities = torch.exp(
            policy.log_probabilities(
                landscape.features(state, landscape.actions(state))
            )
        )
        assert float((probabilities * q).sum()) == pytest.approx(value, abs=1e-12)
        assert exact_optimal_value(landscape, state, HORIZON) >= value - 1e-12
    assert exact_action_values(landscape, policy, _states(landscape)[0], 0).shape == (
        0,
    )
    with pytest.raises(ValueError, match="horizon must be"):
        exact_optimal_value(landscape, _states(landscape)[0], -1)


@pytest.mark.oracle
def test_the_optimal_value_over_a_long_horizon_reaches_the_enumerated_optimum() -> None:
    # With enough decisions the best return from any start is the gap to the
    # enumerated minimum energy, since single flips connect every pair of
    # configurations.
    landscape = _landscape()
    best = optimum(landscape)[1]
    checked = 0
    for state in _states(landscape)[::11]:
        if landscape.is_terminal(state):
            continue
        checked += 1
        # The reward is the rise in the landscape's score, so the best return
        # from a start is the distance up to the enumerated optimum.
        assert exact_optimal_value(landscape, state, 8) == pytest.approx(
            best - landscape.energy(state), abs=1e-9
        )
    assert checked >= 5


@pytest.mark.oracle
@pytest.mark.parametrize("hidden", [None, 16], ids=["linear", "mlp"])
def test_a_fitted_critic_explains_the_enumerated_state_values(
    hidden: int | None,
) -> None:
    # Fitted to V^pi on every configuration: the linear critic explains 0.87
    # of the variance (measured), the MLP 0.996; pinned at the margin below.
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    states = _states(landscape)
    features = torch.stack([state_features(landscape, s) for s in states])
    targets = torch.tensor(
        [float(exact_expected_return(landscape, policy, s, HORIZON)) for s in states]
    )
    critic = Critic(
        n_state_features(landscape),
        hidden=hidden,
        generator=torch.Generator().manual_seed(0),
    )
    fit = fit_critic(critic, features, targets, steps=500)
    with torch.no_grad():
        predicted = critic(features)
    explained = 1.0 - float(
        ((predicted - targets) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    )
    assert explained > (0.8 if hidden is None else 0.98), explained
    assert fit.losses[-1] < fit.losses[0]
    assert features.shape[1] == n_state_features(landscape)


@pytest.mark.mathematical
@pytest.mark.release
def test_the_estimator_with_the_exact_critic_is_unbiased_for_the_exact_gradient() -> (
    None
):
    # 4000 episodes: the advantage-weighted score function with V^pi as the
    # baseline against autodiff through the enumerated J, to the Monte Carlo
    # tolerance the sample size supports (5e-3 measured; asserted at 3e-2).
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    start = _states(landscape)[5]
    exact = exact_policy_gradient(landscape, policy, start, HORIZON)
    rng = np.random.default_rng(0)
    episodes = [
        rollout(landscape, policy, rng, HORIZON, start=start) for _ in range(4000)
    ]
    advantages = [
        [
            g - float(exact_expected_return(landscape, policy, s, HORIZON - t))
            for t, (g, s) in enumerate(
                zip(episode.returns_to_go(), episode.states[:-1], strict=True)
            )
        ]
        for episode in episodes
    ]
    loss = advantage_surrogate_loss(landscape, policy, episodes, advantages)
    (gradient,) = torch.autograd.grad(loss, policy.weights)
    assert_allclose(-gradient.numpy(), exact.numpy(), atol=3e-2)


@pytest.mark.structural
def test_targets_and_advantages_line_up_with_the_decisions() -> None:
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    rng = np.random.default_rng(1)
    episodes = [rollout(landscape, policy, rng, HORIZON) for _ in range(5)]
    n_decisions = sum(len(e.actions) for e in episodes)
    features, targets = monte_carlo_targets(landscape, episodes)
    assert features.shape == (n_decisions, n_state_features(landscape))
    assert targets.tolist() == [g for e in episodes for g in e.returns_to_go()]
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    td_features, td_targets = temporal_difference_targets(landscape, episodes, critic)
    assert td_features.shape == features.shape
    # A zero critic makes the TD target the one-step reward.
    assert td_targets.tolist() == pytest.approx(
        [r for e in episodes for r in e.rewards]
    )
    advantages = critic_advantages(landscape, episodes, critic)
    assert [len(a) for a in advantages] == [len(e.actions) for e in episodes]
    assert [x for a in advantages for x in a] == pytest.approx(targets.tolist())
    with pytest.raises(ValueError, match="at least one episode"):
        advantage_surrogate_loss(landscape, policy, [], [])


@pytest.mark.simulated_truth
def test_actor_critic_reaches_the_optimum_at_least_as_often_as_greedy() -> None:
    # 60 iterations of 32 episodes, the budget #135 trained REINFORCE on:
    # the actor-critic reaches the enumerated optimum from 88.9% of the 81
    # starts against greedy's 80.2% (measured; asserted at greedy's rate).
    landscape = _landscape()
    policy = LinearPolicy(2)
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    training = actor_critic(
        landscape,
        policy,
        critic,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=6,
    )
    assert training.episodes == 1920
    best = optimum(landscape)[1]
    rng = np.random.default_rng(1)
    reached = np.mean(
        [
            abs(
                landscape.energy(rollout(landscape, policy, rng, 6, start=s).states[-1])
                - best
            )
            < 1e-9
            for s in _states(landscape)
        ]
    )
    assert reached >= 65 / 81, reached


@pytest.mark.structural
def test_mlp_policy_is_a_softmax_over_the_available_actions() -> None:
    landscape = _landscape()
    policy = MLPPolicy(2, hidden=8, generator=torch.Generator().manual_seed(0))
    state = _states(landscape)[3]
    features = landscape.features(state, landscape.actions(state))
    log_probabilities = policy.log_probabilities(features)
    assert float(torch.exp(log_probabilities).sum()) == pytest.approx(1.0, abs=1e-12)
    assert len(policy.parameters()) == 5
    assert policy.greedy(features) == int(torch.argmax(log_probabilities))
    with pytest.raises(ValueError, match="expected features of shape"):
        policy.log_probabilities(torch.zeros((3, 5), dtype=torch.float64))
    with pytest.raises(ValueError, match="must be >= 1"):
        MLPPolicy(0, hidden=8, generator=torch.Generator().manual_seed(0))
