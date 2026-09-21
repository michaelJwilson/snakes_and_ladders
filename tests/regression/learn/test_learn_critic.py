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
    PottsEnvironment,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout

from tests.regression.learn.conftest import potts_environment

HORIZON = 3


def _policy(weights: list[float]) -> LinearPolicy:
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(weights, dtype=torch.float64))
    return policy


def _states(environment: PottsEnvironment) -> list[tuple[int, ...]]:
    return list(
        enumerate_configurations(environment.n_states, environment.chain_length)
    )


@pytest.mark.analytic
def test_action_values_satisfy_bellmans_equation() -> None:
    # V^pi(s) = sum_a pi(a | s) Q^pi(s, a), the two sides computed by different
    # recursions; and the optimal value dominates every policy's value.
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    for state in _states(environment)[::7]:
        value = float(exact_expected_return(environment, policy, state, HORIZON))
        q = exact_action_values(environment, policy, state, HORIZON)
        if environment.is_terminal(state):
            # A local optimum ends the episode: nothing is expected from it.
            assert q.shape == (0,)
            assert value == 0.0
            assert exact_optimal_value(environment, state, HORIZON) == 0.0
            continue
        probabilities = torch.exp(
            policy.log_probabilities(
                environment.features(state, environment.actions(state))
            )
        )
        assert float((probabilities * q).sum()) == pytest.approx(value, abs=1e-12)
        assert exact_optimal_value(environment, state, HORIZON) >= value - 1e-12
    assert exact_action_values(
        environment, policy, _states(environment)[0], 0
    ).shape == (0,)
    with pytest.raises(ValueError, match="horizon must be"):
        exact_optimal_value(environment, _states(environment)[0], -1)


@pytest.mark.oracle
def test_the_optimal_value_over_a_long_horizon_reaches_the_enumerated_optimum() -> None:
    # With enough decisions the best return from any start is the gap to the
    # enumerated minimum energy, since single flips connect every pair of
    # configurations.
    environment = potts_environment()
    best = optimum(environment)[1]
    checked = 0
    for state in _states(environment)[::11]:
        if environment.is_terminal(state):
            continue
        checked += 1
        # The reward is the rise in the environment's score, so the best return
        # from a start is the distance up to the enumerated optimum.
        assert exact_optimal_value(environment, state, 8) == pytest.approx(
            best - environment.energy(state), abs=1e-9
        )
    assert checked >= 5


@pytest.mark.oracle
@pytest.mark.parametrize("hidden", [None, 16], ids=["linear", "mlp"])
def test_a_fitted_critic_explains_the_enumerated_state_values(
    hidden: int | None,
) -> None:
    # Fitted to V^pi on every configuration: the linear critic explains 0.87
    # of the variance (measured), the MLP 0.996; pinned at the margin below.
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    states = _states(environment)
    features = torch.stack([state_features(environment, s) for s in states])
    targets = torch.tensor(
        [float(exact_expected_return(environment, policy, s, HORIZON)) for s in states]
    )
    critic = Critic(
        n_state_features(environment),
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
    assert features.shape[1] == n_state_features(environment)


@pytest.mark.analytic
@pytest.mark.release
def test_the_estimator_with_the_exact_critic_is_unbiased_for_the_exact_gradient() -> (
    None
):
    # 4000 episodes: the advantage-weighted score function with V^pi as the
    # baseline against autodiff through the enumerated J, to the Monte Carlo
    # tolerance the sample size supports (5e-3 measured; asserted at 3e-2).
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    start = _states(environment)[5]
    exact = exact_policy_gradient(environment, policy, start, HORIZON)
    rng = np.random.default_rng(0)
    episodes = [
        rollout(environment, policy, rng, HORIZON, start=start) for _ in range(4000)
    ]
    advantages = [
        [
            g - float(exact_expected_return(environment, policy, s, HORIZON - t))
            for t, (g, s) in enumerate(
                zip(episode.returns_to_go(), episode.states[:-1], strict=True)
            )
        ]
        for episode in episodes
    ]
    loss = advantage_surrogate_loss(environment, policy, episodes, advantages)
    (gradient,) = torch.autograd.grad(loss, policy.weights)
    assert_allclose(-gradient.numpy(), exact.numpy(), atol=3e-2)


@pytest.mark.smoke
def test_targets_and_advantages_line_up_with_the_decisions() -> None:
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    rng = np.random.default_rng(1)
    episodes = [rollout(environment, policy, rng, HORIZON) for _ in range(5)]
    n_decisions = sum(len(e.actions) for e in episodes)
    features, targets = monte_carlo_targets(environment, episodes)
    assert features.shape == (n_decisions, n_state_features(environment))
    assert targets.tolist() == [g for e in episodes for g in e.returns_to_go()]
    critic = Critic(
        n_state_features(environment),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    td_features, td_targets = temporal_difference_targets(environment, episodes, critic)
    assert td_features.shape == features.shape
    # A zero critic makes the TD target the one-step reward.
    assert td_targets.tolist() == pytest.approx(
        [r for e in episodes for r in e.rewards]
    )
    advantages = critic_advantages(environment, episodes, critic)
    assert [len(a) for a in advantages] == [len(e.actions) for e in episodes]
    assert [x for a in advantages for x in a] == pytest.approx(targets.tolist())
    with pytest.raises(ValueError, match="at least one episode"):
        advantage_surrogate_loss(environment, policy, [], [])


@pytest.mark.end2end
def test_actor_critic_reaches_the_optimum_at_least_as_often_as_greedy() -> None:
    # 60 iterations of 32 episodes, the budget #135 trained REINFORCE on:
    # the actor-critic reaches the enumerated optimum from 88.9% of the 81
    # starts against greedy's 80.2% (measured; asserted at greedy's rate).
    environment = potts_environment()
    policy = LinearPolicy(2)
    critic = Critic(
        n_state_features(environment),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    training = actor_critic(
        environment,
        policy,
        critic,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=6,
    )
    assert training.episodes == 1920
    best = optimum(environment)[1]
    rng = np.random.default_rng(1)
    reached = np.mean(
        [
            abs(
                environment.energy(
                    rollout(environment, policy, rng, 6, start=s).states[-1]
                )
                - best
            )
            < 1e-9
            for s in _states(environment)
        ]
    )
    assert reached >= 65 / 81, reached


@pytest.mark.smoke
def test_mlp_policy_is_a_softmax_over_the_available_actions() -> None:
    environment = potts_environment()
    policy = MLPPolicy(2, hidden=8, generator=torch.Generator().manual_seed(0))
    state = _states(environment)[3]
    features = environment.features(state, environment.actions(state))
    log_probabilities = policy.log_probabilities(features)
    assert float(torch.exp(log_probabilities).sum()) == pytest.approx(1.0, abs=1e-12)
    assert len(policy.parameters()) == 5
    assert policy.greedy(features) == int(torch.argmax(log_probabilities))
    with pytest.raises(ValueError, match="expected features of shape"):
        policy.log_probabilities(torch.zeros((3, 5), dtype=torch.float64))
    with pytest.raises(ValueError, match="must be >= 1"):
        MLPPolicy(0, hidden=8, generator=torch.Generator().manual_seed(0))


# --- The two oracles issue #729 added -------------------------------------


def _trajectory_sum(
    environment: PottsEnvironment,
    policy: LinearPolicy,
    state: tuple[int, ...],
    horizon: int,
) -> float:
    """``sum over trajectories of P(trajectory) * G(trajectory)``, expanded forwards.

    Brute force, and written here rather than imported: `exact.py` computes
    the same number by a *backward* recursion over ``(state, remaining)``,
    carrying one value per node. This carries a probability and a return per
    trajectory and sums the products at the end, so the two share no algebra
    and agreement between them is evidence rather than a tautology. Costs
    ``|A| ** horizon`` trajectories, 512 at the settings below.
    """
    frontier = [(state, 1.0, 0.0)]
    total = 0.0
    for _ in range(horizon):
        expanded = []
        for current, probability, ret in frontier:
            if environment.is_terminal(current):
                total += probability * ret
                continue
            available = environment.actions(current)
            draw = torch.exp(
                policy.log_probabilities(environment.features(current, available))
            ).detach()
            for index, action in enumerate(available):
                successor, reward = environment.step(current, action)
                expanded.append(
                    (successor, probability * float(draw[index]), ret + reward)
                )
        frontier = expanded
    return total + sum(probability * ret for _, probability, ret in frontier)


@pytest.mark.oracle
def test_the_action_values_are_a_brute_force_over_trajectories() -> None:
    """``Q^pi(s, a)`` equals the reward plus the enumerated return of the successor.

    `exact_action_values` is what a planner's leaf and a critic's target are
    read against, and until now it was checked only against
    `exact_expected_return`, which is the same recursion one level up. Here
    both are compared with :func:`_trajectory_sum`, a forward expansion over
    all 512 trajectories of the 4-site chain at horizon 3.

    Bitwise is not the target: the two routes sum the same terms in different
    orders. Realized over seven states, the largest disagreement is 2.2e-16
    absolute on values of order 1, against a declared 1e-12.
    """
    environment, policy = potts_environment(), _policy([0.3, -0.6])

    for state in _states(environment)[::13]:
        assert float(
            exact_expected_return(environment, policy, state, HORIZON).detach()
        ) == pytest.approx(
            _trajectory_sum(environment, policy, state, HORIZON), abs=1e-12
        )
        values = exact_action_values(environment, policy, state, HORIZON)
        if environment.is_terminal(state):
            assert values.shape == (0,)
            continue
        for index, action in enumerate(environment.actions(state)):
            successor, reward = environment.step(state, action)
            assert float(values[index].detach()) == pytest.approx(
                reward + _trajectory_sum(environment, policy, successor, HORIZON - 1),
                abs=1e-12,
            )


@pytest.mark.oracle
def test_the_bootstrapped_targets_telescope_to_the_closed_form_return() -> None:
    """At ``gamma = 1`` the TD targets sum to the return the energies state.

    The closed form is `learn/CLAUDE.md`'s own: an undiscounted reward
    telescopes, so the return from any step is `energy(s_T) - energy(s_t)`,
    a difference of two *levels* that no reward in the episode appears in.
    Two readings of `temporal_difference_targets` against it, for an episode
    that ends at a local maximum:

    * the last target is ``r_{T-1}`` exactly, the successor being terminal and
      contributing no bootstrap --- equality, not a tolerance, since the
      branch either fired or it did not;
    * summing ``r_t + V(s_{t+1})`` and subtracting ``V(s_t)`` over the episode
      leaves ``V(s_T) - V(s_0)``, so the targets less the critic's own values
      at the features, plus ``V(s_0)``, is the return. Realized over the six
      starts below that take an action: 4.4e-16 absolute at worst against a
      declared 1e-12, on returns of 2.65 to 3.65.

    `monte_carlo_targets` is read against the same closed form, where the
    agreement is term by term rather than in the sum.
    """
    environment = potts_environment()
    critic = Critic(
        n_state_features(environment),
        hidden=None,
        generator=torch.Generator().manual_seed(729),
    )

    for start in _states(environment)[::10]:
        episode = greedy_rollout(environment, start, 12)
        if not episode.actions:
            continue
        assert episode.terminated, "the closed form below needs a finished episode"
        closed = environment.energy(episode.states[-1]) - environment.energy(start)
        assert episode.total_reward == pytest.approx(closed, abs=1e-12)

        features, targets = temporal_difference_targets(environment, [episode], critic)
        assert float(targets[-1]) == episode.rewards[-1]
        with torch.no_grad():
            values = critic(features)
        assert float(targets.sum() - values.sum() + values[0]) == pytest.approx(
            closed, abs=1e-12
        )

        _, returns = monte_carlo_targets(environment, [episode])
        for step, state in enumerate(episode.states[:-1]):
            assert float(returns[step]) == pytest.approx(
                environment.energy(episode.states[-1]) - environment.energy(state),
                abs=1e-12,
            )
