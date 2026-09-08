"""PUCT search and expert iteration against enumeration on the Potts chain (issue #313).

With the exact optimal value as the leaf evaluator and one decision of
depth, every backed-up value is ``Q*`` exactly, so the root's most visited
action must be optimal; deeper, the visit distribution must improve on the
prior it started from, which is the policy-improvement property expert
iteration rests on. The planner's budget is counted in successor
evaluations, and its comparison with greedy states both the rate and the
evaluations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.critic import Critic, n_state_features
from snakes_and_ladders.learn.exact import exact_action_values, exact_optimal_value
from snakes_and_ladders.learn.planning import (
    SearchResult,
    critic_leaf_value,
    expert_iteration,
    plan_episode,
    puct_search,
)
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import (
    PottsLandscape,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.rollout import greedy_rollout

FIELD = np.array([0.4, -0.1, -0.3])


def _landscape() -> PottsLandscape:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=4)


def _states(landscape: PottsLandscape) -> list[tuple[int, ...]]:
    return list(enumerate_configurations(landscape.n_states, landscape.chain_length))


def _policy() -> LinearPolicy:
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    return policy


@pytest.mark.oracle
def test_one_step_search_with_the_exact_leaf_value_picks_an_optimal_action() -> None:
    # Depth one: every backed-up value is r + V*(s'), so the most visited move
    # is an argmax of Q*, ties allowed.
    landscape, policy = _landscape(), _policy()
    leaf = lambda s, remaining: exact_optimal_value(landscape, s, remaining)  # noqa: E731
    checked = 0
    for state in _states(landscape)[::4]:
        if landscape.is_terminal(state):
            continue
        checked += 1
        result = puct_search(
            landscape, policy, state, horizon=1, n_simulations=200, leaf_value=leaf
        )
        q_star = [
            reward + exact_optimal_value(landscape, successor, 0)
            for successor, reward in (landscape.step(state, a) for a in result.actions)
        ]
        assert q_star[result.best()] == pytest.approx(max(q_star), abs=1e-9)
        assert 1 <= result.evaluations <= len(result.actions)
    assert checked >= 10


@pytest.mark.mathematical
def test_the_visit_distribution_improves_on_the_prior_at_depth_three() -> None:
    # Policy improvement: the one-step mixture of the exact action values under
    # the visit distribution is no worse than under the prior on 13 of 14
    # non-terminal states sampled (measured at 400 simulations); pinned at 12.
    landscape, policy = _landscape(), _policy()
    leaf = lambda s, remaining: exact_optimal_value(landscape, s, remaining)  # noqa: E731
    improved, checked = 0, 0
    for state in _states(landscape)[::5]:
        if landscape.is_terminal(state):
            continue
        checked += 1
        result = puct_search(
            landscape, policy, state, horizon=3, n_simulations=400, leaf_value=leaf
        )
        q_pi = exact_action_values(landscape, policy, state, 3).detach().numpy()
        with torch.no_grad():
            prior = torch.exp(
                policy.log_probabilities(
                    landscape.features(state, landscape.actions(state))
                )
            ).numpy()
        improved += int(float(result.distribution @ q_pi) >= float(prior @ q_pi) - 1e-9)
    assert checked == 14
    assert improved >= 12, improved


@pytest.mark.simulated_truth
def test_expert_iteration_makes_the_planner_reach_the_optimum_at_a_fraction_of_greedys_evaluations() -> (
    None
):
    # Measured: an untrained prior with a fresh critic reaches the enumerated
    # optimum from 76.5% of the 81 starts at 57 evaluations per episode against
    # greedy's 80.2% at 48; after 10 iterations of 8 planned episodes the
    # planner reaches it from 92.6% at 8.3 evaluations per episode, and at
    # six simulations (6.3 evaluations) matches greedy's 80.2%.
    landscape = _landscape()
    starts = _states(landscape)
    best = optimum(landscape)[1]
    greedy = float(
        np.mean(
            [
                abs(landscape.energy(greedy_rollout(landscape, s, 6).states[-1]) - best)
                < 1e-9
                for s in starts
            ]
        )
    )
    policy = LinearPolicy(2)
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    training = expert_iteration(
        landscape,
        policy,
        critic,
        np.random.default_rng(0),
        iterations=10,
        batch=8,
        horizon=6,
        n_simulations=20,
    )
    assert training.evaluations > 0
    assert len(training.mean_returns) == 10
    rng = np.random.default_rng(1)
    planned = [
        plan_episode(
            landscape,
            policy,
            s,
            rng,
            horizon=6,
            n_simulations=30,
            leaf_value=critic_leaf_value(landscape, critic),
            sample=False,
        )
        for s in starts
    ]
    reached = float(
        np.mean([abs(landscape.energy(e.states[-1]) - best) < 1e-9 for e in planned])
    )
    evaluations = float(np.mean([e.evaluations for e in planned]))
    assert reached >= greedy, (reached, greedy)
    assert reached >= 0.85
    assert evaluations < 20.0, evaluations
    assert all(len(e.distributions) == len(e.actions) for e in planned)


@pytest.mark.edge_case
def test_the_search_refuses_a_non_positive_budget_and_handles_a_terminal_root() -> None:
    landscape, policy = _landscape(), _policy()
    leaf = critic_leaf_value(
        landscape,
        Critic(
            n_state_features(landscape),
            hidden=None,
            generator=torch.Generator().manual_seed(0),
        ),
    )
    terminal = next(s for s in _states(landscape) if landscape.is_terminal(s))
    result = puct_search(
        landscape, policy, terminal, horizon=3, n_simulations=5, leaf_value=leaf
    )
    assert result.evaluations == 0
    assert isinstance(result, SearchResult)
    with pytest.raises(ValueError, match="must be >= 1"):
        puct_search(
            landscape, policy, terminal, horizon=0, n_simulations=5, leaf_value=leaf
        )
    with pytest.raises(ValueError, match="must be >= 1"):
        expert_iteration(
            landscape,
            policy,
            Critic(3, hidden=None, generator=torch.Generator().manual_seed(0)),
            np.random.default_rng(0),
            iterations=0,
            batch=1,
            horizon=1,
            n_simulations=1,
        )
