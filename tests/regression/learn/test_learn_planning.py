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
from snakes_and_ladders.learn.environment import Episode
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
    PottsEnvironment,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.rollout import greedy_rollout

from tests.regression.learn.conftest import potts_environment


def _states(environment: PottsEnvironment) -> list[tuple[int, ...]]:
    return list(
        enumerate_configurations(environment.n_states, environment.chain_length)
    )


def _policy() -> LinearPolicy:
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    return policy


@pytest.mark.oracle
def test_one_step_search_with_the_exact_leaf_value_picks_an_optimal_action() -> None:
    # Depth one: every backed-up value is r + V*(s'), so the most visited move
    # is an argmax of Q*, ties allowed.
    environment, policy = potts_environment(), _policy()
    leaf = lambda s, remaining: exact_optimal_value(environment, s, remaining)  # noqa: E731
    checked = 0
    for state in _states(environment)[::4]:
        if environment.is_terminal(state):
            continue
        checked += 1
        result = puct_search(
            environment, policy, state, horizon=1, n_simulations=200, leaf_value=leaf
        )
        q_star = [
            reward + exact_optimal_value(environment, successor, 0)
            for successor, reward in (
                environment.step(state, a) for a in result.actions
            )
        ]
        assert q_star[result.best()] == pytest.approx(max(q_star), abs=1e-9)
        assert 1 <= result.evaluations <= len(result.actions)
    assert checked >= 10


@pytest.mark.analytic
def test_the_visit_distribution_improves_on_the_prior_at_depth_three() -> None:
    # Policy improvement: the one-step mixture of the exact action values under
    # the visit distribution is no worse than under the prior on 13 of 14
    # non-terminal states sampled (measured at 400 simulations); pinned at 12.
    environment, policy = potts_environment(), _policy()
    leaf = lambda s, remaining: exact_optimal_value(environment, s, remaining)  # noqa: E731
    improved, checked = 0, 0
    for state in _states(environment)[::5]:
        if environment.is_terminal(state):
            continue
        checked += 1
        result = puct_search(
            environment, policy, state, horizon=3, n_simulations=400, leaf_value=leaf
        )
        q_pi = exact_action_values(environment, policy, state, 3).detach().numpy()
        with torch.no_grad():
            prior = torch.exp(
                policy.log_probabilities(
                    environment.features(state, environment.actions(state))
                )
            ).numpy()
        improved += int(float(result.distribution @ q_pi) >= float(prior @ q_pi) - 1e-9)
    assert checked == 14
    assert improved >= 12, improved


@pytest.mark.end2end
def test_expert_iteration_makes_the_planner_reach_the_optimum_at_a_fraction_of_greedys_evaluations() -> (
    None
):
    # Measured: an untrained prior with a fresh critic reaches the enumerated
    # optimum from 76.5% of the 81 starts at 57 evaluations per episode against
    # greedy's 80.2% at 48; after 10 iterations of 8 planned episodes the
    # planner reaches it from 92.6% at 8.3 evaluations per episode, and at
    # six simulations (6.3 evaluations) matches greedy's 80.2%.
    environment = potts_environment()
    starts = _states(environment)
    best = optimum(environment)[1]
    greedy = float(
        np.mean(
            [
                abs(
                    environment.energy(greedy_rollout(environment, s, 6).states[-1])
                    - best
                )
                < 1e-9
                for s in starts
            ]
        )
    )
    policy = LinearPolicy(2)
    critic = Critic(
        n_state_features(environment),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    training = expert_iteration(
        environment,
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
            environment,
            policy,
            s,
            rng,
            horizon=6,
            n_simulations=30,
            leaf_value=critic_leaf_value(environment, critic),
            sample=False,
        )
        for s in starts
    ]
    reached = float(
        np.mean([abs(environment.energy(e.states[-1]) - best) < 1e-9 for e in planned])
    )
    evaluations = float(np.mean([e.evaluations for e in planned]))
    assert reached >= greedy, (reached, greedy)
    assert reached >= 0.85
    assert evaluations < 20.0, evaluations
    assert all(len(e.distributions) == len(e.actions) for e in planned)


@pytest.mark.smoke
def test_the_search_refuses_a_non_positive_budget_and_handles_a_terminal_root() -> None:
    environment, policy = potts_environment(), _policy()
    leaf = critic_leaf_value(
        environment,
        Critic(
            n_state_features(environment),
            hidden=None,
            generator=torch.Generator().manual_seed(0),
        ),
    )
    terminal = next(s for s in _states(environment) if environment.is_terminal(s))
    result = puct_search(
        environment, policy, terminal, horizon=3, n_simulations=5, leaf_value=leaf
    )
    assert result.evaluations == 0
    assert isinstance(result, SearchResult)
    with pytest.raises(ValueError, match="must be >= 1"):
        puct_search(
            environment, policy, terminal, horizon=0, n_simulations=5, leaf_value=leaf
        )
    with pytest.raises(ValueError, match="must be >= 1"):
        expert_iteration(
            environment,
            policy,
            Critic(3, hidden=None, generator=torch.Generator().manual_seed(0)),
            np.random.default_rng(0),
            iterations=0,
            batch=1,
            horizon=1,
            n_simulations=1,
        )


@pytest.mark.smoke
@pytest.mark.patch
def test_a_planned_episode_is_an_episode_and_its_returns_are_the_recomputed_ones() -> (
    None
):
    # `PlannedEpisode` was a second episode type without `terminated` or
    # `returns_to_go`, so `expert_iteration` recomputed the returns it fits
    # its critic against (issue #862). It is an `Episode` now, and its
    # returns are bitwise the reversed cumulative sum that stood in for
    # them.
    environment = potts_environment()
    critic = Critic(
        n_state_features(environment),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    rng = np.random.default_rng(2)
    planned = [
        plan_episode(
            environment,
            _policy(),
            state,
            rng,
            horizon=5,
            n_simulations=12,
            leaf_value=critic_leaf_value(environment, critic),
        )
        for state in _states(environment)[::9]
    ]

    assert any(episode.actions for episode in planned)
    for episode in planned:
        assert isinstance(episode, Episode)
        assert episode.terminated == environment.is_terminal(episode.states[-1])
        assert len(episode.distributions) == len(episode.actions)
        recomputed = (
            np.cumsum(episode.rewards[::-1])[::-1] if episode.rewards else np.zeros(0)
        )
        assert list(episode.returns_to_go()) == [float(g) for g in recomputed]
        assert episode.total_reward == float(sum(episode.rewards))


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_expert_iteration_curve_is_the_one_a_fixed_seed_produces() -> None:
    # The curve the fold is pinned against (issue #862), seed for seed. A
    # diagnostic and not a result --- `learn/CLAUDE.md` says a sampled
    # return is never one --- so what is asserted is reproduction, to the
    # last bit, of the numbers the same seed produced before the planner's
    # episode became an `Episode`.
    environment = potts_environment()

    def curve() -> tuple[tuple[float, ...], tuple[float, ...], int]:
        policy = LinearPolicy(2)
        policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
        training = expert_iteration(
            environment,
            policy,
            Critic(
                n_state_features(environment),
                hidden=None,
                generator=torch.Generator().manual_seed(7),
            ),
            np.random.default_rng(29),
            iterations=3,
            batch=4,
            horizon=5,
            n_simulations=12,
            critic_steps=10,
        )
        return (
            training.mean_returns,
            training.policy_losses,
            training.evaluations,
        )

    first = curve()
    assert first == curve()
    assert first[0] == (0.8, 1.0875000000000001, 1.25)
    assert first[1] == (
        1.5623207809304243,
        1.4617507081664243,
        1.4698711582623099,
    )
    assert first[2] == 139
