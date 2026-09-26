"""REINFORCE against an enumerated oracle, and against the baseline it must beat.

A rising sampled return is not evidence (root ``CLAUDE.md``). The gradient is
checked by autodiff against central differences of the enumerated ``J`` and
the sampled estimator against that; the baseline's unbiasedness as an exact
identity; learning against the enumerated ``J``; quality against exhaustive
enumeration and greedy at a matched decision budget.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.learn.exact import (
    exact_expected_return,
    exact_policy_gradient,
    finite_difference_gradient,
)
from sal.learn.policy import LinearPolicy
from sal.learn.potts import (
    enumerate_configurations,
    optimum,
)
from sal.learn.reinforce import reinforce, surrogate_loss
from sal.learn.rollout import (
    greedy_rollout,
    log_probabilities_of,
    rollout,
)

from tests.regression.learn.conftest import potts_environment

CHAIN_LENGTH = 4
N_STATES = 3

# The enumerated oracle costs |A| ** horizon trajectories, and |A| is 8 here,
# so the horizon it runs at is not the horizon training runs at. Three is 512
# trajectories per start; six, which is what an episode gets, would be 262144.
EXACT_HORIZON = 3
EPISODE_HORIZON = 6

# Relative, on the gradient's norm rather than entrywise: an entry that is
# near zero makes an entrywise relative comparison meaningless, which is the
# same reason `tests/_objective_checks.py` compares against the norm.
_GRADIENT_TOLERANCE = 1e-8
# Monte Carlo, so this is a sampling tolerance and not a correctness one.
# Measured over 12 independent seeds at 4000 episodes: median 1.9% relative,
# worst 3.4%. 10% at 6000 leaves room for an unlucky seed without leaving
# room for a wrong estimator, which would be off by a factor, not a percent.
_ESTIMATOR_TOLERANCE = 0.10


def _policy(weights: list[float]) -> LinearPolicy:
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(weights, dtype=torch.float64))
    return policy


def _relative_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.norm(actual - expected) / torch.linalg.norm(expected))


# --- the gradient ---------------------------------------------------------


@pytest.mark.analytic
@pytest.mark.oracle
def test_the_enumerated_gradient_matches_finite_differences() -> None:
    # Autodiff against numerical differentiation of the same closed form, which
    # rules out an error in the enumeration's use of autograd and says nothing
    # about the sampled estimator. Realized: 1.5e-11 relative.
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)
    exact = exact_policy_gradient(environment, policy, start, EXACT_HORIZON)
    numerical = finite_difference_gradient(environment, policy, start, EXACT_HORIZON)
    assert _relative_difference(numerical, exact) < 1e-6


@pytest.mark.oracle
@pytest.mark.analytic
def test_the_sampled_estimator_is_unbiased_for_the_enumerated_gradient() -> None:
    # The claim REINFORCE rests on, checked rather than cited. A score-function
    # estimator with a sign error or a missing return-to-go would be wrong by
    # a factor, not by a sampling error. Realized: 9.9e-03 relative.
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)
    exact = exact_policy_gradient(environment, policy, start, EXACT_HORIZON)

    rng = np.random.default_rng(7)
    episodes = [
        rollout(environment, policy, rng, max_steps=EXACT_HORIZON, start=start)
        for _ in range(6000)
    ]
    policy.weights.grad = None
    (-surrogate_loss(environment, policy, episodes, 0.0)).backward()  # type: ignore[no-untyped-call]
    assert policy.weights.grad is not None
    assert _relative_difference(policy.weights.grad, exact) < _ESTIMATOR_TOLERANCE


@pytest.mark.analytic
def test_the_score_function_has_zero_expectation() -> None:
    # Why subtracting a constant baseline leaves the estimator unbiased: it
    # multiplies this, which is exactly zero because the probabilities sum to
    # one whatever the weights are. An identity, not a tolerance.
    environment, policy = potts_environment(), _policy([0.9, -0.4])
    state = (1, 2, 0, 1)
    actions = environment.actions(state)
    log_probabilities = policy.log_probabilities(environment.features(state, actions))
    total = torch.zeros(2, dtype=torch.float64)
    for index in range(len(actions)):
        (score,) = torch.autograd.grad(
            log_probabilities[index], policy.weights, retain_graph=True
        )
        total = total + float(torch.exp(log_probabilities[index]).detach()) * score
    assert_allclose(total.numpy(), [0.0, 0.0], atol=1e-14)


@pytest.mark.analytic
def test_the_baseline_reduces_the_estimator_variance() -> None:
    # A baseline is for variance (sec:policy-gradient): realized ratio 0.90,
    # since returns here are of similar size. Not asserted tightly: a threshold
    # at 0.90 would be tuned to this environment.
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)

    def per_episode_variance(baseline: float) -> float:
        rng = np.random.default_rng(5)
        gradients: list[np.ndarray] = []
        for _ in range(400):
            episode = rollout(
                environment, policy, rng, max_steps=EXACT_HORIZON, start=start
            )
            policy.weights.grad = None
            (-surrogate_loss(environment, policy, [episode], baseline)).backward()  # type: ignore[no-untyped-call]
            assert policy.weights.grad is not None
            gradients.append(policy.weights.grad.numpy().copy())
        return float(np.asarray(gradients).var(axis=0).sum())

    rng = np.random.default_rng(9)
    mean_return = float(
        np.mean(
            [
                rollout(
                    environment, policy, rng, max_steps=EXACT_HORIZON, start=start
                ).total_reward
                for _ in range(2000)
            ]
        )
    )
    assert per_episode_variance(mean_return) < per_episode_variance(0.0)


# --- learning -------------------------------------------------------------


@pytest.mark.oracle
def test_training_raises_the_enumerated_expected_return() -> None:
    # Against the enumerated J, not the sampled mean the training loop
    # reports: that curve is a Monte Carlo estimate under a moving policy and
    # can rise while the estimator is wrong. Realized on 9 probe starts:
    # -0.6245 before, 1.6550 after.
    environment = potts_environment()
    starts = list(enumerate_configurations(N_STATES, CHAIN_LENGTH))[::9]
    policy = LinearPolicy(2)

    def enumerated_return() -> float:
        return float(
            np.mean(
                [
                    float(
                        exact_expected_return(
                            environment, policy, start, EXACT_HORIZON
                        ).detach()
                    )
                    for start in starts
                ]
            )
        )

    before = enumerated_return()
    training = reinforce(
        environment,
        policy,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=EPISODE_HORIZON,
    )
    assert enumerated_return() > before
    assert training.episodes == 60 * 32
    assert len(training.mean_returns) == 60


@pytest.mark.end2end
def test_the_learned_policy_is_at_least_as_good_as_hill_climbing() -> None:
    # Milestone 8, enumerable. Over 8 seeds: 83.6%-87.1% of 81 starts reach the
    # optimum (greedy 80.2%), mean final energy 3.519-3.584 (greedy 3.396,
    # optimum 3.850), 8 of 8 beat greedy. Asserted: "at least as good" (#128).
    environment = potts_environment()
    starts = list(enumerate_configurations(N_STATES, CHAIN_LENGTH))
    policy = LinearPolicy(2)
    reinforce(
        environment,
        policy,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=EPISODE_HORIZON,
    )

    def final_energy(states: tuple[tuple[int, ...], ...]) -> float:
        return environment.log_weight(states[-1])

    greedy = float(
        np.mean(
            [
                final_energy(
                    greedy_rollout(
                        environment, start=start, max_steps=EPISODE_HORIZON
                    ).states
                )
                for start in starts
            ]
        )
    )
    rng = np.random.default_rng(3)
    learned = float(
        np.mean(
            [
                final_energy(
                    rollout(
                        environment, policy, rng, max_steps=EPISODE_HORIZON, start=start
                    ).states
                )
                for start in starts
                for _ in range(16)
            ]
        )
    )
    assert learned >= greedy
    assert learned <= optimum(environment)[1]


@pytest.mark.smoke
def test_training_is_reproducible_from_its_seed() -> None:
    environment = potts_environment()
    runs = [
        reinforce(
            environment,
            LinearPolicy(2),
            np.random.default_rng(4),
            iterations=5,
            batch=8,
            max_steps=EPISODE_HORIZON,
        )
        for _ in range(2)
    ]
    assert_allclose(runs[0].weights, runs[1].weights, atol=0.0, rtol=0.0)
    assert runs[0].mean_returns == runs[1].mean_returns


@pytest.mark.smoke
def test_the_gradient_check_would_catch_a_biased_estimator() -> None:
    # Guards the guard with a myopic estimator (own reward, no credit
    # assignment): 7.1e-01 relative against 9.9e-03 correct. Total-return
    # weighting is unbiased too, 1.4e-02, inside tolerance: return-to-go buys
    # variance, not correctness.
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)
    exact = exact_policy_gradient(environment, policy, start, EXACT_HORIZON)

    rng = np.random.default_rng(7)
    episodes = [
        rollout(environment, policy, rng, max_steps=EXACT_HORIZON, start=start)
        for _ in range(4000)
    ]
    myopic = torch.zeros((), dtype=torch.float64)
    for episode in episodes:
        for step, action in enumerate(episode.actions):
            state = episode.states[step]
            available = environment.actions(state)
            log_probabilities = policy.log_probabilities(
                environment.features(state, available)
            )
            myopic = (
                myopic
                + log_probabilities[available.index(action)] * episode.rewards[step]
            )
    policy.weights.grad = None
    (myopic / len(episodes)).backward()  # type: ignore[no-untyped-call]
    assert policy.weights.grad is not None
    assert _relative_difference(policy.weights.grad, exact) > _ESTIMATOR_TOLERANCE


@pytest.mark.smoke
@pytest.mark.patch
def test_the_shared_decision_loop_reproduces_the_loop_it_replaced() -> None:
    # The one score-function loop five estimators walked (#862), against the
    # loop written out; bitwise, taken entry and whole vector.
    environment, policy = potts_environment(), _policy([0.3, -0.6])
    rng = np.random.default_rng(4)
    episodes = [
        rollout(environment, policy, rng, max_steps=EPISODE_HORIZON) for _ in range(8)
    ]

    replayed = log_probabilities_of(policy, environment, episodes)
    assert [len(decisions) for decisions in replayed] == [
        len(episode.actions) for episode in episodes
    ]
    for episode, decisions in zip(episodes, replayed, strict=True):
        for step, action in enumerate(episode.actions):
            state = episode.states[step]
            available = environment.actions(state)
            log_probabilities = policy.log_probabilities(
                environment.features(state, available)
            )
            assert decisions[step].chosen == available.index(action)
            assert_allclose(
                decisions[step].log_probabilities.detach().numpy(),
                log_probabilities.detach().numpy(),
                rtol=0.0,
                atol=0.0,
            )
            assert float(decisions[step].taken.detach()) == float(
                log_probabilities[available.index(action)].detach()
            )


# --- validation -----------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("iterations", "batch", "message"),
    [(0, 8, "iterations must be >= 1"), (5, 0, "batch must be >= 1")],
)
def test_a_degenerate_budget_is_rejected(
    iterations: int, batch: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reinforce(
            potts_environment(),
            LinearPolicy(2),
            np.random.default_rng(0),
            iterations=iterations,
            batch=batch,
            max_steps=3,
        )


@pytest.mark.smoke
def test_an_estimate_needs_at_least_one_episode() -> None:
    with pytest.raises(ValueError, match="at least one episode"):
        surrogate_loss(potts_environment(), LinearPolicy(2), [], 0.0)


@pytest.mark.smoke
def test_a_negative_horizon_is_rejected_by_the_oracle() -> None:
    with pytest.raises(ValueError, match="horizon must be >= 0"):
        exact_expected_return(potts_environment(), LinearPolicy(2), (0, 1, 0, 1), -1)


@pytest.mark.smoke
def test_the_oracle_returns_zero_at_a_zero_horizon() -> None:
    value = exact_expected_return(potts_environment(), LinearPolicy(2), (0, 1, 2, 0), 0)
    assert float(value.detach()) == 0.0
