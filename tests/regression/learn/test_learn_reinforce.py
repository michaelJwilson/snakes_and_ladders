"""REINFORCE against an enumerated oracle, and against the baseline it must beat.

Root ``CLAUDE.md`` forbids a test that asserts only that something ran, and
"the return went up" is exactly that: a sampled return under a changing
policy rises for reasons that include a broken estimator. So every claim here
is pinned to something computed a different way.

* The gradient is checked twice -- autodiff against central finite
  differences of the same enumerated ``J``, and the sampled estimator against
  that enumerated gradient.
* The baseline's unbiasedness is checked as the identity it rests on, which
  holds exactly rather than to a tolerance.
* Learning is checked against the **enumerated** ``J``, not the sampled mean.
* Quality is checked against exhaustive enumeration of the landscape and
  against the greedy searcher, at a matched decision budget.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.learn.exact import (
    exact_expected_return,
    exact_policy_gradient,
    finite_difference_gradient,
)
from phylo.learn.policy import LinearPolicy
from phylo.learn.potts import PottsLandscape, enumerate_configurations, optimum
from phylo.learn.reinforce import reinforce, surrogate_loss
from phylo.learn.rollout import greedy_rollout, rollout

FIELD = np.array([0.4, -0.1, -0.3])
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


def _landscape() -> PottsLandscape:
    return PottsLandscape(0.75, FIELD, CHAIN_LENGTH)


def _policy(weights: list[float]) -> LinearPolicy:
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(weights, dtype=torch.float64))
    return policy


def _relative_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.norm(actual - expected) / torch.linalg.norm(expected))


# --- the gradient ---------------------------------------------------------


@pytest.mark.mathematical
@pytest.mark.oracle
def test_the_enumerated_gradient_matches_finite_differences() -> None:
    # Autodiff against numerical differentiation of the same closed form.
    # This rules out an error in the enumeration's use of autograd; it says
    # nothing yet about the sampled estimator, which the next test covers.
    # Realized: 1.5e-11 relative.
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)
    exact = exact_policy_gradient(landscape, policy, start, EXACT_HORIZON)
    numerical = finite_difference_gradient(landscape, policy, start, EXACT_HORIZON)
    assert _relative_difference(numerical, exact) < 1e-6


@pytest.mark.oracle
@pytest.mark.mathematical
def test_the_sampled_estimator_is_unbiased_for_the_enumerated_gradient() -> None:
    # The claim REINFORCE rests on, checked rather than cited. A score-function
    # estimator with a sign error or a missing return-to-go would be wrong by
    # a factor, not by a sampling error. Realized: 9.9e-03 relative.
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)
    exact = exact_policy_gradient(landscape, policy, start, EXACT_HORIZON)

    rng = np.random.default_rng(7)
    episodes = [
        rollout(landscape, policy, rng, EXACT_HORIZON, start=start) for _ in range(6000)
    ]
    policy.weights.grad = None
    (-surrogate_loss(landscape, policy, episodes, 0.0)).backward()  # type: ignore[no-untyped-call]
    assert policy.weights.grad is not None
    assert _relative_difference(policy.weights.grad, exact) < _ESTIMATOR_TOLERANCE


@pytest.mark.mathematical
def test_the_score_function_has_zero_expectation() -> None:
    # Why subtracting a constant baseline leaves the estimator unbiased: it
    # multiplies this, and this is exactly zero because the probabilities sum
    # to one whatever the weights are. Stated as an identity rather than a
    # tolerance, since it holds to rounding.
    landscape, policy = _landscape(), _policy([0.9, -0.4])
    state = (1, 2, 0, 1)
    actions = landscape.actions(state)
    log_probabilities = policy.log_probabilities(landscape.features(state, actions))
    total = torch.zeros(2, dtype=torch.float64)
    for index in range(len(actions)):
        (score,) = torch.autograd.grad(
            log_probabilities[index], policy.weights, retain_graph=True
        )
        total = total + float(torch.exp(log_probabilities[index]).detach()) * score
    assert_allclose(total.numpy(), [0.0, 0.0], atol=1e-14)


@pytest.mark.mathematical
def test_the_baseline_reduces_the_estimator_variance() -> None:
    # `docs/tex` gives variance, not bias, as the reason for a baseline, so
    # the variance is what is measured. The reduction here is modest --
    # realized ratio 0.90 -- because at this horizon the returns are all of
    # similar size; the baseline earns its place on problems whose return
    # scale varies, which is the case it is there for. Reported rather than
    # asserted tightly: a threshold tuned to 0.90 would be tuned to this
    # landscape.
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)

    def per_episode_variance(baseline: float) -> float:
        rng = np.random.default_rng(5)
        gradients: list[np.ndarray] = []
        for _ in range(400):
            episode = rollout(landscape, policy, rng, EXACT_HORIZON, start=start)
            policy.weights.grad = None
            (-surrogate_loss(landscape, policy, [episode], baseline)).backward()  # type: ignore[no-untyped-call]
            assert policy.weights.grad is not None
            gradients.append(policy.weights.grad.numpy().copy())
        return float(np.asarray(gradients).var(axis=0).sum())

    rng = np.random.default_rng(9)
    mean_return = float(
        np.mean(
            [
                rollout(landscape, policy, rng, EXACT_HORIZON, start=start).total_reward
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
    landscape = _landscape()
    starts = list(enumerate_configurations(N_STATES, CHAIN_LENGTH))[::9]
    policy = LinearPolicy(2)

    def enumerated_return() -> float:
        return float(
            np.mean(
                [
                    float(
                        exact_expected_return(
                            landscape, policy, start, EXACT_HORIZON
                        ).detach()
                    )
                    for start in starts
                ]
            )
        )

    before = enumerated_return()
    training = reinforce(
        landscape,
        policy,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=EPISODE_HORIZON,
    )
    assert enumerated_return() > before
    assert training.episodes == 60 * 32
    assert len(training.mean_returns) == 60


@pytest.mark.simulated_truth
def test_the_learned_policy_is_at_least_as_good_as_hill_climbing() -> None:
    # Milestone 8's criterion, at a size where the answer is enumerable.
    #
    # Measured over 8 training seeds: the learned policy reaches the global
    # optimum from 83.6%-87.1% of the 81 starts against greedy's 80.2%, at a
    # mean final energy of 3.519-3.584 against greedy's 3.396, with the
    # optimum at 3.850. It beat greedy on both metrics in 8 of 8 seeds.
    #
    # The assertion is deliberately weaker than that: "at least as good".
    # A policy that merely matched hill climbing would still be a true
    # result, and a threshold tuned to the margin measured here would hide
    # the day it stopped holding -- the same reasoning issue #128 applied to
    # the NNI-versus-SPR comparison.
    landscape = _landscape()
    starts = list(enumerate_configurations(N_STATES, CHAIN_LENGTH))
    policy = LinearPolicy(2)
    reinforce(
        landscape,
        policy,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=EPISODE_HORIZON,
    )

    def final_energy(states: tuple[tuple[int, ...], ...]) -> float:
        return landscape.energy(states[-1])

    greedy = float(
        np.mean(
            [
                final_energy(greedy_rollout(landscape, start, EPISODE_HORIZON).states)
                for start in starts
            ]
        )
    )
    rng = np.random.default_rng(3)
    learned = float(
        np.mean(
            [
                final_energy(
                    rollout(landscape, policy, rng, EPISODE_HORIZON, start=start).states
                )
                for start in starts
                for _ in range(16)
            ]
        )
    )
    assert learned >= greedy
    assert learned <= optimum(landscape)[1]


@pytest.mark.structural
def test_training_is_reproducible_from_its_seed() -> None:
    landscape = _landscape()
    runs = [
        reinforce(
            landscape,
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


@pytest.mark.structural
def test_the_gradient_check_would_catch_a_biased_estimator() -> None:
    # Guards the guard: a check that cannot fail reads as evidence while
    # supplying none. The bias planted here is myopia -- weighting each step
    # by its own reward instead of by everything that followed it, which
    # discards precisely the credit assignment `docs/tex` says tree search
    # makes sharp. Realized: 7.1e-01 relative, against 9.9e-03 for the
    # correct estimator on the same episodes.
    #
    # Worth recording what is *not* an error, since it looks like one.
    # Weighting every step by the episode's total return rather than by its
    # return-to-go is also unbiased -- the discarded past rewards are
    # uncorrelated with the action, so they contribute zero in expectation.
    # Measured at 1.4e-02 here, inside the sampling tolerance. Return-to-go
    # buys variance, not correctness, and a test claiming otherwise would be
    # asserting a falsehood.
    landscape, policy = _landscape(), _policy([0.3, -0.6])
    start = (2, 1, 1, 0)
    exact = exact_policy_gradient(landscape, policy, start, EXACT_HORIZON)

    rng = np.random.default_rng(7)
    episodes = [
        rollout(landscape, policy, rng, EXACT_HORIZON, start=start) for _ in range(4000)
    ]
    myopic = torch.zeros((), dtype=torch.float64)
    for episode in episodes:
        for step, action in enumerate(episode.actions):
            state = episode.states[step]
            available = landscape.actions(state)
            log_probabilities = policy.log_probabilities(
                landscape.features(state, available)
            )
            myopic = (
                myopic
                + log_probabilities[available.index(action)] * episode.rewards[step]
            )
    policy.weights.grad = None
    (myopic / len(episodes)).backward()  # type: ignore[no-untyped-call]
    assert policy.weights.grad is not None
    assert _relative_difference(policy.weights.grad, exact) > _ESTIMATOR_TOLERANCE


# --- validation -----------------------------------------------------------


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("iterations", "batch", "message"),
    [(0, 8, "iterations must be >= 1"), (5, 0, "batch must be >= 1")],
)
def test_a_degenerate_budget_is_rejected(
    iterations: int, batch: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reinforce(
            _landscape(),
            LinearPolicy(2),
            np.random.default_rng(0),
            iterations=iterations,
            batch=batch,
            max_steps=3,
        )


@pytest.mark.edge_case
def test_an_estimate_needs_at_least_one_episode() -> None:
    with pytest.raises(ValueError, match="at least one episode"):
        surrogate_loss(_landscape(), LinearPolicy(2), [], 0.0)


@pytest.mark.edge_case
def test_a_negative_horizon_is_rejected_by_the_oracle() -> None:
    with pytest.raises(ValueError, match="horizon must be >= 0"):
        exact_expected_return(_landscape(), LinearPolicy(2), (0, 1, 0, 1), -1)


@pytest.mark.edge_case
def test_the_oracle_returns_zero_at_a_zero_horizon() -> None:
    value = exact_expected_return(_landscape(), LinearPolicy(2), (0, 1, 2, 0), 0)
    assert float(value.detach()) == 0.0
