"""The hidden Markov state path as an `Environment`, against enumeration.

The value of a third instance is that it is not a lattice. `phylo.opt`'s
model-agnosticism is measured rather than asserted --- four instances run
against `Objective` unchanged --- and `phylo.learn.Environment` has until now
had one. So what is checked here is not only that this landscape is correct
but that the *estimator, the policy and the rollout code needed no change to
carry it*, which is the claim `learn/CLAUDE.md` makes for the interface.
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
from phylo.learn.hmm import StatePathLandscape, enumerate_paths, optimum
from phylo.learn.policy import LinearPolicy
from phylo.learn.rollout import greedy_rollout

# Deliberately asymmetric: a near-uniform transition or emission makes the
# hidden states nearly exchangeable, and a search on an almost-flat landscape
# measures the fixture rather than the method (`hmm_params.yaml` says the same
# of the fitting fixture).
INITIAL = np.log(np.array([0.5, 0.3, 0.2]))
TRANSITION = np.log(np.array([[0.7, 0.2, 0.1], [0.15, 0.75, 0.1], [0.2, 0.2, 0.6]]))
EMISSION = np.log(
    np.array([[0.7, 0.15, 0.1, 0.05], [0.05, 0.65, 0.2, 0.1], [0.1, 0.1, 0.25, 0.55]])
)
OBSERVATIONS = np.array([0, 0, 1, 1, 3, 2])

# The enumerated oracle costs |A| ** horizon trajectories, and |A| here is
# `length * (n_states - 1)` = 12, so the horizon it runs at is not the one an
# episode gets: 2 is 144 trajectories, 4 would be 20,736.
EXACT_HORIZON = 2
EPISODE_HORIZON = 12
_GRADIENT_TOLERANCE = 1e-8

# Only 3 of the 729 paths are local optima on this fixture, and from one of
# them the expected return is the constant 0 with no autograd graph, so the
# enumerated gradient has nothing to differentiate. The start is named here
# rather than chosen by eye for that reason.
_NON_TERMINAL_START = (0, 0, 0, 0, 0, 1)


def _landscape() -> StatePathLandscape:
    return StatePathLandscape(INITIAL, TRANSITION, EMISSION, OBSERVATIONS)


@pytest.mark.oracle
def test_the_local_reward_matches_re_evaluating_the_joint_probability() -> None:
    # The failure this class is most exposed to: an O(1) update that
    # disagrees with a full evaluation would be invisible to any test that
    # only checked the search improved.
    landscape = _landscape()
    rng = np.random.default_rng(0)
    for _ in range(30):
        state = landscape.reset(rng)
        for action in landscape.actions(state):
            successor, reward = landscape.step(state, action)
            assert reward == pytest.approx(
                landscape.energy(successor) - landscape.energy(state), abs=1e-12
            )


@pytest.mark.mathematical
def test_the_features_span_the_reward_so_greedy_is_in_the_policy_class() -> None:
    # `learn/CLAUDE.md` requires it. Here the reward is the plain sum of the
    # two features, so the greedy weights carry no parameter at all.
    landscape = _landscape()
    state = landscape.reset(np.random.default_rng(1))
    actions = landscape.actions(state)
    features = landscape.features(state, actions)

    rewards = np.array([landscape.step(state, action)[1] for action in actions])
    scored = (features @ landscape.greedy_weights()).numpy()

    assert_allclose(scored, rewards, atol=1e-12)


@pytest.mark.oracle
def test_enumeration_counts_every_path() -> None:
    assert len(list(enumerate_paths(3, len(OBSERVATIONS)))) == 3 ** len(OBSERVATIONS)


@pytest.mark.oracle
def test_hill_climbing_reaches_the_enumerated_optimum() -> None:
    # 729 paths, so "did the search find the best one" has an answer. Greedy
    # is not guaranteed to reach it and the realized rate is what is
    # reported, not an assumption that it does.
    landscape = _landscape()
    _, best = optimum(landscape)
    rng = np.random.default_rng(2)
    reached = [
        landscape.energy(
            greedy_rollout(landscape, landscape.reset(rng), EPISODE_HORIZON).states[-1]
        )
        for _ in range(40)
    ]

    assert max(reached) == pytest.approx(best)
    assert np.mean([value == pytest.approx(best) for value in reached]) > 0.5


@pytest.mark.mathematical
@pytest.mark.oracle
def test_the_enumerated_gradient_matches_central_differences() -> None:
    # The oracle that makes this an *instance* rather than a second class
    # with the same method names: `phylo.learn.exact` carries it unchanged
    # from the Potts landscape, and the agreement it reaches here is the
    # same claim at 1.5e-11 that one reports.
    landscape = _landscape()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.6, -0.3], dtype=torch.float64))
    start = _NON_TERMINAL_START

    analytic = exact_policy_gradient(landscape, policy, start, EXACT_HORIZON)
    numerical = finite_difference_gradient(landscape, policy, start, EXACT_HORIZON)

    assert_allclose(
        analytic.detach().numpy(),
        numerical.numpy(),
        rtol=_GRADIENT_TOLERANCE,
        atol=_GRADIENT_TOLERANCE * float(np.linalg.norm(numerical.numpy())),
    )


@pytest.mark.mathematical
def test_the_expected_return_is_finite_and_improves_with_greedy_weights() -> None:
    # Not "the return went up": both quantities are *enumerated*, so this is
    # an exact comparison of two closed forms rather than a training curve.
    landscape = _landscape()
    start = _NON_TERMINAL_START
    uniform = LinearPolicy(2)
    greedy = LinearPolicy(2)
    greedy.set_weights(landscape.greedy_weights() * 4.0)

    under_uniform = float(
        exact_expected_return(landscape, uniform, start, EXACT_HORIZON).detach()
    )
    under_greedy = float(
        exact_expected_return(landscape, greedy, start, EXACT_HORIZON).detach()
    )

    assert np.isfinite(under_uniform)
    assert under_greedy > under_uniform


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("initial", "transition", "emission", "observations", "message"),
    [
        (np.log([1.0]), TRANSITION, EMISSION, OBSERVATIONS, "log_initial"),
        (
            INITIAL,
            np.log(np.ones((2, 2)) / 2),
            EMISSION,
            OBSERVATIONS,
            "log_transition",
        ),
        (
            INITIAL,
            TRANSITION,
            np.log(np.ones((2, 4)) / 4),
            OBSERVATIONS,
            "log_emission",
        ),
        (INITIAL, TRANSITION, EMISSION, np.array([0]), "observations must be 1-D"),
        (INITIAL, TRANSITION, EMISSION, np.array([0, 9]), r"lie in \[0, 4\)"),
    ],
)
def test_a_malformed_landscape_is_refused(
    initial: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
    observations: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StatePathLandscape(initial, transition, emission, observations)
