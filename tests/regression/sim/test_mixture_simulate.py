"""The mixture simulator, against the analytic mixture it declares.

Validated against closed forms rather than against a likelihood computed
here: the mixture's mean and variance follow from the law of total variance,
and the fraction of draws from each component follows from the weights, both
to a Monte Carlo bound stated from the sample size (``sim/CLAUDE.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.sim.mixture import MixtureParams, simulate_mixture

WEIGHTS = np.array([0.3, 0.7])
MEAN = np.array([-3.0, 3.0])
SCALE = np.array([1.0, 1.5])
FLOOR = 1e-12
DRAWS = 200_000


def _params(n_samples: int = DRAWS, seed: int = 20260905) -> MixtureParams:
    """The two-component fixture."""
    return MixtureParams(
        weights=WEIGHTS,
        components=GaussianEmission(MEAN, SCALE, FLOOR),
        n_samples=n_samples,
        seed=seed,
        tolerance=1e-12,
    )


def test_the_component_labels_appear_at_their_declared_weights() -> None:
    # Standard error of a proportion at 200000 draws is at most 0.0011, so
    # 0.005 is over four of them.
    dataset = simulate_mixture(_params())

    frequencies = np.bincount(dataset.labels, minlength=2) / dataset.labels.size
    assert_allclose(frequencies, WEIGHTS, atol=0.005)


def test_the_drawn_moments_match_the_law_of_total_variance() -> None:
    # `E[Y] = sum_k w_k mu_k` and
    # `Var[Y] = sum_k w_k (s_k**2 + mu_k**2) - E[Y]**2`, which is the closed
    # form the simulator is checked against rather than a second simulation.
    dataset = simulate_mixture(_params())

    expected_mean = float((WEIGHTS * MEAN).sum())
    expected_variance = float((WEIGHTS * (SCALE**2 + MEAN**2)).sum()) - expected_mean**2
    assert_allclose(
        dataset.observations.mean(),
        expected_mean,
        atol=5.0 * (expected_variance / DRAWS) ** 0.5,
    )
    assert_allclose(dataset.observations.var(ddof=1), expected_variance, atol=0.15)


def test_each_component_s_own_draws_match_that_component() -> None:
    # The labels are retained so this check is possible at all: without them
    # only the mixture's moments could be checked, and a simulator that drew
    # every point from one component would pass that.
    dataset = simulate_mixture(_params())

    for component in (0, 1):
        block = dataset.observations[dataset.labels == component]
        assert_allclose(block.mean(), MEAN[component], atol=0.03)
        assert_allclose(block.std(ddof=1), SCALE[component], atol=0.03)


def test_a_passed_generator_gives_independent_draws_and_a_seed_repeats() -> None:
    # `sim/CLAUDE.md`'s rule, in the pairing it is stated as: two draws from
    # one generator differ, and two generators seeded alike agree.
    params = _params(n_samples=500)
    generator = np.random.default_rng(3)

    first = simulate_mixture(params, generator).observations
    second = simulate_mixture(params, generator).observations

    assert not np.array_equal(first, second)
    assert np.array_equal(
        simulate_mixture(params, np.random.default_rng(3)).observations, first
    )
    assert np.array_equal(
        simulate_mixture(params).observations, simulate_mixture(params).observations
    )


def test_truth_ships_with_the_data() -> None:
    dataset = simulate_mixture(_params(n_samples=64))

    assert dataset.labels.shape == dataset.observations.shape == (64,)
    assert_allclose(dataset.weights, WEIGHTS)
    assert_allclose(dataset.components.mean.numpy(), MEAN)
    assert dataset.seed == 20260905


def test_weights_that_do_not_describe_a_mixture_are_refused() -> None:
    components = GaussianEmission(MEAN, SCALE, FLOOR)
    with pytest.raises(ValueError, match="expected"):
        MixtureParams(np.array([0.2, 0.3, 0.5]), components, 10, 1, 1e-12)
    with pytest.raises(ValueError, match="sum to"):
        MixtureParams(np.array([0.2, 0.3]), components, 10, 1, 1e-12)
    # A zero-weight component is not a component: its parameters would be
    # undefined rather than merely uncertain.
    with pytest.raises(ValueError, match="weight must be positive"):
        MixtureParams(np.array([0.0, 1.0]), components, 10, 1, 1e-12)
