"""What an emission family must be true of, whatever a state emits.

The seam these test is a refactor of live code, so most of its evidence is
elsewhere: every categorical HMM test in the suite passes against the
categorical family without being rewritten, which is what says the
abstraction did not alter a model already validated. What is pinned here is
the part that has no such witness --- the second implementation, and the two
properties that separate a density from a probability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import (
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    pooled_variance_floor,
)

MATRIX = np.array([[0.70, 0.15, 0.10, 0.05], [0.05, 0.65, 0.20, 0.10]])
MEAN = np.array([-2.0, 1.5])
SCALE = np.array([0.5, 1.25])
FLOOR = 1e-8


def _gaussian() -> GaussianEmission:
    """The Gaussian fixture, with a floor low enough not to be in the way."""
    return GaussianEmission(MEAN, SCALE, FLOOR)


def test_both_families_satisfy_the_emission_protocol() -> None:
    assert isinstance(CategoricalEmission(MATRIX), EmissionFamily)
    assert isinstance(_gaussian(), EmissionFamily)


def test_a_categorical_family_scores_a_symbol_as_its_matrix_entry() -> None:
    observations = torch.tensor([[0, 3, 1], [2, 2, 0]])

    scored = CategoricalEmission(MATRIX).log_density(observations).numpy()

    expected = np.log(MATRIX)[:, observations.numpy()].transpose(1, 2, 0)
    assert_allclose(scored, expected, rtol=1e-15)


def test_the_gaussian_density_matches_the_closed_form() -> None:
    # Against the expression written out, not against another call into the
    # same code: `1 / (sigma sqrt(2 pi)) exp(-(y - mu)^2 / 2 sigma^2)`.
    values = np.array([-3.0, -2.0, 0.0, 1.5, 4.0])

    scored = _gaussian().log_density(torch.as_tensor(values)).numpy()

    expected = np.array(
        [
            [
                math.log(
                    math.exp(-((value - mu) ** 2) / (2.0 * sigma**2))
                    / (sigma * math.sqrt(2.0 * math.pi))
                )
                for mu, sigma in zip(MEAN, SCALE, strict=True)
            ]
            for value in values
        ]
    )
    assert_allclose(scored, expected, rtol=1e-13)


def test_a_categorical_score_is_a_probability_and_a_gaussian_one_is_a_density() -> None:
    # The place the discrete assumption was load-bearing. A categorical score
    # is bounded above by zero; a Gaussian one is not, and a test asserting
    # otherwise would fail on correct code. Exhibited rather than asserted
    # from `is_discrete` alone: at scale 0.5 the density at the mean is
    # 1 / (0.5 sqrt(2 pi)) = 0.798 -- still below 1 -- so the fixture that
    # shows it needs a narrower state than the one in this module.
    categorical = CategoricalEmission(MATRIX)
    assert categorical.is_discrete
    assert float(categorical.log_density(torch.tensor([0, 1, 2, 3])).max()) <= 0.0

    narrow = GaussianEmission(np.array([0.0, 5.0]), np.array([0.1, 1.0]), FLOOR)
    assert not narrow.is_discrete
    assert float(narrow.log_density(torch.tensor([0.0]))[0, 0]) > 0.0


def test_the_categorical_m_step_is_the_normalized_expected_counts() -> None:
    rng = np.random.default_rng(11)
    observations = torch.as_tensor(rng.integers(0, 4, size=(5, 7)))
    posterior = torch.as_tensor(rng.dirichlet(np.ones(2), size=(5, 7)))

    fitted = CategoricalEmission(MATRIX).reestimate(observations, posterior)

    weights = posterior.numpy().reshape(-1, 2)
    counts = np.zeros((2, 4))
    for row, symbol in zip(weights, observations.numpy().reshape(-1), strict=True):
        counts[:, symbol] += row
    assert_allclose(
        fitted.matrix.numpy(), counts / counts.sum(axis=1, keepdims=True), rtol=1e-13
    )


def test_the_gaussian_m_step_is_the_posterior_weighted_mean_and_variance() -> None:
    # Checked against the closed form directly rather than only by a
    # monotonically increasing likelihood, which is a strictly stronger
    # statement than the categorical M step gets from an EM run.
    rng = np.random.default_rng(12)
    observations = torch.as_tensor(rng.normal(size=(5, 7)))
    posterior = torch.as_tensor(rng.dirichlet(np.ones(2), size=(5, 7)))

    fitted = _gaussian().reestimate(observations, posterior)

    values = observations.numpy().reshape(-1)
    weights = posterior.numpy().reshape(-1, 2)
    mass = weights.sum(axis=0)
    mean = (weights * values[:, None]).sum(axis=0) / mass
    variance = (weights * (values[:, None] - mean) ** 2).sum(axis=0) / mass
    assert_allclose(fitted.mean.numpy(), mean, rtol=1e-13)
    assert_allclose(fitted.scale.numpy(), np.sqrt(variance), rtol=1e-13)


def test_a_state_collapsed_onto_one_observation_is_refused_not_clamped() -> None:
    # Clamping and returning normally is the failure this exists to prevent:
    # the caller would receive a point estimate at a degenerate optimum, and
    # an interval around it summarizing nothing (issue #122).
    observations = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64)
    # State 0 takes exactly one observation, so its re-estimated variance is
    # zero. State 1 takes the rest and stays well away from the floor.
    posterior = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=torch.float64
    )

    with pytest.raises(ValueError, match="unbounded as a variance goes to zero"):
        _gaussian().reestimate(observations, posterior)


def test_the_gaussian_likelihood_grows_without_bound_as_a_state_collapses() -> None:
    # The pathology exhibited rather than described. With a mean sitting on an
    # observation, the log-density at that observation is -log(sigma) plus a
    # constant, so it grows without bound as sigma falls -- there is no
    # maximum for a fit to find, and any fit that "converges" was stopped by
    # its start or by a floor.
    scales = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    scored = [
        float(
            GaussianEmission(np.array([0.0]), np.array([scale]), 1e-30).log_density(
                torch.tensor([0.0])
            )[0, 0]
        )
        for scale in scales
    ]

    assert scored == sorted(scored)
    # Each tenfold narrowing buys exactly log(10) nats, which is what makes
    # the divergence a property of the model rather than a numerical artefact.
    steps = np.diff(scored)
    assert_allclose(steps, np.full(steps.shape, math.log(10.0)), rtol=1e-12)


def test_the_variance_floor_is_derived_from_the_data_and_scales_with_it() -> None:
    # `s**2 / n**2`: the spacing a state occupying a neighbourhood of the data
    # cannot fall below. Both scalings are asserted, since a floor that
    # ignored either would be a constant wearing a formula.
    rng = np.random.default_rng(13)
    values = rng.normal(size=400)

    floor = pooled_variance_floor(values)

    assert_allclose(floor, values.var(ddof=1) / 400.0**2, rtol=1e-14)
    assert_allclose(pooled_variance_floor(3.0 * values), 9.0 * floor, rtol=1e-13)
    doubled = np.concatenate([values, rng.normal(size=400)])
    assert pooled_variance_floor(doubled) < floor / 3.0


def test_a_floor_cannot_be_derived_without_a_scale_to_derive_it_from() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        pooled_variance_floor(np.array([1.0]))
    with pytest.raises(ValueError, match="zero spread"):
        pooled_variance_floor(np.full(10, 2.5))


def test_each_family_refuses_an_observation_outside_its_support() -> None:
    categorical = CategoricalEmission(MATRIX)
    categorical.validate(np.array([0, 3]))
    with pytest.raises(ValueError, match=r"must lie in \[0, 4\)"):
        categorical.validate(np.array([0, 4]))

    gaussian = _gaussian()
    gaussian.validate(np.array([-1e9, 1e9]))
    with pytest.raises(ValueError, match="must be finite"):
        gaussian.validate(np.array([0.0, np.inf]))


def test_each_family_says_what_distinguishes_its_states() -> None:
    # Aligning a Gaussian fit by anything but the mean would let two states
    # with different means look identical; aligning a categorical one by a
    # single number would throw away the alphabet.
    assert_allclose(CategoricalEmission(MATRIX).alignment_key().numpy(), MATRIX)
    assert_allclose(_gaussian().alignment_key().numpy(), MEAN.reshape(-1, 1))


def test_a_categorical_draw_reproduces_the_declared_row_frequencies() -> None:
    # Monte Carlo standard error at 40000 draws from one row is at most
    # sqrt(0.25 / 40000) = 0.0025, so 0.01 is four of those.
    rng = np.random.default_rng(14)
    states = np.zeros(40_000, dtype=np.int64)

    drawn = CategoricalEmission(MATRIX).sample(states, rng)

    frequencies = np.bincount(drawn, minlength=4) / drawn.size
    assert_allclose(frequencies, MATRIX[0], atol=0.01)


def test_a_gaussian_draw_reproduces_the_declared_moments() -> None:
    # Standard error of the mean at 40000 draws is 1.25 / 200 = 0.00625 for
    # the wider state; 0.03 is under five of those, and the same bound covers
    # the standard deviation, whose error is smaller by sqrt(2).
    rng = np.random.default_rng(15)
    states = np.repeat([0, 1], 40_000)

    drawn = _gaussian().sample(states, rng)

    for state in (0, 1):
        block = drawn[states == state]
        assert_allclose(block.mean(), MEAN[state], atol=0.03)
        assert_allclose(block.std(ddof=1), SCALE[state], atol=0.03)


def test_a_gaussian_family_refuses_parameters_it_cannot_be() -> None:
    with pytest.raises(ValueError, match="same shape"):
        GaussianEmission(np.array([0.0, 1.0]), np.array([1.0]), FLOOR)
    with pytest.raises(ValueError, match="scale must be positive"):
        GaussianEmission(np.array([0.0]), np.array([0.0]), FLOOR)
    with pytest.raises(ValueError, match="variance_floor must be positive"):
        GaussianEmission(np.array([0.0]), np.array([1.0]), 0.0)
