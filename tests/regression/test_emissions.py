"""What an emission family must be true of, whatever a state emits.

The seam these test is a refactor of live code, so most of its evidence is
elsewhere: every categorical HMM test in the suite passes against the
categorical family without being rewritten, which is what says the
abstraction did not alter a model already validated. What is pinned here is
the part that has no such witness --- the second implementation, and the two
properties that separate a density from a probability.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
    identifiable_concentration_bound,
    identifiable_dispersion_bound,
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

    step = CategoricalEmission(MATRIX).reestimate(observations, posterior)

    weights = posterior.numpy().reshape(-1, 2)
    counts = np.zeros((2, 4))
    for row, symbol in zip(weights, observations.numpy().reshape(-1), strict=True):
        counts[:, symbol] += row
    assert_allclose(
        step.emissions.matrix.numpy(),
        counts / counts.sum(axis=1, keepdims=True),
        rtol=1e-13,
    )
    # A closed form has no convergence to report, and says so rather than
    # leaving a caller to infer it.
    assert (step.at_boundary, step.iterations, step.residual) == (False, 0, 0.0)


def test_the_gaussian_m_step_is_the_posterior_weighted_mean_and_variance() -> None:
    # Checked against the closed form directly rather than only by a
    # monotonically increasing likelihood, which is a strictly stronger
    # statement than the categorical M step gets from an EM run.
    rng = np.random.default_rng(12)
    observations = torch.as_tensor(rng.normal(size=(5, 7)))
    posterior = torch.as_tensor(rng.dirichlet(np.ones(2), size=(5, 7)))

    fitted = _gaussian().reestimate(observations, posterior).emissions

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


# --- negative binomial ----------------------------------------------------
#
# The family whose M step is a solve rather than a formula, and whose
# identifiability hazard is a *flat* likelihood where the Gaussian's is an
# unbounded one. It has two exact limits, which is a stronger oracle than a
# generic count model offers, and both are used below.

NB_DISPERSION = np.array([1.5, 10.0])
NB_MEAN = np.array([6.0, 25.0])

#: Monte Carlo standard deviation of a sample variance at 200000 draws,
#: measured over 20 replicates of each fixture: 0.149 at ``(r, mu) = (1.5, 6)``
#: and 0.197 at ``(10, 25)``. The bound below is four of the larger, so the
#: tolerance comes from the sampling noise rather than from what happened to
#: pass -- the discipline ``hmm_params.yaml`` documents for the categorical
#: fixture.
NB_VARIANCE_TOLERANCE = 0.8
NB_DRAWS = 200_000


def _negative_binomial() -> NegativeBinomialEmission:
    """The two-state count fixture."""
    return NegativeBinomialEmission(NB_DISPERSION, NB_MEAN)


def test_the_count_family_satisfies_the_emission_protocol() -> None:
    assert isinstance(_negative_binomial(), EmissionFamily)


def test_a_dispersion_of_one_is_the_geometric_distribution_exactly() -> None:
    # An equality, not a tolerance: at r = 1 the two lgamma terms cancel
    # identically and what is left is log(p) + y log(1 - p). A tolerance here
    # would hide an implementation that is merely close.
    counts = torch.arange(0.0, 12.0, dtype=torch.float64)
    for mean in (3.0, 11.0):
        family = NegativeBinomialEmission([1.0], [mean])
        probability = torch.tensor(1.0 / (1.0 + mean), dtype=torch.float64)

        scored = family.log_density(counts)[:, 0]

        geometric = torch.log(probability) + counts * torch.log(1.0 - probability)
        assert float((scored - geometric).abs().max()) == 0.0


def test_the_poisson_limit_is_approached_at_the_rate_the_expansion_predicts() -> None:
    # The tolerance is *derived* from the truncation rather than chosen. The
    # deviation from the Poisson log-pmf is O(1 / r), so what is asserted is
    # the order -- doubling r halves the deviation -- and the bound at any one
    # r follows from the constant that pins. Realized: r times the deviation
    # is 18.75, 18.88, 18.94, 18.97, 18.98 over r from 500 to 8000, converging
    # rather than drifting, which is what makes the extrapolation legitimate.
    counts = torch.arange(0.0, 12.0, dtype=torch.float64)
    mean = 4.0
    poisson = counts * math.log(mean) - mean - torch.lgamma(counts + 1.0)

    deviations = {
        dispersion: float(
            (
                NegativeBinomialEmission([dispersion], [mean]).log_density(counts)[:, 0]
                - poisson
            )
            .abs()
            .max()
        )
        for dispersion in (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    }

    scaled = [dispersion * deviation for dispersion, deviation in deviations.items()]
    assert_allclose(scaled, np.full(len(scaled), 19.0), atol=0.3)
    ordered = sorted(deviations)
    halvings = [
        deviations[smaller] / deviations[larger]
        for smaller, larger in itertools.pairwise(ordered)
    ]
    assert_allclose(halvings, np.full(len(halvings), 2.0), rtol=0.02)


def test_the_drawn_moments_match_the_mean_variance_relation() -> None:
    # `Var = mu + mu**2 / r` is what makes this a count model rather than a
    # Poisson with a free parameter, so it is checked against the relation
    # itself and not against another call into the family.
    rng = np.random.default_rng(31)
    family = _negative_binomial()

    for state in (0, 1):
        drawn = family.sample(np.full(NB_DRAWS, state, dtype=np.int64), rng)

        expected_variance = NB_MEAN[state] + NB_MEAN[state] ** 2 / NB_DISPERSION[state]
        assert_allclose(float(family.variance[state]), expected_variance, rtol=1e-14)
        # Standard error of the mean is sqrt(Var / n): 0.012 and 0.021 here.
        assert_allclose(
            drawn.mean(),
            NB_MEAN[state],
            atol=5.0 * (expected_variance / NB_DRAWS) ** 0.5,
        )
        assert_allclose(
            drawn.var(ddof=1), expected_variance, atol=NB_VARIANCE_TOLERANCE
        )


def test_the_count_m_step_returns_a_stationary_point_of_the_weighted_likelihood() -> (
    None
):
    # Validated as an *optimization*, not as a formula. A first-order check is
    # what separates a solve that converged from one that stopped: an inner
    # loop returning a non-stationary point would still give a monotone outer
    # likelihood for a while, so the outer likelihood cannot catch it.
    rng = np.random.default_rng(32)
    counts = torch.as_tensor(
        rng.negative_binomial(2.0, 2.0 / 8.0, size=(4, 75)).astype(float)
    )
    posterior = torch.as_tensor(rng.dirichlet(np.ones(2), size=(4, 75)))

    step = _negative_binomial().reestimate(counts, posterior)

    # The residual is the weighted score divided by the total weight, so it is
    # comparable across data sizes; bisection reaches the arithmetic's own
    # floor rather than a tolerance chosen to pass.
    assert step.residual <= 1e-10
    assert not step.at_boundary
    assert step.iterations > 0


def test_the_count_m_step_agrees_with_a_brute_force_grid_search() -> None:
    # The second, independent check the ticket asks for: a grid search shares
    # no root-finding with the solve, so agreement is evidence rather than the
    # solve confirming itself.
    rng = np.random.default_rng(33)
    counts = torch.as_tensor(
        rng.negative_binomial(3.0, 3.0 / 12.0, size=(2, 90)).astype(float)
    )
    posterior = torch.as_tensor(rng.dirichlet(np.ones(2), size=(2, 90)))

    fitted = _negative_binomial().reestimate(counts, posterior).emissions
    dispersion = float(fitted.dispersion[0])
    mean = float(fitted.mean[0])

    weights = posterior[..., 0].reshape(-1)
    values = counts.reshape(-1)

    def weighted_log_likelihood(candidate: float) -> float:
        family = NegativeBinomialEmission([candidate], [mean])
        return float((weights * family.log_density(values)[:, 0]).sum())

    grid = np.exp(
        np.linspace(math.log(dispersion / 4.0), math.log(dispersion * 4.0), 20_001)
    )
    best = max(grid, key=weighted_log_likelihood)
    assert_allclose(dispersion, best, rtol=1e-3)
    assert weighted_log_likelihood(dispersion) >= weighted_log_likelihood(best)


def test_data_that_is_not_overdispersed_reaches_the_bound_and_says_so() -> None:
    # The flat-likelihood hazard, exhibited. Poisson counts carry no
    # overdispersion, so the maximum in `r` is at infinity; the solve returns
    # the bound and flags it rather than running away or pretending to have
    # found a maximum. Under-dispersed data are the same case, more so.
    rng = np.random.default_rng(34)
    for drawn in (
        rng.poisson(5.0, size=4000),
        rng.binomial(10, 0.5, size=4000),
        rng.negative_binomial(2000.0, 2000.0 / 2005.0, size=4000),
    ):
        counts = torch.as_tensor(drawn.astype(float)).reshape(1, -1)
        posterior = torch.ones((1, counts.shape[1], 1), dtype=torch.float64)

        step = NegativeBinomialEmission([1.0], [1.0]).reestimate(counts, posterior)

        assert step.at_boundary
        assert_allclose(
            float(step.emissions.dispersion[0]),
            identifiable_dispersion_bound(float(counts.mean()), float(counts.numel())),
            rtol=1e-12,
        )


def test_overdispersed_data_recovers_its_dispersion_and_does_not_flag() -> None:
    # The paired half of the check above: a guard that flagged everything
    # would pass that test and mean nothing.
    rng = np.random.default_rng(35)
    drawn = rng.negative_binomial(3.0, 3.0 / 13.0, size=4000)
    counts = torch.as_tensor(drawn.astype(float)).reshape(1, -1)
    posterior = torch.ones((1, counts.shape[1], 1), dtype=torch.float64)

    step = NegativeBinomialEmission([1.0], [1.0]).reestimate(counts, posterior)

    assert not step.at_boundary
    assert_allclose(float(step.emissions.dispersion[0]), 3.0, rtol=0.15)
    assert_allclose(float(step.emissions.mean[0]), 10.0, rtol=0.05)


def test_the_dispersion_bound_is_derived_from_the_counts_and_the_sample() -> None:
    # `mu sqrt(W / 2)`: where the overdispersion `mu**2 / r` falls below the
    # sampling noise on a variance. Both scalings are asserted, since a bound
    # ignoring either would be a constant wearing a formula.
    assert_allclose(identifiable_dispersion_bound(5.0, 4000.0), 5.0 * 2000.0**0.5)
    assert_allclose(
        identifiable_dispersion_bound(10.0, 4000.0),
        2.0 * identifiable_dispersion_bound(5.0, 4000.0),
    )
    assert_allclose(
        identifiable_dispersion_bound(5.0, 16000.0),
        2.0 * identifiable_dispersion_bound(5.0, 4000.0),
    )
    with pytest.raises(ValueError, match="must be positive"):
        identifiable_dispersion_bound(0.0, 4000.0)


def test_a_count_family_distinguishes_its_states_by_two_moments() -> None:
    # Mean *and* variance, unlike the Gaussian family's mean alone. The
    # dispersion is the parameter this family exists for, so two states
    # sharing a mean and differing in dispersion are different states; both
    # entries are in the units of the observations, so their absolute
    # differences may be summed.
    family = _negative_binomial()

    key = family.alignment_key().numpy()

    assert key.shape == (2, 2)
    assert_allclose(key[:, 0], NB_MEAN)
    assert_allclose(key[:, 1], NB_MEAN + NB_MEAN**2 / NB_DISPERSION)
    tied_means = NegativeBinomialEmission([1.0, 50.0], [7.0, 7.0]).alignment_key()
    assert float((tied_means[0] - tied_means[1]).abs().sum()) > 0.0


def test_a_count_family_refuses_what_is_not_a_count() -> None:
    family = _negative_binomial()
    family.validate(np.array([0, 3, 17]))
    with pytest.raises(ValueError, match="non-negative integers"):
        family.validate(np.array([0, -1]))
    with pytest.raises(ValueError, match="non-negative integers"):
        family.validate(np.array([0.0, 2.5]))
    with pytest.raises(ValueError, match="dispersion must be positive"):
        NegativeBinomialEmission([0.0], [1.0])
    with pytest.raises(ValueError, match="same shape"):
        NegativeBinomialEmission([1.0, 2.0], [1.0])


def test_the_textbook_parameterization_is_accepted_at_the_boundary() -> None:
    # `(r, p)` is the form a fixture or a reference states, `(r, mu)` the form
    # the M step profiles cleanly in. Converting at the edge keeps one
    # parameterization inside.
    from_probability = NegativeBinomialEmission.from_probability([2.0], [0.25])

    assert_allclose(float(from_probability.mean[0]), 2.0 * 0.75 / 0.25, rtol=1e-14)
    assert_allclose(float(from_probability.dispersion[0]), 2.0, rtol=1e-14)


# --- the rest of the dispersion axis (#260) -------------------------------
#
# Binomial below equidispersion, Poisson exactly at it, negative binomial and
# beta-binomial above. The point of the set is the bracketing: an interface
# exercised only by overdispersed families has not been asked whether it
# assumes overdispersion somewhere.

TRIALS = np.array([12, 12])
POISSON_MEAN = np.array([4.0, 9.0])


def test_the_four_count_families_bracket_equidispersion() -> None:
    # One assertion, and it is the reason the other three families exist.
    binomial = BinomialEmission(TRIALS, [0.3, 0.6])
    poisson = PoissonEmission(POISSON_MEAN)
    negative_binomial = NegativeBinomialEmission([2.0, 5.0], [4.0, 9.0])
    beta_binomial = BetaBinomialEmission(TRIALS, [2.0, 3.0], [4.0, 2.0])

    for family in (binomial, poisson, negative_binomial, beta_binomial):
        assert isinstance(family, EmissionFamily)
        assert family.is_discrete
    assert bool((binomial.variance < binomial.mean).all())
    assert_allclose(poisson.variance.numpy(), poisson.mean.numpy(), rtol=1e-15)
    assert bool((negative_binomial.variance > negative_binomial.mean).all())
    assert bool((beta_binomial.variance > beta_binomial.mean).all())


def test_a_binomial_over_one_trial_is_bernoulli() -> None:
    # An equality up to the arithmetic: at n = 1 the three lgamma terms are
    # lgamma(2), lgamma(y + 1) and lgamma(2 - y), which cancel to zero for
    # both outcomes, leaving log p and log(1 - p).
    outcomes = torch.tensor([0.0, 1.0], dtype=torch.float64)
    probability = 0.3

    scored = BinomialEmission([1], [probability]).log_density(outcomes)[:, 0]

    bernoulli = torch.log(
        torch.tensor([1.0 - probability, probability], dtype=torch.float64)
    )
    assert float((scored - bernoulli).abs().max()) <= 1e-15


def test_a_beta_binomial_at_unit_parameters_is_the_discrete_uniform() -> None:
    # `BetaBinomial(n, 1, 1)` puts equal mass on every one of the n + 1
    # outcomes. Known from outside this repository, and exact.
    trials = 12
    counts = torch.arange(0.0, trials + 1.0, dtype=torch.float64)

    scored = BetaBinomialEmission([trials], [1.0], [1.0]).log_density(counts)[:, 0]

    uniform = torch.full_like(counts, -math.log(trials + 1.0))
    assert float((scored - uniform).abs().max()) <= 1e-14
    assert_allclose(float(torch.exp(scored).sum()), 1.0, rtol=1e-14)


def test_the_beta_binomial_approaches_the_binomial_as_it_concentrates() -> None:
    # The second approached limit, and the tolerance is derived the same way
    # the Poisson one is: the deviation is O(1 / (a + b)), so what is asserted
    # is the order -- doubling the concentration halves it -- and the bound at
    # any one concentration follows from the constant that pins.
    trials, rate = 12, 0.35
    counts = torch.arange(0.0, trials + 1.0, dtype=torch.float64)
    binomial = BinomialEmission([trials], [rate]).log_density(counts)[:, 0]

    deviations = {
        concentration: float(
            (
                BetaBinomialEmission(
                    [trials],
                    [rate * concentration],
                    [(1.0 - rate) * concentration],
                ).log_density(counts)[:, 0]
                - binomial
            )
            .abs()
            .max()
        )
        for concentration in (1e3, 2e3, 4e3, 8e3)
    }

    ordered = sorted(deviations)
    halvings = [
        deviations[smaller] / deviations[larger]
        for smaller, larger in itertools.pairwise(ordered)
    ]
    assert_allclose(halvings, np.full(len(halvings), 2.0), rtol=0.02)


def test_each_count_family_reproduces_its_own_mean_variance_relation() -> None:
    # Every one of these has a closed form for both moments, so the simulated
    # draws are checked against the relation and not against another call.
    # Standard error of a mean at 200000 draws is sqrt(Var / n); the variance
    # bound is the one measured for the negative binomial, which is the widest
    # of the four.
    rng = np.random.default_rng(36)
    families = (
        BinomialEmission(TRIALS, [0.3, 0.6]),
        PoissonEmission(POISSON_MEAN),
        BetaBinomialEmission(TRIALS, [2.0, 3.0], [4.0, 2.0]),
    )

    for family in families:
        for state in (0, 1):
            drawn = family.sample(np.full(NB_DRAWS, state, dtype=np.int64), rng)

            variance = float(family.variance[state])
            assert_allclose(
                drawn.mean(),
                float(family.mean[state]),
                atol=5.0 * (variance / NB_DRAWS) ** 0.5,
            )
            assert_allclose(drawn.var(ddof=1), variance, atol=NB_VARIANCE_TOLERANCE)


def test_the_closed_form_count_m_steps_are_the_weighted_mean() -> None:
    rng = np.random.default_rng(37)
    counts = torch.as_tensor(rng.integers(0, 13, size=(3, 60)).astype(float))
    posterior = torch.as_tensor(rng.dirichlet(np.ones(2), size=(3, 60)))
    weights = posterior.numpy().reshape(-1, 2)
    expected = (weights * counts.numpy().reshape(-1, 1)).sum(axis=0) / weights.sum(
        axis=0
    )

    poisson = PoissonEmission(POISSON_MEAN).reestimate(counts, posterior)
    binomial = BinomialEmission(TRIALS, [0.3, 0.6]).reestimate(counts, posterior)

    assert_allclose(poisson.emissions.mean.numpy(), expected, rtol=1e-13)
    assert_allclose(
        binomial.emissions.probability.numpy(), expected / TRIALS, rtol=1e-13
    )
    for step in (poisson, binomial):
        assert (step.converged, step.at_boundary, step.iterations) == (True, False, 0)


def test_the_beta_binomial_m_step_settles_and_is_a_stationary_point() -> None:
    # Validated as an optimization. Minka's fixed point was tried first and
    # rejected on measurement: monotone but linearly convergent, it was still
    # moving in the third decimal place after 500 iterations at a true
    # concentration of 120, and returned that as though it were an estimate.
    # Alternating bisection settles in single-digit iterations.
    rng = np.random.default_rng(38)
    truth = BetaBinomialEmission([12], [2.0], [5.0])
    counts = torch.as_tensor(
        truth.sample(np.zeros(3000, dtype=np.int64), rng).astype(float)
    ).reshape(1, -1)
    posterior = torch.ones((1, counts.shape[1], 1), dtype=torch.float64)

    step = BetaBinomialEmission([12], [1.0], [1.0]).reestimate(counts, posterior)

    assert step.converged
    assert not step.at_boundary
    assert step.iterations <= 20
    fitted = step.emissions
    assert_allclose(float(fitted.alpha[0]), 2.0, rtol=0.25)
    assert_allclose(float(fitted.beta[0]), 5.0, rtol=0.25)
    # First-order optimality, checked by perturbing each coordinate: the
    # weighted likelihood must not increase in either direction.
    weights = posterior.reshape(-1)
    values = counts.reshape(-1)

    def weighted(alpha: float, beta: float) -> float:
        family = BetaBinomialEmission([12], [alpha], [beta])
        return float((weights * family.log_density(values)[:, 0]).sum())

    best = weighted(float(fitted.alpha[0]), float(fitted.beta[0]))
    for scale in (0.99, 1.01):
        assert weighted(float(fitted.alpha[0]) * scale, float(fitted.beta[0])) <= best
        assert weighted(float(fitted.alpha[0]), float(fitted.beta[0]) * scale) <= best


def test_data_with_no_overdispersion_drives_the_concentration_to_its_bound() -> None:
    # The beta-binomial's flat-likelihood hazard is the negative binomial's,
    # one family over: as `a + b` grows the family becomes a binomial, so
    # binomial data has no concentration to find.
    #
    # **The flag is a property of the sample, not of the population**, and
    # asserting it on one draw would be asserting a coin flip. Binomial data
    # is the boundary case itself, so a given draw is very slightly over- or
    # under-dispersed with roughly equal chance, and only the under-dispersed
    # half has its maximum at infinity. Measured over 12 seeds: 10 reach the
    # bound and the other two stop at 0.40 and 0.45 of it -- deep in the
    # unidentified region either way, which is the claim that survives the
    # draw.
    bound = identifiable_concentration_bound(12.0, 4000.0)
    reached = 0
    fractions = []
    for seed in range(39, 51):
        rng = np.random.default_rng(seed)
        counts = torch.as_tensor(
            rng.binomial(12, 0.4, size=4000).astype(float)
        ).reshape(1, -1)
        posterior = torch.ones((1, counts.shape[1], 1), dtype=torch.float64)

        step = BetaBinomialEmission([12], [1.0], [1.0]).reestimate(counts, posterior)

        reached += step.at_boundary
        fractions.append(float(step.emissions.concentration[0]) / bound)
        if step.at_boundary:
            assert_allclose(float(step.emissions.concentration[0]), bound, rtol=1e-9)

    assert reached >= 8
    assert min(fractions) >= 0.30


def test_the_concentration_bound_is_derived_from_the_trials_and_the_sample() -> None:
    assert_allclose(identifiable_concentration_bound(12.0, 4000.0), 11.0 * 2000.0**0.5)
    assert_allclose(
        identifiable_concentration_bound(12.0, 16000.0),
        2.0 * identifiable_concentration_bound(12.0, 4000.0),
    )
    # At one trial a beta-binomial *is* a Bernoulli whatever its
    # concentration, so nothing identifies it and there is no bound to give.
    with pytest.raises(ValueError, match="trials must be >= 2"):
        identifiable_concentration_bound(1.0, 4000.0)


def test_the_bounded_families_refuse_a_count_above_their_trials() -> None:
    for family in (
        BinomialEmission(TRIALS, [0.3, 0.6]),
        BetaBinomialEmission(TRIALS, [2.0, 3.0], [4.0, 2.0]),
    ):
        family.validate(np.array([0, 12]))
        with pytest.raises(ValueError, match="must not exceed the largest trial count"):
            family.validate(np.array([0, 13]))
    with pytest.raises(ValueError, match="probability must lie strictly"):
        BinomialEmission([5], [1.0])
    with pytest.raises(ValueError, match="trial count must be a positive integer"):
        BinomialEmission([2.5], [0.5])
    with pytest.raises(ValueError, match="mean must be positive"):
        PoissonEmission([0.0])
