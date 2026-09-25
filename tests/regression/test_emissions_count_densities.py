"""The count families against `scipy.stats`, covariate folded in (issue #729).

`scipy.stats` states the four count distributions in its own parameterization,
an answer by another route; the exposure and trial-count branches (#658) are
the same distributions at a rate and a trial count per observation. Relative
tolerances per comparison; docstrings carry the value realized on the 4-core
reference host.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from scipy import stats

#: Counts scored in every test, shape ``(2, 3)``: a batch of two positions
#: over three states, which is the shape the fit passes and which broadcasts
#: a ``(..., 1)`` covariate along the state axis.
COUNTS = np.array([[0, 1, 3], [5, 8, 12]])

#: Three states per family, spread across the parameter range each is used
#: over: a rate under one and one over ten, a dispersion from 0.8 (far from
#: Poisson) to 25 (near it), a probability either side of a half, and a
#: beta-binomial from a U-shaped prior to a concentrated one.
RATE = np.array([0.7, 4.0, 11.5])
DISPERSION = np.array([0.8, 3.0, 25.0])
MEAN = np.array([1.2, 5.0, 9.0])
TRIALS = np.array([12.0, 20.0, 7.0])
PROBABILITY = np.array([0.15, 0.5, 0.82])
BB_TRIALS = np.array([12.0, 20.0, 15.0])
ALPHA = np.array([0.5, 2.0, 6.0])
BETA = np.array([1.5, 2.0, 0.9])

#: The declared tolerance on a log density, relative. Nine `lgamma` calls
#: summed in one order here and in another inside `scipy` do not agree
#: bitwise, and the two are not the same algorithm, so the floor is a
#: tolerance rather than equality (root `CLAUDE.md`).
DENSITY_RTOL = 1e-12

#: The declared tolerance on a moment, relative. Both sides are two
#: arithmetic operations on the same parameters, so this is rounding.
MOMENT_RTOL = 1e-14


def _scipy_negative_binomial(
    counts: np.ndarray, dispersion: np.ndarray, rate: np.ndarray
) -> np.ndarray:
    """``scipy``'s ``(n, p)`` form of the family's ``(r, mu)`` parameters."""
    return np.asarray(
        stats.nbinom.logpmf(counts, dispersion, dispersion / (dispersion + rate))
    )


@pytest.mark.oracle
def test_the_four_count_families_score_the_log_probabilities_scipy_states() -> None:
    """Every family's `log_density` against `scipy.stats`, at 18 points each.

    Widest 1.07e-14 (NB) against 1e-12; counts above trials score ``-inf`` both sides.
    """
    observations = torch.from_numpy(COUNTS)
    counts = COUNTS[..., np.newaxis]

    poisson = PoissonEmission(RATE).log_density(observations).numpy()
    binomial = BinomialEmission(TRIALS, PROBABILITY).log_density(observations).numpy()
    negative_binomial = (
        NegativeBinomialEmission(DISPERSION, MEAN).log_density(observations).numpy()
    )
    beta_binomial = (
        BetaBinomialEmission(BB_TRIALS, ALPHA, BETA).log_density(observations).numpy()
    )

    assert_allclose(poisson, stats.poisson.logpmf(counts, RATE), rtol=DENSITY_RTOL)
    assert_allclose(
        negative_binomial,
        _scipy_negative_binomial(counts, DISPERSION, MEAN),
        rtol=DENSITY_RTOL,
    )
    assert_allclose(
        beta_binomial,
        stats.betabinom.logpmf(counts, BB_TRIALS, ALPHA, BETA),
        rtol=DENSITY_RTOL,
    )
    expected = np.asarray(stats.binom.logpmf(counts, TRIALS, PROBABILITY))
    outside = counts > TRIALS
    assert np.array_equal(np.isneginf(binomial), outside)
    assert np.array_equal(np.isneginf(expected), outside)
    assert_allclose(binomial[~outside], expected[~outside], rtol=DENSITY_RTOL)


@pytest.mark.oracle
def test_the_covariate_branches_score_the_exposure_and_trial_count_folded_in() -> None:
    """The two branches issue #658 added, against the same closed forms.

    NB at ``e_i mu_k``: 4.19e-15; beta-binomial at ``n_i``: 2.39e-15; declared 1e-12.
    """
    observations = torch.from_numpy(COUNTS)
    counts = COUNTS[..., np.newaxis]
    exposures = np.array([[[0.5], [1.0], [2.5]], [[2.0], [0.25], [3.0]]])
    per_observation_trials = np.array(
        [[[14.0], [20.0], [13.0]], [[22.0], [18.0], [16.0]]]
    )

    scored = (
        NegativeBinomialEmission(DISPERSION, MEAN)
        .log_density(observations, torch.from_numpy(exposures))
        .numpy()
    )
    bounded = (
        BetaBinomialEmission(BB_TRIALS, ALPHA, BETA)
        .log_density(observations, torch.from_numpy(per_observation_trials))
        .numpy()
    )

    assert_allclose(
        scored,
        _scipy_negative_binomial(counts, DISPERSION, exposures * MEAN),
        rtol=DENSITY_RTOL,
    )
    assert_allclose(
        bounded,
        stats.betabinom.logpmf(counts, per_observation_trials, ALPHA, BETA),
        rtol=DENSITY_RTOL,
    )


@pytest.mark.oracle
def test_each_count_familys_moments_are_the_moments_scipy_reports() -> None:
    """`mean`, `variance` and `alignment_key` against `scipy.stats`' moments.

    Widest 4.35e-16 (NB variance) against 1e-14; keys are what recovery permutes on.
    """
    poisson = PoissonEmission(RATE)
    binomial = BinomialEmission(TRIALS, PROBABILITY)
    negative_binomial = NegativeBinomialEmission(DISPERSION, MEAN)
    beta_binomial = BetaBinomialEmission(BB_TRIALS, ALPHA, BETA)
    odds = DISPERSION / (DISPERSION + MEAN)

    for family, mean, variance in (
        (poisson, stats.poisson.mean(RATE), stats.poisson.var(RATE)),
        (
            binomial,
            stats.binom.mean(TRIALS, PROBABILITY),
            stats.binom.var(TRIALS, PROBABILITY),
        ),
        (
            negative_binomial,
            stats.nbinom.mean(DISPERSION, odds),
            stats.nbinom.var(DISPERSION, odds),
        ),
        (
            beta_binomial,
            stats.betabinom.mean(BB_TRIALS, ALPHA, BETA),
            stats.betabinom.var(BB_TRIALS, ALPHA, BETA),
        ),
    ):
        assert_allclose(family.mean.numpy(), mean, rtol=MOMENT_RTOL)
        assert_allclose(family.variance.numpy(), variance, rtol=MOMENT_RTOL)
        assert family.is_discrete
        assert family.observation_dtype is torch.float64

    # The Poisson is equidispersed, so its key carries one moment; the other
    # three carry both, since two states may share a mean and differ in the
    # second parameter.
    assert_allclose(
        poisson.alignment_key().numpy(),
        np.asarray(stats.poisson.mean(RATE))[:, np.newaxis],
        rtol=MOMENT_RTOL,
    )
    for family, moments in (
        (binomial, stats.binom.stats(TRIALS, PROBABILITY, moments="mv")),
        (negative_binomial, stats.nbinom.stats(DISPERSION, odds, moments="mv")),
        (beta_binomial, stats.betabinom.stats(BB_TRIALS, ALPHA, BETA, moments="mv")),
    ):
        assert_allclose(
            family.alignment_key().numpy(),
            np.stack([np.asarray(moment) for moment in moments], axis=1),
            rtol=MOMENT_RTOL,
        )

    # The parameters `scipy` was handed, read back off the families: a
    # moment agreeing under a parameter the family renamed would not.
    assert_allclose(negative_binomial.dispersion.numpy(), DISPERSION, rtol=0.0)
    assert_allclose(binomial.trials.numpy(), TRIALS, rtol=0.0)
    assert_allclose(binomial.probability.numpy(), PROBABILITY, rtol=0.0)
    assert_allclose(beta_binomial.trials.numpy(), BB_TRIALS, rtol=0.0)
    assert_allclose(beta_binomial.alpha.numpy(), ALPHA, rtol=0.0)
    assert_allclose(beta_binomial.beta.numpy(), BETA, rtol=0.0)
