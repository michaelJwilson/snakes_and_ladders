"""The two-channel count emission: a depth, and the successes within it.

What a coverage-and-allele-count assay produces, and the first emission in
this repository whose observation is not a scalar. Two claims carry the
family. The first is that each form is a probability distribution: the
log-density sums to one over the support, which is the check a two-channel
density either passes or is not one. The second is that the forms are
*different models* rather than a parameterization choice --- data simulated
under the joint form is preferred by the joint fit on every seed, and data
simulated under the independent form is not, so the preference measures the
coupling and not the extra flexibility of one form over the other.

Sizes and tolerances are stated where they are used, from the sampling noise
at that size rather than from what happened to pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import (
    CountEmissionFamily,
    CountPairEmission,
    EmissionFamily,
)

#: The two-state fixture both forms are built on: a shallow, overdispersed
#: state and a deep, more nearly Poisson one, with allele fractions of 0.25
#: and 0.667 so the states are not exchangeable in either channel.
DISPERSION = [4.0, 12.0]
MEAN = [40.0, 150.0]
ALPHA = [2.0, 6.0]
BETA = [6.0, 3.0]
TRIALS = [20.0, 30.0]


def _family(*, joint: bool) -> CountPairEmission:
    """The fixture in one form or the other."""
    trials = None if joint else TRIALS
    return CountPairEmission(DISPERSION, MEAN, ALPHA, BETA, trials, joint=joint)


@pytest.mark.structural
def test_the_count_pair_family_satisfies_both_protocols() -> None:
    # `CountEmissionFamily` as well as `EmissionFamily`, which is what lets the
    # coupled model and the mixture take it where they take a negative
    # binomial; the moments it satisfies the second with are per channel.
    for joint in (True, False):
        family = _family(joint=joint)
        assert isinstance(family, EmissionFamily)
        assert isinstance(family, CountEmissionFamily)
        assert family.mean.shape == (2, 2)
        assert family.variance.shape == (2, 2)
        assert family.alignment_key().shape == (2, 4)
        assert family.is_discrete


@pytest.mark.oracle
def test_each_form_sums_to_one_over_the_support() -> None:
    # The check a density either passes or is not one, summed over the pairs
    # themselves rather than reduced analytically. The grid is square in both
    # channels and the family scores an impossible pair at `-inf`, so one grid
    # serves both forms: the joint form's `y > n` and the independent form's
    # `y > trials` contribute exactly zero.
    #
    # Truncation is at 900 rather than infinity, and the cap is measured
    # rather than argued: the missing mass of the deeper state (mean 150,
    # dispersion 12) is 2.8e-5 at a cap of 400, 4.0e-10 at 600 and 2.8e-14 at
    # 900, so at 900 the sums are at the round-off of the 1.6 million terms
    # they add and the tolerance below is not a truncation budget.
    grid = np.arange(0.0, 901.0)
    pairs = torch.as_tensor(
        np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    )

    for joint in (True, False):
        mass = torch.exp(_family(joint=joint).log_density(pairs)).sum(dim=0)

        assert_allclose(mass.numpy(), np.ones(2), atol=1e-12)


@pytest.mark.mathematical
def test_the_closed_form_moments_are_the_drawn_ones_in_both_forms() -> None:
    # The joint form's success channel is not a beta-binomial: its variance
    # picks up the varying depth through the law of total variance, and a
    # family that reported the conditional variance alone would understate the
    # spread by `p**2 Var(n)` -- 4.7 of 10.4 for the shallow state here. Both
    # terms are checked against draws, at 200,000 per state, where the
    # standard error of a variance is `sqrt(2 / n)` relative, i.e. 0.3%.
    draws = 200_000
    states = np.repeat([0, 1], draws)

    for joint in (True, False):
        family = _family(joint=joint)
        rng = np.random.default_rng(4022)

        drawn = family.sample(states, rng).astype(float)

        for state in (0, 1):
            sample = drawn[states == state]
            assert_allclose(sample.mean(axis=0), family.mean[state].numpy(), rtol=0.01)
            assert_allclose(
                sample.var(axis=0), family.variance[state].numpy(), rtol=0.02
            )


@pytest.mark.simulated_truth
def test_the_m_step_recovers_the_planted_parameters_in_both_forms() -> None:
    # 4,000 pairs split between the two states by a planted label, scored with
    # that label as the posterior, so what is measured is the M step and not a
    # forward-backward recursion feeding it.
    #
    # The success channel is stated as `(rate, concentration)` rather than
    # `(alpha, beta)`: the rate is what the data resolves -- it is a weighted
    # mean of about 2,000 allele fractions -- and the concentration is the
    # parameter whose likelihood flattens as the family approaches a binomial,
    # exactly as the beta-binomial's own recovery test finds at 3,000 draws.
    # Measured relative errors per state --- joint form: mean 0.008 and 0.012,
    # dispersion 0.021 and 0.059, rate 0.002 and 0.005, concentration 0.117
    # and 0.038; independent form: mean 0.020 and 0.006, dispersion 0.048 and
    # 0.042, rate 0.023 and 0.004, concentration 0.047 and 0.036. The
    # tolerances below are between two and five times the larger of the two
    # forms, so neither form is held to a bound the other set.
    n_samples = 4_000
    rng = np.random.default_rng(4021)
    states = rng.integers(0, 2, size=n_samples)
    posterior = torch.zeros((n_samples, 2), dtype=torch.float64)
    posterior[np.arange(n_samples), states] = 1.0
    rate = np.asarray(ALPHA) / (np.asarray(ALPHA) + np.asarray(BETA))
    concentration = np.asarray(ALPHA) + np.asarray(BETA)

    for joint in (True, False):
        truth = _family(joint=joint)
        observations = torch.as_tensor(truth.sample(states, rng).astype(float))
        start = CountPairEmission(
            [1.0, 1.0],
            [10.0, 10.0],
            [1.0, 1.0],
            [1.0, 1.0],
            None if joint else TRIALS,
            joint=joint,
        )

        step = start.reestimate(observations, posterior)

        assert step.converged
        assert not step.at_boundary
        fitted = step.emissions
        assert fitted.joint is joint
        assert_allclose(fitted.total.mean.numpy(), MEAN, rtol=0.03)
        assert_allclose(fitted.total.dispersion.numpy(), DISPERSION, rtol=0.15)
        assert_allclose(fitted.rate.numpy(), rate, rtol=0.05)
        assert_allclose(fitted.concentration.numpy(), concentration, rtol=0.25)


#: The comparison's size and seeds. 300 pairs is enough that the joint form's
#: advantage on joint data is many times the spread across seeds, and small
#: enough that 40 M steps run in about two seconds.
LRT_SAMPLES = 300
LRT_SEEDS = range(1000, 1010)

#: The single-state truth the comparison simulates from. The depth mean is
#: 120 at dispersion 8, so `P(n < 10)` is 3.4e-6 and the independent form's
#: 10-trial draws lie inside the joint form's support on every one of the
#: 3,000 pairs -- which the test asserts, because a comparison in which one
#: model scores a pair at `-inf` would be settled by the support rather than
#: by the coupling.
LRT_DISPERSION = [8.0]
LRT_MEAN = [120.0]
LRT_ALPHA = [3.0]
LRT_BETA = [5.0]
LRT_TRIALS = [10.0]


def _maximized_log_likelihood(
    start: CountPairEmission, observations: torch.Tensor
) -> float:
    """Fit one state by a single M step, and score the data at it.

    With one state and a posterior of ones the M step *is* the maximum
    likelihood estimate, so no EM loop is needed and the comparison is between
    two maxima rather than between two runs of an iteration.
    """
    posterior = torch.ones((observations.shape[0], 1), dtype=torch.float64)
    fitted = start.reestimate(observations, posterior).emissions
    return float(fitted.log_density(observations)[:, 0].sum())


def _preference(observations: torch.Tensor) -> float:
    """``2 (log L_joint - log L_independent)`` at each form's own maximum.

    The two forms have four free parameters each, so the difference of maxima
    is also the difference of AICs and the comparison needs no penalty. It is
    not a nested test, and no chi-squared quantile is read off it; what is
    asserted is the sign, over seeds.

    The independent form is given the most generous fixed trial count the data
    admits --- the largest success count observed --- so that it is refuted on
    its shape and not on a support that cannot hold the data.
    """
    depth = float(observations[:, 0].mean())
    largest = float(observations[:, 1].max())
    joint = _maximized_log_likelihood(
        CountPairEmission([1.0], [depth], [1.0], [1.0], None, joint=True), observations
    )
    independent = _maximized_log_likelihood(
        CountPairEmission([1.0], [depth], [1.0], [1.0], [largest], joint=False),
        observations,
    )
    return 2.0 * (joint - independent)


@pytest.mark.simulated_truth
def test_the_likelihood_ratio_prefers_the_form_the_data_came_from() -> None:
    # The claim that separates the two forms. On joint data the depth carries
    # information about the allele count and the independent form throws it
    # away; on independent data it carries none and the joint form's
    # conditioning is a misspecification. Measured over the ten seeds:
    # the statistic is 124 to 211 on joint data (joint preferred 10 of 10) and
    # -133 to -77 on independent data (joint preferred 0 of 10), so the two
    # populations do not overlap and neither margin is a coin flip.
    outcome = {}
    for joint in (True, False):
        truth = CountPairEmission(
            LRT_DISPERSION,
            LRT_MEAN,
            LRT_ALPHA,
            LRT_BETA,
            None if joint else LRT_TRIALS,
            joint=joint,
        )
        statistics = []
        for seed in LRT_SEEDS:
            rng = np.random.default_rng(seed)
            drawn = truth.sample(np.zeros(LRT_SAMPLES, dtype=np.int64), rng).astype(
                float
            )
            assert (drawn[:, 1] <= drawn[:, 0]).all()
            statistics.append(_preference(torch.as_tensor(drawn)))
        outcome[joint] = statistics

    assert sum(statistic > 0.0 for statistic in outcome[True]) == 10
    assert min(outcome[True]) > 100.0
    # Indifference to the joint form on independent data, which is what says
    # the preference above is the coupling and not the flexibility.
    assert sum(statistic > 0.0 for statistic in outcome[False]) == 0


@pytest.mark.edge_case
def test_the_form_must_be_stated_and_carry_the_trial_count_it_needs() -> None:
    # `joint` is keyword-only with no default: a default would choose a
    # generative model for the caller, and the two are different models.
    with pytest.raises(TypeError):
        CountPairEmission(DISPERSION, MEAN, ALPHA, BETA, TRIALS)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="trial count is the observed total"):
        CountPairEmission(DISPERSION, MEAN, ALPHA, BETA, TRIALS, joint=True)
    with pytest.raises(ValueError, match="needs a fixed trial count"):
        CountPairEmission(DISPERSION, MEAN, ALPHA, BETA, None, joint=False)
    with pytest.raises(ValueError, match="same number of states"):
        CountPairEmission(DISPERSION, MEAN, [2.0], [6.0], None, joint=True)


@pytest.mark.edge_case
def test_a_pair_outside_the_support_is_refused_and_scored_at_minus_infinity() -> None:
    joint = _family(joint=True)
    independent = _family(joint=False)
    pairs = np.array([[10, 4], [10, 11]], dtype=float)

    joint.validate(pairs[:1])
    with pytest.raises(ValueError, match="cannot exceed the total"):
        joint.validate(pairs)
    with pytest.raises(ValueError, match="2 channels"):
        joint.validate(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="non-negative integers"):
        joint.validate(np.array([[10.0, -1.0]]))
    with pytest.raises(ValueError, match="must not exceed the largest trial count"):
        independent.validate(np.array([[10.0, 31.0]]))

    scored = joint.log_density(torch.as_tensor(pairs))

    assert torch.isfinite(scored[0]).all()
    assert bool(torch.isinf(scored[1]).all())
    assert float(scored[1, 0]) < 0.0
    # Impossible under the model, not merely improbable, and stated as a
    # number a sum can carry rather than as a `nan` out of `lgamma`.
    assert not bool(torch.isnan(scored).any())
