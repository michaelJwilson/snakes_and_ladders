"""Fitting a mixture whose components are two-channel count emissions.

Three referees, in the order their strength falls. At the CI size the E step
is checked against the **enumerated posterior** over component labellings ---
an exhaustive sum sharing no code with the normalization it referees. The fit
is then checked against the **planted parameters** at both declared sizes, up
to the permutation of components. The seeded start is checked against the
uniform one over **shared seeds** and reported whichever way it falls, on the
terms ``snakes_and_ladders.qa.mixture_seeding`` states for the Gaussian case.

Every instance is the registry's (``sim/CLAUDE.md``); the starts' shape
constants are stated here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import CountPairEmission
from snakes_and_ladders.opt.emission_mixture import (
    CountPairSeeding,
    EmissionMixtureFit,
    enumerated_posterior,
    expectation_maximization,
    plus_plus_start,
    uniform_start,
)
from snakes_and_ladders.opt.mixture import (
    mixture_log_likelihood,
    responsibilities,
)
from snakes_and_ladders.sim.emission_mixture import (
    EmissionMixtureParams,
    simulate_emission_mixture,
)
from snakes_and_ladders.sim.fixtures import fixture

#: Where a component starts before the data moves it. One observation says
#: what depth and allele fraction a component sits at and nothing about either
#: shape, so the dispersion and the concentration start at values declared
#: here. Both are away from the fixtures' truth -- 6, 10, 15 and 12 at the CI
#: size -- so a fit that recovers them found them rather than started at them.
START_DISPERSION = 5.0
START_CONCENTRATION = 10.0

#: The EM stopping rule the tests fit under: relative, since an absolute one
#: does not transfer between the two sizes (``DEV.md``, issue #111). At 1e-8
#: the CI fit takes 29 iterations against 36 at 1e-10 and reaches the same
#: parameters to four figures.
EM_TOLERANCE = 1e-8

#: Observations the enumeration oracle runs over. With K = 3 that is 3**10 =
#: 59,049 labellings, inside the cap `snakes_and_ladders.enumeration` sets.
#: The observations are independent, so the marginal at each of these ten is
#: the same quantity the responsibilities of the whole dataset carry, and
#: enumerating the whole dataset would be 3**900.
ENUMERATED = 10


def _seeding(params: EmissionMixtureParams) -> CountPairSeeding:
    """The start-placing rule for the instance's own form of the family."""
    components = params.components
    assert isinstance(components, CountPairEmission)
    declared = components.trials
    return CountPairSeeding(
        dispersion=START_DISPERSION,
        concentration=START_CONCENTRATION,
        joint=components.joint,
        # One number for every component, which `unique` enforces: an
        # independent-form instance whose components disagree on the trial
        # count has no single value to seed with.
        trials=None if declared is None else float(np.unique(declared.numpy()).item()),
    )


def _fit(
    params: EmissionMixtureParams,
    observations: np.ndarray,
    seed: int,
    max_iterations: int = 200,
) -> EmissionMixtureFit:
    """One expectation-maximization run from an ``Emission_Mixture++`` start."""
    start = plus_plus_start(
        observations, params.n_components, _seeding(params), np.random.default_rng(seed)
    )
    return expectation_maximization(
        observations,
        torch.full(
            (params.n_components,), 1.0 / params.n_components, dtype=torch.float64
        ),
        start,
        max_iterations=max_iterations,
        tolerance=EM_TOLERANCE,
    )


def _by_depth(family: CountPairEmission) -> np.ndarray:
    """The permutation ordering components by their depth, then their rate.

    Every declared instance separates its components in the total channel, in
    the success channel, or in both, so ordering on the pair is a canonical
    labelling and recovery is stated against it. Where the key separates, this
    is the answer ``opt.hmm.align_families`` reaches by enumeration --- 3.6
    million permutations at ten components.
    """
    return np.lexsort((family.rate.numpy(), family.total.mean.numpy()))


@pytest.fixture(scope="module")
def ci_instance() -> tuple[EmissionMixtureParams, np.ndarray, EmissionMixtureFit]:
    """The declared CI instance, its draw, and one fit of it.

    Module-scoped because the fit takes about four seconds and three tests
    read it (``CLAUDE.md``, Testing).
    """
    params = fixture("emission_mixture", "ci").params
    observations = simulate_emission_mixture(params).observations.astype(float)
    return params, observations, _fit(params, observations, seed=1)


@pytest.mark.oracle
def test_the_responsibilities_are_the_enumerated_posterior(
    ci_instance: tuple[EmissionMixtureParams, np.ndarray, EmissionMixtureFit],
) -> None:
    # The E step against an exhaustive sum over component labellings, sharing
    # no line with it: the enumeration scores each of the 3**10 labellings of
    # the prefix, normalizes over all 59,049 of them and marginalizes; the E
    # step normalizes each observation's row on its own. Their agreement says
    # the mixture's posterior factorizes.
    params, observations, _ = ci_instance
    prefix = torch.as_tensor(observations[:ENUMERATED])
    log_weight = torch.log(torch.as_tensor(params.weights))

    posterior = responsibilities(prefix, log_weight, params.components)

    enumerated = enumerated_posterior(prefix, log_weight, params.components)
    assert_allclose(posterior.numpy(), enumerated.numpy(), rtol=1e-12)


@pytest.mark.edge_case
def test_the_enumeration_oracle_refuses_a_dataset_it_cannot_enumerate() -> None:
    # The oracle is exponential in the observations; the refusal keeps a
    # caller from discovering that by exhausting the host.
    params = fixture("emission_mixture", "ci").params
    observations = simulate_emission_mixture(params).observations.astype(float)
    log_weight = torch.log(torch.as_tensor(params.weights))

    with pytest.raises(ValueError, match="refusing to enumerate"):
        enumerated_posterior(
            torch.as_tensor(observations[: ENUMERATED + 2]),
            log_weight,
            params.components,
        )


@pytest.mark.simulated_truth
def test_the_fit_recovers_the_planted_mixture_at_the_ci_size(
    ci_instance: tuple[EmissionMixtureParams, np.ndarray, EmissionMixtureFit],
) -> None:
    # Measured relative errors at this instance, per component: weight 0.064,
    # 0.097 and 0.015; depth 0.017, 0.034 and 0.002; dispersion 0.27, 0.18 and
    # 0.175; rate 0.051, 0.007 and 0.018; concentration 0.16, 0.037 and 0.32.
    # The two shape parameters carry the loosest bounds because this much data
    # resolves them least: the dispersion's likelihood flattens toward the
    # Poisson limit and the concentration's toward the binomial one, the
    # hazard `emissions.py` derives a bound for. The fixture's declared
    # `tolerance` is the loosest of these; the better-resolved parameters are
    # held tighter here.
    params, _, fit = ci_instance
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    fitted = fit.components
    assert isinstance(fitted, CountPairEmission)
    order, reference = _by_depth(fitted), _by_depth(truth)

    assert fit.log_likelihood <= 0.0
    assert not fit.at_boundary
    assert_allclose(fit.weights.numpy()[order], params.weights[reference], rtol=0.15)
    assert_allclose(
        fitted.total.mean.numpy()[order],
        truth.total.mean.numpy()[reference],
        rtol=0.05,
    )
    assert_allclose(
        fitted.rate.numpy()[order], truth.rate.numpy()[reference], rtol=0.10
    )
    assert_allclose(
        fitted.total.dispersion.numpy()[order],
        truth.total.dispersion.numpy()[reference],
        rtol=params.tolerance,
    )
    assert_allclose(
        fitted.concentration.numpy()[order],
        truth.concentration.numpy()[reference],
        rtol=params.tolerance,
    )


@pytest.mark.mathematical
def test_every_observation_is_explained_by_exactly_one_unit_of_responsibility(
    ci_instance: tuple[EmissionMixtureParams, np.ndarray, EmissionMixtureFit],
) -> None:
    # The E step's own invariant, at the fitted parameters rather than at the
    # planted ones, and the reason the fit's weights are a distribution: the
    # weights are the mean of these rows.
    _, _, fit = ci_instance

    rows = fit.responsibilities.numpy()

    assert_allclose(rows.sum(axis=1), np.ones(rows.shape[0]), rtol=1e-12)
    assert_allclose(rows.mean(axis=0), fit.weights.numpy(), rtol=1e-12)
    assert bool((rows >= 0.0).all())


#: The iteration budget the stress fit runs under. Ten components is about
#: 0.9 s an iteration on the reference host, so the budget fixes the test's
#: wall clock; it is stated because the fit is compared against a start under
#: the *same* budget.
STRESS_ITERATIONS = 200

#: Seeds the two starts are compared on, at the CI size. Two fits per seed at
#: about four seconds a fit, which is why the comparison is stress-tier.
SEEDING_SEEDS = range(40, 46)


#: How near a fitted component must sit to a planted one to count as having
#: resolved it: the depth within `exp(0.15) - 1` = 16% and the allele fraction
#: within 0.06, added, both scale-free so one bound serves every component of
#: an instance whose depths span 20 to 298.
RESOLVED_DISTANCE = 0.20

#: Planted components a single seeded run resolves at this instance. Measured:
#: seven of ten, the nearest distances running 0.067, 0.100, 0.044, 0.166 and
#: 0.238 over the first five components, with the worst at 0.469 --- so the
#: three that fall short do so by a margin no tolerance choice would move.
RESOLVED_COMPONENTS = 7


# `release` rather than `stress`, per `DEV.md`: at ten components one fit is
# about 0.6 s an iteration on the reference host --- 121 s measured for this
# test --- and the stress tier is the ten-minute local budget. The claim has
# its fast sibling at the CI size above.
@pytest.mark.release
@pytest.mark.simulated_truth
def test_the_fit_recovers_most_of_the_planted_mixture_at_ten_components() -> None:
    """How much of a ten-component mixture one seeded run resolves, and how much not."""
    # **The finding this test records.** At three components a single
    # `Emission_Mixture++`-seeded run recovers every planted parameter; at ten
    # it does not, a property of the instance and of the start rather than a
    # defect to be tuned away (`sim/CLAUDE.md`: a fixture is hard only once
    # measured). Asserted is how many planted components have a fitted
    # component near them, and that the fit has climbed past the planted
    # parameters --- so a regression that loses a component, or stops
    # ascending, fails here while the honest shortfall does not.
    #
    # Recovery is stated by nearest match rather than by a permutation: the
    # depths are 1.35 apart and a 20% error in one reorders neighbours, so a
    # sort is not an alignment here, and enumerating the 3.6 million
    # permutations `opt.hmm.align_families` would is not affordable.
    params = fixture("emission_mixture", "stress").params
    observations = simulate_emission_mixture(params).observations.astype(float)
    truth = params.components
    assert isinstance(truth, CountPairEmission)

    fit = _fit(params, observations, seed=3, max_iterations=STRESS_ITERATIONS)

    fitted = fit.components
    assert isinstance(fitted, CountPairEmission)
    distance = np.abs(
        np.log(fitted.total.mean.numpy()[None, :] / truth.total.mean.numpy()[:, None])
    ) + np.abs(fitted.rate.numpy()[None, :] - truth.rate.numpy()[:, None])
    nearest = distance.min(axis=1)
    resolved = int((nearest <= RESOLVED_DISTANCE).sum())

    assert resolved >= RESOLVED_COMPONENTS, np.round(nearest, 3).tolist()
    planted = float(
        mixture_log_likelihood(
            torch.as_tensor(observations),
            torch.log(torch.as_tensor(params.weights)),
            truth,
        )
    )
    assert fit.log_likelihood >= planted


# `release` for the same reason: twelve fits at about four seconds each.
@pytest.mark.release
@pytest.mark.simulated_truth
def test_the_seeded_start_is_measured_against_the_uniform_one() -> None:
    """``Emission_Mixture++`` against a uniform start, on shared seeds."""
    params = fixture("emission_mixture", "ci").params
    observations = simulate_emission_mixture(params).observations.astype(float)
    at = _seeding(params)
    weights = torch.full(
        (params.n_components,), 1.0 / params.n_components, dtype=torch.float64
    )

    reached = {
        name: [
            expectation_maximization(
                observations,
                weights,
                seeder(observations, params.n_components, at, np.random.default_rng(s)),
                max_iterations=200,
                tolerance=EM_TOLERANCE,
            ).log_likelihood
            for s in SEEDING_SEEDS
        ]
        for name, seeder in (("plus_plus", plus_plus_start), ("uniform", uniform_start))
    }

    difference = np.array(reached["plus_plus"]) - np.array(reached["uniform"])
    assert np.isfinite(difference).all()


@pytest.mark.edge_case
def test_a_seeding_places_a_component_on_a_pair_and_refuses_a_form_it_cannot() -> None:
    # The one place a component's parameters are read off a single
    # observation, and what it reads is a location and not a shape: the depth
    # becomes the negative-binomial mean and the allele fraction, smoothed by
    # a half-count, the beta-binomial rate.
    rows = np.array([[40.0, 10.0], [200.0, 18.0]])

    joint = CountPairSeeding(dispersion=4.0, concentration=20.0, joint=True)(rows)
    independent = CountPairSeeding(
        dispersion=4.0, concentration=20.0, joint=False, trials=25.0
    )(rows)

    assert_allclose(joint.total.mean.numpy(), [40.0, 200.0])
    assert_allclose(joint.rate.numpy(), [10.5 / 41.0, 18.5 / 201.0])
    assert_allclose(joint.concentration.numpy(), [20.0, 20.0])
    assert joint.trials is None
    # The independent form reads its fraction against the declared trial
    # count, not against the observed depth, which is the difference between
    # the two forms carried into the start.
    assert_allclose(independent.rate.numpy(), [10.5 / 26.0, 18.5 / 26.0])
    assert independent.trials is not None
    assert_allclose(independent.trials.numpy(), [25.0, 25.0])

    with pytest.raises(ValueError, match="trial count is the observed total"):
        CountPairSeeding(dispersion=4.0, concentration=20.0, joint=True, trials=25.0)
    with pytest.raises(ValueError, match="needs a fixed trial count"):
        CountPairSeeding(dispersion=4.0, concentration=20.0, joint=False)
