"""The four count-emission HMMs, and the two whose M step is a solve.

Binomial below equidispersion, Poisson exactly at it, negative binomial and
beta-binomial above (issues #229 and #260). The set exists to bracket the
dispersion axis: an interface exercised only by overdispersed families has
never been asked whether it assumes overdispersion somewhere.

Counts are discrete, so ``log P(observations) <= 0`` holds for all four --- the
assertion :mod:`tests.regression.opt.test_opt_hmm_gaussian` had to drop, and
the reason it is worth restoring here rather than leaving implicit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
    PoissonEmission,
    Reestimate,
)
from snakes_and_ladders.likelihood.hmm_paths import enumerate_hidden_paths
from snakes_and_ladders.opt.fit import constrained_standard_errors, covers, fit
from snakes_and_ladders.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
    align_families,
    baum_welch_family,
    forward_log_likelihood_from_density,
)
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences

from tests._objective_checks import assert_gradient_matches_finite_differences

#: The four count families, and the four objectives that fit them. Named as
#: unions rather than as the protocol, so a test may read a moment: the
#: protocol deliberately does not promise one, since a categorical emission has
#: no mean.
CountEmission = (
    PoissonEmission | BinomialEmission | NegativeBinomialEmission | BetaBinomialEmission
)
CountObjective = (
    PoissonHmmObjective
    | BinomialHmmObjective
    | NegativeBinomialHmmObjective
    | BetaBinomialHmmObjective
)

INITIAL = np.array([0.5, 0.5])
TRANSITION = np.array([[0.8, 0.2], [0.2, 0.8]])
TRIALS = np.array([12, 12])

#: Dispersions the negative binomial coverage sweep reports against, spanning
#: the heavy-tailed end and the Poisson limit.
DISPERSIONS = (0.5, 1.0, 2.0, 5.0, 20.0, 100.0)
NB_MEAN = np.array([4.0, 20.0])


def _truth(name: str) -> CountEmission:
    """The two-state fixture for one count family."""
    families: dict[str, CountEmission] = {
        "poisson": PoissonEmission([3.0, 15.0]),
        "binomial": BinomialEmission(TRIALS, [0.2, 0.75]),
        "negative_binomial": NegativeBinomialEmission([2.0, 8.0], [3.0, 20.0]),
        "beta_binomial": BetaBinomialEmission(TRIALS, [2.0, 8.0], [8.0, 2.0]),
    }
    return families[name]


def _start(name: str) -> CountEmission:
    """A symmetry-broken EM start for one count family.

    Broken deliberately: a shared emission leaves the states exchangeable and
    the gradient in that block exactly zero, and an EM run started there
    converges to one state duplicated. That was observed while building this
    module, which is why the starts here are asymmetric and why a test below
    pins the property.
    """
    starts: dict[str, CountEmission] = {
        "poisson": PoissonEmission([2.0, 8.0]),
        "binomial": BinomialEmission(TRIALS, [0.35, 0.6]),
        "negative_binomial": NegativeBinomialEmission([1.0, 1.0], [2.0, 10.0]),
        "beta_binomial": BetaBinomialEmission(TRIALS, [1.0, 3.0], [3.0, 1.0]),
    }
    return starts[name]


def _objective(name: str, observations: np.ndarray) -> CountObjective:
    """The fitting objective for one count family."""
    if name == "poisson":
        return PoissonHmmObjective(observations, 2)
    if name == "binomial":
        return BinomialHmmObjective(observations, 2, TRIALS)
    if name == "negative_binomial":
        return NegativeBinomialHmmObjective(observations, 2)
    return BetaBinomialHmmObjective(observations, 2, TRIALS)


def _params(
    emissions: EmissionFamily, seed: int, n_sequences: int = 60, length: int = 10
) -> HmmParams:
    """A count HMM fixture over ``emissions``."""
    return HmmParams(
        n_states=2,
        sequence_length=length,
        n_sequences=n_sequences,
        initial=INITIAL,
        transition=TRANSITION,
        emissions=emissions,
        seed=seed,
        tolerance=1e-12,
    )


FAMILIES = ("poisson", "binomial", "negative_binomial", "beta_binomial")


@pytest.mark.parametrize("name", FAMILIES)
def test_the_forward_recursion_matches_enumeration_over_every_path(name: str) -> None:
    truth = _truth(name)
    params = _params(truth, seed=101)
    observations = simulate_sequences(params).observations[0][:7]

    enumerated = enumerate_hidden_paths(params, observations)
    recursed = float(
        forward_log_likelihood_from_density(
            truth.log_density(torch.as_tensor(observations)[None, :]),
            torch.log(torch.as_tensor(INITIAL)),
            torch.log(torch.as_tensor(TRANSITION)),
        )
    )

    assert_allclose(recursed, enumerated.log_likelihood, rtol=1e-11)


@pytest.mark.parametrize("name", FAMILIES)
def test_the_evidence_of_a_count_model_is_a_probability(name: str) -> None:
    # The bound the Gaussian case had to give up, restored and asserted where
    # it holds. Pinned per family rather than derived from `is_discrete`, so
    # a family that mislabelled its own support would still be caught.
    truth = _truth(name)
    params = _params(truth, seed=102)
    observations = simulate_sequences(params).observations[0][:7]

    assert truth.is_discrete
    assert enumerate_hidden_paths(params, observations).log_likelihood <= 0.0


@pytest.mark.parametrize("name", FAMILIES)
def test_the_gradient_matches_central_differences(name: str) -> None:
    observations = simulate_sequences(_params(_truth(name), seed=103)).observations
    objective = _objective(name, observations)

    realized = assert_gradient_matches_finite_differences(
        objective, objective.initial(), step=1e-5, rtol=1e-6
    )

    assert realized <= 1e-6


@pytest.mark.parametrize("name", FAMILIES)
def test_the_gradient_fit_and_baum_welch_reach_the_same_optimum(name: str) -> None:
    # Two fitting algorithms sharing only the model. For the negative binomial
    # and the beta-binomial the EM side's M step is a *solve* and the gradient
    # side's is not, so agreement says the solve found the maximum rather than
    # somewhere.
    truth = _truth(name)
    observations = simulate_sequences(_params(truth, seed=104)).observations
    objective = _objective(name, observations)

    result = fit(objective)
    fitted = baum_welch_family(
        observations,
        torch.log(torch.as_tensor(INITIAL)),
        torch.log(torch.as_tensor(TRANSITION)),
        _start(name),
    )

    assert not fitted.emission_at_boundary
    assert_allclose(-float(result.value), fitted.log_likelihood, rtol=1e-8)
    order = list(align_families(objective.emissions(result.theta), fitted.emissions))
    assert order == [0, 1]
    estimate = objective.constrain(result.theta)
    for parameter, value in fitted.emissions.named_parameters().items():
        assert_allclose(estimate[parameter].numpy(), value.numpy(), rtol=1e-3)


@pytest.mark.parametrize("name", FAMILIES)
def test_a_known_truth_round_trips_through_the_unconstrained_coordinates(
    name: str,
) -> None:
    truth = _truth(name)
    observations = simulate_sequences(_params(truth, seed=105)).observations
    objective = _objective(name, observations)
    parameters = truth.named_parameters()

    theta = objective.theta_from_truth(
        INITIAL, TRANSITION, *[value.numpy() for value in parameters.values()]
    )

    estimate = objective.constrain(theta)
    assert_allclose(torch.exp(estimate["log_initial"]).numpy(), INITIAL, rtol=1e-13)
    assert_allclose(
        torch.exp(estimate["log_transition"]).numpy(), TRANSITION, rtol=1e-13
    )
    for parameter, value in parameters.items():
        assert_allclose(estimate[parameter].numpy(), value.numpy(), rtol=1e-13)


@pytest.mark.parametrize("name", FAMILIES)
def test_the_start_places_each_state_on_the_data_and_breaks_the_symmetry(
    name: str,
) -> None:
    observations = simulate_sequences(_params(_truth(name), seed=106)).observations
    objective = _objective(name, observations)

    start = objective.emissions(objective.initial())

    means = start.mean.numpy()
    assert means[0] != means[1]
    assert observations.min() <= means.min()
    assert means.max() <= observations.max()


def test_a_symmetric_start_collapses_the_states_and_the_asymmetric_one_does_not() -> (
    None
):
    # Observed while building this module rather than anticipated: started at
    # `a = b = 1` for both states, beta-binomial EM converges with the two
    # states identical, having never been able to tell them apart. The paired
    # run is what makes it a statement about the start and not about the data.
    truth = _truth("beta_binomial")
    observations = simulate_sequences(_params(truth, seed=107)).observations
    arguments = (
        observations,
        torch.log(torch.as_tensor(INITIAL)),
        torch.log(torch.as_tensor(TRANSITION)),
    )

    symmetric = baum_welch_family(
        *arguments, BetaBinomialEmission(TRIALS, [1.0, 1.0], [1.0, 1.0])
    ).emissions
    asymmetric = baum_welch_family(*arguments, _start("beta_binomial")).emissions
    assert isinstance(symmetric, BetaBinomialEmission)
    assert isinstance(asymmetric, BetaBinomialEmission)

    assert_allclose(symmetric.mean.numpy()[0], symmetric.mean.numpy()[1], rtol=1e-9)
    assert abs(float(asymmetric.mean[0]) - float(asymmetric.mean[1])) > 1.0
    order = list(align_families(asymmetric, truth))
    assert order == [0, 1]
    assert_allclose(asymmetric.mean.numpy(), truth.mean.numpy(), rtol=0.1)


def test_an_m_step_that_did_not_settle_is_refused_rather_than_returned() -> None:
    # `likelihood/CLAUDE.md`: a number read off iterations that never settled
    # is not an estimate, and a caller cannot tell it from one that is. The
    # guard is exercised by starving the solve of iterations, since with its
    # real budget it settles in single digits.
    observations = simulate_sequences(
        _params(_truth("beta_binomial"), seed=108)
    ).observations
    starved = BetaBinomialEmission(TRIALS, [1.0, 3.0], [3.0, 1.0])

    def _one_step(
        self: BetaBinomialEmission, _data: torch.Tensor, _posterior: torch.Tensor
    ) -> Reestimate[BetaBinomialEmission]:
        return Reestimate(self, converged=False, iterations=1, residual=0.5)

    original = BetaBinomialEmission.reestimate
    BetaBinomialEmission.reestimate = _one_step  # type: ignore[method-assign, assignment]
    try:
        with pytest.raises(ValueError, match="did not settle"):
            baum_welch_family(
                observations,
                torch.log(torch.as_tensor(INITIAL)),
                torch.log(torch.as_tensor(TRANSITION)),
                starved,
            )
    finally:
        BetaBinomialEmission.reestimate = original  # type: ignore[method-assign]


def _dispersion_coverage(dispersion: float, replicates: int) -> tuple[int, int, int]:
    """Fit ``replicates`` datasets at one true dispersion and count coverage.

    Returns
    -------
    tuple[int, int, int]
        Intervals covering, intervals checked, and replicates whose observed
        information was too ill-conditioned to invert, which contribute no
        interval and are reported rather than dropped.
    """
    truth = NegativeBinomialEmission(np.array([dispersion, dispersion]), NB_MEAN)
    covered = total = boundary = 0
    for replicate in range(replicates):
        observations = simulate_sequences(
            _params(truth, seed=20260905 + 7919 * replicate, n_sequences=40, length=12)
        ).observations
        objective = NegativeBinomialHmmObjective(observations, 2)
        result = fit(objective)
        estimate = objective.constrain(result.theta)
        try:
            error = constrained_standard_errors(objective, result.theta)
        except ValueError:
            boundary += 1
            continue
        order = list(align_families(objective.emissions(result.theta), truth))
        for name, reference in (
            ("dispersion", truth.dispersion),
            ("mean", truth.mean),
        ):
            hits = covers(estimate[name][order], error[name][order], reference)
            covered += int(hits.sum())
            total += hits.numel()
    return covered, total, boundary


def test_an_interval_stops_existing_at_both_ends_of_the_dispersion_range() -> None:
    # The cheap two-point form of the release sweep below, and the finding it
    # carries: what degrades with the dispersion is not the coverage of the
    # intervals that exist but *whether one exists*, and it degrades at both
    # ends -- a heavy tail at small `r`, a flat likelihood at large.
    # Eight replicates, and the assertion is the *contrast* rather than a
    # rate: the release sweep measures 7 of 16 replicates yielding no interval
    # at a dispersion of 100 against 0 of 16 at 2, so at eight draws the
    # expected count is under four and pinning one would be pinning noise.
    middle = _dispersion_coverage(2.0, replicates=8)
    flat = _dispersion_coverage(100.0, replicates=8)

    assert middle[2] == 0
    assert flat[2] > middle[2]
    assert middle[0] / middle[1] >= 0.85


@pytest.mark.release
def test_coverage_against_the_true_dispersion() -> None:
    # The full sweep behind the table in `STATUS.md`. Marked release: 16
    # replicates at six dispersions is ~80 s, for a measurement that moves
    # only when the model does.
    sweep = {
        dispersion: _dispersion_coverage(dispersion, replicates=16)
        for dispersion in DISPERSIONS
    }

    # Coverage stays near nominal wherever an interval exists at all, which is
    # the half of the result that is *not* obvious: the failure of a flat
    # likelihood shows up as a singular information matrix rather than as an
    # interval in the wrong place.
    for dispersion, (covered, total, _) in sweep.items():
        assert total > 0, dispersion
        assert covered / total >= 0.85, dispersion
    # And the intervals stop existing at both ends, with the middle clean.
    assert sweep[0.5][2] >= 2
    assert sweep[100.0][2] >= 4
    for dispersion in (1.0, 2.0, 5.0, 20.0):
        assert sweep[dispersion][2] == 0, dispersion
