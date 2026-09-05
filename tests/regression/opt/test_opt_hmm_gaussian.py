"""The HMM whose observations are real, and whose likelihood has no maximum.

The categorical instance is checked elsewhere and unchanged. What is pinned
here is what the continuous case brings with it: an evidence that is a
density and so may exceed 1, a fit whose supremum does not exist, and an
identifiable regime that is a *measurement* rather than a fixture choice.

Two independent instruments referee the model, as for the categorical case
and for the same reason. Direct enumeration over all ``k ** T`` hidden paths
shares no recursion with the forward algorithm, and Baum-Welch shares no
optimizer, no parameterization and no constraint map with the gradient fit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.likelihood.hmm_paths import enumerate_hidden_paths
from snakes_and_ladders.opt.fit import (
    constrained_standard_errors,
    covers,
    fit,
)
from snakes_and_ladders.opt.hmm import (
    GaussianHmmObjective,
    align_families,
    baum_welch_family,
    forward_log_likelihood_from_density,
)
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences

from tests._objective_checks import assert_gradient_matches_finite_differences

INITIAL = np.array([0.5, 0.5])
TRANSITION = np.array([[0.75, 0.25], [0.25, 0.75]])
UNIT_SCALE = np.array([1.0, 1.0])

#: A floor far below any variance these fixtures reach, so it is never the
#: thing under test except where a test says it is.
INERT_FLOOR = 1e-12

#: Separations, in units of the common emission standard deviation, that the
#: coverage sweep reports against. Chosen to bracket the boundary rather than
#: to sit on one side of it.
SEPARATIONS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)


def _truth(separation: float = 4.0) -> GaussianEmission:
    """Two states, symmetric about zero, ``separation`` standard deviations apart."""
    return GaussianEmission(
        np.array([-separation / 2.0, separation / 2.0]), UNIT_SCALE, INERT_FLOOR
    )


def _params(
    emissions: GaussianEmission,
    seed: int,
    n_sequences: int = 30,
    length: int = 8,
) -> HmmParams:
    """A Gaussian HMM fixture over ``emissions``."""
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


def _coverage(separation: float, replicates: int) -> tuple[int, int, int]:
    """Fit ``replicates`` datasets at one separation and count covering intervals.

    Returns
    -------
    tuple[int, int, int]
        Intervals covering, intervals checked, and replicates whose observed
        information was too ill-conditioned to invert, which contribute no
        interval and are reported rather than dropped.
    """
    truth = _truth(separation)
    covered = total = boundary = 0
    for replicate in range(replicates):
        observations = simulate_sequences(
            _params(truth, seed=20260905 + 7919 * replicate)
        ).observations
        objective = GaussianHmmObjective(observations, 2)
        result = fit(objective)
        estimate = objective.constrain(result.theta)
        try:
            error = constrained_standard_errors(objective, result.theta)
        except ValueError:
            boundary += 1
            continue
        order = list(align_families(objective.emissions(result.theta), truth))
        for name, reference in (("mean", truth.mean), ("scale", truth.scale)):
            hits = covers(estimate[name][order], error[name][order], reference)
            covered += int(hits.sum())
            total += hits.numel()
    return covered, total, boundary


@pytest.mark.parametrize(
    ("n_states", "length", "seed"), [(2, 5, 1), (2, 7, 2), (3, 4, 3), (3, 6, 4)]
)
def test_the_forward_recursion_matches_enumeration_over_every_path(
    n_states: int, length: int, seed: int
) -> None:
    # The forward algorithm against a sum over all `k ** T` paths, which
    # shares no recursion with it. The generalization the continuous case
    # needed is that the per-site term is a density rather than a table
    # lookup; nothing else about the enumeration changes.
    rng = np.random.default_rng(seed)
    truth = GaussianEmission(
        rng.normal(scale=2.0, size=n_states),
        rng.uniform(0.5, 1.5, size=n_states),
        INERT_FLOOR,
    )
    params = HmmParams(
        n_states=n_states,
        sequence_length=length,
        n_sequences=1,
        initial=rng.dirichlet(np.ones(n_states)),
        transition=rng.dirichlet(np.ones(n_states), size=n_states),
        emissions=truth,
        seed=seed,
        tolerance=1e-12,
    )
    observations = simulate_sequences(params).observations

    enumerated = enumerate_hidden_paths(params, observations[0])
    recursed = float(
        forward_log_likelihood_from_density(
            truth.log_density(torch.as_tensor(observations)),
            torch.log(torch.as_tensor(params.initial)),
            torch.log(torch.as_tensor(params.transition)),
        )
    )

    assert_allclose(recursed, enumerated.log_likelihood, rtol=1e-11)


def test_the_evidence_of_a_continuous_emission_can_exceed_one() -> None:
    # The assertion the categorical case could make and this one cannot. A
    # narrow state sitting on its observations makes the evidence a density
    # above 1, so `log P(observations) <= 0` fails on correct code -- which is
    # why nothing in the enumeration or the recursion asserts it.
    truth = GaussianEmission(np.array([0.0, 10.0]), np.array([0.02, 1.0]), INERT_FLOOR)
    params = HmmParams(
        n_states=2,
        sequence_length=4,
        n_sequences=1,
        initial=np.array([1.0 - 1e-12, 1e-12]),
        transition=np.array([[1.0 - 1e-12, 1e-12], [1e-12, 1.0 - 1e-12]]),
        emissions=truth,
        seed=5,
        tolerance=1e-12,
    )
    observations = simulate_sequences(params).observations

    enumerated = enumerate_hidden_paths(params, observations[0])

    assert enumerated.log_likelihood > 0.0


def test_the_gradient_matches_central_differences() -> None:
    observations = simulate_sequences(_params(_truth(), seed=21)).observations
    objective = GaussianHmmObjective(observations, 2)

    realized = assert_gradient_matches_finite_differences(
        objective, objective.initial(), step=1e-5, rtol=1e-6
    )

    assert realized <= 1e-6


def test_the_gradient_fit_and_baum_welch_reach_the_same_optimum() -> None:
    # Two fitting algorithms sharing only the model: one is L-BFGS in
    # unconstrained coordinates through a constraint map, the other is EM
    # working directly in the parameters. Agreement is evidence; a single
    # algorithm agreeing with itself would not be.
    truth = _truth()
    observations = simulate_sequences(_params(truth, seed=22)).observations
    objective = GaussianHmmObjective(observations, 2)

    result = fit(objective)
    estimate = objective.constrain(result.theta)
    em = baum_welch_family(
        observations,
        torch.log(torch.as_tensor(INITIAL)),
        torch.log(torch.as_tensor(TRANSITION)),
        GaussianEmission(
            np.array([-1.0, 1.0]),
            UNIT_SCALE,
            GaussianHmmObjective(observations, 2).variance_floor,
        ),
    )

    assert_allclose(-float(result.value), em.log_likelihood, rtol=1e-9)
    assert not em.emission_at_boundary
    parameters = em.emissions.named_parameters()
    assert_allclose(estimate["mean"].numpy(), parameters["mean"].numpy(), atol=1e-5)
    assert_allclose(estimate["scale"].numpy(), parameters["scale"].numpy(), atol=1e-5)


def test_the_alignment_recovers_a_known_permutation_of_the_states() -> None:
    # Label switching is unidentifiable, so a recovery comparison is stated up
    # to a permutation and the aligner has to find it. For a Gaussian family
    # the discriminating signal is the mean, not an emission matrix -- there
    # is no matrix -- which is why the signature lives on the family.
    truth = _truth(separation=6.0)
    permuted = GaussianEmission(
        truth.mean.numpy()[[1, 0]], truth.scale.numpy()[[1, 0]], INERT_FLOOR
    )

    assert align_families(permuted, truth) == (1, 0)
    assert align_families(truth, truth) == (0, 1)


def test_the_start_places_the_means_on_the_data_and_breaks_the_symmetry() -> None:
    # A shared mean would leave the states exchangeable and the gradient in
    # that block exactly zero, which is the failure `opt/CLAUDE.md` names. A
    # mean far from every observation is the other failure: the state's
    # density underflows, it is invisible to the E step, and the fit silently
    # becomes one with fewer states.
    observations = simulate_sequences(_params(_truth(), seed=23)).observations
    objective = GaussianHmmObjective(observations, 2)

    start = objective.constrain(objective.initial())

    means = start["mean"].numpy()
    assert means[0] != means[1]
    assert observations.min() <= means.min()
    assert means.max() <= observations.max()
    assert_allclose(torch.exp(start["log_initial"]).numpy(), [0.5, 0.5], rtol=1e-14)


def test_a_known_truth_round_trips_through_the_unconstrained_coordinates() -> None:
    truth = _truth()
    observations = simulate_sequences(_params(truth, seed=24)).observations
    objective = GaussianHmmObjective(observations, 2)

    estimate = objective.constrain(
        objective.theta_from_truth(
            INITIAL, TRANSITION, truth.mean.numpy(), truth.scale.numpy()
        )
    )

    assert_allclose(torch.exp(estimate["log_initial"]).numpy(), INITIAL, rtol=1e-13)
    assert_allclose(
        torch.exp(estimate["log_transition"]).numpy(), TRANSITION, rtol=1e-13
    )
    assert_allclose(estimate["mean"].numpy(), truth.mean.numpy(), rtol=1e-13)
    assert_allclose(estimate["scale"].numpy(), truth.scale.numpy(), rtol=1e-13)


def test_a_collapsing_fit_is_refused_rather_than_returned() -> None:
    # Started with one state's mean on a single observation and a scale far
    # below the floor, EM drives that state's variance down rather than up.
    # The refusal is the deliverable: a run that clamped and returned would
    # report a converged fit at a point where the likelihood has no maximum.
    observations = simulate_sequences(_params(_truth(), seed=25)).observations
    floor = GaussianHmmObjective(observations, 2).variance_floor

    with pytest.raises(ValueError, match="unbounded as a variance goes to zero"):
        baum_welch_family(
            observations,
            torch.log(torch.as_tensor(INITIAL)),
            torch.log(torch.as_tensor(TRANSITION)),
            GaussianEmission(
                np.array([float(observations[0, 0]), 0.0]),
                np.array([np.sqrt(floor) / 100.0, 1.0]),
                floor,
            ),
        )


def test_coverage_reaches_nominal_only_where_the_states_are_separated() -> None:
    # The identifiable regime, measured rather than assumed. At half a
    # standard deviation of separation the two states are nearly the same
    # state: most replicates produce an information matrix too ill-conditioned
    # to invert, so there is no interval at all, and the intervals that do
    # exist under-cover. The cheap two-point form of the sweep the release
    # test tabulates.
    close_covered, close_total, close_boundary = _coverage(0.5, replicates=8)
    far_covered, far_total, far_boundary = _coverage(6.0, replicates=8)

    assert close_boundary >= 4
    assert far_boundary == 0
    assert close_covered / close_total < 0.85
    assert far_covered / far_total >= 0.90
    assert far_total > close_total


@pytest.mark.release
def test_coverage_against_the_separation_of_the_emitting_states() -> None:
    # The full sweep behind the table in `STATUS.md`. Marked release: 24
    # replicates at six separations is ~16 s, which is not a per-pull-request
    # cost for a measurement that moves only when the model does.
    sweep = {
        separation: _coverage(separation, replicates=24) for separation in SEPARATIONS
    }

    rates = {
        separation: covered / total
        for separation, (covered, total, _) in sweep.items()
        if total > 0
    }
    # Below two standard deviations the model is not identifiable from this
    # much data and coverage is visibly below nominal; from two upward it is
    # at nominal within the binomial error of ~0.025 on 96 intervals.
    assert rates[0.5] < 0.75
    assert rates[1.0] < 0.90
    for separation in (2.0, 3.0, 4.0, 6.0):
        assert rates[separation] >= 0.90
    # And the replicates that yield no interval at all vanish over the same
    # range, which is the same boundary seen from the other side.
    assert sweep[0.5][2] >= 12
    assert sweep[1.0][2] >= 4
    for separation in (3.0, 4.0, 6.0):
        assert sweep[separation][2] == 0
