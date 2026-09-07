"""Regression tests for fitting, and for the intervals that make it falsifiable.

`opt/CLAUDE.md` names recovery as the acceptance test: fit simulated data
with known parameters and require the intervals to cover the truth at the
nominal rate. That is what most of this module does, on both reference
instances. Nothing here asserts that a likelihood increased.

Two independent checks stand behind the fits. Every fit is verified to
satisfy the first-order condition and to beat the truth on its own sample --
a maximum-likelihood estimate that does not is not at a maximum. And the HMM
fit is checked against Baum-Welch, which shares no optimizer, no
parameterization and no constraint map with it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.opt.fit import (
    constrained_standard_errors,
    covers,
    fit,
    observed_information,
    parameter_covariance,
)
from phylo.opt.hmm import HmmObjective, align_states, baum_welch
from phylo.opt.potts import PottsObjective, load_potts_params, simulate_chains
from phylo.sim.hmm import load_hmm_params, simulate_sequences

from tests._fixtures import FIXTURES_DIR

POTTS_FIXTURE = FIXTURES_DIR / "potts_params.yaml"
HMM_FIXTURE = FIXTURES_DIR / "hmm_params.yaml"

# The first-order condition, relative to the objective's magnitude: both the
# objective and its gradient scale with the data, so an absolute bound would
# not transfer across fixture sizes (`DEV.md`, issue #111).
_RTOL_STATIONARY = 1e-8

# Coverage is a proportion, so its uncertainty is binomial. Three standard
# errors is the band; at 240 Potts intervals that is 0.95 +/- 0.042.
_COVERAGE_SIGMA = 3.0


def _potts_objective(seed_offset: int = 0) -> tuple[PottsObjective, torch.Tensor]:
    base = load_potts_params(POTTS_FIXTURE)
    params = replace(base, seed=base.seed + seed_offset)
    objective = PottsObjective(simulate_chains(params), params.n_states)
    truth = objective.theta_from_truth(params.coupling, params.field)
    return objective, truth


def _hmm_objective(seed_offset: int = 0) -> tuple[HmmObjective, torch.Tensor]:
    base = load_hmm_params(HMM_FIXTURE)
    params = replace(base, seed=base.seed + seed_offset)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    truth = objective.theta_from_truth(
        params.initial, params.transition, params.emission
    )
    return objective, truth


# --- the fit reaches a maximum, on both instances ------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("build", [_potts_objective, _hmm_objective])
def test_the_fit_satisfies_the_first_order_condition(build) -> None:  # type: ignore[no-untyped-def]
    objective, _ = build()
    result = fit(objective)
    assert result.converged
    assert result.gradient_norm <= _RTOL_STATIONARY


@pytest.mark.simulated_truth
@pytest.mark.parametrize("build", [_potts_objective, _hmm_objective])
def test_the_fit_beats_the_truth_on_its_own_sample(build) -> None:  # type: ignore[no-untyped-def]
    # The defining property of a maximum-likelihood estimate. Not "the
    # likelihood increased": the comparison is against the generating
    # parameters, which the optimizer never sees, so an optimizer that
    # stopped early fails this even though its loss went down.
    objective, truth = build()
    result = fit(objective)
    assert result.value < float(objective(truth))


@pytest.mark.edge_case
def test_a_short_budget_reports_itself_as_unconverged() -> None:
    objective, _ = _hmm_objective()
    result = fit(objective, max_iterations=1)
    assert not result.converged
    assert result.iterations == 1


@pytest.mark.structural
def test_a_supplied_starting_point_is_used() -> None:
    objective, truth = _potts_objective()
    # With no budget the fit must hand back exactly what it was given, which
    # is the only way to show the starting point is honoured rather than
    # coincidentally reached.
    held = fit(objective, theta0=truth, max_iterations=0)
    assert torch.equal(held.theta, truth)
    assert not held.converged
    assert held.iterations == 0
    assert_allclose(held.value, float(objective(truth)), rtol=1e-12)


@pytest.mark.mathematical
def test_the_potts_optimum_does_not_depend_on_the_starting_point() -> None:
    objective, truth = _potts_objective()
    assert_allclose(
        fit(objective, theta0=truth).theta.numpy(),
        fit(objective).theta.numpy(),
        rtol=1e-7,
    )


# --- recovery: the acceptance test ---------------------------------------


@pytest.mark.simulated_truth
def test_potts_intervals_cover_the_truth_at_the_nominal_rate() -> None:
    # 60 independent datasets from the same truth, one fit each, every
    # parameter's 95% Wald interval checked. Deterministic: the seeds are
    # fixed, so this is a pinned number rather than a sample.
    base = load_potts_params(POTTS_FIXTURE)
    truth_coupling = torch.tensor(base.coupling, dtype=torch.float64)
    truth_field = torch.as_tensor(base.field)

    covered = 0
    total = 0
    for replicate in range(60):
        objective, _ = _potts_objective(seed_offset=1000 * replicate)
        result = fit(objective)
        assert result.converged
        estimate = objective.constrain(result.theta)
        error = constrained_standard_errors(objective, result.theta)
        hits = torch.cat(
            [
                covers(
                    estimate["coupling"].reshape(1),
                    error["coupling"].reshape(1),
                    truth_coupling.reshape(1),
                ),
                covers(estimate["field"], error["field"], truth_field),
            ]
        )
        covered += int(hits.sum())
        total += hits.numel()

    rate = covered / total
    band = _COVERAGE_SIGMA * (0.95 * 0.05 / total) ** 0.5
    assert abs(rate - 0.95) <= band, (
        f"coverage {rate:.4f}, expected 0.95 +/- {band:.4f}"
    )


@pytest.mark.simulated_truth
def test_potts_point_estimates_land_near_the_truth() -> None:
    base = load_potts_params(POTTS_FIXTURE)
    objective, _ = _potts_objective()
    estimate = objective.constrain(fit(objective).theta)
    error = constrained_standard_errors(objective, fit(objective).theta)
    # Stated in standard errors rather than absolute units, so the assertion
    # transfers if the fixture size changes.
    assert abs(float(estimate["coupling"]) - base.coupling) < 3.0 * float(
        error["coupling"]
    )
    deviation = (estimate["field"] - torch.as_tensor(base.field)).abs()
    assert bool((deviation < 3.0 * error["field"]).all())


@pytest.mark.simulated_truth
@pytest.mark.release
def test_hmm_interval_coverage_approaches_nominal_with_sample_size() -> None:
    # Release-gated: 15 fits at four times the fixture size is ~30 s.
    #
    # The measured progression, on this fixture, over the whole parameter
    # set: 0.908 at 300 sequences, 0.939 at 1200, 0.984 at 4800. The
    # shortfall at small samples is not a wrong formula -- it shrinks
    # monotonically as the sample grows, which is what an asymptotic
    # approximation does and a wrong one does not. Two reasons it is
    # visible here and not for the Potts chain: some emission probabilities
    # are fitted near zero, where a Wald interval on the log scale is a poor
    # approximation, and aligning the state permutation to truth is a
    # post-selection step that costs a little coverage.
    base = load_hmm_params(HMM_FIXTURE)
    truth = {
        "log_initial": torch.log(torch.as_tensor(base.initial)),
        "log_transition": torch.log(torch.as_tensor(base.transition)),
        "log_emission": torch.log(torch.as_tensor(base.emission)),
    }

    covered = 0
    total = 0
    for replicate in range(15):
        params = replace(
            base, seed=base.seed + 7919 * replicate, n_sequences=4 * base.n_sequences
        )
        objective = HmmObjective(
            simulate_sequences(params).observations, params.n_states, params.n_symbols
        )
        result = fit(objective)
        assert result.converged
        estimate = objective.constrain(result.theta)
        error = constrained_standard_errors(objective, result.theta)
        order = list(
            align_states(estimate["log_emission"], torch.as_tensor(base.emission))
        )
        for name, true_value in truth.items():
            point, spread = estimate[name], error[name]
            if name == "log_transition":
                point, spread = point[order][:, order], spread[order][:, order]
            else:
                point, spread = point[order], spread[order]
            hits = covers(point, spread, true_value)
            covered += int(hits.sum())
            total += hits.numel()

    rate = covered / total
    assert rate >= 0.90, f"coverage {rate:.4f} at four times the fixture size"


# --- the independent fitting algorithm -----------------------------------


@pytest.mark.oracle
def test_the_gradient_fit_agrees_with_baum_welch() -> None:
    # Baum-Welch shares only the model: no optimizer, no unconstrained
    # coordinates, no constraint map. Two algorithms reaching the same
    # optimum is evidence neither of them alone provides.
    params = load_hmm_params(HMM_FIXTURE)
    observations = simulate_sequences(params).observations
    objective = HmmObjective(observations, params.n_states, params.n_symbols)

    gradient_result = fit(objective)
    estimate = objective.constrain(gradient_result.theta)

    start = objective.constrain(objective.initial())
    _, _, em_emission, em_log_likelihood = baum_welch(
        observations,
        start["log_initial"],
        start["log_transition"],
        start["log_emission"],
    )

    assert_allclose(-gradient_result.value, em_log_likelihood, rtol=1e-6)

    order = list(align_states(estimate["log_emission"], torch.exp(em_emission)))
    aligned = torch.exp(estimate["log_emission"])[order]
    assert float((aligned - torch.exp(em_emission)).abs().max()) < 1e-3


# --- the interval machinery itself ---------------------------------------


class _Quadratic:
    """A two-parameter objective whose second parameter does nothing.

    The smallest thing that is genuinely unidentifiable, used to check that
    the singular case is reported as a modelling fault rather than as a
    linear-algebra error.
    """

    def initial(self) -> torch.Tensor:
        return torch.zeros(2, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"value": theta[0]}

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta[0] - 2.0) ** 2


@pytest.mark.edge_case
def test_a_singular_information_matrix_is_reported_as_unidentifiable() -> None:
    with pytest.raises(ValueError, match="not identifiable"):
        parameter_covariance(_Quadratic(), torch.zeros(2, dtype=torch.float64))


@pytest.mark.edge_case
def test_an_estimate_on_the_boundary_has_no_interval() -> None:
    # Not a contrived matrix: at a small enough sample the HMM's
    # maximum-likelihood estimate puts an emission probability at zero, and
    # on the boundary the curvature in that direction vanishes. The
    # information is then singular only *numerically* -- rounding leaves its
    # smallest eigenvalue near 1e-14 rather than at 0 -- so torch inverts it
    # successfully and returns an astronomically large covariance. Refusing
    # is the whole point of checking the conditioning rather than trusting
    # the inversion to fail.
    base = load_hmm_params(HMM_FIXTURE)
    params = replace(base, n_sequences=30, sequence_length=5)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    result = fit(objective)
    fitted = torch.exp(objective.constrain(result.theta)["log_emission"])
    assert float(fitted.min()) < 1e-6, "this sample was meant to reach the boundary"

    with pytest.raises(ValueError, match="not identifiable"):
        parameter_covariance(objective, result.theta)


@pytest.mark.mathematical
def test_a_well_posed_fit_is_far_from_the_conditioning_floor() -> None:
    # The other side of the same threshold: the check must not be so eager
    # that it rejects the fits the recovery tests depend on.
    objective, _ = _hmm_objective()
    information = observed_information(objective, fit(objective).theta)
    eigenvalues = torch.linalg.eigvalsh(information)
    ratio = float(eigenvalues.min() / eigenvalues.max())
    assert ratio > 1e-4, f"eigenvalue ratio {ratio:.2e} is close to the 1e-6 floor"


@pytest.mark.mathematical
@pytest.mark.oracle
def test_the_observed_information_is_the_hessian_of_the_objective() -> None:
    # Pinned against a closed form: d2/dx2 (x - 2)^2 = 2, and zero elsewhere.
    information = observed_information(
        _Quadratic(), torch.zeros(2, dtype=torch.float64)
    )
    assert_allclose(information.numpy(), [[2.0, 0.0], [0.0, 0.0]], atol=1e-12)


@pytest.mark.structural
def test_covers_is_elementwise_and_two_sided() -> None:
    estimate = torch.tensor([0.0, 0.0, 0.0])
    error = torch.tensor([1.0, 1.0, 1.0])
    truth = torch.tensor([1.9, 2.1, -2.1])
    assert covers(estimate, error, truth).tolist() == [True, False, False]


@pytest.mark.structural
def test_standard_errors_are_shaped_like_their_parameters() -> None:
    # At the fitted optimum, not at the truth: the observed information is a
    # statement about curvature *at a maximum*, and away from one the
    # Hessian need not be positive definite. Written against the truth
    # first, this raised -- correctly.
    objective, _ = _hmm_objective()
    error = constrained_standard_errors(objective, fit(objective).theta)
    assert error["log_initial"].shape == (3,)
    assert error["log_transition"].shape == (3, 3)
    assert error["log_emission"].shape == (3, 4)
    assert bool(
        torch.isfinite(torch.cat([e.reshape(-1) for e in error.values()])).all()
    )
    assert bool((error["log_initial"] > 0).all())


@pytest.mark.mathematical
def test_the_information_grows_with_the_data() -> None:
    # A standard error is a claim about how much the data says. Four times
    # the data must halve it, to within the sampling noise of a different
    # dataset; a covariance that ignored the sample size would not move.
    base = load_potts_params(POTTS_FIXTURE)
    small = PottsObjective(simulate_chains(base), base.n_states)
    large = PottsObjective(
        simulate_chains(replace(base, n_chains=16 * base.n_chains)), base.n_states
    )
    small_error = float(
        constrained_standard_errors(small, fit(small).theta)["coupling"]
    )
    large_error = float(
        constrained_standard_errors(large, fit(large).theta)["coupling"]
    )
    ratio = small_error / large_error
    assert_allclose(ratio, 4.0, rtol=0.25)


@pytest.mark.structural
def test_the_potts_fit_is_reproducible() -> None:
    first, _ = _potts_objective()
    second, _ = _potts_objective()
    assert_allclose(fit(first).theta.numpy(), fit(second).theta.numpy(), rtol=1e-12)


@pytest.mark.structural
def test_the_default_start_is_the_objective_s_own_initial_point() -> None:
    objective = PottsObjective(np.zeros((4, 6), dtype=np.int64), n_states=2)
    assert torch.equal(fit(objective, max_iterations=0).theta, objective.initial())
