"""MALA, pinned against HMC where it is exact and against a closed form where it is not.

MALA's proposal is one leapfrog step and its ratio the energy difference, so
the two routes agree to floating point, measured. Then the distribution:
an analytic Gaussian and the 4,096-assignment mixture posterior (issue #749).
Then the correction ablated: unadjusted Langevin's stationary variance has a
closed form that is not the target's. The energy error appears throughout
(`opt/CLAUDE.md`): an uncorrected chain accepts at 1 by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.sample.hmc import (
    Adaptation,
    effective_sample_size,
    sample,
)
from sal.sample.langevin import (
    GRADIENTS_PER_PROPOSAL,
    LANGEVIN_STEPS,
    MALA_TARGET_ACCEPTANCE,
    mala,
)

from tests._objective_checks import AnalyticGaussian
from tests._posteriors import (
    GAUSSIAN,
    assert_recovers_assignment_posterior,
    assert_within_sigmas,
    enumerated_quadrature,
    monte_carlo_sigmas,
    weight_posterior,
)

#: Different arithmetic, same number: over 400 proposals at three steps,
#: 2.7e-15, 1.3e-15, 8.9e-16 on draws and 2.6e-15, 2.7e-15, 1.4e-14 on the
#: energy error; every decision agreed (acceptance 0.985, 0.8425, 0.560).
IDENTITY = 1e-12

#: Standard errors a marginal estimate is allowed from its closed form. Three
#: on the Gaussian, four on the mixture posterior, as issue #756's step 2
#: declared; each is formed from the chain's own spread over its *effective*
#: sample size (`tests/_posteriors.py`).
GAUSSIAN_SIGMAS = 3.0
MIXTURE_SIGMAS = 4.0

#: One unit-variance coordinate: Euler-Maruyama is an AR(1) with a closed-form
#: variance. Steps 1.0 and 1.5: at 0.5 the answers (1.067 against 1.000) are
#: 1.3 standard errors apart (ESS time 15, 6,000 draws); at 1.0, eleven.
ULA_VARIANCE_TRUTH = 1.0
ULA_DRAWS = 6000
ULA_STEPS = (1.0, 1.5)


def ula_stationary_variance(step_size: float, variance: float) -> float:
    """The variance unadjusted Langevin converges to on ``N(0, variance)``.

    AR(1), ``a = 1 - h^2 / (2 s^2)``: ``s^2 / (1 - h^2 / (4 s^2))``, unbounded at ``2s``.
    """
    return variance / (1.0 - step_size * step_size / (4.0 * variance))


@pytest.mark.analytic
@pytest.mark.critical
def test_one_langevin_step_is_the_hamiltonian_transition() -> None:
    # At one leapfrog step HMC's proposal and ratio are MALA's: two
    # implementations (transition densities against the trajectory), one chain.
    for step in (0.3, 0.7, 1.0):
        hamiltonian_route = sample(
            GAUSSIAN,
            rng=torch.Generator().manual_seed(5),
            n_samples=400,
            step_size=step,
            n_steps=LANGEVIN_STEPS,
        )
        langevin_route = mala(
            GAUSSIAN,
            rng=torch.Generator().manual_seed(5),
            n_samples=400,
            step_size=step,
        )

        difference = float((hamiltonian_route.draws - langevin_route.draws).abs().max())
        energy = float(
            (hamiltonian_route.energy_error - langevin_route.energy_error).abs().max()
        )
        assert difference < IDENTITY, (step, difference)
        assert energy < IDENTITY, (step, energy)
        # Exact, not to a tolerance: a proposal either was accepted or was
        # not, and a difference of 1e-15 in the ratio that decided it would
        # have to land inside a uniform's last bits to move one.
        assert langevin_route.acceptance_rate == hamiltonian_route.acceptance_rate, step
        assert langevin_route.spent == hamiltonian_route.spent, step
        assert langevin_route.spent == 400 * GRADIENTS_PER_PROPOSAL


@pytest.mark.oracle
def test_the_langevin_chain_recovers_an_analytic_gaussian() -> None:
    # Closed form, fixed step (no warm-up between claim and failure). Two
    # seeds, in chain errors: mean 0.40, 0.87; variance 0.45, 0.47 (3.0).
    # Acceptance 0.575, 0.559; mean energy error 1.61, 1.57, worst 21.7, 25.1.
    for seed in (11, 12):
        chain = mala(
            GAUSSIAN,
            rng=torch.Generator().manual_seed(seed),
            n_samples=4000,
            step_size=1.0,
            burn_in=400,
        )

        assert_within_sigmas(
            chain.draws,
            GAUSSIAN.mean.numpy(),
            np.diag(GAUSSIAN.covariance.numpy()),
            GAUSSIAN_SIGMAS,
            seed,
        )
        assert chain.corrected
        # The diagnostic `opt/CLAUDE.md` requires beside the acceptance rate:
        # a step too large biases the spread while acceptance looks healthy.
        assert float(chain.energy_error.mean()) < 2.5, seed


@pytest.mark.oracle
@pytest.mark.release
def test_the_langevin_chain_recovers_the_enumerated_assignment_posterior() -> None:
    # The rung below (#734): the enumerated posterior over 4,096 assignments.
    # Two seeds: weight 1.15 and 0.11 errors, worst marginal 1.14 and 0.15
    # (4.0 declared); acceptance 0.793, 0.787.
    target, observations, components = weight_posterior()
    reference = enumerated_quadrature(observations, components)

    for seed in (756, 757):
        chain = mala(
            target,
            rng=torch.Generator().manual_seed(seed),
            n_samples=1500,
            step_size=0.9,
            burn_in=200,
        )
        assert chain.acceptance_rate > 0.6, (seed, chain.acceptance_rate)

        assert_recovers_assignment_posterior(
            chain.draws,
            observations,
            components,
            reference,
            sigmas=MIXTURE_SIGMAS,
            size=float(effective_sample_size(chain.draws)[0]),
            context=seed,
        )


@pytest.mark.analytic
def test_unadjusted_langevin_realizes_the_closed_form_discretization_bias() -> None:
    # ULA converges to its own stationary law, not the target: 0.14 errors
    # from the closed form at step 1.0, 0.98 at 1.5 (2.246 against 2.286;
    # target 1.000).
    target = AnalyticGaussian([0.0], [[ULA_VARIANCE_TRUTH]])
    for step in ULA_STEPS:
        chain = mala(
            target,
            rng=torch.Generator().manual_seed(3),
            n_samples=ULA_DRAWS,
            step_size=step,
            burn_in=500,
            corrected=False,
        )
        closed_form = ula_stationary_variance(step, ULA_VARIANCE_TRUTH)

        _, variance_sigmas = monte_carlo_sigmas(
            chain.draws, np.zeros(1), np.array([closed_form])
        )

        assert not chain.corrected
        # 1 by construction, which is the reason `corrected` is on the result.
        assert chain.acceptance_rate == 1.0
        assert float(variance_sigmas.max()) < GAUSSIAN_SIGMAS, (step, variance_sigmas)


@pytest.mark.analytic
def test_dropping_the_correction_misses_the_variance_that_mala_recovers() -> None:
    # Ablation at `h = s = 1`: uncorrected is 33% overdispersed, 11.09 errors
    # off; corrected 1.45. Acceptance 1.00 against 0.927 sees none of it;
    # mean energy error 0.1835 against 0.1524.
    target = AnalyticGaussian([0.0], [[ULA_VARIANCE_TRUTH]])
    truth = np.array([ULA_VARIANCE_TRUTH])
    uncorrected = mala(
        target,
        rng=torch.Generator().manual_seed(3),
        n_samples=ULA_DRAWS,
        step_size=1.0,
        burn_in=500,
        corrected=False,
    )
    corrected = mala(
        target,
        rng=torch.Generator().manual_seed(3),
        n_samples=ULA_DRAWS,
        step_size=1.0,
        burn_in=500,
    )

    _, missed = monte_carlo_sigmas(uncorrected.draws, np.zeros(1), truth)
    _, recovered = monte_carlo_sigmas(corrected.draws, np.zeros(1), truth)

    assert float(missed.max()) > GAUSSIAN_SIGMAS, missed
    assert float(recovered.max()) < GAUSSIAN_SIGMAS, recovered
    # And the diagnostic that does see it, which is why it is reported.
    assert float(uncorrected.energy_error.mean()) > float(corrected.energy_error.mean())


@pytest.mark.analytic
def test_the_warm_up_adapts_the_step_to_the_langevin_acceptance() -> None:
    # HMC's two windows driven to MALA's target: pooled over four seeds 0.537
    # against 0.574 (0.594, 0.542, 0.472, 0.542).
    adaptation = Adaptation(
        warmup=300, target_acceptance=MALA_TARGET_ACCEPTANCE, step_jitter=0.4
    )
    accepted = []
    for seed in range(4):
        chain = mala(
            GAUSSIAN,
            rng=torch.Generator().manual_seed(seed),
            n_samples=2000,
            step_size=0.5,
            burn_in=200,
            adaptation=adaptation,
        )
        assert chain.adapted is not None
        assert chain.adapted.step_size > 0.0
        assert bool((chain.adapted.mass_diagonal > 0.0).all())
        # The warm-up's gradients are in the bill, so a cost per effective
        # sample is the whole cost and not the recorded part of it.
        assert chain.spent == (2000 + 200 + 300) * GRADIENTS_PER_PROPOSAL
        accepted.append(chain.acceptance_rate)

    assert abs(float(np.mean(accepted)) - MALA_TARGET_ACCEPTANCE) < 0.1, accepted


@pytest.mark.smoke
@pytest.mark.critical
def test_a_langevin_chain_refuses_a_step_or_a_temperature_that_is_not_positive() -> (
    None
):
    # Refused rather than clamped: a zero step proposes the current point,
    # accepts at rate 1 and samples nothing, which looks healthy by every
    # diagnostic on the result.
    with pytest.raises(ValueError, match="step_size must be positive"):
        mala(
            GAUSSIAN,
            rng=torch.Generator().manual_seed(1),
            n_samples=4,
            step_size=0.0,
        )
    with pytest.raises(ValueError, match="temperature must be positive"):
        mala(
            GAUSSIAN,
            rng=torch.Generator().manual_seed(1),
            n_samples=4,
            step_size=0.5,
            temperature=0.0,
        )
