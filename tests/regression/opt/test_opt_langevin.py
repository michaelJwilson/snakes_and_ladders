"""MALA, pinned against HMC where it is exact and against a closed form where it is not.

Three statements, in that order. The one-step identity is arithmetic: MALA's
proposal *is* one leapfrog step and its Metropolis ratio *is* the energy
difference, so the two routes run on one generator state agree to floating
point and the agreement is measured rather than assumed. Then the
distribution, against an analytic Gaussian and against the mixture posterior
whose evidence is a sum over all 4,096 assignments (issue #749). Then the
correction, ablated: without it the chain is the Langevin diffusion's Euler
discretization, whose stationary variance on a Gaussian has a closed form
that is not the target's.

`opt/CLAUDE.md` is why the energy error appears in every one of them. An
acceptance rate near its target says the step is not absurd and says nothing
about the spread, and for an uncorrected chain it is 1 by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.constrain import log_simplex
from snakes_and_ladders.opt.hmc import (
    Adaptation,
    effective_sample_size,
    sample,
)
from snakes_and_ladders.opt.langevin import (
    GRADIENTS_PER_PROPOSAL,
    LANGEVIN_STEPS,
    MALA_TARGET_ACCEPTANCE,
    mala,
)
from snakes_and_ladders.opt.mixture import responsibilities

from tests._objective_checks import AnalyticGaussian
from tests._posteriors import (
    enumerated_quadrature,
    monte_carlo_sigmas,
    weight_posterior,
)

GAUSSIAN = AnalyticGaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])

#: The tolerance the identity is declared at, and what it realizes. The two
#: routes compute the same number by different arithmetic --- a trajectory's
#: kick-drift-kick against two Gaussian log densities --- so bitwise agreement
#: is not available and the difference is the reordering's. Measured over 400
#: proposals at three step sizes: 2.7e-15, 1.3e-15 and 8.9e-16 on the draws
#: and 2.6e-15, 2.7e-15 and 1.4e-14 on the energy error, against the 1e-12
#: declared here. Every accept/reject decision agreed exactly, at acceptances
#: of 0.985, 0.8425 and 0.560.
IDENTITY = 1e-12

#: Standard errors a marginal estimate is allowed from its closed form. Three
#: on the Gaussian, four on the mixture posterior, as issue #756's step 2
#: declared; each is formed from the chain's own spread over its *effective*
#: sample size (`tests/_posteriors.py`).
GAUSSIAN_SIGMAS = 3.0
MIXTURE_SIGMAS = 4.0

#: The uncorrected chain's target: a unit-variance Gaussian in one coordinate,
#: where the Euler-Maruyama discretization is an AR(1) and its stationary
#: variance is :func:`ula_stationary_variance` exactly. One coordinate because
#: the closed form is written out below, and a matrix version of it would be
#: the test asserting its own algebra.
#:
#: **The steps are 1.0 and 1.5 and not smaller.** At 0.5 the closed form is
#: 1.067 against a target of 1.000, and 6,000 draws of a chain whose
#: integrated autocorrelation time is 15 carry a standard error of 0.05 --- so
#: the two candidate answers are 1.3 standard errors apart and the comparison
#: decides nothing. At 1.0 they are 1.333 against 1.000, eleven of them.
ULA_VARIANCE_TRUTH = 1.0
ULA_DRAWS = 6000
ULA_STEPS = (1.0, 1.5)


def ula_stationary_variance(step_size: float, variance: float) -> float:
    """The variance unadjusted Langevin converges to on ``N(0, variance)``.

    The proposal ``x' = x - (h^2 / 2) x / s^2 + h z`` is an AR(1) with
    coefficient ``a = 1 - h^2 / (2 s^2)`` and innovation variance ``h^2``, so
    its stationary variance is ``h^2 / (1 - a^2) = s^2 / (1 - h^2 / (4 s^2))``
    --- above the target for every step, and unbounded as ``h`` approaches
    ``2 s``, where the recursion stops being stable at all. Written here so
    the test asserts the closed form rather than a recorded number.
    """
    return variance / (1.0 - step_size * step_size / (4.0 * variance))


@pytest.mark.analytic
@pytest.mark.critical
def test_one_langevin_step_is_the_hamiltonian_transition() -> None:
    # The identity the module rests on: at one leapfrog step HMC's proposal is
    # the Langevin proposal and its acceptance ratio is MALA's, so the two
    # routes on one generator state are one chain. This module writes the
    # transition densities and `hmc` writes the trajectory, so the comparison
    # is between two implementations rather than of one against a copy.
    for step in (0.3, 0.7, 1.0):
        hamiltonian_route = sample(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(5),
            n_samples=400,
            step_size=step,
            n_steps=LANGEVIN_STEPS,
        )
        langevin_route = mala(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(5),
            n_samples=400,
            step_size=step,
        )

        difference = float((hamiltonian_route.theta - langevin_route.theta).abs().max())
        energy = float(
            (hamiltonian_route.energy_error - langevin_route.energy_error).abs().max()
        )
        assert difference < IDENTITY, (step, difference)
        assert energy < IDENTITY, (step, energy)
        # Exact, not to a tolerance: a proposal either was accepted or was
        # not, and a difference of 1e-15 in the ratio that decided it would
        # have to land inside a uniform's last bits to move one.
        assert langevin_route.acceptance_rate == hamiltonian_route.acceptance_rate, step
        assert (
            langevin_route.force_evaluations == hamiltonian_route.force_evaluations
        ), step
        assert langevin_route.force_evaluations == 400 * GRADIENTS_PER_PROPOSAL


@pytest.mark.oracle
def test_the_langevin_chain_recovers_an_analytic_gaussian() -> None:
    # Mean and covariance in closed form, so nothing rests on a second
    # sampler. Two seeds, at a fixed step rather than an adapted one: what is
    # being asserted is the kernel's stationary distribution, and a warm-up
    # would put a second thing between the claim and the failure.
    #
    # Realized over the two seeds, in the chain's own standard errors: mean
    # 0.40 and 0.87 worst, variance 0.45 and 0.47 worst, against the 3.0
    # declared. Acceptance 0.575 and 0.559 -- the Langevin optimum, reached
    # at this step without a warm-up -- and the mean energy error 1.61 and
    # 1.57 against a worst proposal of 21.7 and 25.1, which is what a
    # single-step method at its optimal acceptance looks like and what the
    # acceptance rate alone would not have said.
    for seed in (11, 12):
        chain = mala(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(seed),
            n_samples=4000,
            step_size=1.0,
            burn_in=400,
        )

        mean_sigmas, variance_sigmas = monte_carlo_sigmas(
            chain.theta,
            GAUSSIAN.mean.numpy(),
            np.diag(GAUSSIAN.covariance.numpy()),
        )

        assert chain.corrected
        assert float(mean_sigmas.max()) < GAUSSIAN_SIGMAS, (seed, mean_sigmas)
        assert float(variance_sigmas.max()) < GAUSSIAN_SIGMAS, (seed, variance_sigmas)
        # The diagnostic `opt/CLAUDE.md` requires beside the acceptance rate:
        # a step too large biases the spread while acceptance looks healthy.
        assert float(chain.energy_error.mean()) < 2.5, seed


@pytest.mark.oracle
@pytest.mark.release
def test_the_langevin_chain_recovers_the_enumerated_assignment_posterior() -> None:
    # The rung below (issue #734): assignment enumeration. Every number the
    # chain is judged against is a sum over all 4,096 whole assignments,
    # integrated over the one free coordinate; the responsibilities come from
    # the factorized E step and are compared against the enumeration's own.
    #
    # Realized over the two seeds: the posterior mean weight 1.15 and 0.11
    # standard errors away and the worst marginal 1.14 and 0.15, against the
    # 4.0 declared, at acceptances of 0.793 and 0.787.
    target, observations, components = weight_posterior()
    quadrature_weight, quadrature_marginal = enumerated_quadrature(
        observations, components
    )

    for seed in (756, 757):
        chain = mala(
            target,
            generator=torch.Generator().manual_seed(seed),
            n_samples=1500,
            step_size=0.9,
            burn_in=200,
        )
        assert chain.acceptance_rate > 0.6, (seed, chain.acceptance_rate)

        log_weights = log_simplex(chain.theta)
        drawn_weight = torch.exp(log_weights)[:, 0].numpy()
        drawn_marginal = np.stack(
            [
                responsibilities(
                    torch.as_tensor(observations, dtype=torch.float64), row, components
                ).numpy()[:, 0]
                for row in log_weights
            ]
        )
        size = float(effective_sample_size(chain.theta)[0])
        weight_tolerance = MIXTURE_SIGMAS * float(drawn_weight.std()) / np.sqrt(size)
        marginal_tolerance = (
            MIXTURE_SIGMAS * float(drawn_marginal.std(axis=0).max()) / np.sqrt(size)
        )

        assert abs(drawn_weight.mean() - quadrature_weight) < weight_tolerance, (
            seed,
            drawn_weight.mean(),
            quadrature_weight,
        )
        assert (
            np.abs(drawn_marginal.mean(axis=0) - quadrature_marginal).max()
            < marginal_tolerance
        ), (seed, np.abs(drawn_marginal.mean(axis=0) - quadrature_marginal).max())


@pytest.mark.analytic
def test_unadjusted_langevin_realizes_the_closed_form_discretization_bias() -> None:
    # ULA converges, and not to the target: the discretization has its own
    # stationary distribution and `ula_stationary_variance` writes it out.
    # Asserting the chain against *that* rather than against the target is
    # what makes this a statement about the discretization instead of a
    # restatement of the ablation below. Realized: 0.14 standard errors from
    # the closed form at a step of 1.0 and 0.98 at 1.5, where the chain's
    # variance is 2.246 against a closed form of 2.286 and a target of 1.000.
    target = AnalyticGaussian([0.0], [[ULA_VARIANCE_TRUTH]])
    for step in ULA_STEPS:
        chain = mala(
            target,
            generator=torch.Generator().manual_seed(3),
            n_samples=ULA_DRAWS,
            step_size=step,
            burn_in=500,
            corrected=False,
        )
        closed_form = ula_stationary_variance(step, ULA_VARIANCE_TRUTH)

        _, variance_sigmas = monte_carlo_sigmas(
            chain.theta, np.zeros(1), np.array([closed_form])
        )

        assert not chain.corrected
        # 1 by construction, which is the reason `corrected` is on the result.
        assert chain.acceptance_rate == 1.0
        assert float(variance_sigmas.max()) < GAUSSIAN_SIGMAS, (step, variance_sigmas)


@pytest.mark.analytic
def test_dropping_the_correction_misses_the_variance_that_mala_recovers() -> None:
    # The ablation. At a step the size of the target's standard deviation the
    # uncorrected chain is 33% overdispersed --- `1 / (1 - h^2 / 4)` at
    # `h = s = 1` --- and that is many standard errors, while the same step
    # with the correction lands inside three. The acceptance rate sees none of
    # it: the uncorrected chain reads 1.00 and the corrected one 0.927.
    # Realized: the uncorrected variance is 11.09 standard errors from the
    # truth and the corrected one 1.45, and the mean energy error 0.1835
    # against 0.1524.
    target = AnalyticGaussian([0.0], [[ULA_VARIANCE_TRUTH]])
    truth = np.array([ULA_VARIANCE_TRUTH])
    uncorrected = mala(
        target,
        generator=torch.Generator().manual_seed(3),
        n_samples=ULA_DRAWS,
        step_size=1.0,
        burn_in=500,
        corrected=False,
    )
    corrected = mala(
        target,
        generator=torch.Generator().manual_seed(3),
        n_samples=ULA_DRAWS,
        step_size=1.0,
        burn_in=500,
    )

    _, missed = monte_carlo_sigmas(uncorrected.theta, np.zeros(1), truth)
    _, recovered = monte_carlo_sigmas(corrected.theta, np.zeros(1), truth)

    assert float(missed.max()) > GAUSSIAN_SIGMAS, missed
    assert float(recovered.max()) < GAUSSIAN_SIGMAS, recovered
    # And the diagnostic that does see it, which is why it is reported.
    assert float(uncorrected.energy_error.mean()) > float(corrected.energy_error.mean())


@pytest.mark.analytic
def test_the_warm_up_adapts_the_step_to_the_langevin_acceptance() -> None:
    # The same two windows HMC's warm-up runs, driven to MALA's target rather
    # than HMC's. What is pinned is what the adaptation achieves on the drawn
    # chain, not the constants behind it: pooled over four seeds the drawn
    # acceptance is 0.537 against the 0.574 asked for --- 0.594, 0.542, 0.472
    # and 0.542 by seed --- and every chain reports the step and the mass it
    # ran at.
    adaptation = Adaptation(
        warmup=300, target_acceptance=MALA_TARGET_ACCEPTANCE, step_jitter=0.4
    )
    accepted = []
    for seed in range(4):
        chain = mala(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(seed),
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
        assert chain.force_evaluations == (2000 + 200 + 300) * GRADIENTS_PER_PROPOSAL
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
            generator=torch.Generator().manual_seed(1),
            n_samples=4,
            step_size=0.0,
        )
    with pytest.raises(ValueError, match="temperature must be positive"):
        mala(
            GAUSSIAN,
            generator=torch.Generator().manual_seed(1),
            n_samples=4,
            step_size=0.5,
            temperature=0.0,
        )
