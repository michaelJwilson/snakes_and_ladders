"""What a draw costs, in the unit each sampler spends, on one target at a time.

Effective samples per evaluation (`opt/CLAUDE.md`): HMC buys ``n_steps + 1``
gradients per trajectory, MALA two per step, slice sampling objective
evaluations only. The units differ, so wall clock is recorded beside them
(experiment 024). The ordering is asserted over both targets, sizes and seeds;
the numbers are in comments and `STATUS.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.sample.hmc import Adaptation, effective_sample_size, sample
from sal.sample.langevin import MALA_TARGET_ACCEPTANCE, mala
from sal.sample.slice import slice_sample

from tests._posteriors import GAUSSIAN, weight_posterior
from tests._scale import at_scale

#: Each sampler at its own warm-up, since a comparison between a tuned method
#: and an untuned one measures the tuning. HMC and MALA adapt the step and the
#: mass by the same two windows at the acceptance each is optimal at; slice
#: sampling has nothing to adapt, which is its claim.
HMC_ADAPTATION = Adaptation(warmup=300, target_acceptance=0.65, step_jitter=0.4)
MALA_ADAPTATION = Adaptation(
    warmup=300, target_acceptance=MALA_TARGET_ACCEPTANCE, step_jitter=0.4
)


def effective_per_thousand(draws: torch.Tensor, evaluations: int) -> float:
    """Effective samples per 1,000 evaluations, on the worst coordinate."""
    return 1000.0 * float(effective_sample_size(draws).numpy().min()) / evaluations


@pytest.mark.analytic
@pytest.mark.release
@at_scale("draws", ci=(2000, 800), stress=(8000, 3200))
def test_one_gradient_buys_more_from_a_langevin_step_than_from_a_trajectory(
    draws: tuple[int, int],
) -> None:
    # Effective samples per 1,000 evaluations, worst coordinate, two seeds,
    # own warm-up; load 3.90 to 4.32 on a shared 4-core host.
    #
    #                        HMC         MALA        slice      wall (HMC/MALA/slice)
    #   Gaussian, 2,000   25.3 / 22.0  59.1 / 71.8  30.8 / 29.4   4.8 / 1.0 / 0.65 s
    #   Gaussian, 8,000   27.8 / 30.5  53.0 / 54.7  31.4 / 30.3  17.5 / 3.8 / 2.6 s
    #   mixture,    800   19.4 / 20.5  191.8 / 166.5  100.8 / 116.5  4.6 / 1.2 / 0.61 s
    #   mixture,  3,200   38.8 / 30.2  196.2 / 235.0  125.2 / 115.1 15.1 / 3.8 / 2.5 s
    #
    # MALA: 1.7x-2.6x HMC per gradient (Gaussian), 5.1x-7.8x (mixture).
    gaussian_draws, mixture_draws = draws
    target, _, _ = weight_posterior()

    for objective, n_samples, step, trajectory, width in (
        (GAUSSIAN, gaussian_draws, 0.25, 12, 2.0),
        (target, mixture_draws, 0.9, 10, 1.0),
    ):
        burn_in = n_samples // 10
        hamiltonian = sample(
            objective,
            generator=torch.Generator().manual_seed(1),
            n_samples=n_samples,
            step_size=step,
            n_steps=trajectory,
            burn_in=burn_in,
            adaptation=HMC_ADAPTATION,
        )
        langevin = mala(
            objective,
            generator=torch.Generator().manual_seed(1),
            n_samples=n_samples,
            step_size=0.5,
            burn_in=burn_in,
            adaptation=MALA_ADAPTATION,
        )
        sliced = slice_sample(
            objective,
            generator=torch.Generator().manual_seed(1),
            n_samples=n_samples,
            width=width,
            max_steps_out=10,
            burn_in=burn_in,
        )

        per_gradient = effective_per_thousand(
            hamiltonian.draws, hamiltonian.force_evaluations
        )
        langevin_per_gradient = effective_per_thousand(
            langevin.draws, langevin.force_evaluations
        )
        slice_per_evaluation = effective_per_thousand(
            sliced.draws, sliced.objective_evaluations
        )

        assert langevin_per_gradient > per_gradient, (
            n_samples,
            langevin_per_gradient,
            per_gradient,
        )
        # In its own unit, and stated as such: the tuning-free sampler is
        # ahead of the tuned trajectory per evaluation on both targets.
        assert slice_per_evaluation > per_gradient, (
            n_samples,
            slice_per_evaluation,
            per_gradient,
        )
        # Every number above is a chain that mixed, not a chain that stood
        # still: an effective sample size at the draw count would make the
        # ratios meaningless.
        assert np.all(effective_sample_size(hamiltonian.draws).numpy() > 10.0)
