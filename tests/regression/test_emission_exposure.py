"""An exposure per observation, and the per-state rate that survives it (#631).

Step 4. ``mu_k`` stays the state's own rate --- the per-state association the
fit targets and recovery checks --- and the *emission's* mean at observation
``i`` in state ``k`` is ``e_i mu_k``. So :attr:`NegativeBinomialEmission.mean`
keeps a value under a varying exposure rather than losing one: the exposure is
conditioned on and never fitted, and unit exposure is the reference the family
declares itself at.

What makes the conserved family a referee here is an identity rather than a
tolerance. Where ``e_i`` is constant the two models **are** the same model,
absorbing the exposure as ``log mu - log(c)``, so the covariate-aware family at
constant ``e`` is the conserved family at ``mu e`` --- checked bitwise, not
approximately. Recovery under a varying exposure, where no oracle exists, is
``test_emission_covariate_recovery.py``'s.
"""

from __future__ import annotations

import pytest
import torch
from snakes_and_ladders.emissions import NegativeBinomialEmission
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.sandbox.count_emissions import (
    NegativeBinomialEmission as Conserved,
)

DISPERSION = torch.tensor([2.0, 5.0], dtype=torch.float64)
MEAN = torch.tensor([1.0, 4.0], dtype=torch.float64)
SEED = 20260916


@pytest.fixture
def observations() -> tuple[torch.Tensor, torch.Tensor]:
    """Counts and a normalized posterior, from one seeded generator."""
    generator = torch.Generator().manual_seed(SEED)
    counts = torch.randint(0, 15, (1, 300), generator=generator).to(torch.float64)
    posterior = torch.rand(1, 300, 2, generator=generator, dtype=torch.float64)
    return counts, posterior / posterior.sum(dim=-1, keepdim=True)


@pytest.mark.critical
@pytest.mark.oracle
def test_a_constant_exposure_is_the_conserved_family_at_a_rescaled_mean() -> None:
    # The identity the whole step rests on, and the reason a conserved family
    # referees a covariate it cannot itself express: at constant `c` the two
    # are the same model, so this is an equality and not a tolerance.
    counts = torch.tensor([0, 1, 3, 9])
    live = NegativeBinomialEmission(DISPERSION, MEAN)
    constant = torch.full((4, 1), 3.0, dtype=torch.float64)

    assert torch.equal(
        live.log_density(counts, constant),
        Conserved(DISPERSION, MEAN * 3.0).log_density(counts),
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_unit_exposure_changes_nothing_in_scoring_or_the_m_step(
    observations: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # `e = 1` is the degenerate case. **Scoring is bitwise** and stays the
    # strongest claim available. The M step backs off to the repository's
    # declared float64 tolerance, because its profiled mean is now a matmul --
    # 13.72x faster at the declared scale and 80.7 MB of temporary lighter --
    # and a matmul reduces in a different order. Measured 2.7e-16 relative,
    # five orders inside the 1e-11 declared; root `CLAUDE.md` permits that
    # trade and forbids loosening a bound to admit a result, so the bound here
    # is the declared one and not one fitted to this measurement (#649).
    counts, posterior = observations
    live = NegativeBinomialEmission(DISPERSION, MEAN)
    conserved = Conserved(DISPERSION, MEAN)
    ones = torch.ones_like(counts)

    reference = conserved.reestimate(counts, posterior).emissions
    fitted = live.reestimate(counts, posterior, ones).emissions

    assert torch.equal(
        live.log_density(counts[0], torch.ones(300, 1, dtype=torch.float64)),
        conserved.log_density(counts[0]),
    )
    assert torch.allclose(fitted.mean, reference.mean, rtol=CROSS_DEVICE_RTOL_FLOAT64)
    assert torch.allclose(
        fitted.dispersion, reference.dispersion, rtol=CROSS_DEVICE_RTOL_FLOAT64
    )


@pytest.mark.critical
@pytest.mark.infra
def test_the_per_state_rate_is_what_the_moments_are_stated_at() -> None:
    # `mu_k` is the per-state association the fit targets; the emission's mean
    # is `e_i mu_k`. So `mean` and `variance` keep a value under a varying
    # exposure instead of losing one, stated at the unit-exposure reference.
    live = NegativeBinomialEmission(DISPERSION, MEAN)

    assert torch.equal(live.mean, MEAN)
    assert torch.equal(live.variance, MEAN + MEAN**2 / DISPERSION)


@pytest.mark.critical
@pytest.mark.smoke
def test_an_exposure_is_non_negative_and_broadcasts_along_the_states() -> None:
    # The layout rule the trial count already carries, for the same reason, and
    # a support the model has no meaning outside.
    live = NegativeBinomialEmission(DISPERSION, MEAN)
    counts = torch.tensor([1, 2])

    with pytest.raises(ValueError, match="singleton axis"):
        live.log_density(counts, torch.full((2, 2), 1.0))
    # Zero is admitted since #933: it marks the total unobserved.
    with pytest.raises(ValueError, match="non-negative"):
        live.log_density(counts, torch.tensor([[1.0], [-1.0]]))
