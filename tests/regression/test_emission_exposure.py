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
approximately.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import NegativeBinomialEmission
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
    # `e = 1` is the degenerate case and must not merely agree to a tolerance.
    # The M step needs the constant-exposure mass to factor out exactly to get
    # here: formed as a matmul instead it leaves a one-ULP residue in the mean.
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
    assert torch.equal(fitted.mean, reference.mean)
    assert torch.equal(fitted.dispersion, reference.dispersion)


@pytest.mark.critical
@pytest.mark.simulated_truth
def test_a_varying_exposure_recovers_the_planted_rate_and_dispersion() -> None:
    # The regime the conserved family cannot express, so it is refereed by the
    # parameters that generated the data rather than by an oracle. The exposure
    # spans an eightfold range, which is what makes this a test of the offset
    # rather than of a constant absorbed into the mean.
    rng = np.random.default_rng(SEED)
    offsets = torch.tensor(rng.uniform(0.5, 4.0, 2000), dtype=torch.float64)
    planted_mean, planted_dispersion = 2.5, 3.0
    rate = (offsets * planted_mean).numpy()
    draws = torch.tensor(
        rng.negative_binomial(
            planted_dispersion, planted_dispersion / (planted_dispersion + rate)
        ),
        dtype=torch.float64,
    ).reshape(1, -1)

    start = NegativeBinomialEmission(torch.tensor([1.0]), torch.tensor([1.0]))
    fitted = start.reestimate(
        draws, torch.ones(1, 2000, 1, dtype=torch.float64), offsets.reshape(1, -1)
    ).emissions

    assert float(fitted.mean[0]) == pytest.approx(planted_mean, rel=0.05)
    assert float(fitted.dispersion[0]) == pytest.approx(planted_dispersion, rel=0.10)


@pytest.mark.critical
@pytest.mark.structural
def test_the_per_state_rate_is_what_the_moments_are_stated_at() -> None:
    # `mu_k` is the per-state association the fit targets; the emission's mean
    # is `e_i mu_k`. So `mean` and `variance` keep a value under a varying
    # exposure instead of losing one, stated at the unit-exposure reference.
    live = NegativeBinomialEmission(DISPERSION, MEAN)

    assert torch.equal(live.mean, MEAN)
    assert torch.equal(live.variance, MEAN + MEAN**2 / DISPERSION)


@pytest.mark.critical
@pytest.mark.edge_case
def test_an_exposure_is_positive_and_broadcasts_along_the_states() -> None:
    # The layout rule the trial count already carries, for the same reason, and
    # a support the model has no meaning outside.
    live = NegativeBinomialEmission(DISPERSION, MEAN)
    counts = torch.tensor([1, 2])

    with pytest.raises(ValueError, match="singleton axis"):
        live.log_density(counts, torch.full((2, 2), 1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        live.log_density(counts, torch.tensor([[1.0], [0.0]]))
