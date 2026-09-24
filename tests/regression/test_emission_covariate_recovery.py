"""Planted parameters recovered with the covariates varying (issue #631).

Each family is fitted back to its generating values while its covariate varies
--- an exposure over eightfold, a trial count over sixteenfold --- the regime
no conserved family expresses. Refereed by the simulated truth: no independent
covariate-aware implementation exists. The exact half is at a constant
covariate, bitwise, in ``test_emission_exposure.py`` and
``test_emission_trial_covariate.py``. The spreads are the assertion: a
constant covariate is absorbed into the mean.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    NegativeBinomialEmission,
)

SEED = 20260916
#: The planted negative-binomial rate and dispersion.
MEAN, DISPERSION = 2.5, 3.0


def _exposed_draws() -> tuple[torch.Tensor, torch.Tensor]:
    """2,000 exposures over 0.5 to 4.0 and the counts drawn under them."""
    rng = np.random.default_rng(SEED)
    offsets = torch.tensor(rng.uniform(0.5, 4.0, 2000), dtype=torch.float64)
    rate = (offsets * MEAN).numpy()
    draws = torch.tensor(
        rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + rate)),
        dtype=torch.float64,
    ).reshape(1, -1)
    return offsets, draws


@pytest.mark.critical
@pytest.mark.end2end
def test_the_exposure_does_not_hide_the_rate_or_the_dispersion() -> None:
    # 2,000 draws, exposure over 0.5 to 4.0 -- an eightfold spread, so no single
    # rescaling of `mu` explains the data.
    offsets, draws = _exposed_draws()
    mean, dispersion = MEAN, DISPERSION

    start = NegativeBinomialEmission(torch.tensor([1.0]), torch.tensor([1.0]))
    fitted = start.reestimate(
        draws, torch.ones(1, 2000, 1, dtype=torch.float64), offsets.reshape(1, -1)
    ).emissions

    assert float(fitted.mean[0]) == pytest.approx(mean, rel=0.05)
    assert float(fitted.dispersion[0]) == pytest.approx(dispersion, rel=0.10)


@pytest.mark.critical
@pytest.mark.end2end
def test_the_trial_count_does_not_hide_the_beta_parameters() -> None:
    # 4,000 draws, trial counts over 5 to 80 -- a sixteenfold spread. The
    # declared per-state count is deliberately 20, which none of the draws use,
    # so a fit that quietly ignored the covariate would not land here.
    rng = np.random.default_rng(SEED)
    trials = torch.tensor(rng.integers(5, 81, 4000), dtype=torch.float64)
    alpha, beta = 3.0, 7.0
    rates = rng.beta(alpha, beta, 4000)
    draws = torch.tensor(
        rng.binomial(trials.numpy().astype(np.int64), rates), dtype=torch.float64
    ).reshape(1, -1)

    start = BetaBinomialEmission(
        torch.tensor([20.0]), torch.tensor([1.0]), torch.tensor([1.0])
    )
    fitted = start.reestimate(
        draws, torch.ones(1, 4000, 1, dtype=torch.float64), trials.reshape(1, -1)
    ).emissions

    assert float(fitted.alpha[0]) == pytest.approx(alpha, rel=0.05)
    assert float(fitted.beta[0]) == pytest.approx(beta, rel=0.05)


@pytest.mark.critical
@pytest.mark.end2end
def test_ignoring_a_varying_covariate_does_not_recover_the_truth() -> None:
    # What makes the two above a test of the covariate and not of the fit. The
    # same draws, refitted with the covariate withheld, must miss -- otherwise
    # the exposure was never load-bearing and the recovery proves nothing.
    offsets, draws = _exposed_draws()
    mean, dispersion = MEAN, DISPERSION
    posterior = torch.ones(1, 2000, 1, dtype=torch.float64)
    start = NegativeBinomialEmission(torch.tensor([1.0]), torch.tensor([1.0]))

    blind = start.reestimate(draws, posterior).emissions

    # The mean recovers the average rate (5.684 against `mu E[e]` = 5.631); the
    # dispersion reads 1.729 against a planted 3.0: heterogeneity as overdispersion.
    average_rate = mean * float(offsets.mean())

    assert float(blind.mean[0]) == pytest.approx(average_rate, rel=0.02)
    assert float(blind.mean[0]) != pytest.approx(mean, rel=0.05)
    assert float(blind.dispersion[0]) < 0.7 * dispersion
