"""Planted parameters recovered with the covariates varying (issue #631).

Step 5, and the ticket's Done-when. Each family is fitted back to the values
that generated its data while its covariate varies over a stated spread --- an
exposure over eightfold, a trial count over sixteenfold --- which is the regime
neither conserved family can express and therefore the one nothing else here
referees.

**Refereed by the simulated truth, not by an oracle.** There is no independent
implementation of a covariate-aware count family to check against, so what
these assert is recovery of the generating parameters to a stated tolerance,
which root ``CLAUDE.md`` admits where no oracle is affordable. The *oracle*
half of the claim is elsewhere and is exact: at a constant covariate both
families reduce to the conserved ones bitwise, in
``test_emission_exposure.py`` and ``test_emission_trial_covariate.py``.

The spreads are stated because they are the whole assertion. A covariate
varying by a factor of one is a constant absorbed into the mean, which those
other files already cover; these need it to vary enough that absorbing it is
not available.
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


@pytest.mark.critical
@pytest.mark.simulated_truth
def test_the_exposure_does_not_hide_the_rate_or_the_dispersion() -> None:
    # 2,000 draws, exposure over 0.5 to 4.0 -- an eightfold spread, so no single
    # rescaling of `mu` explains the data.
    rng = np.random.default_rng(SEED)
    offsets = torch.tensor(rng.uniform(0.5, 4.0, 2000), dtype=torch.float64)
    mean, dispersion = 2.5, 3.0
    rate = (offsets * mean).numpy()
    draws = torch.tensor(
        rng.negative_binomial(dispersion, dispersion / (dispersion + rate)),
        dtype=torch.float64,
    ).reshape(1, -1)

    start = NegativeBinomialEmission(torch.tensor([1.0]), torch.tensor([1.0]))
    fitted = start.reestimate(
        draws, torch.ones(1, 2000, 1, dtype=torch.float64), offsets.reshape(1, -1)
    ).emissions

    assert float(fitted.mean[0]) == pytest.approx(mean, rel=0.05)
    assert float(fitted.dispersion[0]) == pytest.approx(dispersion, rel=0.10)


@pytest.mark.critical
@pytest.mark.simulated_truth
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
@pytest.mark.simulated_truth
def test_ignoring_a_varying_covariate_does_not_recover_the_truth() -> None:
    # What makes the two above a test of the covariate and not of the fit. The
    # same draws, refitted with the covariate withheld, must miss -- otherwise
    # the exposure was never load-bearing and the recovery proves nothing.
    rng = np.random.default_rng(SEED)
    offsets = torch.tensor(rng.uniform(0.5, 4.0, 2000), dtype=torch.float64)
    mean, dispersion = 2.5, 3.0
    rate = (offsets * mean).numpy()
    draws = torch.tensor(
        rng.negative_binomial(dispersion, dispersion / (dispersion + rate)),
        dtype=torch.float64,
    ).reshape(1, -1)
    posterior = torch.ones(1, 2000, 1, dtype=torch.float64)
    start = NegativeBinomialEmission(torch.tensor([1.0]), torch.tensor([1.0]))

    blind = start.reestimate(draws, posterior).emissions

    # Both halves land where the mathematics says. The mean recovers the
    # *average* rate -- 5.684 against `mu E[e]` = 5.631, within 1% -- and so
    # says nothing about `mu`. The dispersion absorbs the spread the exposure
    # would have explained and comes back at 1.729 against a planted 3.0, a
    # 42% underestimate: unmodelled heterogeneity reads as overdispersion.
    average_rate = mean * float(offsets.mean())

    assert float(blind.mean[0]) == pytest.approx(average_rate, rel=0.02)
    assert float(blind.mean[0]) != pytest.approx(mean, rel=0.05)
    assert float(blind.dispersion[0]) < 0.7 * dispersion
