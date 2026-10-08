"""The factored count densities against the families' own ``log_density`` (issue #1064).

:mod:`sal.emissions.nb` tabulates the negative binomial by count and leaves
the exposure to the caller; :mod:`sal.emissions.bb` tabulates the
beta-binomial's rising factorials each by its own integer (issue #1332). Each is
reassembled here per observation and judged against the family scoring the
same draws, simulated from the family at a seeded per-observation covariate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import BetaBinomialEmission, NegativeBinomialEmission
from sal.emissions.bb import beta_binomial_log_pmf, log_factorial, trial_tables
from sal.emissions.nb import (
    count_log_factor,
    exposure_table,
    negative_binomial_log_pmf,
)
from scipy.special import gammaln

SEED = 20260925

#: Draws per state, each at its own covariate.
N_DRAWS = 4_000

#: The negative binomial's tables completed in :mod:`sal.emissions.nb`'s order
#: against the family's ``log_density``, in ulp of the summed magnitudes of
#: the terms either forms (issue #1335): the two differ in arithmetic, so the
#: pin that was bitwise (the family's order) is restated at the tabulated
#: order's declared 4 ulp of magnitudes, not re-recorded. Measured 1.05 ulp.
_ULPS = 4


def _negative_binomial() -> tuple[NegativeBinomialEmission, np.ndarray, np.ndarray]:
    """A three-state family, its draws and the log-normal exposure each was drawn at."""
    family = NegativeBinomialEmission([4.0, 7.0, 12.0], [20.0, 80.0, 200.0])
    rng = np.random.default_rng(SEED)
    states = np.repeat(np.arange(3), N_DRAWS)
    exposure = np.exp(0.5 * rng.standard_normal(states.size))
    counts = family.sample(states, rng, covariate=exposure[:, None])
    return family, counts, exposure


def _beta_binomial() -> tuple[BetaBinomialEmission, np.ndarray, np.ndarray]:
    """A three-state family, its draws and the trial count each was drawn at.

    The trial counts are ``round(40 exp(0.25 z))``, the ci fixture's, with a
    tenth set to zero, which marks the successes unobserved.
    """
    family = BetaBinomialEmission(
        [40.0, 40.0, 40.0], [2.4, 7.0, 19.5], [9.6, 7.0, 10.5]
    )
    rng = np.random.default_rng(SEED)
    states = np.repeat(np.arange(3), N_DRAWS)
    trials = np.rint(40.0 * np.exp(0.25 * rng.standard_normal(states.size)))
    trials[rng.random(states.size) < 0.1] = 0.0
    successes = family.sample(states, rng, covariate=trials[:, None])
    return family, successes, trials


@pytest.mark.oracle
def test_the_count_log_factor_completed_in_order_is_the_pmf_bitwise_and_the_density_within_tolerance() -> (
    None
):
    # `(T + y log(lambda / (1 + q))) - r log1p(q)`, the order and the calls
    # `negative_binomial_log_pmf` makes, so the same bits; `log_density`'s own
    # arithmetic agrees within NB_ROUTE_TOLERANCE.
    family, counts, exposure = _negative_binomial()
    r = family.dispersion.numpy()
    rate = family.mean.numpy() * exposure[:, None]
    q = rate / r
    y = counts[:, None]
    assembled = (
        count_log_factor(family, counts)
        + np.where(y == 0.0, 0.0, y * np.log(rate / (1.0 + q)))
    ) - r * np.log1p(q)
    pmf = negative_binomial_log_pmf(y, r, rate)
    assert np.array_equal(assembled, pmf)

    want = family.log_density(
        torch.as_tensor(counts, dtype=torch.float64),
        covariate=torch.as_tensor(exposure)[:, None],
    ).numpy()
    magnitude = (
        np.abs(gammaln(y + r))
        + np.abs(gammaln(r))
        + np.abs(gammaln(y + 1.0))
        + np.abs(y * np.log(r))
        + np.abs(y * np.log(rate / (r + rate)))
        + np.abs(r * np.log1p(q))
    )
    worst = float(np.max(np.abs(assembled - want) / magnitude))
    assert worst <= _ULPS * np.finfo(np.float64).eps, (
        f"{worst / np.finfo(np.float64).eps:.2f} ulp"
    )


@pytest.mark.oracle
def test_the_exposure_table_is_the_count_log_factor_by_count_bitwise() -> None:
    family, counts, _ = _negative_binomial()
    extent = int(counts.max()) + 1
    assert np.array_equal(
        exposure_table(family, extent)[counts.astype(np.int64)],
        count_log_factor(family, counts),
    )


#: The scaled rising-factorial pmf against the family's ``lgamma``
#: arithmetic, over ``max(|f|, 1)`` (issue #1332). Both routes form ``log C``
#: from three ``lgamma`` of size up to ``lgamma(41)``, each rounded to half an
#: ulp: 3.7e-14 per route, 7.3e-14 between two, 1e-13 with the rest of the
#: sum. Derived, not fitted.
DENSITY_TOLERANCE = 1e-13


@pytest.mark.oracle
def test_the_trial_tables_summed_in_order_are_the_numpy_pmf_bitwise() -> None:
    # `log C + U + V - W`, the order `beta_binomial_log_pmf` sums in and the
    # same `log_rising` and `gammaln` calls, so the same bits; the family's
    # `log_density` agrees within DENSITY_TOLERANCE (measured 4.2e-14).
    family, successes, trials = _beta_binomial()
    z, n = successes.astype(np.int64), trials.astype(np.int64)
    tables = trial_tables(family, int(z.max()) + 1, int(n.max()) + 1)
    factorial = log_factorial(int(n.max()) + 1)
    observed = n > 0
    zo, no = z[observed], n[observed]

    assembled = np.zeros((z.size, family.n_states))
    log_p, log_q = tables.log_rate
    binomial = (
        ((factorial[no] - factorial[zo]) - factorial[no - zo])[:, None]
        + zo[:, None] * log_p
    ) + (no - zo)[:, None] * log_q
    assembled[observed] = (
        (binomial + tables.success[zo]) + tables.failure[no - zo]
    ) - tables.trial[no]
    pmf = beta_binomial_log_pmf(
        zo[:, None], no[:, None], family.alpha.numpy(), family.beta.numpy()
    )
    want = family.log_density(
        torch.as_tensor(successes, dtype=torch.float64),
        covariate=torch.as_tensor(trials)[:, None],
    ).numpy()

    assert bool((~observed).any())
    assert np.array_equal(assembled[observed], pmf)
    assert np.array_equal(assembled[~observed], want[~observed])
    worst = float((np.abs(assembled - want) / np.maximum(np.abs(want), 1.0)).max())
    assert worst <= DENSITY_TOLERANCE, worst


@pytest.mark.oracle
def test_the_log_factorial_is_the_factorial() -> None:
    # `lgamma(j + 1) = log j!`, against the integers themselves.
    exact = np.log([float(np.prod(np.arange(1, j + 1))) for j in range(21)])

    np.testing.assert_allclose(log_factorial(21), exact, rtol=1e-15, atol=0)
