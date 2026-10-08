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
from sal.emissions.nb import count_log_factor, exposure_table

SEED = 20260925

#: Draws per state, each at its own covariate.
N_DRAWS = 4_000

#: Agreement of the table ``B`` completed as ``B + y ln c - (y + r) ln t``
#: with the family's ``log_density``, which forms
#: ``A + r ln(r / t) + y ln(mu c / t)``, in ulp of the three terms' summed
#: magnitudes: ``y ln mu`` in ``B`` cancels against ``-y ln t``, so the error
#: is the rounding of the terms and not of their difference. Measured 1.08 ulp
#: on these draws, where relative to the score it is 1,380 ulp (262.9 at the ci
#: instance of ``spatio_sequential_counts_covariate``, whose means are lower).
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
def test_the_count_log_factor_completed_in_the_familys_order_is_its_density_bitwise() -> (
    None
):
    # `A` is `log_density`'s first three terms; the two exposure terms added
    # in its order are the same operations on the same numbers.
    family, counts, exposure = _negative_binomial()
    y = torch.as_tensor(counts, dtype=torch.float64)
    c = torch.as_tensor(exposure)[:, None]
    r, mu = family.dispersion, family.mean
    t = r + c * mu

    assembled = (
        count_log_factor(family, y)
        + r * torch.log(r / t)
        + y[:, None] * torch.log(c * mu / t)
    )

    assert torch.equal(assembled, family.log_density(y, covariate=c))


@pytest.mark.oracle
def test_the_exposure_table_completed_per_observation_is_the_density_to_its_rounding() -> (
    None
):
    family, counts, exposure = _negative_binomial()
    table = exposure_table(family, int(counts.max()) + 1)
    y = torch.as_tensor(counts, dtype=torch.float64)
    c = torch.as_tensor(exposure)[:, None]
    r, mu = family.dispersion, family.mean

    terms = (
        table[counts],
        y[:, None] * torch.log(c),
        (y[:, None] + r) * torch.log(r + mu * c),
    )
    assembled = terms[0] + terms[1] - terms[2]
    want = family.log_density(y, covariate=c)

    magnitude = sum(term.abs() for term in terms)
    worst = float(((assembled - want).abs() / magnitude).max())
    assert worst <= _ULPS * np.finfo(np.float64).eps, (
        f"{worst / np.finfo(np.float64).eps:.2f} ulp"
    )


#: The rising-factorial pmf against the family's ``lgamma`` arithmetic, over
#: ``max(|f|, 1)``: the bitwise pin of #1064 restated as a tolerance (issue
#: #1332), since the two now round differently. The bound is the sum of the
#: two routes' own errors against ``mpmath`` at 50 digits on the dense
#: fixture, 4.0e-14 and 2.8e-14, rounded up; measured below.
DENSITY_TOLERANCE = 1e-13


@pytest.mark.oracle
def test_the_trial_tables_summed_in_order_are_the_numpy_pmf_bitwise() -> None:
    # `log C + U + V - W`, the order `beta_binomial_log_pmf` sums in and the
    # same `log_rising` and `gammaln` calls, so the same bits; the family's
    # `log_density` agrees within DENSITY_TOLERANCE (measured 1.9e-14).
    family, successes, trials = _beta_binomial()
    z, n = successes.astype(np.int64), trials.astype(np.int64)
    tables = trial_tables(family, int(z.max()) + 1, int(n.max()) + 1)
    factorial = log_factorial(int(n.max()) + 1)
    observed = n > 0
    zo, no = z[observed], n[observed]

    assembled = np.zeros((z.size, family.n_states))
    assembled[observed] = (
        (
            ((factorial[no] - factorial[zo]) - factorial[no - zo])[:, None]
            + tables.success[zo]
        )
        + tables.failure[no - zo]
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
