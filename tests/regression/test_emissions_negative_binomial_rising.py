"""The negative binomial from scaled rising factorials, against ``mpmath``, its sum and its limit (issue #1335).

:func:`~sal.emissions.nb.negative_binomial_log_pmf` and the tables
:func:`~sal.emissions.nb.exposure_table` completed in :mod:`sal.emissions.nb`'s
order are judged against ``mpmath`` at 50 digits from ``r = 10`` to ``1e16``,
against normalization, against the Poisson at ``r = inf``; the Rust dense
kernel and the torch ``log_density`` against the pmf; and the beta-binomial
M step's concentration score against ``mpmath``.
"""

from __future__ import annotations

import mpmath  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from sal.emissions import NegativeBinomialEmission
from sal.emissions.coded import Dense, log_emission
from sal.emissions.mstep import _beta_binomial_concentration_score
from sal.emissions.nb import exposure_table, negative_binomial_log_pmf
from scipy.stats import poisson

#: The dispersions judged, 10 to 1e16.
DISPERSIONS = (10.0, 1e3, 1e5, 1e8, 1e12, 1e14, 1e16)

#: The pmf against ``mpmath`` over ``max(|f|, 1)``: #1332's derived bound,
#: ``T``'s ``lgamma(y + 1)`` and the ``y log`` term each to an ulp of terms up
#: to ``lgamma(200) = 857``. Measured 1.8e-14 at ``r = 10``, 1.5e-14 at
#: ``r >= 1e3``, where the raw ``lgamma`` form lost 2.2e-3 at ``r = 1e12``.
RELATIVE = 5e-14

#: Two routes over ``max(|f|, 1)``, #1332's: the Rust kernel's ``ln`` and
#: NumPy's ``log`` differ by an ulp (7.8e-15 measured), torch's form by
#: 8.6e-14 measured.
ROUTE_TOLERANCE = 1e-13

COUNTS = np.arange(200.0)


def _exact(y: float, r: float, rate: float) -> float:
    """The log pmf at 50 digits, from ``loggamma`` of the exact arguments."""
    with mpmath.workdps(50):
        y_, r_, l_ = mpmath.mpf(y), mpmath.mpf(r), mpmath.mpf(rate)
        t = r_ + l_
        value = (
            mpmath.loggamma(y_ + r_)
            - mpmath.loggamma(r_)
            - mpmath.loggamma(y_ + 1)
            + r_ * mpmath.log(r_ / t)
            + y_ * mpmath.log(l_ / t)
        )
        return float(value)


def _family(r: float) -> NegativeBinomialEmission:
    return NegativeBinomialEmission(
        dispersion=torch.full((4,), r, dtype=torch.float64),
        mean=torch.tensor([2.0, 9.0, 30.0, 80.0], dtype=torch.float64),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("r", DISPERSIONS)
@pytest.mark.parametrize("rate", [3.0, 40.0])
def test_the_pmf_is_mpmath_within_the_derived_bound(r: float, rate: float) -> None:
    exact = np.array([_exact(y, r, rate) for y in COUNTS])
    got = negative_binomial_log_pmf(COUNTS, r, rate)
    assert np.max(np.abs(got - exact) / np.maximum(np.abs(exact), 1.0)) <= RELATIVE


@pytest.mark.analytic
def test_infinite_dispersion_is_the_poisson_and_the_pmf_sums_to_one() -> None:
    assert np.array_equal(
        negative_binomial_log_pmf(COUNTS, np.inf, 40.0), poisson.logpmf(COUNTS, 40.0)
    )
    for r, extent in ((10.0, 3000), (1e16, 400)):
        total = np.exp(
            negative_binomial_log_pmf(np.arange(float(extent)), r, 40.0)
        ).sum()
        assert abs(total - 1.0) <= 1e-13


@pytest.mark.oracle
@pytest.mark.parametrize("r", [10.0, 1e12])
def test_the_tables_are_the_pmf_bitwise_and_rust_and_torch_within_the_route_tolerance(
    r: float,
) -> None:
    family = _family(r)
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 150, 5000).astype(np.float64)
    exposure = rng.uniform(0.5, 2.0, (5000, 1))
    dispersion = family.dispersion.numpy()[:, None]
    rate = family.mean.numpy()[:, None] * exposure[:, 0][None, :]
    reference = negative_binomial_log_pmf(counts[None, :], dispersion, rate)
    table = exposure_table(family, 150)[counts.astype(np.int64)].T
    q = rate / dispersion
    tabulated = (
        table + np.where(counts == 0.0, 0.0, counts * np.log(rate / (1.0 + q)))
    ) - dispersion * np.log1p(q)
    assert np.array_equal(tabulated, reference)
    scale = np.maximum(np.abs(reference), 1.0)
    family_order = log_emission(family, Dense(counts, exposure[:, 0]))
    assert np.max(np.abs(family_order - reference) / scale) <= ROUTE_TOLERANCE
    torch_route = (
        family.log_density(torch.tensor(counts), torch.tensor(exposure)).numpy().T
    )
    assert np.max(np.abs(torch_route - reference) / scale) <= ROUTE_TOLERANCE


def _exact_concentration_score(
    values: np.ndarray, trials: float, rate: float, concentration: float
) -> float:
    with mpmath.workdps(50):
        p = mpmath.mpf(rate)
        a, b, m = p * concentration, (1 - p) * concentration, mpmath.mpf(concentration)
        psi = mpmath.digamma
        total = mpmath.mpf(0)
        for v in values.tolist():
            total += (
                p * (psi(v + a) - psi(a))
                + (1 - p) * (psi(trials - v + b) - psi(b))
                - (psi(trials + m) - psi(m))
            )
        return float(total)


#: The score is three terms of size ``n / M`` summed to ``n^2 / M^2``, so its
#: relative rounding grows as ``M eps``: measured 1.7e-12 at ``M = 1e3`` and
#: 1.1e-8 at 1e8, where the raw ``digamma`` differences were 2.6e-10 and 3.7.
@pytest.mark.oracle
@pytest.mark.parametrize(
    ("concentration", "bound"), [(10.0, 1e-13), (1e3, 1e-11), (1e5, 1e-9), (1e8, 1e-6)]
)
def test_the_beta_binomial_concentration_score_is_mpmath(
    concentration: float, bound: float
) -> None:
    rng = np.random.default_rng(0)
    values = rng.binomial(40, 0.3, 200).astype(np.float64)
    exact = _exact_concentration_score(values, 40.0, 0.3, concentration)
    got = _beta_binomial_concentration_score(
        torch.tensor(values),
        torch.ones(200, dtype=torch.float64),
        40.0,
        0.3,
        concentration,
    )
    assert abs(got - exact) <= bound * abs(exact)
