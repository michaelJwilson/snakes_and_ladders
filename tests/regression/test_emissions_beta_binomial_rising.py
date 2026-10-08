"""The beta-binomial from rising factorials, against ``mpmath``, its sum and its limit (issue #1332).

:func:`~sal.emissions.bb.beta_binomial_log_pmf` and
:func:`~sal.emissions.bb.trial_tables` form ``log C + R(a, k) + R(b, n - k)
- R(a + b, n)``; each is judged against ``mpmath`` at 50 digits, against
normalization at ``tau = 1e16``, against the binomial at ``tau = inf``, and
the torch :meth:`~sal.emissions.counts.BetaBinomialEmission.log_density` and
the NumPy helpers against the pmf.
"""

from __future__ import annotations

import mpmath  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from sal.emissions import BetaBinomialEmission
from sal.emissions.bb import (
    beta_binomial_log_pmf,
    binomial_log_pmf,
    log_factorial,
    trial_tables,
)
from sal.emissions.rising import (
    log_rising,
    log_rising_into,
    on_distinct,
    on_distinct_array,
)
from scipy.special import logsumexp

#: The concentrations ``tau = a + b`` judged, 10 to 1e16.
TAUS = (10.0, 1e3, 1e5, 1e8, 1e12, 1e16)

#: The issue's absolute bound against ``mpmath``, in nats.
ABSOLUTE = 1e-11

#: The pmf against torch's ``log_density``, over ``max(|f|, 1)``: the sum of
#: the two routes' errors against ``mpmath`` at 50 digits for ``tau <= 1e16``,
#: ``n <= 40`` (2.0e-13 and 8.4e-15), rounded up. Not the issue's 1e-14: the
#: unscaled rising factorials cancel, see the test below.
TORCH_TOLERANCE = 3e-13

#: The rate ``p = a / tau`` at every concentration.
RATE = 0.3


def _exact(k: int, n: int, a: float, b: float) -> float:
    """The log pmf at 50 digits, from ``loggamma`` of the exact arguments."""
    with mpmath.workdps(50):
        a_, b_ = mpmath.mpf(a), mpmath.mpf(b)
        value = (
            mpmath.log(mpmath.binomial(n, k))
            + mpmath.loggamma(a_ + k)
            - mpmath.loggamma(a_)
            + mpmath.loggamma(b_ + n - k)
            - mpmath.loggamma(b_)
            - mpmath.loggamma(a_ + b_ + n)
            + mpmath.loggamma(a_ + b_)
        )
        return float(value)


@pytest.mark.oracle
@pytest.mark.parametrize("tau", TAUS, ids=str)
def test_the_pmf_is_mpmath_to_1e_11_at_every_concentration(tau: float) -> None:
    # Before #1332 the tables differenced lgamma of size tau log tau: 5e-3
    # nats at 1e12. Every (k, n) with n <= 40, three of them per n.
    a, b = RATE * tau, (1.0 - RATE) * tau
    pairs = [(k, n) for n in range(41) for k in sorted({0, n // 3, n})]
    k = np.array([p[0] for p in pairs], dtype=np.float64)
    n = np.array([p[1] for p in pairs], dtype=np.float64)

    got = beta_binomial_log_pmf(k, n, a, b)
    want = np.array([_exact(int(i), int(j), a, b) for i, j in pairs])

    assert float(np.abs(got - want).max()) <= ABSOLUTE


@pytest.mark.oracle
def test_the_pmf_sums_to_one_at_a_concentration_of_1e16() -> None:
    # At tau = 1e16 the lgamma difference summed to about e^132 over n <= 100.
    a, b = RATE * 1e16, (1.0 - RATE) * 1e16
    worst = 0.0
    for n in range(101):
        k = np.arange(n + 1, dtype=np.float64)
        worst = max(worst, abs(float(logsumexp(beta_binomial_log_pmf(k, n, a, b)))))
    assert worst <= 1e-12


@pytest.mark.analytic
def test_an_infinite_concentration_is_the_binomial() -> None:
    # The binomial is the tau -> inf limit; at tau = 1e16 the beta-binomial
    # is within (n^2 / tau) of it, far below the tolerance.
    k = np.arange(31, dtype=np.float64)
    binomial = binomial_log_pmf(k, 30, RATE)
    near = beta_binomial_log_pmf(k, 30, RATE * 1e16, (1.0 - RATE) * 1e16)
    with mpmath.workdps(50):
        exact = np.array(
            [
                float(
                    mpmath.log(mpmath.binomial(30, int(j)))
                    + int(j) * mpmath.log(mpmath.mpf(RATE))
                    + (30 - int(j)) * mpmath.log(1 - mpmath.mpf(RATE))
                )
                for j in k
            ]
        )

    np.testing.assert_allclose(binomial, exact, rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(near, binomial, rtol=0, atol=1e-11)


@pytest.mark.analytic
def test_off_support_scores_minus_infinity() -> None:
    # k past n, or negative, has probability zero in both pmfs.
    k = np.array([-1.0, 4.0, 2.0])
    assert np.array_equal(
        beta_binomial_log_pmf(k, 3, 2.0, 5.0)[:2], np.array([-np.inf, -np.inf])
    )
    assert np.isfinite(beta_binomial_log_pmf(k, 3, 2.0, 5.0)[2])
    assert np.array_equal(binomial_log_pmf(k, 3, 0.4)[:2], [-np.inf, -np.inf])


@pytest.mark.oracle
@pytest.mark.parametrize("tau", [10.0, 1e12], ids=str)
def test_the_tables_are_the_pmf_bitwise_and_torch_within_its_promise(
    tau: float,
) -> None:
    # The tables summed as `log C + U + V - W` are the NumPy pmf's operations
    # on the same numbers. Torch's `log_density` keeps its own arithmetic and
    # agrees within TORCH_TOLERANCE: the rising factorials are each of size
    # n log tau and cancel to the pmf, so the tables carry eps n log tau
    # absolute, measured 9.1e-14 over max(|f|, 1) against mpmath at 1e12 and
    # 2.0e-13 at 1e16, where torch's scaled series is 7.2e-15.
    rates = np.array([0.2, 0.5, 0.8])
    family = BetaBinomialEmission(
        [40.0] * 3, list(rates * tau), list((1.0 - rates) * tau)
    )
    rng = np.random.default_rng(1332)
    n = rng.integers(1, 41, size=500)
    z = rng.integers(0, n + 1)
    tables = trial_tables(family, 41, 41)
    factorial = log_factorial(41)

    assembled = (
        (
            ((factorial[n] - factorial[z]) - factorial[n - z])[:, None]
            + tables.success[z]
        )
        + tables.failure[n - z]
    ) - tables.trial[n]
    pmf = beta_binomial_log_pmf(
        z[:, None], n[:, None], family.alpha.numpy(), family.beta.numpy()
    )
    torch_scores = family.log_density(
        torch.as_tensor(z, dtype=torch.float64),
        covariate=torch.as_tensor(n, dtype=torch.float64)[:, None],
    ).numpy()

    assert np.array_equal(assembled, pmf)
    scale = np.maximum(np.abs(pmf), 1.0)
    assert float((np.abs(torch_scores - pmf) / scale).max()) <= TORCH_TOLERANCE


@pytest.mark.oracle
def test_log_rising_into_is_log_rising_bitwise() -> None:
    # The same compiled kernel on the same arrays; nothing broadcast.
    rng = np.random.default_rng(13320)
    x = 10.0 ** rng.uniform(-3, 16, size=1000)
    m = np.floor(10.0 ** rng.uniform(0, 3, size=1000))
    out = np.empty_like(x)

    log_rising_into(x, m, out)

    assert np.array_equal(out, log_rising(x, m))
    with pytest.raises(ValueError, match="one shape"):
        log_rising_into(x, m[:-1], out)


@pytest.mark.oracle
def test_the_numpy_on_distinct_is_the_torch_one_bitwise() -> None:
    # Both gather an elementwise table over the distinct values.
    rng = np.random.default_rng(13321)
    values = rng.integers(0, 30, size=(500, 1)).astype(np.float64)
    shapes = np.array([0.5, 3.0, 1e6])

    got = on_distinct_array(values, lambda v: log_rising(shapes, v))
    want = on_distinct(
        torch.from_numpy(values),
        lambda v: torch.from_numpy(log_rising(shapes, v.numpy())),
    ).numpy()

    assert np.array_equal(got, want)
    assert np.array_equal(got, log_rising(shapes, values))
