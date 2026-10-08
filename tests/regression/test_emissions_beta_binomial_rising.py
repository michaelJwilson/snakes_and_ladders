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
    scaled_rising_array,
    scaled_rising_table,
)
from scipy.special import logsumexp

#: The concentrations ``tau = a + b`` judged, 10 to 1e16.
TAUS = (10.0, 1e3, 1e5, 1e8, 1e12, 1e16)

#: The issue's absolute bound against ``mpmath``, in nats.
ABSOLUTE = 1e-11

#: The pmf against torch's ``log_density`` or the Rust count kernel, over ``max(|f|, 1)``, at
#: ``n <= 40``. Both form ``log C`` from three ``lgamma`` of size up to
#: ``lgamma(41) = 110.3``, each rounded to half an ulp: 3.7e-14 per route,
#: 7.3e-14 between two, 1e-13 with the rest of the sum. Derived from the
#: arithmetic, not fitted; the issue's 1e-14 is below this rounding, and
#: torch alone is 1.15e-14 from ``mpmath`` at ``n <= 100``.
ROUTE_TOLERANCE = 1e-13

#: The pmf against ``mpmath`` over ``max(|f|, 1)`` at ``n <= 40``: ``log C``'s
#: three ``lgamma`` rounded to half an ulp, ``1.5 eps lgamma(41) = 3.7e-14``,
#: plus ``n eps = 0.9e-14`` from the two rates' logarithms. Derived; measured
#: 1.4e-14 to 1.6e-14 from tau = 10 to 1e16, where the issue asked 1e-14.
RELATIVE = 5e-14

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
    scale = np.maximum(np.abs(want), 1.0)
    assert float((np.abs(got - want) / scale).max()) <= RELATIVE


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
    # The tables summed in bb's order are the NumPy pmf's operations on the
    # same numbers. Torch's `log_density` keeps its own arithmetic and agrees
    # within ROUTE_TOLERANCE; measured 1.1e-14 at tau = 10 and 1.7e-14 at
    # 1e12 over max(|f|, 1).
    rates = np.array([0.2, 0.5, 0.8])
    family = BetaBinomialEmission(
        [40.0] * 3, list(rates * tau), list((1.0 - rates) * tau)
    )
    rng = np.random.default_rng(1332)
    n = rng.integers(1, 41, size=500)
    z = rng.integers(0, n + 1)
    tables = trial_tables(family, 41, 41)
    factorial = log_factorial(41)

    log_p, log_q = tables.log_rate
    binomial = (
        ((factorial[n] - factorial[z]) - factorial[n - z])[:, None] + z[:, None] * log_p
    ) + (n - z)[:, None] * log_q
    assembled = ((binomial + tables.success[z]) + tables.failure[n - z]) - tables.trial[
        n
    ]
    pmf = beta_binomial_log_pmf(
        z[:, None], n[:, None], family.alpha.numpy(), family.beta.numpy()
    )
    torch_scores = family.log_density(
        torch.as_tensor(z, dtype=torch.float64),
        covariate=torch.as_tensor(n, dtype=torch.float64)[:, None],
    ).numpy()

    assert np.array_equal(assembled, pmf)
    scale = np.maximum(np.abs(pmf), 1.0)
    assert float((np.abs(torch_scores - pmf) / scale).max()) <= ROUTE_TOLERANCE


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


@pytest.mark.oracle
@pytest.mark.parametrize("tau", TAUS, ids=str)
def test_the_rust_count_kernel_is_the_pmf_within_the_route_tolerance(
    tau: float,
) -> None:
    # `count_mixture::Rising`'s compensated prefix sums, one observation and
    # one state at a time so the value is that observation's log pmf;
    # measured 2.7e-14 over max(|f|, 1), the worst at tau = 10.
    from sal import oxisal

    a, b = RATE * tau, (1.0 - RATE) * tau
    worst = 0.0
    for n in range(0, 41, 3):
        for z in range(n + 1):
            value = oxisal.count_mixture_value_and_gradient(
                np.zeros(1),
                np.zeros(1),
                successes=np.array([z], dtype=np.uint32),
                alpha=np.array([a]),
                beta=np.array([b]),
                trials=np.array([float(n)]),
                grad_alpha=np.zeros(1),
                grad_beta=np.zeros(1),
            )
            pmf = float(beta_binomial_log_pmf(z, n, a, b))
            worst = max(worst, abs(value - pmf) / max(abs(pmf), 1.0))
    assert worst <= ROUTE_TOLERANCE, worst


@pytest.mark.analytic
def test_the_scaled_rise_is_zero_at_an_infinite_shape() -> None:
    # `R(x, m) - m log x -> 0` as `x -> inf` for finite `m`, as the docstring
    # states; the kernel formed `inf / inf` there (found by #1335).
    m = np.array([0.0, 1.0, 7.0, 1e6])
    assert np.array_equal(scaled_rising_array(np.full(4, np.inf), m), np.zeros(4))


@pytest.mark.oracle
def test_the_scaled_rising_table_is_the_elementwise_kernel_bitwise() -> None:
    # Hoisting gammaln(x) and log x per row reads the same values the
    # per-element kernel forms, so the table is it bit for bit: across both
    # routes, the shift below the series, x = inf and m = 0 (issue #1341).
    rng = np.random.default_rng(1341)
    shapes = np.concatenate(
        [10.0 ** rng.uniform(-3, 16, size=200), [1.4616, 9.99, 10.0, 99.9, np.inf]]
    )
    counts = np.concatenate([np.arange(300.0), 10.0 ** rng.uniform(-6, 6, size=200)])

    table = scaled_rising_table(shapes, counts)

    want = scaled_rising_array(shapes[:, None], counts[None, :])
    assert table.shape == (205, 500)
    assert table.flags.c_contiguous
    assert np.array_equal(table.view(np.int64), want.view(np.int64))
    assert np.array_equal(table[-1], np.zeros(500))
    with pytest.raises(ValueError, match="1-D"):
        scaled_rising_table(shapes[:, None], counts)
