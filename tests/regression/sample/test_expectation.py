"""Operator expectations by a Kalman filter, and a chain run without its draws (issue #988).

The filter's mean and variance are the conditional GLS estimator under AR(1)
noise, computed from the stored stream, within 1e-12; over 400 AR(1) streams at
phi = 0.9 its 95% intervals cover within 3 binomial standard errors of 0.95
(white noise: about 0.35); HMC at d = 1,000 without its chain returns ``x``
within 4.5 standard errors of 0 and ``x^2`` of ``1 / p``, and no draws; an
operator leaves the default chain bitwise unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.sample import hmc
from sal.sample.expectation import KalmanMean
from sal.validation.gaussian import GaussianTarget, diagonal_precision

from tests._posteriors import assert_gaussian_moments


def _ar1(
    rng: np.random.Generator, n: int, phi: float, mu: float, width: int
) -> np.ndarray:
    """``width`` AR(1) streams of length ``n`` about ``mu``, started stationary."""
    noise = rng.normal(size=(n, width))
    stream = np.empty((n, width))
    stream[0] = noise[0] / np.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        stream[t] = phi * stream[t - 1] + noise[t]
    return mu + stream


def _filtered(stream: np.ndarray) -> KalmanMean:
    kalman = KalmanMean()
    for row in stream:
        kalman.update(row)
    return kalman


@pytest.mark.oracle
def test_the_filter_is_the_ar1_gls_estimator() -> None:
    stream = _ar1(np.random.default_rng(988), 5_000, 0.7, 2.0, 3)
    estimate = _filtered(stream).estimate()
    n = stream.shape[0]
    centred = stream - stream.mean(axis=0)
    gamma0 = (centred**2).mean(axis=0)
    gamma1 = (centred[1:] * centred[:-1]).sum(axis=0) / (n - 1)
    phi = gamma1 / gamma0
    whitened = stream[1:] - phi * stream[:-1]
    gls = whitened.sum(axis=0) / ((n - 1) * (1.0 - phi))
    variance = gamma0 * (1.0 - phi**2) / ((n - 1) * (1.0 - phi) ** 2)
    np.testing.assert_allclose(estimate.phi, phi, rtol=1e-12)
    np.testing.assert_allclose(estimate.mean, gls, rtol=1e-12)
    np.testing.assert_allclose(estimate.standard_error**2, variance, rtol=1e-12)
    assert np.abs(estimate.phi - 0.7).max() < 0.05


@pytest.mark.oracle
def test_the_intervals_cover_at_their_level_where_white_noise_does_not() -> None:
    rng = np.random.default_rng(9880)
    streams = _ar1(rng, 2_000, 0.9, 1.0, 400)
    estimate = _filtered(streams).estimate()
    z = (estimate.mean - 1.0) / estimate.standard_error
    coverage = float(np.mean(np.abs(z) < 1.96))
    error = np.sqrt(0.95 * 0.05 / 400)
    assert abs(coverage - 0.95) < 3 * error, coverage
    white = streams.std(axis=0) / np.sqrt(streams.shape[0])
    white_coverage = float(np.mean(np.abs(streams.mean(axis=0) - 1.0) < 1.96 * white))
    assert white_coverage < 0.6, white_coverage


@pytest.mark.end2end
def test_a_chain_without_its_draws_returns_the_expectations() -> None:
    dimension = 1_000
    precision = diagonal_precision(dimension)
    chain = hmc.sample(
        GaussianTarget(precision),
        torch.Generator().manual_seed(988),
        1_000,
        step_size=0.9 / (2.0 * dimension**0.25),
        n_steps=10,
        store_chain=False,
        operators={"x": lambda x: x, "x2": lambda x: x * x},
    )
    assert chain.draws.shape == (0, dimension)
    first, second = chain.expectations["x"], chain.expectations["x2"]
    assert first.n == second.n == 1_000
    assert_gaussian_moments(first, second, precision)


@pytest.mark.smoke
def test_an_operator_leaves_the_chain_bitwise_as_it_was() -> None:
    target = GaussianTarget(diagonal_precision(20))

    def run(**options: object) -> hmc.HmcChain:
        return hmc.sample(
            target,
            torch.Generator().manual_seed(9881),
            200,
            step_size=0.3,
            n_steps=5,
            # The torch route: an operator keeps a chain on it, so the plain
            # chain it is compared with must take it too (issue #986).
            backend=Backend.PYTHON,
            **options,  # type: ignore[arg-type]
        )

    plain = run()
    observed = run(operators={"x": lambda x: x})
    assert torch.equal(plain.draws, observed.draws)
    assert plain.expectations == {}
    np.testing.assert_allclose(
        observed.expectations["x"].mean,
        _filtered(plain.draws.numpy()).estimate().mean,
        rtol=0.0,
        atol=0.0,
    )
