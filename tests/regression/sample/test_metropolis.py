"""Random-walk Metropolis on both routes, its warm-up, and its operators (issue #1006).

The Python kernel is :func:`metropolis.replay` bitwise, the step BlackJAX's
``rmh`` is pinned to (`tests/validation/test_blackjax.py`). Each route's
moments are within 4.5 standard errors (``KalmanMean``) of 0 and ``1 / p``,
``2 / p`` at ``T = 2``. The warm-ups settle on one acceptance to 0.03 and on
Rosenbrock (``b = 1``) agree within 4.5 combined errors. An operator's filter,
in Rust or per block, is ``KalmanMean`` bitwise (#1011); ``store_chain=False``
keeps no draw and the same expectations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.opt.testfunctions import Rosenbrock
from sal.sample import hmc, metropolis
from sal.sample.declared import Power
from sal.sample.expectation import KalmanMean
from sal.validation.gaussian import GaussianTarget, diagonal_precision

from tests._posteriors import assert_gaussian_moments

DIMENSION = 20
PRECISION = diagonal_precision(DIMENSION)
STEP = 1.2 / np.sqrt(DIMENSION)


def _chain(backend: Backend, n: int = 40_000, **options: object) -> hmc.Chain:
    return metropolis.random_walk(
        GaussianTarget(PRECISION),
        np.random.default_rng(1006),
        n,
        step_size=STEP,
        backend=backend,
        **options,  # type: ignore[arg-type]
    )


def _assert_moments(chain: hmc.Chain, temperature: float = 1.0) -> None:
    first, second = chain.expectations["x"], chain.expectations["x2"]
    assert_gaussian_moments(first, second, PRECISION, temperature=temperature)


@pytest.mark.oracle
def test_the_python_kernel_is_the_replay_on_its_own_draws() -> None:
    chain = _chain(Backend.PYTHON, n=500)
    generator = np.random.default_rng(1006)
    increments, uniforms = [], []
    for _ in range(500):
        increments.append(STEP * generator.standard_normal(DIMENSION))
        uniforms.append(generator.random())
    replayed = metropolis.replay(
        GaussianTarget(PRECISION),
        np.zeros(DIMENSION),
        1.0,
        np.stack(increments),
        np.asarray(uniforms),
    )
    np.testing.assert_array_equal(chain.draws.numpy(), replayed.draws)
    assert chain.acceptance_rate == replayed.accepted.mean()


@pytest.mark.end2end
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON], ids=str)
def test_each_route_recovers_the_gaussian_moments(backend: Backend) -> None:
    chain = _chain(
        backend,
        store_chain=False,
        operators={"x": Power(1), "x2": Power(2)},
    )
    assert chain.draws.shape == (0, DIMENSION)
    assert 0.2 < chain.acceptance_rate < 0.5
    _assert_moments(chain)


@pytest.mark.analytic
def test_the_tempered_chain_samples_the_tempered_target() -> None:
    chain = _chain(
        Backend.RUST,
        temperature=2.0,
        store_chain=False,
        operators={"x": Power(1), "x2": Power(2)},
    )
    _assert_moments(chain, temperature=2.0)


@pytest.mark.oracle
@pytest.mark.backend
def test_both_warm_ups_settle_on_the_target_acceptance() -> None:
    adaptation = hmc.Adaptation(2_000, metropolis.RWM_TARGET_ACCEPTANCE, 0.0)
    rates = []
    for backend in (Backend.RUST, Backend.PYTHON):
        chain = _chain(backend, n=4_000, adaptation=adaptation, store_chain=False)
        assert chain.adapted is not None
        assert chain.adapted.force_evaluations == 2_000
        assert chain.force_evaluations == 6_000
        rates.append(chain.acceptance_rate)
    assert abs(rates[0] - metropolis.RWM_TARGET_ACCEPTANCE) < 0.05
    assert abs(rates[0] - rates[1]) < 0.03


@pytest.mark.oracle
@pytest.mark.backend
def test_both_routes_sample_one_rosenbrock_density() -> None:
    # b = 1 so the chain mixes in the draws a test affords; at b = 100 the
    # between-seed spread of the acceptance alone is 0.06-0.08 at 20,000.
    target = Rosenbrock(2, b=1.0)
    means = [
        metropolis.random_walk(
            target,
            np.random.default_rng(1006),
            30_000,
            step_size=1.0,
            burn_in=1_000,
            store_chain=False,
            operators={"x": Power(1), "x2": Power(2)},
            backend=backend,
        ).expectations
        for backend in (Backend.RUST, Backend.PYTHON)
    ]
    for name in ("x", "x2"):
        rust, python = means[0][name], means[1][name]
        spread = np.hypot(rust.standard_error, python.standard_error)
        assert np.all(np.abs(rust.mean - python.mean) < 4.5 * spread)


@pytest.mark.oracle
@pytest.mark.backend
def test_operators_are_filtered_as_kalman_mean_filters_the_stored_draws() -> None:
    operators: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "x2": Power(2),
        "sum": lambda x: x.sum().reshape(1),
    }
    stored = _chain(Backend.RUST, n=3_000, operators=operators)
    free = _chain(Backend.RUST, n=3_000, operators=operators, store_chain=False)
    assert free.draws.shape == (0, DIMENSION)
    for name, operator in operators.items():
        kalman = KalmanMean()
        for draw in stored.draws:
            kalman.update(operator(draw))
        expected = kalman.estimate()
        for chain in (stored, free):
            got = chain.expectations[name]
            np.testing.assert_array_equal(got.mean, expected.mean)
            np.testing.assert_array_equal(got.phi, expected.phi)
            assert got.n == 3_000


@pytest.mark.analytic
def test_a_block_update_is_the_one_at_a_time_update() -> None:
    # Bitwise: a block is added row by row in the order `update` adds
    # (issue #1011); torch's cascade sum had held this to 1e-12.
    values = np.random.default_rng(1006).normal(size=(2_500, 3))
    one, block = KalmanMean(), KalmanMean()
    for value in values:
        one.update(value)
    block.update_block(values[:1_000])
    block.update_block(values[1_000:])
    for field in ("mean", "standard_error", "phi"):
        np.testing.assert_array_equal(
            getattr(block.estimate(), field), getattr(one.estimate(), field)
        )


@pytest.mark.smoke
def test_the_compiled_route_is_reproducible_from_the_generator() -> None:
    first, second = (_chain(Backend.RUST, n=200) for _ in range(2))
    np.testing.assert_array_equal(first.draws.numpy(), second.draws.numpy())


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON], ids=str)
def test_an_array_start_is_the_objectives_initial_point_bitwise(
    backend: Backend,
) -> None:
    # Issue #1059: `theta0` is an array, as the chain takes no derivative;
    # the zero start `GaussianTarget.initial` gives is the same chain.
    implicit = _chain(backend, n=200)
    explicit = _chain(backend, n=200, theta0=np.zeros(DIMENSION))
    np.testing.assert_array_equal(implicit.draws.numpy(), explicit.draws.numpy())


@pytest.mark.smoke
def test_a_bad_step_and_an_unknown_backend_are_refused() -> None:
    with pytest.raises(ValueError, match="step_size"):
        metropolis.random_walk(
            GaussianTarget(PRECISION), np.random.default_rng(), 1, step_size=0.0
        )
    with pytest.raises(ValueError, match="random_walk"):
        _chain(Backend.TORCH, n=1)
