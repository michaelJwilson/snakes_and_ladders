"""Random-walk Metropolis on both routes, its warm-up, and its operators (issue #1006).

Referees:

- the torch kernel is :func:`metropolis.replay` on the same draws, bitwise:
  the replay is the step BlackJAX's ``rmh`` is pinned to draw for draw
  (`tests/validation/test_blackjax.py`), so this chains the two;
- each route recovers the Gaussian's moments: every coordinate's mean within
  4.5 standard errors of 0 and its second moment within 4.5 of ``1 / p``,
  the errors from the chain's own AR(1) fit (``KalmanMean``);
- at ``T = 2`` the torch route's second moment is ``2 / p``, the tempered
  target's, at the same 4.5;
- the compiled route's warm-up and the torch route's settle on the same
  acceptance to 0.03, and on Rosenbrock at ``b = 1`` their first two moments
  agree within 4.5 combined standard errors;
- a declared operator's filter kept in Rust is the Python ``KalmanMean`` over
  the stored draws, bitwise; an undeclared one, filtered per block, agrees
  to 1e-12; ``store_chain=False`` keeps no draw and the same expectations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.opt.testfunctions import Rosenbrock
from snakes_and_ladders.sample import hmc, metropolis
from snakes_and_ladders.sample.declared import Power
from snakes_and_ladders.sample.expectation import KalmanMean
from snakes_and_ladders.validation.gaussian import GaussianTarget, diagonal_precision

DIMENSION = 20
PRECISION = diagonal_precision(DIMENSION)
STEP = 1.2 / np.sqrt(DIMENSION)


def _chain(backend: Backend, n: int = 40_000, **options: object) -> hmc.Chain:
    return metropolis.random_walk(
        GaussianTarget(PRECISION),
        torch.Generator().manual_seed(1006),
        n,
        step_size=STEP,
        backend=backend,
        **options,  # type: ignore[arg-type]
    )


def _assert_moments(chain: hmc.Chain, temperature: float = 1.0) -> None:
    first, second = chain.expectations["x"], chain.expectations["x2"]
    assert np.all(np.abs(first.mean.numpy()) < 4.5 * first.standard_error.numpy())
    target = temperature / PRECISION
    assert np.all(
        np.abs(second.mean.numpy() - target) < 4.5 * second.standard_error.numpy()
    )


@pytest.mark.oracle
def test_the_torch_kernel_is_the_replay_on_its_own_draws() -> None:
    chain = _chain(Backend.PYTHON, n=500)
    generator = torch.Generator().manual_seed(1006)
    increments, uniforms = [], []
    for _ in range(500):
        increments.append(
            STEP * torch.randn(DIMENSION, generator=generator, dtype=torch.float64)
        )
        uniforms.append(float(torch.rand(1, generator=generator)))
    replayed = metropolis.replay(
        GaussianTarget(PRECISION),
        np.zeros(DIMENSION),
        1.0,
        torch.stack(increments).numpy(),
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
            torch.Generator().manual_seed(1006),
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
        spread = np.hypot(rust.standard_error.numpy(), python.standard_error.numpy())
        assert np.all(np.abs(rust.mean.numpy() - python.mean.numpy()) < 4.5 * spread)


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
            if isinstance(operator, Power):
                np.testing.assert_array_equal(got.mean.numpy(), expected.mean.numpy())
                np.testing.assert_array_equal(got.phi.numpy(), expected.phi.numpy())
            else:
                np.testing.assert_allclose(
                    got.mean.numpy(), expected.mean.numpy(), rtol=1e-12
                )
            assert got.n == 3_000


@pytest.mark.analytic
def test_a_block_update_is_the_one_at_a_time_update() -> None:
    values = torch.as_tensor(np.random.default_rng(1006).normal(size=(2_500, 3)))
    one, block = KalmanMean(), KalmanMean()
    for value in values:
        one.update(value)
    block.update_block(values[:1_000])
    block.update_block(values[1_000:])
    for field in ("mean", "standard_error", "phi"):
        np.testing.assert_allclose(
            getattr(block.estimate(), field).numpy(),
            getattr(one.estimate(), field).numpy(),
            rtol=1e-12,
        )


@pytest.mark.smoke
def test_the_compiled_route_is_reproducible_from_the_generator() -> None:
    first, second = (_chain(Backend.RUST, n=200) for _ in range(2))
    np.testing.assert_array_equal(first.draws.numpy(), second.draws.numpy())


@pytest.mark.smoke
def test_a_bad_step_and_an_unknown_backend_are_refused() -> None:
    with pytest.raises(ValueError, match="step_size"):
        metropolis.random_walk(
            GaussianTarget(PRECISION), torch.Generator(), 1, step_size=0.0
        )
    with pytest.raises(ValueError, match="random_walk"):
        _chain(Backend.TORCH, n=1)
