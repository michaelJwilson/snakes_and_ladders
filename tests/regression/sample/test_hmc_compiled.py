"""The compiled Gaussian HMC chain against the torch route (issue #986).

Referees:

- ``gaussian_leapfrog`` is ``hmc.leapfrog`` on the same position and
  momentum, diagonal and dense, within 1e-12: the trajectory is the only
  arithmetic the two routes share, since their streams differ;
- the compiled chain at d = 200 returns every coordinate's mean within 4.5
  standard errors of 0 and variance within 4.5 of ``1 / p``, the standard
  errors from the chain's own AR(1) fit (``KalmanMean``);
- it is reproducible from the generator, and a chain the compiled route
  cannot run (an operator, a temperature) takes the torch route.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.expectation import KalmanMean
from snakes_and_ladders.validation.gaussian import (
    GaussianTarget,
    dense_precision,
    diagonal_precision,
)


@pytest.mark.oracle
@pytest.mark.parametrize("dense", [False, True], ids=["diagonal", "dense"])
def test_the_compiled_trajectory_is_the_torch_leapfrog(dense: bool) -> None:
    rng = np.random.default_rng(986)
    precision = dense_precision(12, rng) if dense else diagonal_precision(12)
    theta, momentum = rng.normal(size=12), rng.normal(size=12)
    torch_end = hmc.leapfrog(
        GaussianTarget(precision),
        torch.as_tensor(theta),
        torch.as_tensor(momentum),
        0.21,
        17,
    )
    position, velocity = oxi_snakes_and_ladders.gaussian_leapfrog(
        np.ascontiguousarray(precision).reshape(-1), theta, momentum, 0.21, 17
    )
    np.testing.assert_allclose(position, torch_end.position.numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(velocity, torch_end.momentum.numpy(), rtol=0, atol=1e-12)


@pytest.mark.end2end
def test_the_compiled_chain_recovers_the_gaussian_moments() -> None:
    dimension = 200
    precision = diagonal_precision(dimension)
    chain = hmc.sample(
        GaussianTarget(precision),
        torch.Generator().manual_seed(986),
        4_000,
        step_size=0.9 / (2.0 * dimension**0.25),
        n_steps=10,
        burn_in=200,
    )
    assert chain.theta.shape == (4_000, dimension)
    assert chain.acceptance_rate > 0.8
    first, second = KalmanMean(), KalmanMean()
    for row in chain.theta:
        first.update(row)
        second.update(row * row)
    mean, square = first.estimate(), second.estimate()
    assert np.abs(mean.mean.numpy() / mean.standard_error.numpy()).max() < 4.5
    residual = (square.mean.numpy() - 1.0 / precision) / square.standard_error.numpy()
    assert np.abs(residual).max() < 4.5


@pytest.mark.smoke
def test_the_compiled_chain_is_reproducible_and_routes_what_it_cannot_run() -> None:
    target = GaussianTarget(diagonal_precision(8))

    def run(**options: object) -> hmc.HmcChain:
        return hmc.sample(
            target,
            torch.Generator().manual_seed(9860),
            50,
            step_size=0.3,
            n_steps=4,
            **options,  # type: ignore[arg-type]
        )

    assert torch.equal(run().theta, run().theta)
    # A temperature is the torch route's alone: the compiled and torch
    # streams differ, so the tempered chain matches the explicit torch one.
    tempered = run(temperature=2.0)
    assert torch.equal(
        tempered.theta, run(temperature=2.0, backend=Backend.PYTHON).theta
    )
    assert not torch.equal(run().theta, run(backend=Backend.PYTHON).theta)
