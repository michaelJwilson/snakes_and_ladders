"""The compiled Gaussian HMC and MALA chains against the torch route (issues #986, #997).

Referees:

- ``leapfrog_trajectory`` is ``hmc.leapfrog`` on the same position and
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
from snakes_and_ladders import oxisal
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.opt.mixture import GaussianMixtureObjective
from snakes_and_ladders.opt.objective import DeclaredGradient
from snakes_and_ladders.opt.testfunctions import Rosenbrock
from snakes_and_ladders.sample import hmc, langevin
from snakes_and_ladders.sample.declared import Power, declared_energy
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
    position, velocity = oxisal.leapfrog_trajectory(
        0, np.ascontiguousarray(precision).reshape(-1), theta, momentum, 0.21, 17
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


@pytest.mark.oracle
@pytest.mark.parametrize("dense", [False, True], ids=["diagonal", "dense"])
def test_a_declared_gradient_is_autograd_s(dense: bool) -> None:
    # `gradient_at` reads a DeclaredGradient's own closed form (issue #986);
    # it must be the gradient autograd takes through `__call__`.
    rng = np.random.default_rng(9862)
    precision = dense_precision(15, rng) if dense else diagonal_precision(15)
    target = GaussianTarget(precision)
    assert isinstance(target, DeclaredGradient)
    theta = torch.as_tensor(rng.normal(size=15))
    point = theta.clone().requires_grad_(True)
    (autograd,) = torch.autograd.grad(target(point), point)
    np.testing.assert_allclose(
        hmc.gradient_at(target, theta).numpy(), autograd.numpy(), rtol=0, atol=1e-13
    )


@pytest.mark.end2end
def test_the_compiled_mala_chain_recovers_the_gaussian_moments() -> None:
    # Issue #997: MALA on a declared Gaussian runs the compiled chain at one
    # leapfrog step, the identity `sample.langevin` keeps; judged against the
    # target's moments as the HMC chain is.
    dimension = 50
    precision = diagonal_precision(dimension)
    chain = langevin.mala(
        GaussianTarget(precision),
        torch.Generator().manual_seed(997),
        20_000,
        step_size=0.9 / (2.0 * dimension**0.25),
        burn_in=500,
    )
    assert chain.theta.shape == (20_000, dimension)
    assert chain.corrected
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
def test_the_compiled_mala_chain_routes_what_it_cannot_run() -> None:
    target = GaussianTarget(diagonal_precision(8))

    def run(**options: object) -> langevin.LangevinChain:
        return langevin.mala(
            target,
            torch.Generator().manual_seed(9970),
            50,
            step_size=0.3,
            **options,  # type: ignore[arg-type]
        )

    assert torch.equal(run().theta, run().theta)
    # ULA and a temperature are the torch route's alone.
    for options in ({"corrected": False}, {"temperature": 2.0}):
        assert torch.equal(
            run(**options).theta, run(**options, backend=Backend.PYTHON).theta
        )
    assert not torch.equal(run().theta, run(backend=Backend.PYTHON).theta)


@pytest.mark.oracle
def test_the_compiled_trajectory_is_the_torch_leapfrog_on_rosenbrock() -> None:
    # Issue #1008: the declared Rosenbrock force, step for step.
    rng = np.random.default_rng(1008)
    theta, momentum = 0.3 * rng.normal(size=10), rng.normal(size=10)
    torch_end = hmc.leapfrog(
        Rosenbrock(10), torch.as_tensor(theta), torch.as_tensor(momentum), 0.002, 25
    )
    position, velocity = oxisal.leapfrog_trajectory(
        1, np.asarray([1.0, 100.0]), theta, momentum, 0.002, 25
    )
    np.testing.assert_allclose(position, torch_end.position.numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(velocity, torch_end.momentum.numpy(), rtol=0, atol=1e-10)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_compiled_warm_up_settles_where_the_torch_one_does() -> None:
    # Issue #1008: `Adaptation` on the compiled chain is `hmc._warm_up`'s
    # arithmetic on its own stream, so the two routes agree in distribution:
    # the acceptance each reaches, and the mass diagonal's estimate of the
    # precision.
    precision = diagonal_precision(20)
    adaptation = hmc.Adaptation(1_000, 0.65, 0.2)
    chains = [
        hmc.sample(
            GaussianTarget(precision),
            torch.Generator().manual_seed(1008),
            2_000,
            step_size=0.1,
            n_steps=10,
            adaptation=adaptation,
            store_chain=False,
            backend=backend,
        )
        for backend in (Backend.RUST, Backend.PYTHON)
    ]
    for chain in chains:
        assert chain.adapted is not None
        ratio = chain.adapted.mass_diagonal.numpy() / precision
        assert ratio.min() > 0.5
        assert ratio.max() < 2.0
        assert chain.force_evaluations == 3_000 * hmc.leapfrog.force_evaluations(10)
    assert abs(chains[0].acceptance_rate - chains[1].acceptance_rate) < 0.05


@pytest.mark.oracle
@pytest.mark.backend
def test_compiled_operators_are_kalman_mean_over_the_stored_draws() -> None:
    target = GaussianTarget(diagonal_precision(12))
    operators = {"x": Power(1), "x2": Power(2)}
    chain = hmc.sample(
        target,
        torch.Generator().manual_seed(1008),
        1_000,
        step_size=0.3,
        n_steps=5,
        operators=operators,
    )
    for name, operator in operators.items():
        kalman = KalmanMean()
        for row in chain.theta:
            kalman.update(operator(row))
        np.testing.assert_array_equal(
            chain.expectations[name].mean.numpy(), kalman.estimate().mean.numpy()
        )


@pytest.mark.oracle
@pytest.mark.backend
def test_both_routes_sample_one_rosenbrock_density() -> None:
    # b = 1 so 20,000 transitions mix; the first two moments agree within
    # 4.5 combined standard errors.
    target = Rosenbrock(2, b=1.0)
    expectations = [
        hmc.sample(
            target,
            torch.Generator().manual_seed(1008),
            8_000,
            step_size=0.2,
            n_steps=5,
            burn_in=200,
            store_chain=False,
            operators={"x": Power(1), "x2": Power(2)},
            backend=backend,
        ).expectations
        for backend in (Backend.RUST, Backend.PYTHON)
    ]
    for name in ("x", "x2"):
        rust, python = expectations[0][name], expectations[1][name]
        spread = np.hypot(rust.standard_error.numpy(), python.standard_error.numpy())
        assert np.all(np.abs(rust.mean.numpy() - python.mean.numpy()) < 4.5 * spread)


@pytest.mark.end2end
def test_compiled_mala_with_its_warm_up_reaches_its_target() -> None:
    chain = langevin.mala(
        GaussianTarget(diagonal_precision(50)),
        torch.Generator().manual_seed(1008),
        4_000,
        step_size=0.1,
        adaptation=hmc.Adaptation(1_000, langevin.MALA_TARGET_ACCEPTANCE, 0.0),
    )
    assert chain.adapted is not None
    assert abs(chain.acceptance_rate - langevin.MALA_TARGET_ACCEPTANCE) < 0.06


def _mixture(n: int) -> GaussianMixtureObjective:
    rng = np.random.default_rng(1008)
    labels = rng.choice(3, size=n, p=[0.3, 0.3, 0.4])
    values = np.array([-4.0, 0.0, 5.0])[labels] + rng.normal(size=n)
    return GaussianMixtureObjective(values, 3)


@pytest.mark.oracle
def test_the_compiled_mixture_trajectory_is_the_torch_leapfrog() -> None:
    # Issue #1008: the declared mixture's force is the objective's streamed
    # gradient in `theta`'s layout, step for step through a trajectory.
    objective = _mixture(5_000)
    declared = declared_energy(objective)
    assert declared is not None
    rng = np.random.default_rng(10080)
    theta = np.array([0.0, 0.3, -4.0, 0.0, 5.0, 0.0, 0.1, 0.0]) + 0.01 * rng.normal(
        size=8
    )
    momentum = rng.normal(size=8)
    torch_end = hmc.leapfrog(
        objective, torch.as_tensor(theta), torch.as_tensor(momentum), 0.002, 10
    )
    position, velocity = oxisal.leapfrog_trajectory(
        declared[0], declared[1], theta, momentum, 0.002, 10
    )
    np.testing.assert_allclose(
        position, torch_end.position.numpy(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        velocity, torch_end.momentum.numpy(), rtol=1e-10, atol=1e-8
    )


@pytest.mark.oracle
@pytest.mark.backend
def test_both_routes_sample_the_mixture_posterior_alike() -> None:
    objective = _mixture(2_000)
    theta0 = torch.tensor(
        [0.0, 0.3, -4.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    expectations = [
        hmc.sample(
            objective,
            torch.Generator().manual_seed(1008),
            1_500,
            step_size=0.01,
            n_steps=8,
            theta0=theta0,
            burn_in=200,
            store_chain=False,
            operators={"x": Power(1)},
            backend=backend,
        ).expectations["x"]
        for backend in (Backend.RUST, Backend.PYTHON)
    ]
    rust, python = expectations
    spread = np.hypot(rust.standard_error.numpy(), python.standard_error.numpy())
    assert np.all(np.abs(rust.mean.numpy() - python.mean.numpy()) < 4.5 * spread)
