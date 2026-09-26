"""The compiled Gaussian HMC and MALA chains against the torch route (issues #986, #997).

``hmc.compiled_trajectory`` (``oxisal.leapfrog_trajectory``) is
``hmc.leapfrog`` on the same position and momentum, diagonal and dense, within
1e-12 (the streams differ). At d = 200 every mean is within 4.5 standard errors
of 0 and variance of ``1 / p`` (``KalmanMean``). Reproducible from the
generator; an operator or temperature takes the torch route.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.opt.hmm import GaussianHmmObjective, PoissonHmmObjective
from sal.opt.mixture import GaussianMixtureObjective
from sal.opt.objective import DeclaredGradient
from sal.opt.testfunctions import Rosenbrock
from sal.sample import hmc, langevin
from sal.sample.declared import Power
from sal.sample.expectation import KalmanMean
from sal.validation.gaussian import (
    GaussianTarget,
    dense_precision,
    diagonal_precision,
)

from tests._posteriors import assert_gaussian_moments, draw_moments


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
    compiled = hmc.compiled_trajectory(
        GaussianTarget(precision),
        torch.as_tensor(theta),
        torch.as_tensor(momentum),
        0.21,
        17,
    )
    position, velocity = compiled.position.numpy(), compiled.momentum.numpy()
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
    assert chain.draws.shape == (4_000, dimension)
    assert chain.acceptance_rate > 0.8
    assert_gaussian_moments(*draw_moments(chain.draws), precision)


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

    assert torch.equal(run().draws, run().draws)
    # A temperature is the torch route's alone: the compiled and torch
    # streams differ, so the tempered chain matches the explicit torch one.
    tempered = run(temperature=2.0)
    assert torch.equal(
        tempered.draws, run(temperature=2.0, backend=Backend.PYTHON).draws
    )
    assert not torch.equal(run().draws, run(backend=Backend.PYTHON).draws)


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
    assert chain.draws.shape == (20_000, dimension)
    assert chain.corrected
    assert chain.acceptance_rate > 0.8
    assert_gaussian_moments(*draw_moments(chain.draws), precision)


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

    assert torch.equal(run().draws, run().draws)
    # ULA and a temperature are the torch route's alone.
    for options in ({"corrected": False}, {"temperature": 2.0}):
        assert torch.equal(
            run(**options).draws, run(**options, backend=Backend.PYTHON).draws
        )
    assert not torch.equal(run().draws, run(backend=Backend.PYTHON).draws)


@pytest.mark.oracle
def test_the_compiled_trajectory_is_the_torch_leapfrog_on_rosenbrock() -> None:
    # Issue #1008: the declared Rosenbrock force, step for step.
    rng = np.random.default_rng(1008)
    theta, momentum = 0.3 * rng.normal(size=10), rng.normal(size=10)
    torch_end = hmc.leapfrog(
        Rosenbrock(10), torch.as_tensor(theta), torch.as_tensor(momentum), 0.002, 25
    )
    compiled = hmc.compiled_trajectory(
        Rosenbrock(10), torch.as_tensor(theta), torch.as_tensor(momentum), 0.002, 25
    )
    position, velocity = compiled.position.numpy(), compiled.momentum.numpy()
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
        assert chain.spent == 3_000 * hmc.leapfrog.force_evaluations(10)
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
        for row in chain.draws:
            kalman.update(operator(row))
        np.testing.assert_array_equal(
            chain.expectations[name].mean, kalman.estimate().mean
        )


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON], ids=str)
def test_mala_operators_are_kalman_mean_over_the_stored_draws(
    backend: Backend,
) -> None:
    # Issue #1059: `mala` takes the `operators` `sample` and `random_walk`
    # take, filtered bitwise as `KalmanMean` filters the stored draws on
    # either route; unstored, the expectations are the same numbers.
    target = GaussianTarget(diagonal_precision(12))
    operators = {"x": Power(1), "x2": Power(2)}

    def run(store_chain: bool) -> langevin.LangevinChain:
        return langevin.mala(
            target,
            torch.Generator().manual_seed(1059),
            1_000,
            step_size=0.3,
            operators=operators,
            store_chain=store_chain,
            backend=backend,
        )

    stored, free = run(True), run(False)
    assert free.draws.shape == (0, 12)
    for name, operator in operators.items():
        kalman = KalmanMean()
        for row in stored.draws:
            kalman.update(operator(row))
        expected = kalman.estimate()
        for chain in (stored, free):
            np.testing.assert_array_equal(chain.expectations[name].mean, expected.mean)
            np.testing.assert_array_equal(chain.expectations[name].phi, expected.phi)


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
        spread = np.hypot(rust.standard_error, python.standard_error)
        assert np.all(np.abs(rust.mean - python.mean) < 4.5 * spread)


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
    rng = np.random.default_rng(10080)
    theta = np.array([0.0, 0.3, -4.0, 0.0, 5.0, 0.0, 0.1, 0.0]) + 0.01 * rng.normal(
        size=8
    )
    momentum = rng.normal(size=8)
    torch_end = hmc.leapfrog(
        objective, torch.as_tensor(theta), torch.as_tensor(momentum), 0.002, 10
    )
    compiled = hmc.compiled_trajectory(
        objective, torch.as_tensor(theta), torch.as_tensor(momentum), 0.002, 10
    )
    position, velocity = compiled.position.numpy(), compiled.momentum.numpy()
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
            start=theta0,
            burn_in=200,
            store_chain=False,
            operators={"x": Power(1)},
            backend=backend,
        ).expectations["x"]
        for backend in (Backend.RUST, Backend.PYTHON)
    ]
    rust, python = expectations
    spread = np.hypot(rust.standard_error, python.standard_error)
    assert np.all(np.abs(rust.mean - python.mean) < 4.5 * spread)


def _sequences(n: int, length: int, counts: bool = False) -> np.ndarray:
    rng = np.random.default_rng(1008)
    states = np.zeros((n, length), dtype=int)
    for t in range(1, length):
        stay = rng.random(n) < 0.9
        states[:, t] = np.where(stay, states[:, t - 1], 1 - states[:, t - 1])
    means = np.array([2.0, 9.0]) if counts else np.array([-1.0, 1.0])
    if counts:
        drawn: np.ndarray = rng.poisson(means[states]).astype(np.float64)
    else:
        drawn = means[states] + 0.6 * rng.normal(size=states.shape)
    return drawn


_HMM_START = [
    0.0,
    np.log(0.1 / 0.9),
    np.log(0.9 / 0.1),
    -1.0,
    1.0,
    np.log(0.6),
    np.log(0.6),
]


@pytest.mark.oracle
def test_the_compiled_hmm_trajectory_is_the_torch_leapfrog() -> None:
    # Issue #1008: the declared Gaussian HMM's force is Fisher's identity
    # over the streamed statistics, step for step with autograd's leapfrog.
    objective = GaussianHmmObjective(_sequences(20, 50), 2, backend=Backend.TORCH)
    rng = np.random.default_rng(10081)
    theta = np.asarray(_HMM_START) + 0.05 * rng.normal(size=7)
    momentum = rng.normal(size=7)
    torch_end = hmc.leapfrog(
        objective, torch.as_tensor(theta), torch.as_tensor(momentum), 0.01, 10
    )
    compiled = hmc.compiled_trajectory(
        objective, torch.as_tensor(theta), torch.as_tensor(momentum), 0.01, 10
    )
    position, velocity = compiled.position.numpy(), compiled.momentum.numpy()
    np.testing.assert_allclose(
        position, torch_end.position.numpy(), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        velocity, torch_end.momentum.numpy(), rtol=1e-9, atol=1e-9
    )


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("family", ["gaussian-rust", "poisson-jax"])
def test_both_routes_sample_the_hmm_posterior_alike(family: str) -> None:
    # The Gaussian HMM runs the Rust walk, the Poisson HMM the JAX one
    # (`sample.hmc.jax`), each against the torch route in distribution.
    if family == "gaussian-rust":
        objective: Any = GaussianHmmObjective(_sequences(10, 50), 2)
        theta0 = torch.as_tensor(_HMM_START)
        step = 0.02
    else:
        objective = PoissonHmmObjective(_sequences(10, 50, counts=True), 2)
        theta0 = torch.as_tensor(
            [0.0, np.log(0.1 / 0.9), np.log(0.9 / 0.1), np.log(2.0), np.log(9.0)]
        )
        step = 0.02
    expectations = [
        hmc.sample(
            objective,
            torch.Generator().manual_seed(1008),
            500,
            step_size=step,
            n_steps=8,
            start=theta0,
            burn_in=100,
            store_chain=False,
            operators={"x": Power(1)},
            backend=backend,
        ).expectations["x"]
        for backend in (Backend.RUST, Backend.PYTHON)
    ]
    rust, python = expectations
    spread = np.hypot(rust.standard_error, python.standard_error)
    assert np.all(np.abs(rust.mean - python.mean) < 4.5 * spread)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_jax_walk_filters_and_warms_up_as_the_rust_one_does() -> None:
    objective = PoissonHmmObjective(_sequences(10, 40, counts=True), 2)
    theta0 = torch.as_tensor([0.0, -2.0, 2.0, np.log(2.0), np.log(9.0)])
    chain = hmc.sample(
        objective,
        torch.Generator().manual_seed(1008),
        400,
        step_size=0.05,
        n_steps=6,
        start=theta0,
        adaptation=hmc.Adaptation(200, 0.65, 0.2),
        operators={"x2": Power(2)},
    )
    assert chain.adapted is not None
    assert chain.adapted.mass_diagonal.shape == (5,)
    assert abs(chain.acceptance_rate - 0.65) < 0.2
    kalman = KalmanMean()
    for row in chain.draws:
        kalman.update(row**2)
    np.testing.assert_allclose(
        chain.expectations["x2"].mean,
        kalman.estimate().mean,
        rtol=1e-12,
    )
