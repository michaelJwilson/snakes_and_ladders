"""Benchmarks for the mixture objective and its seeding.

See tests/regression/opt/test_opt_mixture.py for correctness. The seeding is
timed beside the objective because the two are what a caller trades off:
`search/CLAUDE.md`'s budget rule counts a restart as a fit, and a seeding that
cost as much as the fit it starts would not be worth the choice.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.opt.mixture import (
    GaussianMixtureObjective,
    kmeans_plus_plus,
)
from snakes_and_ladders.sim.mixture import MixtureParams, simulate_mixture

WEIGHTS = np.array([0.2, 0.3, 0.5])
MEAN = np.array([-6.0, 0.0, 6.0])
SCALE = np.array([1.0, 1.5, 1.0])


def _observations(n_samples: int = 4000) -> np.ndarray:
    """A three-component fixture at a size a real fit would see."""
    params = MixtureParams(
        weights=WEIGHTS,
        components=GaussianEmission(MEAN, SCALE, 1e-12),
        n_samples=n_samples,
        seed=20260905,
        tolerance=1e-12,
    )
    return simulate_mixture(params).observations


def test_mixture_objective_benchmark(benchmark: BenchmarkFixture) -> None:
    objective = GaussianMixtureObjective(_observations(), 3)
    theta = objective.theta_from_truth(WEIGHTS, MEAN, SCALE)

    result = benchmark(objective, theta)

    # Benchmarks only assert finiteness -- correctness is pinned in
    # tests/regression/opt/test_opt_mixture.py.
    assert math.isfinite(float(result))


def test_mixture_objective_and_gradient_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    objective = GaussianMixtureObjective(_observations(), 3)
    theta = objective.theta_from_truth(WEIGHTS, MEAN, SCALE)

    def _value_and_gradient() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    assert math.isfinite(benchmark(_value_and_gradient))


def test_kmeans_plus_plus_seeding_benchmark(benchmark: BenchmarkFixture) -> None:
    observations = _observations()
    rng = np.random.default_rng(20260905)

    centres = benchmark(kmeans_plus_plus, observations, 3, rng)

    assert centres.shape == (3,)
