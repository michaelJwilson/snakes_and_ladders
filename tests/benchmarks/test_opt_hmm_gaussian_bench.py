"""Benchmarks for the Gaussian-emission HMM objective and its gradient.

See tests/regression/opt/test_opt_hmm_gaussian.py for correctness. Measured
beside the categorical benchmark at the same shape, because the pair is the
only statement about what the emission seam costs: the recursion is shared,
so any difference between the two is the emission term and nothing else.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.opt.hmm import GaussianHmmObjective
from snakes_and_ladders.sim.hmm import HmmParams, load_hmm_params, simulate_sequences

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "hmm_params.yaml"


def _objective() -> tuple[GaussianHmmObjective, torch.Tensor]:
    """The Gaussian instance at the categorical fixture's shape, and a truth point."""
    categorical = load_hmm_params(FIXTURE)
    mean = np.linspace(-3.0, 3.0, categorical.n_states)
    scale = np.full(categorical.n_states, 1.0)
    params = HmmParams(
        n_states=categorical.n_states,
        sequence_length=categorical.sequence_length,
        n_sequences=categorical.n_sequences,
        initial=categorical.initial,
        transition=categorical.transition,
        emissions=GaussianEmission(mean, scale, 1e-12),
        seed=categorical.seed,
        tolerance=categorical.tolerance,
    )
    objective = GaussianHmmObjective(
        simulate_sequences(params).observations, params.n_states
    )
    return objective, objective.theta_from_truth(
        params.initial, params.transition, mean, scale
    )


def test_gaussian_hmm_objective_benchmark(benchmark: BenchmarkFixture) -> None:
    objective, theta = _objective()

    result = benchmark(objective, theta)

    # Benchmarks only assert finiteness -- correctness is pinned in
    # tests/regression/opt/test_opt_hmm_gaussian.py.
    assert math.isfinite(float(result))


def test_gaussian_hmm_objective_and_gradient_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    objective, theta = _objective()

    def _value_and_gradient() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    assert math.isfinite(benchmark(_value_and_gradient))
