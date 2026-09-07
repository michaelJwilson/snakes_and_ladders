"""Benchmarks for the Potts-chain objective and its gradient.

See tests/regression/test_opt_potts.py for correctness. Two shapes are
measured: the objective alone, and one objective-plus-backward pass, which is
the unit a fitting loop actually repeats -- the ratio between them is what
says whether the reverse pass is worth optimizing.
"""

from __future__ import annotations

import math

import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.potts import (
    PottsObjective,
    load_potts_params,
    simulate_chains,
)

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "potts_params.yaml"


def test_potts_objective_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_potts_params(FIXTURE)
    objective = PottsObjective(simulate_chains(params), params.n_states)
    theta = objective.theta_from_truth(params.coupling, params.field)

    result = benchmark(objective, theta)

    # Benchmarks only assert finiteness -- correctness is pinned in
    # tests/regression/test_opt_potts.py.
    assert math.isfinite(float(result))


def test_potts_objective_and_gradient_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_potts_params(FIXTURE)
    objective = PottsObjective(simulate_chains(params), params.n_states)
    theta = objective.theta_from_truth(params.coupling, params.field)

    def _value_and_gradient() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    assert math.isfinite(benchmark(_value_and_gradient))
