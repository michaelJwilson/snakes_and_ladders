"""Benchmarks for the HMM forward objective and its gradient.

See tests/regression/test_opt_hmm.py for correctness. The forward recursion
is sequential in the chain length and batched over sequences, so both shapes
below are measured at the full fixture rather than a slice -- a per-sequence
number would hide the part that does not parallelize.
"""

from __future__ import annotations

import math

import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.hmm import HmmObjective
from snakes_and_ladders.sim.hmm import load_hmm_params, simulate_sequences

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "hmm_params.yaml"


def test_hmm_objective_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_hmm_params(FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    theta = objective.theta_from_truth(
        params.initial, params.transition, params.emission
    )

    result = benchmark(objective, theta)

    # Benchmarks only assert finiteness -- correctness is pinned in
    # tests/regression/test_opt_hmm.py.
    assert math.isfinite(float(result))


def test_hmm_objective_and_gradient_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_hmm_params(FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    theta = objective.theta_from_truth(
        params.initial, params.transition, params.emission
    )

    def _value_and_gradient() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    assert math.isfinite(benchmark(_value_and_gradient))
