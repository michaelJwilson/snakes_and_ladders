"""Benchmarks for the Potts-chain objective and its gradient.

See tests/regression/test_opt_potts.py for correctness. Two shapes are
measured: the objective alone, and one objective-plus-backward pass, which is
the unit a fitting loop actually repeats -- the ratio between them is what
says whether the reverse pass is worth optimizing.

The normalizer's two routes are measured beside them at the stress instance
the ranking reads -- ``q = 3``, a chain of 64 -- and not at the fixture's own
length of 12, where ``squaring_is_cheaper`` takes the recursion and a ratio
would decide nothing (root ``CLAUDE.md``: a speedup is established at stress
sizes alone). Issue #754.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.fixtures import load_params
from sal.opt.potts import (
    PottsObjective,
    log_partition_by_recursion,
    log_partition_by_squaring,
)
from sal.sim.potts_chain import PottsParams, simulate_chains

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "potts_chain/ci.yaml"

#: The chain the stress-tier ranking profiles `hmc.sample` on (issue #754).
STRESS_LENGTH = 64


def test_potts_objective_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_params(FIXTURE, PottsParams)
    objective = PottsObjective(simulate_chains(params), params.n_states)
    theta = objective.theta_from_truth(params.coupling, params.field)

    result = benchmark(objective, theta)

    # Benchmarks only assert finiteness -- correctness is pinned in
    # tests/regression/test_opt_potts.py.
    assert math.isfinite(float(result))


def test_potts_objective_and_gradient_benchmark(benchmark: BenchmarkFixture) -> None:
    params = load_params(FIXTURE, PottsParams)
    objective = PottsObjective(simulate_chains(params), params.n_states)
    theta = objective.theta_from_truth(params.coupling, params.field)

    def _value_and_gradient() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    assert math.isfinite(benchmark(_value_and_gradient))


@pytest.mark.parametrize(
    "route",
    [log_partition_by_recursion, log_partition_by_squaring],
    ids=lambda f: f.__name__,
)
def test_log_partition_route_benchmark(
    benchmark: BenchmarkFixture,
    route: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> None:
    params = load_params(FIXTURE, PottsParams)
    coupling = torch.tensor(params.coupling, dtype=torch.float64)
    field = torch.as_tensor(params.field, dtype=torch.float64)

    result = benchmark(route, coupling, field, STRESS_LENGTH)

    assert math.isfinite(float(result))


@pytest.mark.parametrize(
    "route",
    [log_partition_by_recursion, log_partition_by_squaring],
    ids=lambda f: f.__name__,
)
def test_log_partition_route_and_gradient_benchmark(
    benchmark: BenchmarkFixture,
    route: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> None:
    # The tape is the half of the cost the forward pass does not show: the
    # recursion builds one node per site and autograd walks every one of them.
    params = load_params(FIXTURE, PottsParams)
    field = torch.as_tensor(params.field, dtype=torch.float64)

    def _value_and_gradient() -> float:
        coupling = torch.tensor(
            params.coupling, dtype=torch.float64, requires_grad=True
        )
        value = route(coupling, field, STRESS_LENGTH)
        torch.autograd.grad(value, coupling)
        return float(value.detach())

    assert math.isfinite(benchmark(_value_and_gradient))
