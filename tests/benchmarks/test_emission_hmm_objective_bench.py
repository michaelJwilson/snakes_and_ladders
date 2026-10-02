"""`EmissionHmmObjective` value and gradient: the compiled E step against autograd (issue #1169).

The declared route runs ``oxisal.ragged_posteriors`` and one backward pass of
the Fisher-identity surrogate; the autograd route differentiates the torch
forward recursion. `tests/regression/opt/test_opt_hmm_emission_objective.py`
pins the two at 1e-10; this measures the ratio. Four-state sticky
negative-binomial chains, the fixture of `test_hmm_ragged_estep_bench.py`.
The ratio is read at the stress size, 200 chains of 100-3,000 positions,
marked ``release``; the gate size times the same two routes per PR and
decides nothing (root ``CLAUDE.md``, Measurement).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.emissions import NegativeBinomialEmission
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.objective import autograd_value_and_gradient
from sal.ragged import Ragged

ROUTES: dict[
    str,
    Callable[[EmissionHmmObjective, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
] = {
    "compiled": lambda objective, theta: objective.value_and_gradient(theta),
    "autograd": autograd_value_and_gradient,
}


def _objective(
    n_chains: int, shortest: int, longest: int
) -> tuple[EmissionHmmObjective, torch.Tensor]:
    """Sticky negative-binomial chains over four states, seeded, and a point off the start."""
    rng = np.random.default_rng(1169)
    lengths = rng.integers(shortest, longest, n_chains)
    means = np.geomspace(2.0, 60.0, 4)
    chains = []
    for length in lengths:
        states = np.zeros(length, dtype=np.int64)
        for t in range(1, length):
            states[t] = states[t - 1] if rng.random() < 0.97 else rng.integers(0, 4)
        chains.append(rng.negative_binomial(8.0, 8.0 / (8.0 + means[states])))
    batch = Ragged(np.concatenate(chains).astype(float), tuple(int(x) for x in lengths))
    objective = EmissionHmmObjective(
        batch, NegativeBinomialEmission([3.0] * 4, means * 1.3)
    )
    theta = objective.initial()
    return objective, theta + 0.01 * torch.arange(theta.numel(), dtype=torch.float64)


@pytest.fixture(scope="module")
def gate() -> tuple[EmissionHmmObjective, torch.Tensor]:
    # The gate size: thirty chains of 40-300 positions.
    return _objective(30, 40, 300)


@pytest.mark.benchmark
@pytest.mark.parametrize("route", list(ROUTES))
def test_value_and_gradient_at_gate_size(
    benchmark: object, gate: tuple[EmissionHmmObjective, torch.Tensor], route: str
) -> None:
    objective, theta = gate
    value, _ = benchmark(lambda: ROUTES[route](objective, theta))  # type: ignore[operator]
    assert math.isfinite(float(value))


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("route", list(ROUTES))
def test_value_and_gradient_at_stress_size(benchmark: object, route: str) -> None:
    # 200 chains of 100-3,000 positions, the size the ragged E step was kept for.
    objective, theta = _objective(200, 100, 3000)
    value, _ = benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: ROUTES[route](objective, theta), rounds=5, iterations=1
    )
    assert math.isfinite(float(value))
