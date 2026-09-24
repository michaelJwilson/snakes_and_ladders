"""The count M-step solves, batched torch against the compiled kernel (issue #922).

At the `spatio_sequential_counts/release` projection's size, 100 components
over 4,000 pairs, the size #892 and #918 were measured at: the success
channel's beta-binomial solve (one trial count per component) and the total
channel's dispersion solve, each by the batched torch oracle and by
`src/count_mstep.rs`. They agree to #648's floor
(`tests/regression/test_emissions_count_mstep_kernel.py`); this measures the
ratio root `CLAUDE.md` keeps a port on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import BetaBinomialEmission, mstep
from snakes_and_ladders.opt.mixture import responsibilities
from snakes_and_ladders.search.projection import flatten, project
from snakes_and_ladders.sim.count_pairs import binned_model
from snakes_and_ladders.sim.fixtures import fixture

#: Observations the projection draws, as #892 and #918 measured.
N_OBSERVATIONS = 4_000

Problem = tuple[
    torch.Tensor, torch.Tensor, list[float], list[float], list[float], list[float]
]


@pytest.fixture(scope="module")
def problem() -> Problem:
    """Both channels, the responsibilities at the truth, and each solve's inputs."""
    params = binned_model(
        fixture("spatio_sequential_counts", "release").params.model, 1
    )
    instance = project(params, N_OBSERVATIONS, np.random.default_rng([541, 2]))
    truth = flatten(params)
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    k = truth.n_states
    posterior = responsibilities(
        values, torch.full((k,), -float(np.log(k)), dtype=torch.float64), truth
    )
    channel = truth._successes
    assert isinstance(channel, BetaBinomialEmission)
    named = channel.named_parameters()
    totals = [float(a + b) for a, b in zip(named["alpha"], named["beta"], strict=True)]
    rates = [float(a) / t for a, t in zip(named["alpha"], totals, strict=True)]
    means = [float(m) for m in (posterior.T @ values[:, 0]) / posterior.sum(dim=0)]
    trials = [float(t) for t in channel.trials]
    return values, posterior, trials, rates, totals, means


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("route", ["batched", "rust"])
def test_beta_binomial_solve(benchmark: object, problem: Problem, route: str) -> None:
    values, posterior, trials, rates, totals, _ = problem
    solve = getattr(mstep, f"solve_beta_binomial_{route}")
    benchmark(solve, values[:, 1], posterior, trials, rates, totals)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("route", ["batched", "rust"])
def test_dispersion_solve(benchmark: object, problem: Problem, route: str) -> None:
    values, posterior, _, _, _, means = problem
    solve = getattr(mstep, f"solve_dispersion_{route}")
    benchmark(solve, values[:, 0], posterior, means)  # type: ignore[operator]
