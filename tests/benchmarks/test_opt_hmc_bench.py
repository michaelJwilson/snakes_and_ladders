"""Benchmarks for HMC, in gradient evaluations rather than seconds.

Correctness is pinned in `tests/regression/opt/`, per the repo's division of
labor between the two directories.

What decides whether a sampler is affordable is the cost of one gradient of
the objective times the trajectory length, and `search/CLAUDE.md`'s rule that
a budget is counted in work rather than wall-clock applies here for the same
reason: a chain's cost is `n_samples * n_steps` gradients, which is
reproducible from the seed, and the seconds are not.
"""

from __future__ import annotations

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.opt.hmc import sample
from snakes_and_ladders.opt.testfunctions import Rosenbrock


@pytest.mark.parametrize("n_steps", [5, 20])
@pytest.mark.parametrize("dimension", [2, 10])
def test_hmc_sample_benchmark(
    benchmark: BenchmarkFixture, dimension: int, n_steps: int
) -> None:
    # Rosenbrock as a density is a banana-shaped target, which is the standard
    # hard case for a sampler with an isotropic momentum: the step size is
    # bounded by the narrow direction while the long direction needs many
    # steps to traverse.
    objective = Rosenbrock(dimension=dimension)

    chain = benchmark(
        sample, objective, 1, 200, step_size=0.01, n_steps=n_steps, burn_in=20
    )

    assert torch.isfinite(chain.theta).all()
