"""One emission M step: the family's closed form against `LbfgsMStep` (issue #1171).

The L-BFGS step is for a family with no closed form or with parameters held;
on a closed-form family it is the cost of not using the formula. Timed on a
three-state Gaussian over 20 chains of 1,000 positions, an HMM posterior's
shape. See tests/regression/opt/test_opt_m_step.py for correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.emissions import GaussianEmission
from sal.opt.m_step import LbfgsMStep


def _problem() -> tuple[GaussianEmission, torch.Tensor, torch.Tensor]:
    """The family, 2 x 10^4 observations and a Dirichlet posterior over them."""
    rng = np.random.default_rng(1171)
    observations = torch.as_tensor(rng.normal(size=(20, 1000)))
    posterior = torch.as_tensor(rng.dirichlet(np.ones(3), size=(20, 1000)))
    family = GaussianEmission(
        np.array([-1.0, 0.0, 1.0]), np.ones(3), variance_floor=1e-6
    )
    return family, observations, posterior


@pytest.mark.parametrize("route", ["closed_form", "lbfgs"])
def test_gaussian_m_step_benchmark(benchmark: BenchmarkFixture, route: str) -> None:
    family, observations, posterior = _problem()
    if route == "closed_form":
        result = benchmark(family.reestimate, observations, posterior)
    else:
        result = benchmark(LbfgsMStep(), family, observations, posterior, None)
    assert result.converged
