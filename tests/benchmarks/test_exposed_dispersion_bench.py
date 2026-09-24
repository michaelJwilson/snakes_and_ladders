"""The negative-binomial dispersion solve under an exposure, torch against the compiled kernel (issue #933, R4).

Four states over 20,000 counts at exposures in [0.5, 2], state means 25-200
(tails about 1,100 wide): the per-state torch oracle and
`src/count_mstep.rs::negative_binomial_dispersions_exposed`, which agree to
#648's floor (`tests/regression/test_emissions_exposed_kernel.py`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import mstep


@pytest.fixture(scope="module")
def problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    rng = np.random.default_rng(933)
    n = 20_000
    means = np.array([25.0, 50.0, 100.0, 200.0])
    states = rng.integers(0, 4, n)
    exposure = rng.uniform(0.5, 2.0, n)
    counts = rng.negative_binomial(6.0, 6.0 / (6.0 + exposure * means[states]))
    posterior = np.full((n, 4), 0.02)
    posterior[np.arange(n), states] = 0.94
    values = torch.as_tensor(counts, dtype=torch.float64)
    weights = torch.as_tensor(posterior)
    offsets = torch.as_tensor(exposure)
    return (
        values,
        weights,
        offsets,
        ((weights.T @ values) / (weights.T @ offsets)).tolist(),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("route", ["torch", "rust"])
def test_exposed_dispersion_solve(
    benchmark: object,
    problem: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]],
    route: str,
) -> None:
    values, weights, offsets, means = problem
    if route == "rust":
        solved = benchmark(  # type: ignore[operator]
            mstep.solve_dispersion_exposed_rust, values, weights, means, offsets
        )
    else:
        solved = benchmark(  # type: ignore[operator]
            lambda: [
                mstep.solve_dispersion(values, weights[:, k], offsets * means[k])
                for k in range(4)
            ]
        )
    assert len(solved) == 4
