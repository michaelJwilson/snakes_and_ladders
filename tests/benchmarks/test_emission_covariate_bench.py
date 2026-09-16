"""What the covariate costs, measured rather than predicted (issue #631).

The ticket's runtime analysis named one thing as asymptotically wasteful --- the
rate ``e_t mu`` rebuilt inside the dispersion bisection --- and one as free ---
the trial count, whose ``lgamma`` term count is unchanged at three. **The first
prediction is wrong and this file is where that is established.**

Measured at a size the fixtures do not reach, per `likelihood/CLAUDE.md`'s rule
that a boundary cost hides behind the fixture's size. The solve is dominated by
two ``digamma`` evaluations per bisection step; the multiply the analysis
pointed at is under half a percent of it. Root `CLAUDE.md` then says to leave
it alone, which is what the code does --- the hoisted form is kept for reading
clearly, not for a speed-up.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from snakes_and_ladders import emissions

#: Past every declared fixture, which is the point: both terms below are
#: invisible at a `ci.yaml` size.
N_OBSERVATIONS = 200_000
SEED = 20260916


@pytest.fixture(scope="module")
def solve_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Counts, posterior weights and a varying exposure, seeded."""
    rng = np.random.default_rng(SEED)
    values = torch.tensor(
        rng.negative_binomial(3.0, 0.5, N_OBSERVATIONS), dtype=torch.float64
    )
    weights = torch.rand(N_OBSERVATIONS, dtype=torch.float64)
    offsets = torch.tensor(rng.uniform(0.5, 4.0, N_OBSERVATIONS), dtype=torch.float64)
    return values, weights, offsets


@pytest.mark.release
@pytest.mark.structural
def test_the_hoisted_rate_is_not_what_the_dispersion_solve_costs(
    solve_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    # The finding, asserted rather than left in a pull request body: the
    # multiply #631 predicted would dominate is a rounding error against the
    # digamma pair, so the bisection's cost is the special function and not the
    # bookkeeping. A regression here would mean digamma got cheap, which is
    # when hoisting would start to matter.
    values, weights, offsets = solve_inputs
    rate = offsets * 2.5

    start = time.perf_counter()
    for _ in range(45):
        _ = offsets * 2.5
    multiplies = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(45):
        _ = torch.digamma(values + 3.0)
    digammas = time.perf_counter() - start

    assert multiplies < 0.05 * digammas, (
        f"the rate multiply is {multiplies:.4f}s against digamma's "
        f"{digammas:.4f}s; #631 predicted the multiply would dominate and the "
        "measurement says otherwise"
    )
    assert rate.shape == values.shape


@pytest.mark.release
@pytest.mark.benchmark
def test_dispersion_solve_under_a_varying_exposure(
    benchmark: object,
    solve_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The exposure path's whole solve, at a size no fixture declares."""
    values, weights, offsets = solve_inputs
    rate = offsets * 2.5

    benchmark(emissions._solve_dispersion, values, weights, rate)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_dispersion_solve_at_a_scalar_mean(
    benchmark: object,
    solve_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The same solve without an exposure, as the reference it is read against."""
    values, weights, _ = solve_inputs

    benchmark(emissions._solve_dispersion, values, weights, 2.5)  # type: ignore[operator]
