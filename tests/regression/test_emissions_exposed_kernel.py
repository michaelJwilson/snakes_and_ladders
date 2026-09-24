"""The compiled negative-binomial dispersion solve under an exposure (issue #933, R4).

`src/count_mstep.rs::negative_binomial_dispersions_exposed` sums the digamma
half on the count weights' tails and the log half, with the term a varying
exposure keeps, per observation; `_solve_dispersion`, one torch solve per
state over every observation, stays as the oracle. The sums are reordered,
so bitwise is the target and #648's 2e-6 the floor; each draw's measured
difference is stated. Past `EXPOSED_TAIL_RATIO` the route falls back to the
oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders import emissions
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.emissions import NegativeBinomialEmission, mstep

#: The declared floor for a reordered sum through a bisection (#648).
FLOOR = 2e-06


def _draw(
    n: int, scale: float, seed: int, varying: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Counts from four states at ``scale`` times (0.5, 1, 2, 4), their exposures and a posterior."""
    rng = np.random.default_rng([933, 4, seed])
    means = scale * np.array([0.5, 1.0, 2.0, 4.0])
    states = rng.integers(0, 4, n)
    exposure = rng.uniform(0.5, 2.0, n) if varying else np.full(n, 1.5)
    counts = rng.negative_binomial(6.0, 6.0 / (6.0 + exposure * means[states]))
    posterior = np.full((n, 4), 0.02)
    posterior[np.arange(n), states] = 0.94
    return (
        torch.as_tensor(counts, dtype=torch.float64),
        torch.as_tensor(posterior),
        torch.as_tensor(exposure),
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize(
    ("scale", "varying"), [(5.0, True), (500.0, True), (50.0, False)]
)
def test_the_kernel_is_the_per_state_solve(scale: float, varying: bool) -> None:
    # Measured: bitwise at scale 5, a relative 1.8e-12 at 500, bitwise at a
    # constant exposure; every state's steps and boundary decision agree.
    values, weights, offsets = _draw(2_000, scale, 0, varying)
    means = ((weights.T @ values) / (weights.T @ offsets)).tolist()
    rust = mstep.solve_dispersion_exposed_rust(values, weights, means, offsets)
    oracle = [
        mstep.solve_dispersion(values, weights[:, k], offsets * means[k])
        for k in range(4)
    ]
    assert [r.at_boundary for r in rust] == [o.at_boundary for o in oracle]
    assert [r.iterations for r in rust] == [o.iterations for o in oracle]
    moved = max(
        abs(r.value - o.value) / o.value for r, o in zip(rust, oracle, strict=True)
    )
    assert moved < FLOOR


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_m_step_routes_through_the_kernel_and_falls_back_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = NegativeBinomialEmission([3.0] * 4, [1.0, 2.0, 4.0, 8.0])
    values, weights, offsets = _draw(2_000, 50.0, 1)
    covariate = offsets.unsqueeze(-1)
    compiled = family.reestimate(values, weights, covariate).emissions
    monkeypatch.setattr(emissions, "M_STEP_BACKEND", Backend.PYTHON)
    oracle = family.reestimate(values, weights, covariate).emissions
    assert torch.equal(compiled.mean, oracle.mean)
    moved = float(
        ((compiled.dispersion - oracle.dispersion).abs() / oracle.dispersion).max()
    )
    assert moved < FLOOR
    # Counts wider than the cap: the compiled route declines and the oracle
    # answers, so the two backends agree bitwise.
    wide, wide_weights, wide_offsets = _draw(200, 5_000.0, 2)
    assert float(wide.max()) > emissions.EXPOSED_TAIL_RATIO * wide.numel()
    with pytest.raises(mstep.NoTails):
        mstep.solve_dispersion_exposed_rust(
            wide, wide_weights, [1.0, 2.0, 3.0, 4.0], wide_offsets
        )
    fallback = family.reestimate(
        wide, wide_weights, wide_offsets.unsqueeze(-1)
    ).emissions
    monkeypatch.setattr(emissions, "M_STEP_BACKEND", Backend.RUST)
    routed = family.reestimate(wide, wide_weights, wide_offsets.unsqueeze(-1)).emissions
    assert torch.equal(routed.dispersion, fallback.dispersion)
