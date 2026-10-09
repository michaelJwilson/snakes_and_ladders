"""The compiled negative-binomial dispersion solve under an exposure (issue #933, R4).

`src/count_mstep.rs::negative_binomial_dispersions_exposed` sums the digamma
half on the count weights' tails and the log half, with the term a varying
exposure keeps, per observation; `_solve_dispersion`, one torch solve per
state over every observation, stays as the oracle. The sums are reordered,
so bitwise is the target and #648's 2e-6 the floor; each draw's measured
difference is stated. Past `EXPOSED_TAIL_RATIO` the route falls back to the
oracle. Since #1410 the kernel steps by safeguarded Newton from a start, and
the oracle bisects: each lands within half the solve's 1e-12 of the root in
``log r``, so the two are held to that 1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal import emissions
from sal.backend import Backend
from sal.emissions import NegativeBinomialEmission, mstep

from tests._rows import every_row

#: The declared floor for a reordered sum through a bisection (#648).
FLOOR = 2e-06
#: The solve's own tolerance, on the bracket in ``log r``.
SOLVE_TOLERANCE = 1e-12


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
def test_the_kernel_is_the_per_state_solve() -> None:
    # Measured (#1410): a relative 2.4e-13 at scale 5, 6.0e-13 at 500 and
    # 8.6e-13 at a constant exposure from the bracket's midpoint, in 8 to 11
    # score evaluations where the bisection takes 45; from 1.1 times the
    # answer, 2.6e-13, 3.9e-13 and 8.6e-13 in 5 or 6. Every boundary decision
    # agrees.
    def check(scale: float, varying: bool) -> None:
        values, weights, offsets = _draw(2_000, scale, 0, varying)
        means = ((weights.T @ values) / (weights.T @ offsets)).tolist()
        oracle = [
            mstep.solve_dispersion(values, weights[:, k], offsets * means[k])
            for k in range(4)
        ]
        assert [o.iterations for o in oracle] == [45] * 4
        for starts in (None, [1.1 * o.value for o in oracle]):
            rust = mstep.solve_dispersion_exposed_rust(
                values, weights, means, offsets, starts=starts
            )
            assert [r.at_boundary for r in rust] == [o.at_boundary for o in oracle]
            assert max(r.iterations for r in rust) <= (11 if starts is None else 6)
            moved = max(
                abs(r.value - o.value) / o.value
                for r, o in zip(rust, oracle, strict=True)
            )
            assert moved < SOLVE_TOLERANCE

    every_row([(5.0, True), (500.0, True), (50.0, False)], check)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_m_step_routes_through_the_kernel_and_falls_back_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = NegativeBinomialEmission([3.0] * 4, [1.0, 2.0, 4.0, 8.0])
    values, weights, offsets = _draw(2_000, 50.0, 1)
    covariate = offsets.unsqueeze(-1)
    compiled = family.reestimate(values, weights, covariate).components
    monkeypatch.setattr(emissions, "M_STEP_BACKEND", Backend.PYTHON)
    oracle = family.reestimate(values, weights, covariate).components
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
    ).components
    monkeypatch.setattr(emissions, "M_STEP_BACKEND", Backend.RUST)
    routed = family.reestimate(
        wide, wide_weights, wide_offsets.unsqueeze(-1)
    ).components
    assert torch.equal(routed.dispersion, fallback.dispersion)
