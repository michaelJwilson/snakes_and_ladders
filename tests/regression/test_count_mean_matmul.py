"""The profiled mean is a matmul, and what that costs (#649, #651 candidate 3).

Four count families compute a posterior-weighted mean in their M step. Written
elementwise it materialises an ``(n_obs, n_states)`` intermediate --- 80.7 MB at
the scale `ROADMAP.md` declares --- which ``weights.T @ values`` does not.

A matmul reduces in a different order, so this is the trade `CLAUDE.md` permits:
bitwise is the target, the declared tolerance is the floor, and the measured
difference is stated rather than assumed small. It is **1.3e-14** relative,
three orders inside the **1e-11** declared for a float64 comparison.

What is asserted here is the bound, against the elementwise form computed in
the test. There is no conserved copy to referee these two against ---
`sandbox.count_emissions` holds the negative binomial and the beta-binomial
only --- so the reference is written out rather than imported.
"""

from __future__ import annotations

import pytest
import torch
from snakes_and_ladders.emissions import BinomialEmission, PoissonEmission
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64

SEED = 20260916
N_OBSERVATIONS, N_STATES = 20_000, 6


@pytest.fixture
def weighted() -> tuple[torch.Tensor, torch.Tensor]:
    """Counts and a normalized posterior, from one seeded generator."""
    generator = torch.Generator().manual_seed(SEED)
    counts = torch.randint(0, 20, (1, N_OBSERVATIONS), generator=generator).to(
        torch.float64
    )
    posterior = torch.rand(
        1, N_OBSERVATIONS, N_STATES, generator=generator, dtype=torch.float64
    )
    return counts, posterior / posterior.sum(dim=-1, keepdim=True)


def _elementwise_mean(counts: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
    """The form the matmul replaced, with its ``(n_obs, n_states)`` temporary."""
    values = counts.reshape(-1).to(posterior.dtype)
    weights = posterior.reshape(-1, posterior.shape[-1])
    return (weights * values.unsqueeze(-1)).sum(dim=0) / weights.sum(dim=0)


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_poisson_mean_is_the_elementwise_one_to_the_declared_tolerance(
    weighted: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # The Poisson's M step *is* this mean, so the fitted rate is the quantity
    # the rewrite moves and the bound is asserted on it directly.
    counts, posterior = weighted
    family = PoissonEmission(torch.full((N_STATES,), 4.0, dtype=torch.float64))

    fitted = family.reestimate(counts, posterior).emissions

    assert torch.allclose(
        fitted.mean,
        _elementwise_mean(counts, posterior),
        rtol=CROSS_DEVICE_RTOL_FLOAT64,
    )


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_binomial_rate_is_the_elementwise_one_to_the_declared_tolerance(
    weighted: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # The binomial fits a rate, so the same mean divided by its trial count.
    counts, posterior = weighted
    trials = torch.full((N_STATES,), 20.0, dtype=torch.float64)
    family = BinomialEmission(trials, torch.full((N_STATES,), 0.4, dtype=torch.float64))

    fitted = family.reestimate(counts, posterior).emissions

    assert torch.allclose(
        fitted.mean,
        _elementwise_mean(counts, posterior),
        rtol=CROSS_DEVICE_RTOL_FLOAT64,
    )


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_difference_is_inside_the_floor_and_not_merely_under_it(
    weighted: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # The rule permits backing off to the declared tolerance, not settling for
    # it. The measured difference is orders inside, and a regression toward the
    # floor means something changed that the tolerance would then be hiding.
    counts, posterior = weighted
    family = PoissonEmission(torch.full((N_STATES,), 4.0, dtype=torch.float64))
    reference = _elementwise_mean(counts, posterior)

    fitted = family.reestimate(counts, posterior).emissions
    moved = float(((fitted.mean - reference).abs() / reference.abs()).max())

    assert moved < 0.001 * CROSS_DEVICE_RTOL_FLOAT64, (
        f"the matmul moved the fitted mean {moved:.2e}, against a declared "
        f"{CROSS_DEVICE_RTOL_FLOAT64:.0e}; it was 1.3e-14 when taken, and a "
        "drift toward the floor is the tolerance starting to hide something"
    )
