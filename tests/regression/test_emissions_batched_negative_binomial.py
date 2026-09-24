"""The batched negative-binomial dispersion solve against the per-state solve it replaced (issue #918).

`mstep.solve_dispersion_batched` runs every state's bisection on
``log r`` at once, on the distinct counts weighted by the responsibility
summed at each. `_solve_dispersion`, one state at a time over every
observation, is kept as the oracle. The sums are reordered, so bitwise is the
target and #648's 2e-06 the declared floor; the difference each draw measures
is asserted at that floor and stated beside it. Two call paths: the count
pair's total channel on `emission_mixture`, and the family's own M step,
which is what EM calls.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import (
    CountPairEmission,
    NegativeBinomialEmission,
    mstep,
)
from snakes_and_ladders.opt.mixture import responsibilities
from snakes_and_ladders.sim.emission_mixture import simulate_emission_mixture
from snakes_and_ladders.sim.fixtures import fixture

#: The declared floor for a reordered sum through a bisection (#648).
FLOOR = 2e-06


def _draw(tier: str) -> tuple[torch.Tensor, torch.Tensor, NegativeBinomialEmission]:
    """The totals of a draw, the responsibilities at the truth, and the total channel."""
    params = fixture("emission_mixture", tier).params
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    values = torch.as_tensor(
        simulate_emission_mixture(params).observations, dtype=torch.float64
    )
    posterior = responsibilities(
        values, torch.log(torch.as_tensor(params.weights, dtype=torch.float64)), truth
    )
    total = truth.total
    assert isinstance(total, NegativeBinomialEmission)
    return values[:, 0], posterior, total


def _oracle(
    values: torch.Tensor, posterior: torch.Tensor
) -> tuple[torch.Tensor, list[mstep.SolvedDispersion]]:
    """The profiled means and the per-state solve, state by state, as before #918."""
    mean = (posterior.T @ values) / posterior.sum(dim=0)
    return mean, [
        mstep.solve_dispersion(values, posterior[:, k], float(mean[k]))
        for k in range(posterior.shape[1])
    ]


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("tier", ["ci", "stress"])
def test_the_batched_dispersion_is_the_per_state_solve_at_the_floor(tier: str) -> None:
    # K = 3 over 900 totals, K = 10 over 3,000. Measured: bitwise on ci; on
    # stress 7 of 10 bitwise, three within 5.9e-13 relative. Brackets, boundary
    # decisions and iteration counts are the oracle's.
    values, posterior, _ = _draw(tier)
    mean, oracle = _oracle(values, posterior)
    batched = mstep.solve_dispersion_batched(
        values, posterior, [float(m) for m in mean]
    )
    assert [one.at_boundary for one in batched] == [o.at_boundary for o in oracle]
    assert [one.iterations for one in batched] == [o.iterations for o in oracle]
    moved = max(
        abs(one.value - o.value) / o.value
        for one, o in zip(batched, oracle, strict=True)
    )
    assert moved < FLOOR, f"the solve moved {moved:.3e}"


@pytest.mark.oracle
@pytest.mark.backend
def test_the_family_m_step_runs_the_batched_solve() -> None:
    # Through `reestimate`, which EM calls: the mean is the closed form it
    # was, and the dispersion is the oracle's at the floor.
    values, posterior, family = _draw("stress")
    mean, oracle = _oracle(values, posterior)
    fitted = family.reestimate(values, posterior)
    assert torch.equal(fitted.emissions.mean, mean)
    expected = torch.tensor([o.value for o in oracle], dtype=torch.float64)
    moved = float(((fitted.emissions.dispersion - expected).abs() / expected).max())
    assert moved < FLOOR
    assert fitted.iterations == max(o.iterations for o in oracle)


@pytest.mark.analytic
def test_a_state_past_the_identifiable_bound_is_reported_at_it() -> None:
    # A sample with no spread at all has no overdispersion to find, so its
    # score is still positive at the bound and the solve stops there, flagged,
    # as the oracle does; an overdispersed one is solved inside the bracket.
    rng = np.random.default_rng(918)
    constant = np.full(2_000, 4)
    spread = rng.negative_binomial(2.0, 2.0 / (2.0 + 4.0), 2_000)
    values = torch.as_tensor(np.concatenate([constant, spread]), dtype=torch.float64)
    weights = torch.zeros((4_000, 2), dtype=torch.float64)
    weights[:2_000, 0] = 1.0
    weights[2_000:, 1] = 1.0
    means = [float(values[:2_000].mean()), float(values[2_000:].mean())]
    batched = mstep.solve_dispersion_batched(values, weights, means)
    oracle = [mstep.solve_dispersion(values, weights[:, k], means[k]) for k in (0, 1)]
    assert [one.at_boundary for one in batched] == [True, False]
    assert [one.at_boundary for one in oracle] == [True, False]
    assert batched[0].value == oracle[0].value
    assert abs(batched[1].value - oracle[1].value) / oracle[1].value < FLOOR
