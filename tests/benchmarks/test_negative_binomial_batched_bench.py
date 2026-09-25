"""The negative-binomial M step, per state against batched on distinct counts (issue #918).

At the `spatio_sequential_counts/release` projection's size, 100 components
over 4,000 pairs, the size #918's profile was taken at: the total channel's
dispersion solve, one bisection per state against one lockstep bisection. The
two agree to #648's floor (`tests/regression/test_emissions_batched_negative_binomial.py`);
this measures what the batched solve saves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import NegativeBinomialEmission, mstep
from sal.opt.mixture import responsibilities
from sal.search.projection import flatten, project
from sal.sim.count_pairs import binned_model
from sal.sim.fixtures import fixture

#: Observations the projection draws, as #892 and #918 measured.
N_OBSERVATIONS = 4_000


@pytest.fixture(scope="module")
def m_step() -> tuple[torch.Tensor, torch.Tensor, NegativeBinomialEmission]:
    """The totals, the responsibilities at the truth, and the total channel."""
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
    channel = truth._total
    assert isinstance(channel, NegativeBinomialEmission)
    return values[:, 0], posterior, channel


def _per_state(
    values: torch.Tensor, posterior: torch.Tensor
) -> list[mstep.SolvedDispersion]:
    mean = (posterior.T @ values) / posterior.sum(dim=0)
    return [
        mstep.solve_dispersion(values, posterior[:, k], float(mean[k]))
        for k in range(posterior.shape[1])
    ]


@pytest.mark.release
@pytest.mark.benchmark
def test_negative_binomial_m_step_per_state(
    benchmark: object,
    m_step: tuple[torch.Tensor, torch.Tensor, NegativeBinomialEmission],
) -> None:
    """The oracle: one bisection per state over every observation."""
    values, posterior, _ = m_step
    benchmark(_per_state, values, posterior)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_negative_binomial_m_step_batched(
    benchmark: object,
    m_step: tuple[torch.Tensor, torch.Tensor, NegativeBinomialEmission],
) -> None:
    """The family's M step, which runs the batched solve."""
    values, posterior, channel = m_step
    benchmark(channel.reestimate, values, posterior)  # type: ignore[operator]
