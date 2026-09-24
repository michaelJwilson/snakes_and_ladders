"""The beta-binomial M step, per component against batched on distinct values (issue #892).

At the `spatio_sequential_counts/release` projection's size, 100 components
over 4,000 pairs, the size #892's profile was taken at. The two produce the
same parameters bitwise (`tests/regression/test_emissions_batched_beta_binomial.py`);
this measures what the batched solve saves.
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

#: Observations the projection draws, as #892 measured.
N_OBSERVATIONS = 4_000


@pytest.fixture(scope="module")
def m_step() -> tuple[torch.Tensor, torch.Tensor, BetaBinomialEmission]:
    """The success counts, the responsibilities at the truth, and the success channel."""
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
    return values[:, 1], posterior, channel


def _per_component(
    values: torch.Tensor, posterior: torch.Tensor, channel: BetaBinomialEmission
) -> list[mstep.SolvedBetaBinomial]:
    alpha, beta = (
        channel.named_parameters()["alpha"],
        channel.named_parameters()["beta"],
    )
    return [
        mstep.solve_beta_binomial(
            values,
            posterior[:, k],
            float(channel.trials[k]),
            float(alpha[k]) / float(alpha[k] + beta[k]),
            float(alpha[k] + beta[k]),
        )
        for k in range(posterior.shape[1])
    ]


@pytest.mark.release
@pytest.mark.benchmark
def test_beta_binomial_m_step_per_component(
    benchmark: object,
    m_step: tuple[torch.Tensor, torch.Tensor, BetaBinomialEmission],
) -> None:
    """The oracle: one solve per component over every observation."""
    benchmark(_per_component, *m_step)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_beta_binomial_m_step_batched(
    benchmark: object,
    m_step: tuple[torch.Tensor, torch.Tensor, BetaBinomialEmission],
) -> None:
    """The family's M step, which runs the batched solve."""
    values, posterior, channel = m_step
    benchmark(channel.reestimate, values, posterior)  # type: ignore[operator]
