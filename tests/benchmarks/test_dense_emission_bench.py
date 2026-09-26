"""The dense count-pair log-emission, tabulated in Rust against ``log_density`` (issue #1132).

A pair family of six states over 20,000 positions and 16 replicates, at a
log-normal exposure and a varying trial count: 320,000 observations, the
shape of a count-HMM whose M step scores the emission once per perturbed
parameter (the port's profile: 789 calls, 8.5 s of a 60 s fit). The
reference is the family's own ``log_density``, torch, with its state axis
moved first; ``tests/regression/test_emissions_dense.py`` pins the two.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import CountPairEmission
from sal.emissions.dense import Order, log_emission

#: States, positions and replicates.
N_STATES, N_POSITIONS, N_REPLICATES = 6, 20_000, 16

Instance = tuple[CountPairEmission, np.ndarray, np.ndarray]


@pytest.fixture(scope="module")
def instance() -> Instance:
    """The family, its draws and the covariate each was drawn at."""
    rng = np.random.default_rng(1132)
    family = CountPairEmission(
        rng.uniform(5.0, 30.0, N_STATES),
        rng.uniform(20.0, 300.0, N_STATES),
        rng.uniform(1.0, 20.0, N_STATES),
        rng.uniform(1.0, 20.0, N_STATES),
        np.full(N_STATES, 60.0),
        joint=False,
    )
    shape = (N_POSITIONS, N_REPLICATES, 1)
    covariate = np.concatenate(
        [
            np.exp(0.5 * rng.standard_normal(shape)),
            np.rint(60.0 * np.exp(0.25 * rng.standard_normal(shape))),
        ],
        axis=-1,
    )
    states = rng.integers(0, N_STATES, shape[:-1])
    return family, family.sample(states, rng, covariate=covariate), covariate


def _reference(
    family: CountPairEmission, observations: torch.Tensor, covariate: torch.Tensor
) -> np.ndarray:
    return np.ascontiguousarray(
        np.moveaxis(family.log_density(observations, covariate).numpy(), -1, 0)
    )


@pytest.mark.release
@pytest.mark.benchmark
def test_dense_emission_log_density(benchmark: object, instance: Instance) -> None:
    """The oracle: the family's torch ``log_density``, state axis first."""
    family, observations, covariate = instance
    values = torch.as_tensor(observations, dtype=torch.float64)
    benchmark(_reference, family, values, torch.as_tensor(covariate))  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("order", list(Order), ids=str)
def test_dense_emission_tabulated(
    benchmark: object, instance: Instance, order: Order
) -> None:
    """The tables and the Rust completion, in either order."""
    family, observations, covariate = instance
    benchmark(log_emission, family, observations, covariate, order=order)  # type: ignore[operator]
