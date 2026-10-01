"""The count densities below and above the large-shape threshold (issue #1136).

A negative binomial and a beta-binomial of four states over 100,000
observations, once at shapes below :data:`sal.emissions.rising.LARGE_SHAPE`,
where the arithmetic is unchanged, and once above it, where the differenced
Stirling series replaces ``lgamma``. ``tests/regression/test_emissions_large_shape.py``
pins both against ``mpmath``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import BetaBinomialEmission, NegativeBinomialEmission

#: Observations scored per call.
N_OBSERVATIONS = 100_000


@pytest.fixture(scope="module")
def counts() -> torch.Tensor:
    """Negative-binomial counts and beta-binomial successes out of 40, as floats."""
    rng = np.random.default_rng(1136)
    return torch.as_tensor(rng.integers(0, 41, N_OBSERVATIONS), dtype=torch.float64)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [10.0, 1e6], ids=["below", "above"])
def test_negative_binomial_density(
    benchmark: object, counts: torch.Tensor, shape: float
) -> None:
    """Four states at dispersion ``shape``."""
    family = NegativeBinomialEmission([shape] * 4, [5.0, 10.0, 20.0, 40.0])
    benchmark(family.log_density, counts)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [10.0, 1e6], ids=["below", "above"])
def test_beta_binomial_density(
    benchmark: object, counts: torch.Tensor, shape: float
) -> None:
    """Four states at concentration ``shape``."""
    rate = np.array([0.2, 0.4, 0.6, 0.8])
    family = BetaBinomialEmission([40.0] * 4, rate * shape, (1.0 - rate) * shape)
    benchmark(family.log_density, counts)  # type: ignore[operator]
