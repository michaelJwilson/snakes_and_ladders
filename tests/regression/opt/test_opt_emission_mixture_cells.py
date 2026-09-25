"""The emission-mixture EM on distinct counts against the per-observation route (issue #997).

Integer counts in one channel are fitted on their distinct values weighted
by multiplicity; the same counts as floats take the per-observation route,
which is the oracle: every parameter, the log-likelihood and every
observation's responsibilities, at the declared float64 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.emission_mixture import expectation_maximization


def _case(name: str) -> tuple[np.ndarray, EmissionFamily]:
    rng = np.random.default_rng(997)
    component = rng.choice(3, size=4_000, p=[0.3, 0.3, 0.4])
    if name == "poisson":
        return rng.poisson(np.array([1.0, 5.0, 12.0])[component]), PoissonEmission(
            [2.0, 4.0, 10.0]
        )
    if name == "negative_binomial":
        r = np.array([2.0, 5.0, 10.0])[component]
        mu = np.array([1.0, 5.0, 12.0])[component]
        return rng.negative_binomial(r, r / (r + mu)), NegativeBinomialEmission(
            [1.0, 3.0, 5.0], [2.0, 4.0, 10.0]
        )
    if name == "binomial":
        p = np.array([0.2, 0.5, 0.8])[component]
        return rng.binomial(30, p), BinomialEmission([30.0] * 3, [0.3, 0.5, 0.7])
    p = rng.beta(
        np.array([2.0, 5.0, 8.0])[component], np.array([8.0, 5.0, 2.0])[component]
    )
    return rng.binomial(30, p), BetaBinomialEmission(
        [30.0] * 3, [1.5, 4.0, 6.0], [6.0, 4.0, 1.5]
    )


@pytest.mark.oracle
@pytest.mark.parametrize(
    "name", ["poisson", "negative_binomial", "binomial", "beta_binomial"]
)
def test_the_distinct_count_fit_is_the_per_observation_one(name: str) -> None:
    counts, family = _case(name)
    weights = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    cells, oracle = (
        expectation_maximization(
            values, weights, family, max_iterations=15, tolerance=-np.inf
        )
        for values in (counts, counts.astype(np.float64))
    )
    assert_allclose(cells.log_likelihood, oracle.log_likelihood, rtol=1e-12)
    assert_allclose(cells.weights.numpy(), oracle.weights.numpy(), rtol=1e-10)
    for key, value in oracle.components.named_parameters().items():
        assert_allclose(
            cells.components.named_parameters()[key].numpy(),
            value.numpy(),
            rtol=1e-9,
            err_msg=key,
        )
    assert_allclose(
        cells.responsibilities.numpy(),
        oracle.responsibilities.numpy(),
        rtol=0.0,
        atol=1e-10,
    )
    assert cells.iterations == oracle.iterations
    assert cells.at_boundary == oracle.at_boundary
