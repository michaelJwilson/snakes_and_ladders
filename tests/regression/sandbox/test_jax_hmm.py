"""The sandboxed JAX twins of the HMM objectives against PyTorch's autograd (issue #1000).

Referee: each objective's value and its autograd gradient through
``__call__``, at three points away from the start, within 1e-10 relative:
every HMM objective the package defines, with the covariates it takes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    GaussianHmmObjective,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
)
from snakes_and_ladders.sandbox import jax_hmm

# The twins import JAX when they build; the module itself does not.
pytest.importorskip("jax")


def _objectives() -> list[jax_hmm.Twinned]:
    rng = np.random.default_rng(1000)
    counts = rng.poisson(6.0, size=(8, 25))
    return [
        HmmObjective(rng.integers(0, 4, size=(8, 25)), 3, 4),
        GaussianHmmObjective(rng.normal(size=(8, 25)), 3),
        PoissonHmmObjective(counts, 3),
        NegativeBinomialHmmObjective(counts, 3),
        BetaBinomialHmmObjective(
            rng.binomial(20, 0.4, size=(8, 25)), 3, np.full(3, 20.0)
        ),
        BinomialHmmObjective(rng.binomial(20, 0.4, size=(8, 25)), 3, np.full(3, 20.0)),
        # The package's covariates: an exposure per observation for the
        # negative binomial, a trial count per observation for the
        # beta-binomial.
        NegativeBinomialHmmObjective(
            counts, 3, covariate=rng.uniform(0.5, 2.0, size=(8, 25))
        ),
        BetaBinomialHmmObjective(
            rng.binomial(15, 0.4, size=(8, 25)),
            3,
            np.full(3, 20.0),
            covariate=rng.integers(15, 30, size=(8, 25)),
        ),
    ]


@pytest.mark.oracle
@pytest.mark.parametrize("index", range(8))
def test_the_jax_twin_is_autograd(index: int) -> None:
    objective = _objectives()[index]
    twin = jax_hmm.value_and_grad(objective)
    rng = np.random.default_rng(index)
    for _ in range(3):
        theta = objective.initial() + 0.2 * torch.as_tensor(
            rng.normal(size=objective.n_parameters)
        )
        point = theta.clone().requires_grad_(True)
        value = objective(point)
        (gradient,) = torch.autograd.grad(value, point)
        ours_value, ours_gradient = twin(theta.numpy())
        assert_allclose(ours_value, float(value.detach()), rtol=1e-10)
        assert_allclose(
            ours_gradient,
            gradient.numpy(),
            rtol=1e-10,
            atol=1e-10 * float(gradient.abs().max()),
        )
