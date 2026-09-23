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


@pytest.mark.oracle
@pytest.mark.parametrize("index", [2, 3, 4, 5])
def test_the_count_table_is_the_per_position_density(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Referee: the same twin with the table refused, scored per position.
    objective = _objectives()[index]
    theta = objective.initial().numpy() + 0.1
    tabled = jax_hmm.value_and_grad(objective)(theta)
    assert jax_hmm._prepared(objective)[0].tabled
    monkeypatch.setattr(jax_hmm, "TABLE_SHARE", 0.0)
    assert not jax_hmm._prepared(objective)[0].tabled
    whole = jax_hmm.value_and_grad(objective)(theta)
    # The scatter-add sums the posteriors in another order: a tolerance.
    assert_allclose(tabled[0], whole[0], rtol=1e-12)
    assert_allclose(tabled[1], whole[1], rtol=1e-10, atol=1e-10)


@pytest.mark.patch
def test_one_structure_compiles_once() -> None:
    # Two objectives of one structure and shape share the program; the table
    # is padded to a power of two, so a table of another length does too.
    rng = np.random.default_rng(1)
    first, second = (
        BetaBinomialHmmObjective(rng.binomial(20, p, size=(8, 25)), 3, np.full(3, 20.0))
        for p in (0.3, 0.6)
    )
    shapes = [jax_hmm._prepared(o)[1]["y"].shape for o in (first, second)]
    assert shapes[0] == shapes[1]
    jax_hmm.value_and_grad(first)(first.initial().numpy())
    compiled = jax_hmm._compiled(jax_hmm._prepared(first)[0])
    before = compiled._cache_size()
    jax_hmm.value_and_grad(second)(second.initial().numpy())
    assert compiled._cache_size() == before
