"""The HMM objectives' JAX gradient, their default, against PyTorch's autograd (issue #1000).

Referee: each objective's value and its autograd gradient through
``__call__``, at three points away from the start, within 1e-10 relative:
every HMM objective the package defines, with the covariates it takes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    GaussianHmmObjective,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
)
from sal.opt.jax import hmm as hmm_jax

from tests._rows import every_value


def _objectives() -> list[hmm_jax.Twinned]:
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
    twin = hmm_jax.value_and_grad(objective)
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
    tabled = hmm_jax.value_and_grad(objective)(theta)
    assert hmm_jax._prepared(objective)[0].tabled
    monkeypatch.setattr(hmm_jax, "TABLE_SHARE", 0.0)
    assert not hmm_jax._prepared(objective)[0].tabled
    whole = hmm_jax.value_and_grad(objective)(theta)
    # The scatter-add sums the posteriors in another order: a tolerance.
    assert_allclose(tabled[0], whole[0], rtol=1e-12)
    assert_allclose(tabled[1], whole[1], rtol=1e-10, atol=1e-10)


@pytest.mark.smoke
def test_one_structure_compiles_once() -> None:
    # Two objectives of one structure and shape share the program; the table
    # is padded to a power of two, so a table of another length does too.
    rng = np.random.default_rng(1)
    first, second = (
        BetaBinomialHmmObjective(rng.binomial(20, p, size=(8, 25)), 3, np.full(3, 20.0))
        for p in (0.3, 0.6)
    )
    shapes = [hmm_jax._prepared(o)[1]["y"].shape for o in (first, second)]
    assert shapes[0] == shapes[1]
    hmm_jax.value_and_grad(first)(first.initial().numpy())
    compiled = hmm_jax._compiled(hmm_jax._prepared(first)[0])
    before = compiled._cache_size()
    hmm_jax.value_and_grad(second)(second.initial().numpy())
    assert compiled._cache_size() == before


@pytest.mark.oracle
def test_the_default_gradient_is_the_torch_backend() -> None:
    # Referee: the same objective built with Backend.TORCH, autograd.
    def check(index: int) -> None:
        from sal.backend import Backend
        from sal.opt.objective import value_and_gradient

        objective = _objectives()[index]
        theta = objective.initial() + 0.1
        value, gradient = value_and_gradient(objective, theta)
        objective._backend = Backend.TORCH
        torch_value, torch_gradient = value_and_gradient(objective, theta)
        assert_allclose(float(value), float(torch_value), rtol=1e-10)
        assert_allclose(
            gradient.numpy(),
            torch_gradient.numpy(),
            rtol=1e-10,
            atol=1e-10 * float(torch_gradient.abs().max()),
        )

    every_value(range(8), check)


@pytest.mark.oracle
def test_a_fit_under_either_backend_reaches_one_optimum() -> None:
    # Referee: the L-BFGS fit under Backend.TORCH from the same start.
    from sal.backend import Backend
    from sal.opt.fit import fit

    rng = np.random.default_rng(3)
    counts = rng.poisson(np.repeat([2.0, 9.0], 50), size=(4, 100))
    fits = [
        fit(PoissonHmmObjective(counts, 2, backend=backend), max_iterations=200)
        for backend in (Backend.JAX, Backend.TORCH)
    ]
    assert all(f.converged for f in fits)
    assert_allclose(fits[0].value, fits[1].value, rtol=1e-9)


@pytest.mark.oracle
def test_the_rising_factorial_holds_where_the_difference_cancels() -> None:
    # Referee: SciPy's `gammaln(k) - betaln(k, x)`, accurate at every `x`
    # here, where `gammaln(k + x) - gammaln(x)` in float64 is off by 1e-3 at
    # 4e11 (issue #1000).
    import jax
    from scipy.special import betaln, gammaln

    counts = np.array([0.0, 1.0, 7.0, 30.0])[:, None]
    x = np.array([2.0, 95.0, 999.0, 1.001e3, 5.8e7, 4e11, 1e15])[None, :]
    expected = np.where(
        counts > 0,
        gammaln(np.maximum(counts, 1.0)) - betaln(np.maximum(counts, 1.0), x),
        0.0,
    )
    got = np.asarray(hmm_jax._rising(counts, x, jax))
    assert_allclose(got, expected, rtol=1e-12, atol=1e-8)
