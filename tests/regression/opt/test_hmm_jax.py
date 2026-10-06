"""`EmissionHmmObjective`'s JAX twin against PyTorch's autograd (issues #1000, #1189).

Referee: each objective's value and its autograd gradient through
``__call__``, at three points away from the start, within 1e-10 relative:
every family the twin covers, with the covariates it takes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.hmm import EmissionHmmObjective, family_start
from sal.opt.hmm import jax as hmm_jax
from sal.ragged import Ragged

from tests._rows import every_value


def _hmm(
    kind: type,
    data: np.ndarray,
    covariate: np.ndarray | None = None,
    backend: Backend = Backend.JAX,
    **constants: object,
) -> EmissionHmmObjective:
    """Three states at :func:`family_start`, the covariate one value per position."""
    return EmissionHmmObjective(
        data,
        family_start(kind, data, 3, **constants),  # type: ignore[arg-type]
        covariate=None if covariate is None else covariate[..., None],
        backend=backend,
    )


def _objectives(backend: Backend = Backend.JAX) -> list[EmissionHmmObjective]:
    rng = np.random.default_rng(1000)
    counts = rng.poisson(6.0, size=(8, 25))
    trials = np.full(3, 20.0)
    return [
        _hmm(
            CategoricalEmission,
            rng.integers(0, 4, size=(8, 25)),
            backend=backend,
            n_symbols=4,
        ),
        _hmm(GaussianEmission, rng.normal(size=(8, 25)), backend=backend),
        _hmm(PoissonEmission, counts, backend=backend),
        _hmm(NegativeBinomialEmission, counts, backend=backend),
        _hmm(
            BetaBinomialEmission,
            rng.binomial(20, 0.4, size=(8, 25)),
            backend=backend,
            trials=trials,
        ),
        _hmm(
            BinomialEmission,
            rng.binomial(20, 0.4, size=(8, 25)),
            backend=backend,
            trials=trials,
        ),
        # The package's covariates: an exposure per observation for the
        # negative binomial, a trial count per observation for the
        # beta-binomial.
        _hmm(
            NegativeBinomialEmission,
            counts,
            rng.uniform(0.5, 2.0, size=(8, 25)),
            backend=backend,
        ),
        _hmm(
            BetaBinomialEmission,
            rng.binomial(15, 0.4, size=(8, 25)),
            rng.integers(15, 30, size=(8, 25)).astype(np.float64),
            backend=backend,
            trials=trials,
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
        _hmm(
            BetaBinomialEmission,
            rng.binomial(20, p, size=(8, 25)),
            trials=np.full(3, 20.0),
        )
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
def test_the_jax_backend_s_gradient_is_the_torch_backend_s() -> None:
    # Referee: the same objective built with Backend.TORCH, autograd.
    def check(index: int) -> None:
        from sal.opt.objective import value_and_gradient

        objective = _objectives()[index]
        theta = objective.initial() + 0.1
        value, gradient = value_and_gradient(objective, theta)
        torch_value, torch_gradient = value_and_gradient(
            _objectives(Backend.TORCH)[index], theta
        )
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
    from sal.opt.fit import fit

    rng = np.random.default_rng(3)
    counts = rng.poisson(np.repeat([2.0, 9.0], 50), size=(4, 100))
    fits = [
        fit(
            EmissionHmmObjective(
                counts, family_start(PoissonEmission, counts, 2), backend=backend
            ),
            max_iterations=200,
        )
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


@pytest.mark.oracle
def test_the_twin_reads_the_family_from_the_start() -> None:
    # Issue #1189: the JAX route is keyed on `EmissionHmmObjective`'s start
    # family. Referee: the objective's own autograd value, for a twinned
    # family --- on segments of one length, on segments of two (issue #1206),
    # and for a count pair --- and `twinned` refuses what no twin covers, a
    # joint pair given a covariate its family refuses, so it neither declares
    # a JAX energy nor admits Backend.JAX.
    rng = np.random.default_rng(1189)
    counts = rng.poisson(6.0, size=48)
    pair = CountPairEmission(
        [5.0, 12.0], [6.0, 30.0], [2.0, 4.0], [3.0, 2.0], None, joint=True
    )
    ragged = EmissionHmmObjective(
        Ragged(counts.astype(np.float64), (20, 28)),
        family_start(PoissonEmission, counts, 3),
    )
    values = np.asarray(pair.sample(rng.integers(0, 2, 40), rng), dtype=np.float64)
    joint = EmissionHmmObjective(Ragged(values, (20, 20)), pair)
    covaried = EmissionHmmObjective(
        Ragged(values, (20, 20)), pair, covariate=np.ones((40, 2))
    )
    assert not hmm_jax.twinned(covaried)
    assert covaried.jax_energy() is None
    with pytest.raises(ValueError, match="no JAX twin"):
        EmissionHmmObjective(
            Ragged(values, (20, 20)),
            pair,
            covariate=np.ones((40, 2)),
            backend=Backend.JAX,
        )
    for twinned in (_objectives()[2], ragged, joint):
        assert hmm_jax.twinned(twinned)
        theta = twinned.initial() + 0.1
        energy, data = twinned.jax_energy()  # type: ignore[misc]
        assert_allclose(
            float(energy(theta.numpy(), data)), float(twinned(theta)), rtol=1e-12
        )
