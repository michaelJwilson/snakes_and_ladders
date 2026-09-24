"""A covariate that depends on the parameters, recomputed before every E step (issue #933, R8).

`baum_welch_family(..., update=)` hands each E step the covariate the update
returns from the current family and the previous posterior;
`ExpectedRateNormalizer` divides each sequence's exposure by ``Z_c = sum_g
lambda_g E[mu_{s_c(g)}]``. Referees: an update that returns its covariate is
the fit without one, bitwise; two iterations on a tiny instance are pinned by
enumerating every state path, the normalizer included; and the rate ratios of
a planted normalized chain are recovered.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import NegativeBinomialEmission
from snakes_and_ladders.opt.hmm import ExpectedRateNormalizer, baum_welch_family


def _kernel(stay: float, m: int) -> torch.Tensor:
    matrix = np.full((m, m), (1.0 - stay) / (m - 1))
    np.fill_diagonal(matrix, stay)
    return torch.log(torch.as_tensor(matrix))


def _enumerated(
    counts: np.ndarray,
    exposure: np.ndarray,
    family: NegativeBinomialEmission,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
) -> tuple[float, np.ndarray]:
    """The log-likelihood and posterior marginals, summed over every state path."""
    n, length = counts.shape
    m = family.n_states
    scores = family.log_density(
        torch.as_tensor(counts, dtype=torch.float64),
        covariate=torch.as_tensor(exposure)[..., None],
    ).numpy()
    total = 0.0
    marginals = np.zeros((n, length, m))
    for c in range(n):
        weights = []
        paths = list(itertools.product(range(m), repeat=length))
        for path in paths:
            value = float(log_initial[path[0]]) + scores[c, 0, path[0]]
            for t in range(1, length):
                value += (
                    float(log_transition[path[t - 1], path[t]]) + scores[c, t, path[t]]
                )
            weights.append(value)
        top = max(weights)
        mass = np.exp(np.array(weights) - top)
        total += top + math.log(mass.sum())
        for path, share in zip(paths, mass / mass.sum(), strict=True):
            for t, state in enumerate(path):
                marginals[c, t, state] += share
    return total, marginals


@pytest.mark.critical
@pytest.mark.oracle
def test_an_update_returning_its_covariate_is_the_fit_without_one() -> None:
    rng = np.random.default_rng(933)
    counts = rng.poisson(5.0, (3, 12)).astype(float)
    exposure = rng.uniform(0.5, 2.0, (3, 12))
    family = NegativeBinomialEmission([4.0, 4.0], [2.0, 8.0])
    initial = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    plain = baum_welch_family(
        counts, initial, _kernel(0.8, 2), family, max_iterations=5, covariate=exposure
    )
    held = baum_welch_family(
        counts,
        initial,
        _kernel(0.8, 2),
        family,
        max_iterations=5,
        covariate=exposure,
        update=lambda _emissions, _posterior, covariate: covariate,
    )
    assert plain.log_likelihood == held.log_likelihood
    for name, value in plain.emissions.named_parameters().items():
        assert torch.equal(value, held.emissions.named_parameters()[name])


@pytest.mark.critical
@pytest.mark.oracle
def test_two_iterations_are_the_enumerated_plug_in_fit() -> None:
    # 2 sequences x 4 positions x 2 states: 16 paths each. The first E step is
    # scored at Z from a uniform posterior and the start's means; the second
    # at Z from the first E step's posterior, enumerated, and the first M
    # step's means.
    rng = np.random.default_rng(8)
    counts = rng.poisson(6.0, (2, 4)).astype(float)
    exposure = rng.uniform(0.5, 2.0, (2, 4))
    lam = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    normalizer = ExpectedRateNormalizer(lam)
    family = NegativeBinomialEmission([5.0, 5.0], [3.0, 9.0])
    initial = torch.log(torch.tensor([0.4, 0.6], dtype=torch.float64))
    kernel = _kernel(0.7, 2)

    means = family.mean.numpy()
    first_z = (lam.numpy() * means.mean()).sum()
    one = baum_welch_family(
        counts,
        initial,
        kernel,
        family,
        max_iterations=1,
        covariate=exposure,
        update=normalizer,
    )
    expected_one, marginals = _enumerated(
        counts, exposure / first_z, family, initial, kernel
    )
    assert abs(one.log_likelihood - expected_one) <= 1e-10 * abs(expected_one)

    fitted = one.emissions
    assert isinstance(fitted, NegativeBinomialEmission)
    second_z = (lam.numpy() * (marginals @ fitted.mean.numpy())).sum(axis=1)
    two = baum_welch_family(
        counts,
        initial,
        kernel,
        family,
        max_iterations=2,
        tolerance=0.0,
        covariate=exposure,
        update=normalizer,
    )
    expected_two, _ = _enumerated(
        counts,
        exposure / second_z[:, None],
        fitted,
        one.log_initial,
        one.log_transition,
    )
    assert abs(two.log_likelihood - expected_two) <= 1e-10 * abs(expected_two)


@pytest.mark.end2end
def test_the_rate_ratios_of_a_normalized_chain_are_recovered() -> None:
    # Planted: rates mu = (1, 2, 4) per unit of lambda, 60 sequences of 120
    # positions, each drawn at exposure N_c lambda_g mu_s / Z_c with Z_c
    # the sequence's own sum over its path. Only the ratios are identified;
    # measured 2.017 and 4.146 in 34 iterations, 1.1 s. With the exposure
    # held fixed and no normalizer the same fit lands at 1.876 and 3.331.
    rng = np.random.default_rng(933)
    n, length, m = 60, 120, 3
    lam = rng.uniform(0.5, 1.5, length)
    mu = np.array([1.0, 2.0, 4.0])
    states = np.empty((n, length), dtype=np.int64)
    states[:, 0] = rng.integers(0, m, n)
    for t in range(1, length):
        move = rng.random(n) > 0.95
        states[:, t] = np.where(move, rng.integers(0, m, n), states[:, t - 1])
    depth = rng.uniform(4_000.0, 8_000.0, n)
    z = (lam * mu[states]).sum(axis=1)
    rate = depth[:, None] * lam * mu[states] / z[:, None]
    counts = rng.negative_binomial(20.0, 20.0 / (20.0 + rate)).astype(float)
    exposure = depth[:, None] * lam[None, :]
    start = NegativeBinomialEmission([5.0] * 3, [0.8, 2.5, 3.0])
    fit = baum_welch_family(
        counts,
        torch.full((m,), -math.log(m), dtype=torch.float64),
        _kernel(0.9, m),
        start,
        max_iterations=200,
        covariate=exposure,
        update=ExpectedRateNormalizer(torch.as_tensor(lam)),
    )
    means = np.sort(fit.emissions.alignment_key()[:, 0].numpy())
    np.testing.assert_allclose(means / means[0], [1.0, 2.0, 4.0], rtol=0.05)


@pytest.mark.smoke
def test_an_update_without_a_covariate_is_refused() -> None:
    family = NegativeBinomialEmission([5.0, 5.0], [3.0, 9.0])
    with pytest.raises(ValueError, match="needs a covariate"):
        baum_welch_family(
            np.ones((1, 3)),
            torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)),
            _kernel(0.7, 2),
            family,
            update=ExpectedRateNormalizer(torch.ones(3, dtype=torch.float64)),
        )
