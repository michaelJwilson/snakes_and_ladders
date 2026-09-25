"""Benchmarks for the relabelling methods and the conjugate Gibbs sweep (issue #964).

Correctness is pinned in `tests/regression/sample/`. The size is the
notebook's: 2,000 draws of 500 observations and 5 components. One STEPHENS
pass and one SJW EM iteration are timed rather than a whole run, since the
number of passes is the data's and not the implementation's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.emissions import GaussianEmission
from sal.sample.mixture_gibbs import (
    GaussianMixturePrior,
    gibbs_gaussian_mixture,
)
from sal.sample.relabel import ecr, pra, sjw, stephens

M, N, K = 2_000, 500, 5


def _draws() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(964)
    observations = rng.normal(np.linspace(-3.0, 3.0, K)[rng.integers(0, K, N)], 1.0)
    means = np.linspace(-3.0, 3.0, K) + rng.normal(0.0, 0.3, (M, K))
    logits = -0.5 * (observations[None, :, None] - means[:, None, :]) ** 2
    probabilities = np.exp(logits - logits.max(axis=2, keepdims=True))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    allocations = (
        np.cumsum(probabilities, axis=2)[..., :-1] < rng.random((M, N))[..., None]
    ).sum(axis=2)
    parameters = np.stack([means, np.ones((M, K)), np.full((M, K), 1.0 / K)], axis=2)
    return {
        "observations": observations,
        "probabilities": probabilities,
        "allocations": allocations,
        "parameters": parameters,
    }


@pytest.fixture(scope="module")
def draws() -> dict[str, np.ndarray]:
    return _draws()


def test_stephens_pass_benchmark(
    benchmark: BenchmarkFixture, draws: dict[str, np.ndarray]
) -> None:
    result = benchmark(stephens, draws["probabilities"], max_iterations=1)
    assert result.permutations.shape == (M, K)


def test_ecr_benchmark(
    benchmark: BenchmarkFixture, draws: dict[str, np.ndarray]
) -> None:
    result = benchmark(ecr, draws["allocations"], draws["allocations"][0], K)
    assert result.permutations.shape == (M, K)


def test_pra_benchmark(
    benchmark: BenchmarkFixture, draws: dict[str, np.ndarray]
) -> None:
    result = benchmark(pra, draws["parameters"], draws["parameters"][0])
    assert result.permutations.shape == (M, K)


def test_sjw_iteration_benchmark(
    benchmark: BenchmarkFixture, draws: dict[str, np.ndarray]
) -> None:
    values = torch.as_tensor(draws["observations"])

    def scores(estimate: np.ndarray) -> np.ndarray:
        family = GaussianEmission(estimate[:, 0], estimate[:, 1], 1e-12)
        return family.log_density(values).numpy() + np.log(estimate[:, 2])

    result = benchmark(
        sjw, draws["parameters"], draws["allocations"], scores, max_iterations=1
    )
    assert result.permutations.shape == (M, K)


def test_gibbs_sweeps_benchmark(
    benchmark: BenchmarkFixture, draws: dict[str, np.ndarray]
) -> None:
    observations = draws["observations"]
    prior = GaussianMixturePrior.weakly_informative(observations, K)

    def run() -> int:
        chain = gibbs_gaussian_mixture(
            observations, K, prior, np.random.default_rng(0), 100
        )
        return int(chain.weights.shape[0])

    assert benchmark(run) == 100
