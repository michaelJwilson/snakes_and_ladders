"""Benchmarks for the Gumbel-softmax relaxation.

Correctness is pinned in `tests/regression/learn/`, per the repo's division of
labor between the two directories.

The comparison worth watching is the deterministic relaxation against the
sampled one: they take the same number of gradient steps, so the difference is
the Gumbel draw and the extra graph it builds. `tests/regression/` measures
which finds the optimum more often; this measures what each costs to get
there.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from phylo.learn.potts import PottsLandscape
from phylo.learn.relaxed import (
    RelaxationMode,
    RelaxedPotts,
    estimate_gradient,
    exact_expected_gradient,
    gumbel_softmax,
    optimize,
)
from pytest_benchmark.fixture import BenchmarkFixture


def _objective(chain_length: int) -> RelaxedPotts:
    return RelaxedPotts(PottsLandscape(-0.9, np.array([0.4, 0.35, -0.6]), chain_length))


@pytest.mark.parametrize("chain_length", [8, 32, 128])
def test_relaxed_score_benchmark(
    benchmark: BenchmarkFixture, chain_length: int
) -> None:
    objective = _objective(chain_length)
    probabilities = torch.full((chain_length, 3), 1.0 / 3.0, dtype=torch.float64)

    value = benchmark(objective.relaxed, probabilities)

    assert float(value) == pytest.approx(
        objective.landscape.coupling * (chain_length - 1) / 3.0
        + chain_length * float(objective.landscape.field.mean())
    )


@pytest.mark.parametrize("mode", list(RelaxationMode))
def test_gumbel_softmax_benchmark(
    benchmark: BenchmarkFixture, mode: RelaxationMode
) -> None:
    generator = torch.Generator().manual_seed(1)
    logits = torch.zeros((64, 3), dtype=torch.float64)

    sample = benchmark(gumbel_softmax, logits, 0.5, generator, mode=mode)

    assert sample.shape == (64, 3)


@pytest.mark.parametrize("stochastic", [False, True])
def test_optimize_benchmark(benchmark: BenchmarkFixture, stochastic: bool) -> None:
    # The number the deterministic-versus-sampled result has to be read
    # against: the same 100 gradient steps, differing only by the Gumbel draw.
    objective = _objective(7)

    result = benchmark(
        optimize, objective, 1, temperature=0.5, steps=100, stochastic=stochastic
    )

    assert len(result.configuration) == 7


@pytest.mark.parametrize("chain_length", [4, 6, 8])
def test_exact_expected_gradient_benchmark(
    benchmark: BenchmarkFixture, chain_length: int
) -> None:
    # `3 ** L` configurations, each differentiated through. This is the oracle
    # and its cost is the reason the fixtures stay small; the benchmark is
    # where that stays visible.
    objective = _objective(chain_length)
    logits = torch.zeros((chain_length, 3), dtype=torch.float64)

    gradient = benchmark(exact_expected_gradient, objective, logits)

    assert gradient.shape == (chain_length, 3)


def test_estimate_gradient_benchmark(benchmark: BenchmarkFixture) -> None:
    objective = _objective(7)
    generator = torch.Generator().manual_seed(1)
    logits = torch.zeros((7, 3), dtype=torch.float64)

    gradient = benchmark(
        estimate_gradient, objective, logits, 0.5, generator, n_samples=8
    )

    assert gradient.shape == (7, 3)
