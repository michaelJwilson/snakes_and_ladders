"""Benchmarks for Felsenstein pruning's hot path.

See tests/regression/test_likelihood_pruning.py for correctness, pinned
separately per the repo's division of labor between the two test
directories. Runs across a few (n, L) points, the denominator
CLAUDE.md's >=10x GPU threshold is measured against once a backend exists.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FIXTURES_DIR


@pytest.mark.parametrize(
    "fixture_name",
    [
        "simulation_params.yaml",  # 4 taxa, 200_000 sites
        "simulation_params_small_sites.yaml",  # 4 taxa, 20_000 sites
        "simulation_params_8taxa.yaml",  # 8 taxa, 200_000 sites
    ],
)
def test_log_likelihood_benchmark(
    benchmark: BenchmarkFixture, fixture_name: str
) -> None:
    params = load_simulation_params(FIXTURES_DIR / fixture_name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )

    result = benchmark(
        log_likelihood, params.tau, params.k, params.pi, dataset.alignment
    )

    # Benchmarks only assert finiteness -- numerical correctness is pinned
    # separately in tests/regression/test_likelihood_pruning.py.
    assert math.isfinite(result)
    assert result < 0.0  # a log-likelihood, never positive
