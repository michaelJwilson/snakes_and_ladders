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
from sal.fixtures import load_params
from sal.likelihood.pruning import log_likelihood
from sal.sim.params import SimulationParams
from sal.sim.simulator import simulate_tree

from tests._fixtures import FIXTURES_DIR


@pytest.mark.parametrize(
    "fixture_name",
    [
        "tree_jc/stress.yaml",  # 4 taxa, 200_000 sites
        "tree_jc/ci.yaml",  # 4 taxa, 20_000 sites
        "tree_jc/release.yaml",  # 8 taxa, 200_000 sites
    ],
)
def test_log_likelihood_benchmark(
    benchmark: BenchmarkFixture, fixture_name: str
) -> None:
    params = load_params(FIXTURES_DIR / fixture_name, SimulationParams)
    dataset = simulate_tree(params, np.random.default_rng(params.seed))

    result = benchmark(
        log_likelihood, params.tau, params.n_states, params.pi, dataset.alignment
    )

    # Benchmarks only assert finiteness -- numerical correctness is pinned
    # separately in tests/regression/test_likelihood_pruning.py.
    assert math.isfinite(result)
    assert result < 0.0  # a log-likelihood, never positive
