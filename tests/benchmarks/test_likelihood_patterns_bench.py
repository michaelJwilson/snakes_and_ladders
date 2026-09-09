"""Pruning over site patterns against pruning over columns (issue #408).

Correctness is pinned in
tests/regression/likelihood/test_likelihood_patterns.py; this measures what
the identity buys. Both members of each pair evaluate the same
log-likelihood, so the ratio of their times is the speedup the compression
earns at that fixture's ratio -- and the Rust pair is the one to watch,
since its wrapper pays one crossing per distinct weight rather than one per
call (see ``pruning_rust``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood import pruning, pruning_rust, pruning_torch
from snakes_and_ladders.likelihood.patterns import compress
from snakes_and_ladders.likelihood.pruning_torch import branch_lengths_from_tree
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import load_fixture

FIXTURES = (
    "tree_jc/ci.yaml",  # 4 taxa, 20 000 sites, 256 patterns
    "tree_jc/release.yaml",  # 8 taxa, 200 000 sites, 19 646 patterns
)


@pytest.mark.parametrize("fixture_name", FIXTURES)
@pytest.mark.parametrize("backend", ["numpy", "torch", "rust"])
@pytest.mark.parametrize("compressed", [False, True], ids=["columns", "patterns"])
def test_pruning_over_patterns_benchmark(
    benchmark: BenchmarkFixture, fixture_name: str, backend: str, compressed: bool
) -> None:
    params = load_fixture(fixture_name)
    alignment = dict(
        simulate_alignment(
            tau=params.tau,
            k=params.k,
            pi=params.pi,
            rng=np.random.default_rng(params.seed),
            n_sites=params.n_sites,
        ).alignment
    )
    patterns = compress(alignment)
    columns = patterns.alignment if compressed else alignment
    weights = patterns.weights if compressed else None
    pi = np.asarray(params.pi)
    lengths = branch_lengths_from_tree(params.tau)

    calls: dict[str, Callable[[], float]] = {
        "numpy": lambda: pruning.log_likelihood(
            params.tau, params.k, pi, columns, weights=weights
        ),
        "torch": lambda: float(
            pruning_torch.log_likelihood(
                params.tau, params.k, pi, columns, lengths, weights=weights
            )
        ),
        "rust": lambda: pruning_rust.log_likelihood(
            params.tau, params.k, pi, columns, weights=weights
        ),
    }

    result = benchmark(calls[backend])

    # Finiteness only -- the two paths are pinned against each other in the
    # regression suite, and a benchmark that re-asserted it would run the
    # comparison at the timing loop's cost.
    assert math.isfinite(result)
    assert result < 0.0
