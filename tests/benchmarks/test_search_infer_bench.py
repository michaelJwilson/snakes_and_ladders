"""Benchmarks for topology search.

See tests/regression/test_search_infer.py for correctness. The unit that
matters is one candidate fit, because that is what the budget counts and
where a search spends its time: generating a neighbourhood is combinatorics
on a handful of nodes, while fitting one candidate is an optimization. Both
are measured here, so the ratio between them says whether that assumption
still holds.
"""

from __future__ import annotations

import math

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.infer import MoveSet, infer, score_topology
from snakes_and_ladders.search.topology import nni_neighbours, random_topology
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import SMALL_SITES, load_fixture

_SITES = 2000


def _alignment() -> tuple[dict[str, np.ndarray], int]:
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=_SITES
    )
    return dict(dataset.alignment), params.k


def test_one_candidate_fit_benchmark(benchmark: BenchmarkFixture) -> None:
    """The unit the search budget is denominated in."""
    alignment, k = _alignment()
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    result = benchmark(score_topology, topology, alignment, k)

    # Benchmarks assert finiteness only; correctness is pinned in
    # tests/regression/test_search_infer.py.
    assert math.isfinite(result)
    assert result < 0.0


def test_neighbourhood_generation_benchmark(benchmark: BenchmarkFixture) -> None:
    """Move generation alone, to show it is not where the time goes."""
    alignment, _ = _alignment()
    topology = random_topology(sorted(alignment), np.random.default_rng(1))

    neighbours = benchmark(lambda: list(nni_neighbours(topology)))

    assert len(neighbours) > 0


def test_hill_climb_benchmark(benchmark: BenchmarkFixture) -> None:
    """A whole search, so the per-fit number can be checked against a run."""
    alignment, k = _alignment()

    result = benchmark(infer, alignment, k, seed=1, moves=MoveSet.NNI)

    assert result.converged
    assert math.isfinite(result.log_likelihood)
