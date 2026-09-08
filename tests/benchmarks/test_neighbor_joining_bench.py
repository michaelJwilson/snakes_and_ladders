"""Benchmarks for ``snakes_and_ladders.search.neighbor_joining`` at 50 taxa.

See ``tests/regression/search/test_neighbor_joining.py`` for correctness.
The algorithm is ``O(n^3)``; 50 taxa is the largest size the exactness test
runs at, and the four-point check beside it is ``O(n^4)``, which is why it
is a check and not a step.
"""

from __future__ import annotations

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.distance import tree_distances
from snakes_and_ladders.search.neighbor_joining import (
    four_point_violation,
    neighbor_joining,
)
from snakes_and_ladders.search.topology import leaf_bipartitions

from tests.regression.search.test_neighbor_joining import random_tree

N_TAXA = 50


def test_neighbor_joining_benchmark(benchmark: BenchmarkFixture) -> None:
    truth = random_tree(N_TAXA, np.random.default_rng(0))
    names, distances = tree_distances(truth)

    tree = benchmark(neighbor_joining, names, distances)

    assert leaf_bipartitions(tree) == leaf_bipartitions(truth)


def test_four_point_violation_benchmark(benchmark: BenchmarkFixture) -> None:
    truth = random_tree(N_TAXA, np.random.default_rng(0))
    _, distances = tree_distances(truth)

    violation = benchmark(four_point_violation, distances)

    assert violation < 1e-12
