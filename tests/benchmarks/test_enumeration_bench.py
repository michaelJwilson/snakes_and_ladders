"""Benchmarks for the shared enumeration, whose cost is every oracle's floor.

Correctness is pinned in `tests/regression/test_enumeration.py`, per the
repo's division of labor between the two directories.

The number worth watching is the table: `k ** n` assignments times `n` sites,
doubling with every site added, and paid before any consumer's log weight is
evaluated. Issue #387 replaced a materialized `itertools.product` with the
base-`k` digits of the assignment index, which is the same array without the
Python-level loop; these cases are the ones that measurement was taken over,
so a regression in it shows up here rather than in whichever oracle happens
to be timed next.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.enumeration import (
    accumulate,
    assignment_table,
    best_assignment,
)

CASES = [(3, 4), (3, 8), (2, 16), (4, 8)]


@pytest.mark.parametrize(("n_states", "n_sites"), CASES)
def test_assignment_table_benchmark(
    benchmark: BenchmarkFixture, n_states: int, n_sites: int
) -> None:
    table = benchmark(assignment_table, n_states, n_sites, what="assignments")

    assert table.shape == (n_states**n_sites, n_sites)


@pytest.mark.parametrize(("n_states", "n_sites"), CASES)
def test_product_table_reference_benchmark(
    benchmark: BenchmarkFixture, n_states: int, n_sites: int
) -> None:
    # The implementation replaced, kept as the thing the speedup is against:
    # a ratio between two runs of this file is comparable, where a ratio
    # against a number in a commit message is not.
    def product_table() -> np.ndarray:
        return np.array(
            list(itertools.product(range(n_states), repeat=n_sites)), dtype=np.int64
        ).reshape(-1, n_sites)

    table = benchmark(product_table)

    assert table.shape == (n_states**n_sites, n_sites)


@pytest.mark.parametrize(("n_states", "n_sites"), CASES)
def test_accumulate_benchmark(
    benchmark: BenchmarkFixture, n_states: int, n_sites: int
) -> None:
    # The marginalization three weighted enumerations run once per call, and
    # the coupled one runs once per class.
    table = assignment_table(n_states, n_sites, what="assignments")
    weight = np.asarray(np.random.default_rng(0).random(table.shape[0]))

    marginals = benchmark(accumulate, table, weight, n_states)

    assert marginals.shape == (n_sites, n_states)


@pytest.mark.parametrize(("n_states", "n_sites"), [(3, 4), (3, 8), (4, 8)])
def test_best_assignment_benchmark(
    benchmark: BenchmarkFixture, n_states: int, n_sites: int
) -> None:
    # A Python-level loop by construction: the score is a caller's callable,
    # so there is nothing to vectorize without taking the score away from the
    # three landscapes that own it. This tracks the call overhead, which is
    # what such a loop costs.
    def score(candidate: tuple[int, ...]) -> float:
        return float(sum(candidate))

    # The product is rebuilt inside the timed callable: a generator passed in
    # would be exhausted after the first round and every later round would
    # time an empty loop, which is the way a benchmark over an iterator
    # silently reports zero.
    def run() -> tuple[tuple[int, ...], float]:
        return best_assignment(
            itertools.product(range(n_states), repeat=n_sites), score
        )

    best, value = benchmark(run)

    assert value == float(n_sites * (n_states - 1))
    assert best == (n_states - 1,) * n_sites
