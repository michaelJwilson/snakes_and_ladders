"""Benchmark for the tropical Grassmannian relaxation's objective.

Correctness is pinned in ``tests/regression/search/test_search_tropical.py``,
per the repo's division of labour between the two directories.

What is timed here is :func:`relaxed_score`, and only it. A run of the
relaxation is two costs of very different kinds: ``3 C(n, 4)`` four-taxon
fits to build the quartet table, and 300 evaluations of the objective to
ascend it. The first is fits, whose cost is
``tests/benchmarks/test_search_infer_bench.py``'s subject already and is not
re-timed here; the second touches no alignment, grows with ``C(n, 4)`` and
not with the site count, and had no number until this one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.tropical import (
    QuartetTable,
    pairing_positions,
    quartet_indices,
    relaxed_score,
)

#: Taxon counts the objective is timed at. 8 is the largest size enumeration
#: referees; 20 is past it, where the quartet count is 4,845 and the
#: evaluation is the term that grew.
TAXON_COUNTS = (5, 8, 20)


def _table(n_taxa: int, rng: np.random.Generator) -> QuartetTable:
    """A table of the right shape with arbitrary scores.

    The objective reads the score table and never the data behind it, so
    timing it needs no fit --- and a table built from one would time the fit.
    """
    quartets = quartet_indices(n_taxa)
    return QuartetTable(
        names=tuple(f"t{index}" for index in range(n_taxa)),
        quartets=quartets,
        scores=rng.normal(size=(quartets.shape[0], 3)),
    )


@pytest.mark.parametrize("n_taxa", TAXON_COUNTS)
def test_relaxed_score_benchmark(benchmark: BenchmarkFixture, n_taxa: int) -> None:
    rng = np.random.default_rng(20260909)
    table = _table(n_taxa, rng)
    positions = pairing_positions(n_taxa)
    distances = torch.from_numpy(rng.uniform(0.1, 1.0, size=n_taxa * (n_taxa - 1) // 2))

    value = benchmark(relaxed_score, table, positions, distances, 0.05)

    # The softmin weights are a probability vector per quartet, so the value
    # lies between summing each quartet's smallest score and its largest.
    assert float(table.scores.min(axis=1).sum()) <= float(value)
    assert float(value) <= float(table.scores.max(axis=1).sum())
