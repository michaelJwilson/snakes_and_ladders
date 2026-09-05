"""Benchmarks for Fitch parsimony against the likelihood it is compared with.

Correctness is pinned in `tests/regression/likelihood/`, per the repo's
division of labor between the two directories.

The comparison worth having is structural rather than a ranking: parsimony is
one post-order pass of integer bit operations with no branch lengths to fit,
while a likelihood score is a full continuous optimization per topology. The
two are different amounts of work for different claims, and the benchmark says
how different.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node


def _tree() -> Node:
    return Node(
        "root",
        None,
        (
            Node("i1", 0.05, (Node("A", 0.2), Node("B", 0.2))),
            Node("i2", 0.05, (Node("C", 0.2), Node("D", 0.2))),
        ),
    )


@pytest.mark.parametrize("n_sites", [2_000, 20_000, 200_000])
def test_fitch_score_benchmark(benchmark: BenchmarkFixture, n_sites: int) -> None:
    # Linear in sites by construction: the recursion is vectorized over the
    # whole alignment, so the Python cost is per *node* rather than per site.
    dataset = simulate_alignment(_tree(), 4, np.full(4, 0.25), 11, n_sites)

    score = benchmark(fitch_score, _tree(), dataset.alignment, 4)

    assert score > 0
