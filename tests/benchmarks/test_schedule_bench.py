"""Benchmarks for the schedule plan, and the cost issue #592 named.

Correctness is pinned in `tests/regression/likelihood/test_schedule.py`, per
the repo's division of labor between the two directories.

The plan is built once per `sum_product` call and was a fifth to a quarter of
it on a chain: one message per level, and the grouping machinery run per level
to group one thing. The two fixes are measured here rather than asserted ---
the axis carried from the breadth-first walk, and the group arrays built once
per shape instead of once per level. The second pays and the first does not,
and both numbers are in the pull request.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.message_passing import (
    MessageScheduleName,
    sum_product,
)
from snakes_and_ladders.likelihood.schedule import Layout
from snakes_and_ladders.sim.factor_graph import Factor, FactorGraph, Variable


def _chain(n: int, cardinality: int = 4) -> FactorGraph:
    rng = np.random.default_rng(3)
    return FactorGraph(
        [Variable(f"v{i}", cardinality) for i in range(n)],
        [
            Factor(
                f"f{i}",
                (f"v{i}", f"v{i + 1}"),
                rng.normal(size=(cardinality, cardinality)),
            )
            for i in range(n - 1)
        ],
    )


def _star(n: int, cardinality: int = 4) -> FactorGraph:
    rng = np.random.default_rng(3)
    return FactorGraph(
        [Variable("hub", cardinality)]
        + [Variable(f"v{i}", cardinality) for i in range(n)],
        [
            Factor(
                f"f{i}",
                ("hub", f"v{i}"),
                rng.normal(size=(cardinality, cardinality)),
            )
            for i in range(n)
        ],
    )


@pytest.mark.parametrize(
    "graph",
    [_chain(500), _chain(2000), _star(2000)],
    ids=["chain-500", "chain-2000", "star-2000"],
)
def test_tree_plan_benchmark(benchmark: BenchmarkFixture, graph: FactorGraph) -> None:
    # The plan alone: what #592 measured at 18.2 and 24.4 per cent of the run.
    passes = benchmark(lambda: Layout(graph).tree_passes())

    up, down = passes
    assert len(up) > 0
    assert len(down) > 0


@pytest.mark.parametrize(
    "graph", [_chain(500), _chain(2000)], ids=["chain-500", "chain-2000"]
)
def test_tree_schedule_benchmark(
    benchmark: BenchmarkFixture, graph: FactorGraph
) -> None:
    # The whole call, so the plan's share stays visible as the denominator.
    marginals = benchmark(lambda: sum_product(graph, schedule=MessageScheduleName.TREE))

    assert marginals.exact


@pytest.mark.parametrize("schedule", ["tree", "upward"])
def test_upward_is_half_the_work_benchmark(
    benchmark: BenchmarkFixture, schedule: str
) -> None:
    # The claim the partial schedule earns its place on: `log Z` for one pass
    # rather than two. Both rows are here so the ratio is read, not asserted.
    graph = _chain(1000)

    marginals = benchmark(lambda: sum_product(graph, schedule=schedule))

    assert np.isfinite(marginals.log_partition)
