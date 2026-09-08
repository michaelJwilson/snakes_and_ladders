"""The cost of the seam itself, beside the speedups it is meant to buy.

See tests/regression/test_parallel.py for correctness. What is timed here is
the overhead a caller pays for going through ``map_tasks`` at one worker --
the list materialization, the spawned generators, the note on each call --
against the bare loop it replaces. The speedup at four workers is a
fixed-hardware measurement, recorded in ``STATUS.md`` and never taken here
(``DEV.md``, No CI Profiling).
"""

from __future__ import annotations

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.parallel import map_tasks

N_TASKS = 256


def _draw(item: int, rng: np.random.Generator) -> float:
    return item + float(rng.random())


def test_serial_map_tasks_benchmark(benchmark: BenchmarkFixture) -> None:
    items = list(range(N_TASKS))

    results = benchmark(
        map_tasks,
        _draw,
        items,
        workers=1,
        backend="serial",
        intra_op_threads=None,
        generator=np.random.default_rng(0),
    )

    assert len(results) == N_TASKS


def test_bare_loop_benchmark(benchmark: BenchmarkFixture) -> None:
    def loop() -> list[float]:
        children = np.random.default_rng(0).spawn(N_TASKS)
        return [_draw(item, child) for item, child in enumerate(children)]

    results = benchmark(loop)

    assert len(results) == N_TASKS
