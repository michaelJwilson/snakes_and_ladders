"""The one seam for CPU parallelism, pinned on the contract that makes it usable.

Issue #344. What is asserted is the part a speedup cannot show: that a run at
four workers is the run at one -- results in input order, task ``i`` drawing
the stream spawned for it and no other -- and that a failure is reported with
the task that failed rather than as a pool's timeout. The speedups themselves
are measured on fixed hardware and recorded in ``STATUS.md``, per ``DEV.md``'s
No CI Profiling rule.
"""

from __future__ import annotations

import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sal import parallel
from sal.parallel import POOLS, Pool, map_tasks

WORKERS = (1, 4)


def _square(item: int) -> int:
    return item * item


def _draw(item: int, rng: np.random.Generator) -> float:
    """A task whose answer is its stream, so scheduling cannot hide in it."""
    return item + float(rng.random())


def _refuse_three(item: int) -> int:
    if item == 3:
        msg = "three is refused"
        raise ValueError(msg)
    return item


def _thread_count(_item: int) -> int:
    return torch.get_num_threads()


@pytest.mark.infra
@pytest.mark.parametrize("pool", list(POOLS))
def test_results_come_back_in_input_order_under_every_backend(pool: Pool) -> None:
    # Input order, not completion order: the items are unequal enough in
    # cost that a pool returning as tasks finish would permute them.
    items = list(range(12))
    workers = 1 if pool == "serial" else 4

    results = map_tasks(
        _square, items, workers=workers, pool=pool, intra_op_threads=None
    )

    assert results == [_square(item) for item in items]


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("pool", ["threads", "processes"])
def test_four_workers_draw_the_streams_one_worker_draws(pool: Pool) -> None:
    # The ticket's first rule: a parallel run is bitwise the serial run. Each
    # task's generator is spawned from the caller's, in item order, so what
    # a task draws depends on its index and on nothing about scheduling.
    items = list(range(8))

    serial = map_tasks(
        _draw,
        items,
        workers=1,
        pool="serial",
        intra_op_threads=None,
        generator=np.random.default_rng(7),
    )
    pooled = map_tasks(
        _draw,
        items,
        workers=4,
        pool=pool,
        intra_op_threads=None,
        generator=np.random.default_rng(7),
    )

    assert pooled == serial
    children = np.random.default_rng(7).spawn(len(items))
    assert serial == [
        _draw(item, child) for item, child in zip(items, children, strict=True)
    ]


@pytest.mark.infra
def test_a_second_call_on_the_same_generator_draws_fresh_children() -> None:
    # `Generator.spawn` advances the spawn counter, not the stream: two calls
    # on one generator are two replicate sets, and a fresh generator with the
    # same seed reproduces the first. Both halves matter -- a bootstrap called
    # twice must not silently repeat itself.
    rng = np.random.default_rng(3)

    first = map_tasks(
        _draw, [0, 1], workers=1, pool="serial", intra_op_threads=None, generator=rng
    )
    second = map_tasks(
        _draw, [0, 1], workers=1, pool="serial", intra_op_threads=None, generator=rng
    )
    again = map_tasks(
        _draw,
        [0, 1],
        workers=1,
        pool="serial",
        intra_op_threads=None,
        generator=np.random.default_rng(3),
    )

    assert first != second
    assert again == first


@pytest.mark.smoke
@pytest.mark.parametrize("pool", list(POOLS))
def test_a_task_that_raises_propagates_with_the_item_that_raised(
    pool: Pool,
) -> None:
    workers = 1 if pool == "serial" else 4

    with pytest.raises(ValueError, match="three is refused") as excinfo:
        map_tasks(
            _refuse_three,
            [1, 2, 3, 4],
            workers=workers,
            pool=pool,
            intra_op_threads=None,
        )

    assert "task 2 of 4 raised on item 3" in "".join(excinfo.value.__notes__)


@pytest.mark.smoke
def test_an_unusable_worker_count_or_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        map_tasks(_square, [1], workers=0, pool="serial", intra_op_threads=None)
    with pytest.raises(ValueError, match="one of"):
        map_tasks(_square, [1], workers=1, pool="fibres", intra_op_threads=None)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="serial pool runs one worker"):
        map_tasks(_square, [1], workers=2, pool="serial", intra_op_threads=None)


@pytest.mark.smoke
def test_no_items_is_an_empty_result_and_spawns_nothing() -> None:
    assert map_tasks(_square, [], workers=4, pool="processes", intra_op_threads=1) == []


@pytest.mark.infra
@pytest.mark.parametrize("pool", ["serial", "threads"])
def test_the_intra_op_thread_count_is_applied_inside_and_restored_after(
    pool: Pool,
) -> None:
    # The thread rule DEV.md states: a worker runs at the count the caller
    # named, and the calling process is left as it was found.
    before = torch.get_num_threads()
    workers = 1 if pool == "serial" else 2

    inside = map_tasks(
        _thread_count, [0, 1], workers=workers, pool=pool, intra_op_threads=1
    )

    assert inside == [1, 1]
    assert torch.get_num_threads() == before


@pytest.mark.infra
def test_a_spawned_process_runs_at_the_thread_count_it_was_given() -> None:
    inside = map_tasks(
        _thread_count, [0, 1], workers=2, pool="processes", intra_op_threads=1
    )

    assert inside == [1, 1]


def _square_and_draw(item: int, rng: np.random.Generator) -> tuple[float, float]:
    """``i * i`` in float64, and the item's own stream, from one pool pass."""
    return float(item) * float(item), item + float(rng.random())


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("pool", ["threads", "processes"])
def test_each_backend_maps_the_seeded_tasks_onto_the_serial_map_bitwise(
    pool: Pool,
) -> None:
    """The three backends are one map, refereed from outside it (issue #729).

    ``sum i^2`` = 22,140 at ``n = 41`` and `Generator.spawn`'s children; ``==``, 0.0.
    """
    items = list(range(41))
    closed_form = len(items) * (len(items) - 1) * (2 * len(items) - 1) / 6

    serial = map_tasks(
        _square_and_draw,
        items,
        workers=1,
        pool="serial",
        intra_op_threads=1,
        generator=np.random.default_rng(11),
    )
    pooled = map_tasks(
        _square_and_draw,
        items,
        workers=4,
        pool=pool,
        intra_op_threads=1,
        generator=np.random.default_rng(11),
    )

    assert math.fsum(square for square, _ in pooled) == closed_form == 22140.0
    assert pooled == serial
    children = np.random.default_rng(11).spawn(len(items))
    assert pooled == [
        _square_and_draw(item, child)
        for item, child in zip(items, children, strict=True)
    ]


@pytest.mark.smoke
def test_a_pool_is_named_pool_and_never_backend() -> None:
    # #860 renamed the type `Pool`, the word `Backend` naming which
    # implementation runs a kernel; #1059 renamed the keyword `pool=` and
    # dropped the old alias, so `backend` means `sal.backend.Backend` alone.
    assert not hasattr(parallel, "Backend")
    assert set(POOLS) == {"serial", "threads", "processes"}
    assert "pool" in inspect.signature(parallel.map_tasks).parameters
    assert "backend" not in inspect.signature(parallel.map_tasks).parameters


_BODIES = """
import sys


def torch_loaded(_item: int) -> bool:
    return "torch" in sys.modules


def torch_threads(_item: int) -> int:
    import torch

    return torch.get_num_threads()
"""

_PROBE = """
import sys

import _bodies
from sal.parallel import map_tasks

pool = sys.argv[1]
workers = 1 if pool == "serial" else 2
numpy_only = map_tasks(
    _bodies.torch_loaded, [0, 1], workers=workers, pool=pool, intra_op_threads=1
)
print(numpy_only, "torch" in sys.modules)
pinned = map_tasks(
    _bodies.torch_threads, [0, 1], workers=workers, pool=pool, intra_op_threads=1
)
import torch

print(pinned, torch.get_num_threads())
"""


@pytest.mark.infra
@pytest.mark.parametrize("pool", list(POOLS))
def test_torch_is_imported_only_by_a_body_that_imports_it(
    pool: Pool, tmp_path: Path
) -> None:
    """A NumPy body loads no ``torch``; a ``torch`` body runs at the count (issue #1011).

    Fresh process, thread variables at 3: workers read 1, the caller 3 afterwards.
    """
    (tmp_path / "_bodies.py").write_text(_BODIES)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), os.environ.get("PYTHONPATH", "")]
        ),
    }

    result = subprocess.run(
        [sys.executable, "-c", _PROBE, pool],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )

    assert result.stdout.splitlines() == ["[False, False] False", "[1, 1] 3"]


@pytest.mark.analytic
def test_the_defaults_are_the_serial_loop() -> None:
    # Issue #1085: `map_tasks(f, items, generator=rng)` needs no pool spelled
    # out, and is the spelled-out serial call bitwise.
    def draw(item: int, rng: np.random.Generator) -> float:
        return item + float(rng.random())

    defaulted = map_tasks(draw, range(5), generator=np.random.default_rng(3))
    spelled = map_tasks(
        draw,
        range(5),
        workers=1,
        pool="serial",
        intra_op_threads=None,
        generator=np.random.default_rng(3),
    )

    assert defaulted == spelled
