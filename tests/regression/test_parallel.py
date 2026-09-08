"""The one seam for CPU parallelism, pinned on the contract that makes it usable.

Issue #344. What is asserted is the part a speedup cannot show: that a run at
four workers is the run at one -- results in input order, task ``i`` drawing
the stream spawned for it and no other -- and that a failure is reported with
the task that failed rather than as a pool's timeout. The speedups themselves
are measured on fixed hardware and recorded in ``STATUS.md``, per ``DEV.md``'s
No CI Profiling rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.parallel import BACKENDS, Backend, map_tasks

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


@pytest.mark.structural
@pytest.mark.parametrize("backend", list(BACKENDS))
def test_results_come_back_in_input_order_under_every_backend(backend: Backend) -> None:
    # Input order, not completion order: the items are unequal enough in
    # cost that a pool returning as tasks finish would permute them.
    items = list(range(12))
    workers = 1 if backend == "serial" else 4

    results = map_tasks(
        _square, items, workers=workers, backend=backend, intra_op_threads=None
    )

    assert results == [_square(item) for item in items]


@pytest.mark.structural
@pytest.mark.parametrize("backend", ["threads", "processes"])
def test_four_workers_draw_the_streams_one_worker_draws(backend: Backend) -> None:
    # The ticket's first rule: a parallel run is bitwise the serial run. Each
    # task's generator is spawned from the caller's, in item order, so what
    # a task draws depends on its index and on nothing about scheduling.
    items = list(range(8))

    serial = map_tasks(
        _draw,
        items,
        workers=1,
        backend="serial",
        intra_op_threads=None,
        generator=np.random.default_rng(7),
    )
    pooled = map_tasks(
        _draw,
        items,
        workers=4,
        backend=backend,
        intra_op_threads=None,
        generator=np.random.default_rng(7),
    )

    assert pooled == serial
    children = np.random.default_rng(7).spawn(len(items))
    assert serial == [
        _draw(item, child) for item, child in zip(items, children, strict=True)
    ]


@pytest.mark.structural
def test_a_second_call_on_the_same_generator_draws_fresh_children() -> None:
    # `Generator.spawn` advances the spawn counter, not the stream: two calls
    # on one generator are two replicate sets, and a fresh generator with the
    # same seed reproduces the first. Both halves matter -- a bootstrap called
    # twice must not silently repeat itself.
    rng = np.random.default_rng(3)

    first = map_tasks(
        _draw, [0, 1], workers=1, backend="serial", intra_op_threads=None, generator=rng
    )
    second = map_tasks(
        _draw, [0, 1], workers=1, backend="serial", intra_op_threads=None, generator=rng
    )
    again = map_tasks(
        _draw,
        [0, 1],
        workers=1,
        backend="serial",
        intra_op_threads=None,
        generator=np.random.default_rng(3),
    )

    assert first != second
    assert again == first


@pytest.mark.edge_case
@pytest.mark.parametrize("backend", list(BACKENDS))
def test_a_task_that_raises_propagates_with_the_item_that_raised(
    backend: Backend,
) -> None:
    workers = 1 if backend == "serial" else 4

    with pytest.raises(ValueError, match="three is refused") as excinfo:
        map_tasks(
            _refuse_three,
            [1, 2, 3, 4],
            workers=workers,
            backend=backend,
            intra_op_threads=None,
        )

    assert "task 2 of 4 raised on item 3" in "".join(excinfo.value.__notes__)


@pytest.mark.edge_case
def test_an_unusable_worker_count_or_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        map_tasks(_square, [1], workers=0, backend="serial", intra_op_threads=None)
    with pytest.raises(ValueError, match="one of"):
        map_tasks(_square, [1], workers=1, backend="fibres", intra_op_threads=None)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="serial backend runs one worker"):
        map_tasks(_square, [1], workers=2, backend="serial", intra_op_threads=None)


@pytest.mark.edge_case
def test_no_items_is_an_empty_result_and_spawns_nothing() -> None:
    assert (
        map_tasks(_square, [], workers=4, backend="processes", intra_op_threads=1) == []
    )


@pytest.mark.structural
@pytest.mark.parametrize("backend", ["serial", "threads"])
def test_the_intra_op_thread_count_is_applied_inside_and_restored_after(
    backend: Backend,
) -> None:
    # The thread rule DEV.md states: a worker runs at the count the caller
    # named, and the calling process is left as it was found.
    before = torch.get_num_threads()
    workers = 1 if backend == "serial" else 2

    inside = map_tasks(
        _thread_count, [0, 1], workers=workers, backend=backend, intra_op_threads=1
    )

    assert inside == [1, 1]
    assert torch.get_num_threads() == before


@pytest.mark.structural
def test_a_spawned_process_runs_at_the_thread_count_it_was_given() -> None:
    inside = map_tasks(
        _thread_count, [0, 1], workers=2, backend="processes", intra_op_threads=1
    )

    assert inside == [1, 1]
