"""One seam for CPU parallelism over independent tasks (issue #344).

Every loop this repository runs over independent work -- restarts of a fit,
seeds of a budgeted comparison, bootstrap replicates -- ran serially, and each
would have grown its own pool. This module is the one place a pool is made,
under two rules that come before any speedup:

* **A parallel run is bitwise equal to the serial run.** Randomness enters as
  one generator per task, spawned from the caller's generator with
  :meth:`numpy.random.Generator.spawn`, so task ``i`` draws the same stream
  under every backend and worker count and the result cannot depend on
  scheduling. Results are returned in input order.
* **Workers are an explicit argument.** ``workers=1`` is serial; there is no
  default and no environment override, so a run does not change on a bigger
  machine (root ``CLAUDE.md``, Runtime Optimization Opportunities).

``torch`` and BLAS already multithread inside a kernel, so a pool of workers
each running a multithreaded kernel oversubscribes the machine. The intra-op
thread count is set per worker, explicitly, by ``intra_op_threads``; the
serial backend applies the same count, so the two runs execute the same
kernels with the same reduction order. ``DEV.md`` carries the measured rule
and the hardware it was measured on.

The process backend uses the ``spawn`` start method, the one that works with
``torch`` and on Apple Silicon. Spawned workers import the package afresh, a
fixed cost per pool that ``STATUS.md`` reports beside each speedup; a
function sent to it must be importable by name, and its items picklable.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Literal, TypeVar, overload

import numpy as np

Backend = Literal["serial", "threads", "processes"]
BACKENDS: tuple[Backend, ...] = ("serial", "threads", "processes")

T = TypeVar("T")
R = TypeVar("R")


def _set_intra_op_threads(count: int) -> int:
    """Set ``torch``'s intra-op thread count and return the one it replaces."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    return previous


@contextmanager
def _intra_op_threads(count: int | None) -> Iterator[None]:
    """Run the block with ``torch`` at ``count`` intra-op threads, then restore.

    ``None`` leaves the setting alone, so a caller that does not name a count
    keeps whatever the process had.
    """
    if count is None:
        yield
        return
    previous = _set_intra_op_threads(count)
    try:
        yield
    finally:
        _set_intra_op_threads(previous)


def _configure_worker(count: int | None) -> None:
    """Process-pool initializer: pin the worker's intra-op threads."""
    if count is not None:
        _set_intra_op_threads(count)


def _call(
    function: Callable[..., R], item: object, generator: np.random.Generator | None
) -> R:
    """Apply ``function`` to one item, with its generator where one was spawned."""
    if generator is None:
        return function(item)
    return function(item, generator)


def _executor(backend: Backend, workers: int, intra_op_threads: int | None) -> Executor:
    if backend == "threads":
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_configure_worker,
        initargs=(intra_op_threads,),
    )


@overload
def map_tasks(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    backend: Backend,
    intra_op_threads: int | None,
    generator: None = None,
) -> list[R]: ...


@overload
def map_tasks(
    function: Callable[[T, np.random.Generator], R],
    items: Iterable[T],
    *,
    workers: int,
    backend: Backend,
    intra_op_threads: int | None,
    generator: np.random.Generator,
) -> list[R]: ...


def map_tasks(
    function: Callable[..., R],
    items: Iterable[T],
    *,
    workers: int,
    backend: Backend,
    intra_op_threads: int | None,
    generator: np.random.Generator | None = None,
) -> list[R]:
    """Apply ``function`` to every item, in parallel where asked, results in input order.

    Parameters
    ----------
    function : Callable
        ``function(item)``, or ``function(item, generator)`` when
        ``generator`` is given. Under the process backend it must be
        importable by name, so a closure or a lambda is refused by pickling.
    items : Iterable
        The independent tasks. Materialized once, so a generator expression
        is consumed here and the count is known before any task runs.
    workers : int
        At least one. ``1`` runs serially in the calling thread whatever
        ``backend`` says, so a caller passing ``1`` gets the loop it had.
    backend : {"serial", "threads", "processes"}
        ``"serial"`` refuses ``workers > 1`` rather than ignoring it.
        ``"threads"`` suits a body that releases the GIL -- a ``torch`` op on
        a tensor large enough to parallelize, a Rust kernel under
        ``allow_threads``. ``"processes"`` suits everything else, at the cost
        of pickling the item and the result and of spawning the workers.
    intra_op_threads : int | None
        ``torch.set_num_threads`` inside every worker, and in this process
        for the serial and thread backends, restored afterwards. ``None``
        leaves the setting alone. Stated by the caller rather than defaulted
        because it decides whether the pool oversubscribes the machine;
        ``DEV.md`` carries the measured rule.
    generator : np.random.Generator | None
        When given, one child generator per item is spawned from it, in item
        order, and passed as the task's second argument. The parent's spawn
        counter advances and its stream does not, so a second call with the
        same generator draws fresh children and a fresh generator with the
        same seed reproduces the first.

    Returns
    -------
    list
        ``function``'s result per item, in the order the items were given.

    Raises
    ------
    ValueError
        If ``workers < 1``, ``backend`` is not one of the three, or
        ``backend="serial"`` is asked for more than one worker.
    Exception
        Whatever a task raised, re-raised in the caller with a note naming
        the task's index and item (:meth:`BaseException.add_note`). Tasks
        not yet started are cancelled; tasks already running finish.
    """
    if workers < 1:
        msg = f"workers is at least one, got {workers}"
        raise ValueError(msg)
    if backend not in BACKENDS:
        msg = f"backend is one of {BACKENDS}, got {backend!r}"
        raise ValueError(msg)
    if backend == "serial" and workers > 1:
        msg = f"the serial backend runs one worker, asked for {workers}"
        raise ValueError(msg)
    tasks = list(items)
    generators: list[np.random.Generator | None] = (
        list(generator.spawn(len(tasks)))
        if generator is not None
        else [None] * len(tasks)
    )

    if workers == 1 or not tasks:
        with _intra_op_threads(intra_op_threads):
            return [
                _annotated(function, index, item, child, len(tasks))
                for index, (item, child) in enumerate(
                    zip(tasks, generators, strict=True)
                )
            ]

    pool_threads = intra_op_threads if backend == "threads" else None
    with _intra_op_threads(pool_threads):
        executor = _executor(backend, min(workers, len(tasks)), intra_op_threads)
        try:
            futures: list[Future[R]] = [
                executor.submit(_call, function, item, child)
                for item, child in zip(tasks, generators, strict=True)
            ]
            results: list[R] = []
            for index, (future, item) in enumerate(zip(futures, tasks, strict=True)):
                try:
                    results.append(future.result())
                except Exception as error:
                    error.add_note(_note(index, item, len(tasks)))
                    raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    return results


def _note(index: int, item: object, count: int) -> str:
    return f"map_tasks: task {index} of {count} raised on item {item!r}"


def _annotated(
    function: Callable[..., R],
    index: int,
    item: object,
    generator: np.random.Generator | None,
    count: int,
) -> R:
    """The serial path's call, carrying the same note a pooled task would."""
    try:
        return _call(function, item, generator)
    except Exception as error:
        error.add_note(_note(index, item, count))
        raise
