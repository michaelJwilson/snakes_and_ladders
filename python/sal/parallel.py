"""One seam for CPU parallelism over independent tasks (issue #344).

Every loop this repository runs over independent work -- restarts of a fit,
seeds of a budgeted comparison, bootstrap replicates -- ran serially, and each
would have grown its own pool. This is the one place a pool is made, under two
rules that come before any speedup:

* **A parallel run is bitwise equal to the serial run.** Randomness enters as
  one generator per task, spawned from the caller's generator with
  :meth:`numpy.random.Generator.spawn`, so task ``i`` draws the same stream
  under every pool and worker count and the result cannot depend on
  scheduling. Results are returned in input order.
* **Workers are an explicit argument.** ``workers=1`` is serial; there is no
  default and no environment override, so a run does not change on a bigger
  machine (root ``CLAUDE.md``, Runtime Optimization Opportunities).

``torch`` and BLAS already multithread inside a kernel, so a pool of workers
each running a multithreaded kernel oversubscribes the machine.
``intra_op_threads`` sets the count per worker explicitly, and the serial
pool applies the same count, so the two runs execute the same kernels with
the same reduction order. ``DEV.md`` carries the measured rule and the hardware.

The count is ``torch``'s, and this module never imports ``torch`` (issue
#1011): a body that runs NumPy alone runs without it, in the caller and in
every worker. Where ``torch`` is already loaded the count is set at once;
where it is not, an import hook sets it the moment ``torch`` finishes
importing, so a body that imports ``torch`` itself -- as a spawned worker does
when it unpickles a function from a module that imports it -- runs its first
kernel at the count. The serial and thread pools restore what they
replaced, including the default of a ``torch`` first imported inside the call.
BLAS behind NumPy is not set here: it reads its thread count when NumPy
loads, which in a spawned worker is before any initializer runs, and changing
it afterwards needs ``threadpoolctl``, which is not a dependency. It follows
``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and ``MKL_NUM_THREADS``, which
a spawned worker inherits from the caller's environment.

The process pool uses the ``spawn`` start method, which works with ``torch``
and on Apple Silicon. Spawned workers import the package afresh, a fixed cost
per pool that ``STATUS.md`` reports beside each speedup; a function sent to it
must be importable by name, and its items picklable.
"""

from __future__ import annotations

import importlib.abc
import multiprocessing
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, Literal, TypeVar, overload

import numpy as np

Pool = Literal["serial", "threads", "processes"]
"""Which pool runs the tasks. `Pool` and not `Backend` (issue #860): the word
`Backend` names which *implementation* runs a kernel
(:class:`sal.backend.Backend`, an enum in 47 files), and one
word for two choices had ``search.infer`` annotating a pool with it. The
keyword that takes one is ``pool=`` since #1059, so ``backend=`` means a
:class:`~sal.backend.Backend` wherever it is written."""

POOLS: tuple[Pool, ...] = ("serial", "threads", "processes")

T = TypeVar("T")
R = TypeVar("R")


def _set_intra_op_threads(torch: Any, count: int) -> int:
    """Set ``torch``'s intra-op thread count and return the one it replaces."""
    previous: int = torch.get_num_threads()
    torch.set_num_threads(count)
    return previous


class _PinAtImport(importlib.abc.MetaPathFinder):
    """Sets ``torch``'s intra-op thread count when ``torch`` finishes importing.

    Placed first on ``sys.meta_path``, it answers for ``torch`` alone: it asks
    the other finders for the real spec and wraps its loader, so the
    import is the one the process would have made and the count is set
    before any caller can run a kernel. One-shot: it leaves ``sys.meta_path``
    when it fires. ``replaced`` is the count ``torch`` started with, ``None``
    until then.
    """

    def __init__(self, count: int) -> None:
        self.count = count
        self.replaced: int | None = None

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != "torch":
            return None
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if finder is self or find is None:
                continue
            spec: ModuleSpec | None = find(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PinningLoader(spec.loader, self)
                return spec
        return None

    def pin(self, torch: Any) -> None:
        self.replaced = _set_intra_op_threads(torch, self.count)
        self.remove()

    def remove(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)


class _PinningLoader(importlib.abc.Loader):
    """``torch``'s own loader, with the count set once its module has run."""

    def __init__(self, loader: Any, finder: _PinAtImport) -> None:
        self._loader = loader
        self._finder = finder

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        module: ModuleType | None = self._loader.create_module(spec)
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The real loader goes back on the module before `torch` runs, so
        # nothing it reads during or after its import sees this wrapper.
        module.__loader__ = self._loader
        if module.__spec__ is not None:
            module.__spec__.loader = self._loader
        self._loader.exec_module(module)
        self._finder.pin(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


@contextmanager
def _intra_op_threads(count: int | None) -> Iterator[None]:
    """Run the block with ``torch`` at ``count`` intra-op threads, then restore.

    ``None`` leaves the setting alone, so a caller naming no count keeps
    whatever the process had. Where ``torch`` is not loaded, it is not
    imported: the count is set if the block imports it, and the default it
    replaced is restored afterwards.
    """
    if count is None:
        yield
        return
    torch = sys.modules.get("torch")
    if torch is not None:
        previous = _set_intra_op_threads(torch, count)
        try:
            yield
        finally:
            _set_intra_op_threads(torch, previous)
        return
    hook = _PinAtImport(count)
    sys.meta_path.insert(0, hook)
    try:
        yield
    finally:
        hook.remove()
        torch = sys.modules.get("torch")
        if hook.replaced is not None and torch is not None:
            _set_intra_op_threads(torch, hook.replaced)


def _configure_worker(count: int | None) -> None:
    """Process-pool initializer: pin the worker's intra-op threads, now or at ``torch``'s import."""
    if count is None:
        return
    torch = sys.modules.get("torch")
    if torch is not None:
        _set_intra_op_threads(torch, count)
    else:
        sys.meta_path.insert(0, _PinAtImport(count))


def _call(
    function: Callable[..., R], item: object, generator: np.random.Generator | None
) -> R:
    """Apply ``function`` to one item, with its generator where one was spawned."""
    if generator is None:
        return function(item)
    return function(item, generator)


def _executor(pool: Pool, workers: int, intra_op_threads: int | None) -> Executor:
    if pool == "threads":
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
    workers: int = 1,
    pool: Pool = "serial",
    intra_op_threads: int | None = None,
    generator: None = None,
) -> list[R]: ...


@overload
def map_tasks(
    function: Callable[[T, np.random.Generator], R],
    items: Iterable[T],
    *,
    workers: int = 1,
    pool: Pool = "serial",
    intra_op_threads: int | None = None,
    generator: np.random.Generator,
) -> list[R]: ...


def map_tasks(
    function: Callable[..., R],
    items: Iterable[T],
    *,
    workers: int = 1,
    pool: Pool = "serial",
    intra_op_threads: int | None = None,
    generator: np.random.Generator | None = None,
) -> list[R]:
    """Apply ``function`` to every item, in parallel where asked, results in input order.

    Serial by default (issue #1085): ``map_tasks(f, items, generator=rng)``
    runs every item in the calling thread on its own spawned stream, the loop
    a caller would write, and a parallel call states its pool.

    Parameters
    ----------
    function : Callable
        ``function(item)``, or ``function(item, generator)`` when
        ``generator`` is given. Under the process pool it must be
        importable by name, so a closure or a lambda is refused by pickling.
    items : Iterable
        The independent tasks. Materialized once, so a generator expression
        is consumed here and the count is known before any task runs.
    workers : int
        At least one. ``1`` runs serially in the calling thread whatever
        ``pool`` says, so a caller passing ``1`` gets the loop it had.
    pool : {"serial", "threads", "processes"}
        ``"serial"`` refuses ``workers > 1`` rather than ignoring it.
        ``"threads"`` suits a body that releases the GIL -- a ``torch`` op on
        a tensor large enough to parallelize, a Rust kernel under
        ``allow_threads``. ``"processes"`` suits everything else, at the cost
        of pickling the item and the result and of spawning the workers.
    intra_op_threads : int | None
        ``torch.set_num_threads`` inside every worker, and in this process
        for the serial and thread pools, restored afterwards; set at
        ``torch``'s import where ``torch`` is not yet loaded, so it is never
        imported here. ``None``, the default, leaves the setting alone, which is what a serial
        call wants; a parallel call states it, because it decides whether the
        pool oversubscribes the machine, and ``DEV.md`` carries the measured
        rule.
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
        If ``workers < 1``, ``pool`` is not one of the three, or
        ``pool="serial"`` is asked for more than one worker.
    Exception
        Whatever a task raised, re-raised in the caller with a note naming
        the task's index and item (:meth:`BaseException.add_note`). Tasks
        not yet started are cancelled; tasks already running finish.
    """
    if workers < 1:
        msg = f"workers is at least one, got {workers}"
        raise ValueError(msg)
    if pool not in POOLS:
        msg = f"pool is one of {POOLS}, got {pool!r}"
        raise ValueError(msg)
    if pool == "serial" and workers > 1:
        msg = f"the serial pool runs one worker, asked for {workers}"
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

    pool_threads = intra_op_threads if pool == "threads" else None
    with _intra_op_threads(pool_threads):
        executor = _executor(pool, min(workers, len(tasks)), intra_op_threads)
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
