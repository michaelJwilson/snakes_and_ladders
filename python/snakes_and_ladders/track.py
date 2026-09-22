"""The run seam: Aim's own interface, a metrics set per problem, one record per sweep (issue #778).

**The store is ``aim.Run`` with no adapter between it and a sampler.**
:class:`Run` is a ``runtime_checkable`` Protocol of exactly the three members
a hook uses --- ``track``, ``__setitem__``, ``close`` --- written with Aim's
signatures, so ``aim.Run`` satisfies it structurally and
``tests/regression/test_track.py`` asserts the ``isinstance``. Nothing in the
package imports ``aim`` at module scope.

A sampler reports its diagnostics once, at the end, in the dataclass it
returns. That is enough to judge a run and not enough to watch one: an
annealer that stalls at sweep 400 of 10,000, a chain whose acceptance
collapses when the step size is adapted, a search whose resident set grows
with every round, all look the same from the outside until they finish.

Libraries emit, entry points configure --- the division
:mod:`snakes_and_ladders.log` takes, for the same reason. A loop calls
:func:`current` once at its top and :meth:`TrackedOptimization.record` once
per iteration, sweep, round or figure; an entry point opens :func:`track` and
names the store::

    with track(aim.Run(repo="runs"), metrics=PottsMetrics(graph, field)):
        anneal_potts(graph, field, schedule, rng)

Outside any block the bound object is :data:`NULL`, whose run is
:data:`NULL_RUN`: :meth:`TrackedOptimization.record` returns on its first
line, no metric is computed, and the default run is bitwise the run before
this module existed.

**A tracked number is a number the run already reports.** Every hook records
a quantity its result dataclass returns, so the last value of a series equals
the field and the series reads as the field's history rather than as a second
definition to keep in step. A *metric* is the other half: a :class:`Metrics`
instance, bound by the entry point and never by a loop, reports what the
problem means --- an energy, a likelihood, a distance to a known truth ---
beside the objective, each metric computed by a function the package already
has.

The bound object is a :class:`contextvars.ContextVar` rather than a module
global, so two contexts in one process do not write into each other. A
``ContextVar`` does not cross into a pool worker: a task run on
:mod:`snakes_and_ladders.parallel`'s thread or process backend starts from the
default and records nothing, and a run to be tracked per task opens
:func:`track` inside the task.
"""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - the import is for the annotation only
    from matplotlib.figure import Figure

#: The state a :class:`Metrics` reads. Contravariant, because a metrics set is
#: consumed and never produced through the Protocol: one written for a
#: labelling is usable wherever a set over anything it accepts is asked for.
S_contra = TypeVar("S_contra", contravariant=True)


@runtime_checkable
class Run(Protocol):
    """Where a run's numbers go: Aim's three members, with Aim's signatures.

    Written from ``aim.Run`` (3.29.1, ``aim/sdk/run.py``) rather than from
    what a sampler would have asked for, so the optional store implements the
    Protocol as it stands and no adapter sits between them. Nothing reads a
    run back through this interface --- a store that only writes satisfies it
    --- and :class:`MemoryRun` adds the readers the tests need.
    """

    def track(
        self,
        value: Any,
        name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Append ``value`` to the sequence ``(name, context)`` at ``step``."""

    def __setitem__(self, key: str, value: Any) -> None:
        """Record one of the run's parameters: what was run, at what settings."""

    def close(self) -> None:
        """Close the run, flushing what it holds."""


class NullRun:
    """The default: every method a no-op, so an untracked run pays one returned call a sweep."""

    def track(
        self,
        value: Any,
        name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Discard ``value``."""

    def __setitem__(self, key: str, value: Any) -> None:
        """Discard ``value``."""

    def close(self) -> None:
        """Do nothing: there is nothing to flush."""


#: The run outside any :func:`track` block, and the identity
#: :meth:`TrackedOptimization.record` returns on. One shared instance rather
#: than one per lookup: it holds no state, so nothing distinguishes two.
NULL_RUN: Run = NullRun()


def _key(name: str | None, context: Mapping[str, Any] | None) -> tuple[Any, ...]:
    """The sequence key ``(name, frozen context)``, as Aim keys a sequence."""
    if context is None:
        return (name, ())
    return (name, tuple(sorted(context.items())))


class MemoryRun:
    """The referee for the tests: a :class:`Run` that keeps what it is given, in order.

    Keyed by ``(name, context)`` as Aim keys a sequence, so a per-rung or
    per-replica series recorded under a context reads back under that context
    rather than merged into the series of the same name.

    Attributes
    ----------
    params : dict[str, Any]
        What ``__setitem__`` was given, in insertion order.
    """

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self._series: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}

    def track(
        self,
        value: Any,
        name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,  # noqa: ARG002 - Aim's signature; see below
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Append ``(step, value)`` to the sequence ``(name, context)``.

        ``step`` is auto-incremented where it is omitted, as Aim does: the
        next index of the sequence it lands in. ``epoch`` is accepted and
        discarded: the signature is Aim's, no hook passes one, and a referee
        that stored it would be refereeing a field nothing writes.
        """
        entries = self._series.setdefault(_key(name, context), [])
        entries.append((len(entries) if step is None else step, value))

    def __setitem__(self, key: str, value: Any) -> None:
        """Record the parameter ``key``."""
        self.params[key] = value

    def close(self) -> None:
        """Do nothing: the record is the object."""

    def series(
        self, name: str, context: Mapping[str, Any] | None = None
    ) -> list[tuple[int, Any]]:
        """The sequence ``(name, context)``, as ``(step, value)`` in the order recorded.

        Raises
        ------
        KeyError
            If nothing was recorded under that key: an assertion against a
            series that was never written is a test that passes on nothing.
        """
        return self._series[_key(name, context)]

    def last(self, name: str, context: Mapping[str, Any] | None = None) -> Any:
        """The last value of the sequence ``(name, context)``."""
        return self.series(name, context)[-1][1]


def as_aim(fig: Figure) -> Any:
    """``aim.Image(fig)`` where ``aim`` imports, and ``fig`` itself where it does not.

    Aim's ``aim.Figure`` takes a Plotly figure; ``aim.Image`` is the one that
    accepts a matplotlib ``Figure``, which is what :mod:`snakes_and_ladders.qa`
    renders. The import is here rather than at module scope because ``aim`` is
    the optional ``track`` extra and this module is imported by every sampler.
    It names the defining module rather than the package: ``aim`` binds
    ``Image`` through a lazy ``__getattr__``, so ``aim.Image`` type-checks
    where the package is absent and fails where it is installed, and the
    package is absent from every job but the one a reader runs by hand.

    Parameters
    ----------
    fig : Figure
        The rendered figure.

    Returns
    -------
    Any
        ``aim.Image`` wrapping ``fig``, or ``fig`` itself, which is what
        :class:`MemoryRun` keeps and :class:`NullRun` discards.
    """
    try:
        from aim.sdk.objects.image import Image
    except ImportError:
        return fig
    return Image(fig)


@runtime_checkable
class Metrics(Protocol[S_contra]):
    """What a problem's state means, in numbers a person reads.

    One instance per problem class, defined in the module owning that class's
    objective or energy, and bound by an entry point --- never by a loop,
    which knows its own counters and not what they are about. Every metric is
    computed by a function the package already has, cited in the
    implementation's docstring: a metrics set restates the science, it does
    not define it.
    """

    @property
    def names(self) -> tuple[str, ...]:
        """The series this set records, in the order :meth:`__call__` returns them.

        So an entry point can name them before a run starts. Declared
        read-only: every implementation is a frozen dataclass, and a settable
        member would refuse them all.
        """

    def __call__(self, state: S_contra) -> Mapping[str, float]:
        """The metrics of ``state``, keyed by :attr:`names`."""


@dataclass(frozen=True)
class TrackedOptimization:
    """The run and its metrics bound together: the one object a loop holds.

    Parameters
    ----------
    run : Run
        Where the numbers go. :data:`NULL_RUN` is the default, and the
        identity :meth:`record` returns on.
    metrics : Metrics[Any] | None
        What the state means, or ``None`` for the counters alone. Evaluated
        only inside a :func:`track` block and only where a hook passes a
        ``state``, so an untracked run never computes one.
    started : float
        :func:`time.perf_counter` when the object was bound, which
        :func:`track` does on entering its block: the origin of the
        ``seconds`` :meth:`record_cost` records (issue #891).
    """

    run: Run
    metrics: Metrics[Any] | None = None
    started: float = field(default_factory=time.perf_counter)

    @property
    def is_null(self) -> bool:
        """Whether this records nothing, so a loop can skip assembling what it would pass."""
        return self.run is NULL_RUN

    def record(
        self,
        step: int,
        *,
        state: Any = None,
        objective: float | None = None,
        context: Mapping[str, Any] | None = None,
        **diagnostics: float,
    ) -> None:
        """Record one iteration: the objective, the metrics of ``state``, and the diagnostics.

        One call per iteration, sweep, round or figure, never one per site:
        the whole cost of the seam on an untracked run is this call and its
        first line.

        Every call also records ``seconds``, the wall clock since the
        enclosing :func:`track` block opened, at the same step and context,
        unless ``diagnostics`` names ``seconds`` itself: each sample of a run
        then carries the time it was taken, so a caller plots a series against
        seconds from the run alone (issue #891).

        Parameters
        ----------
        step : int
            The iteration the numbers belong to, shared by every series
            written here.
        state : Any
            The problem's state --- ``theta``, a labelling, a topology ---
            passed to :attr:`metrics`, and ignored where either is absent.
        objective : float | None
            The value being minimized, recorded under the name ``objective``.
        context : Mapping[str, Any] | None
            Aim's per-sequence key, for a series that exists once per rung or
            per replica.
        **diagnostics : float
            One series per keyword, under the keyword's own name.
        """
        if self.run is NULL_RUN:
            return
        if "seconds" not in diagnostics:
            diagnostics["seconds"] = time.perf_counter() - self.started
        if objective is not None:
            self.run.track(
                float(objective), name="objective", step=step, context=context
            )
        if self.metrics is not None and state is not None:
            for metric, measured in self.metrics(state).items():
                self.run.track(float(measured), name=metric, step=step, context=context)
        for name, value in diagnostics.items():
            self.run.track(float(value), name=name, step=step, context=context)

    def record_cost(self, step: int, state_bytes: int) -> None:
        """Record what the run cost the machine: peak resident bytes, the bytes its state holds, and seconds.

        Called once, at the end of a run, at the step its last series entry
        carries, so the three land beside the last number rather than past it.
        ``seconds`` is the wall clock since the enclosing :func:`track` block
        was entered, so it covers whatever the block ran before the hook ---
        a seeding, a burn-in --- as well as the loop that records it; a block
        holding two runs reports the second's as the sum (issue #891).
        """
        if self.run is NULL_RUN:
            return
        self.record(
            step,
            peak_rss_bytes=float(peak_rss_bytes()),
            state_bytes=float(state_bytes),
        )


#: What :func:`current` returns outside any block: the shared null run, and no
#: metrics. Bound once rather than built per lookup.
NULL = TrackedOptimization(NULL_RUN, None)

_CURRENT: ContextVar[TrackedOptimization] = ContextVar("sal_tracked", default=NULL)


def current() -> TrackedOptimization:
    """The :class:`TrackedOptimization` of the enclosing :func:`track` block, or :data:`NULL`.

    Read once at the top of a loop and held, not called per step: the lookup
    is cheap and a hook that repeats it per sweep is a hook that can be seen
    in a profile.
    """
    return _CURRENT.get()


@contextmanager
def track(
    run: Run | None = None,
    *,
    metrics: Metrics[Any] | None = None,
    **params: Any,
) -> Iterator[TrackedOptimization]:
    """Run the block recording into ``run``, restoring the previous binding on exit.

    Parameters
    ----------
    run : Run | None
        Where the numbers go; ``None`` is a fresh :class:`MemoryRun`, which is
        what a test wants and what an entry point overrides with an
        ``aim.Run``.
    metrics : Metrics[Any] | None
        The problem's metrics set, or ``None``.
    **params : Any
        The run's parameters, set on it through ``__setitem__``.

    Yields
    ------
    TrackedOptimization
        The bound object, so a caller can read what its run recorded.
    """
    active = TrackedOptimization(MemoryRun() if run is None else run, metrics)
    token = _CURRENT.set(active)
    for key, value in params.items():
        active.run[key] = value
    try:
        yield active
    finally:
        _CURRENT.reset(token)
        active.run.close()


def peak_rss_bytes() -> int:
    """The process's peak resident set size, in bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS, so the platform
    decides the factor; both are converted here rather than at each call
    site. It is a *peak over the process*, not over the block: a second run
    in one process reports the high-water mark of both.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


__all__ = [
    "NULL",
    "NULL_RUN",
    "MemoryRun",
    "Metrics",
    "NullRun",
    "Run",
    "TrackedOptimization",
    "as_aim",
    "current",
    "peak_rss_bytes",
    "track",
]
