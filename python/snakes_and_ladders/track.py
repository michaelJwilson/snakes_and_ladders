"""The run tracker: what a sweep did, what it cost, and where the numbers go (issue #778).

A sampler reports its diagnostics once, at the end, in the dataclass it
returns. That is enough to judge a run and not enough to watch one: an
annealer that stalls at sweep 400 of 10,000, a chain whose acceptance
collapses when the step size is adapted, a search whose resident set grows
with every round, all look the same from the outside until they finish. This
is the seam that lets a run say so as it goes, without any sampler learning
where the numbers are stored.

Libraries emit, entry points configure --- the division
:mod:`snakes_and_ladders.log` takes, for the same reason. A sampler calls
:func:`current` once at the top of its loop and :meth:`Tracker.scalar` once
per sweep; an entry point --- a notebook, a QA script, an experiment --- opens
:func:`track` and names the store. Outside any context the tracker is
:class:`NullTracker`, whose every method is a no-op, so the default run
records nothing, allocates nothing and is bitwise the run before this module
existed.

**A tracked number is a number the run already reports.** Every hook records
a quantity its result dataclass returns, so the last value of a series equals
the field, and the series can be read as the field's history rather than as a
second definition of it to keep in step
(``tests/regression/test_track.py`` pins that equality).

The active tracker is a :class:`contextvars.ContextVar` rather than a module
global, so two contexts in one process do not write into each other. A
``ContextVar`` does not cross into a pool worker: a task run on
:mod:`snakes_and_ladders.parallel`'s thread or process backend starts from the
default and records nothing, and a run to be tracked per task opens
:func:`track` inside the task.
"""

from __future__ import annotations

import resource
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - the import is for the annotation only
    from matplotlib.figure import Figure


@runtime_checkable
class Tracker(Protocol):
    """Where a run's metrics go: three writes, no reads.

    A sampler holds one of these and calls it; nothing in the package asks a
    tracker what it recorded, so a store that only writes --- Aim, a file, a
    socket --- satisfies it as well as :class:`MemoryTracker` does. An
    optional ``close`` is called by :func:`track` on exit where the
    implementation defines one.
    """

    def params(self, values: Mapping[str, object]) -> None:
        """Record the run's parameters: what was run, at what settings."""

    def scalar(self, name: str, value: float, step: int) -> None:
        """Record one number of the series ``name`` at ``step``."""

    def figure(self, name: str, fig: Figure) -> None:
        """Record a rendered figure under ``name``."""


class NullTracker:
    """The default: every method a no-op, so an untracked run pays a call and nothing else."""

    def params(self, values: Mapping[str, object]) -> None:
        """Discard ``values``."""

    def scalar(self, name: str, value: float, step: int) -> None:
        """Discard ``value``."""

    def figure(self, name: str, fig: Figure) -> None:
        """Discard ``fig``."""


class MemoryTracker:
    """The referee for the tests: keeps what it is given, in order.

    Attributes
    ----------
    scalars : dict[str, list[tuple[int, float]]]
        Per series name, ``(step, value)`` in the order recorded.
    parameters : dict[str, object]
        The parameters, accumulated over every :meth:`params` call. Named
        ``parameters`` and not ``params`` because ``params`` is the method
        that writes it.
    figures : dict[str, Figure]
        The last figure recorded under each name.
    """

    def __init__(self) -> None:
        self.scalars: dict[str, list[tuple[int, float]]] = {}
        self.parameters: dict[str, object] = {}
        self.figures: dict[str, Figure] = {}

    def params(self, values: Mapping[str, object]) -> None:
        """Merge ``values`` into :attr:`parameters`."""
        self.parameters.update(values)

    def scalar(self, name: str, value: float, step: int) -> None:
        """Append ``(step, value)`` to the series ``name``."""
        self.scalars.setdefault(name, []).append((step, value))

    def figure(self, name: str, fig: Figure) -> None:
        """Keep ``fig`` under ``name``."""
        self.figures[name] = fig

    def last(self, name: str) -> float:
        """The last value of the series ``name``.

        Raises
        ------
        KeyError
            If nothing was recorded under ``name``: an assertion against a
            series that was never written is a test that passes on nothing.
        """
        return self.scalars[name][-1][1]


class AimTracker:
    """An Aim run as a :class:`Tracker`: the optional store, imported where it is used.

    ``aim`` is declared in no extra --- ``pyproject.toml`` states the two
    advisories standing against its current release --- so it is installed by
    hand where it is wanted, and the import is inside ``__init__`` rather than
    at module scope: this module is imported by every sampler and must import
    without it.

    A figure is recorded as ``aim.Image(fig)``. Aim's ``aim.Figure`` takes a
    Plotly figure; ``aim.Image`` is the one that accepts a matplotlib
    ``Figure``, which is what ``qa`` renders.

    Parameters
    ----------
    repo : str | Path | None
        The Aim repository to write into; ``None`` is Aim's default, the
        ``.aim`` directory of the working directory. A run written here is
        readable back only after the repository is indexed
        (``aim storage --repo <path> reindex``), which ``DEV.md`` states
        beside the viewer.
    experiment : str | None
        The experiment the run belongs to.
    """

    def __init__(
        self, repo: str | Path | None = None, experiment: str | None = None
    ) -> None:
        import aim

        self._aim = aim
        self._run = aim.Run(
            repo=None if repo is None else str(repo), experiment=experiment
        )

    @property
    def run_hash(self) -> str:
        """Aim's identifier for this run, which is how it is read back."""
        return str(self._run.hash)

    def params(self, values: Mapping[str, object]) -> None:
        """Set each entry on the run."""
        for key, value in values.items():
            self._run[key] = value

    def scalar(self, name: str, value: float, step: int) -> None:
        """Track ``value`` on the sequence ``name`` at ``step``."""
        self._run.track(value, name=name, step=step)

    def figure(self, name: str, fig: Figure) -> None:
        """Track ``fig`` as an ``aim.Image`` under ``name``."""
        self._run.track(self._aim.Image(fig), name=name)

    def close(self) -> None:
        """Close the run, flushing what it holds."""
        self._run.close()


#: The tracker outside any :func:`track` block. One shared instance rather
#: than one per lookup: it holds no state, so nothing distinguishes two.
_NULL: Tracker = NullTracker()

_CURRENT: ContextVar[Tracker] = ContextVar("sal_tracker", default=_NULL)


def current() -> Tracker:
    """The tracker of the enclosing :func:`track` block, or :class:`NullTracker`.

    Read once at the top of a loop and held, not called per step: the lookup
    is cheap and a hook that repeats it per sweep is a hook that can be seen
    in a profile.
    """
    return _CURRENT.get()


@contextmanager
def track(
    name: str, *, tracker: Tracker | None = None, **params: object
) -> Iterator[Tracker]:
    """Run the block with ``tracker`` active, restoring the previous one on exit.

    Parameters
    ----------
    name : str
        What the run is, recorded as the parameter ``name``.
    tracker : Tracker | None
        Where the metrics go; ``None`` is a fresh :class:`MemoryTracker`,
        which is what a test wants and what an entry point overrides with
        :class:`AimTracker`.
    **params : object
        Further parameters, recorded with ``name``.

    Yields
    ------
    Tracker
        The active tracker, so a caller can read what it recorded.
    """
    active: Tracker = MemoryTracker() if tracker is None else tracker
    token = _CURRENT.set(active)
    active.params({"name": name, **params})
    try:
        yield active
    finally:
        _CURRENT.reset(token)
        close = getattr(active, "close", None)
        if callable(close):
            close()


def peak_rss_bytes() -> int:
    """The process's peak resident set size, in bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS, so the platform
    decides the factor; both are converted here rather than at each call
    site. It is a *peak over the process*, not over the block: a second run
    in one process reports the high-water mark of both.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def record_cost(tracker: Tracker, step: int, state_bytes: int) -> None:
    """Record what a run cost the machine: peak resident bytes, and the bytes its state holds.

    Called once, at the end of a run, at the step its last series entry
    carries, so the two land beside the last scalar rather than past it.
    """
    tracker.scalar("peak_rss_bytes", float(peak_rss_bytes()), step)
    tracker.scalar("state_bytes", float(state_bytes), step)


__all__ = [
    "AimTracker",
    "MemoryTracker",
    "NullTracker",
    "Tracker",
    "current",
    "peak_rss_bytes",
    "record_cost",
    "track",
]
