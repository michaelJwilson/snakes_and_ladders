"""Hybrid solvers as stages chained under one budget (issue #1077).

A hybrid runs one solver, hands its answer to the next as that solver's
start, and is charged what every part spent. Three Potts arms wrote that loop
by hand, each with its own budget split and spend sum. :class:`Then` is the
one loop: a sequence of :class:`Step`, each a stage and what it holds back.

A stage takes a problem, a :class:`~sal.opt.budget.Budget` and a generator,
``start`` by keyword, and returns a run carrying ``spent`` and ``seconds``,
the shape `search.ground_state`'s solvers already have. Step ``k`` gets the
budget less what steps before it spent and less its own ``reserve``, the
units kept for the steps after it. All steps draw from the one generator in
order, so a chain is the hand-written loop's draws, bitwise. The chain
returns the last step's run with ``spent`` summed and ``seconds`` the whole
chain's wall time; it keeps the last answer, not the lowest, so a chain of
stages that each return the best they visited never raises the value.

Keyword options (a schedule, a step count) go to the steps that declare them
in ``takes``. An option no step takes is refused, as is a step left no
budget.

:class:`BestOf` is the other combinator: independent realizations from one
start, each on an equal share of the budget and its own spawned generator,
the lowest value kept.

Model-agnostic, per ``opt/CLAUDE.md``: nothing here knows what a problem is.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from sal.opt.budget import Budget


class Staged(Protocol):
    """A stage's run: a frozen dataclass carrying what it spent and how long it took."""

    @property
    def spent(self) -> int:
        """Units spent, in the budget's unit."""
        ...

    @property
    def seconds(self) -> float:
        """Wall seconds."""
        ...


#: A solver: ``stage(problem, budget, rng, *, start=None, **options)`` to a
#: :class:`Staged` run. A ``Callable[..., Any]`` rather than a ``Protocol``,
#: because a stage takes only the options it declares and a ``Protocol`` with
#: ``**options`` admits no function without them.
Stage = Callable[..., Any]


@dataclass(frozen=True)
class Step:
    """One stage of a chain, and the options and budget it takes.

    Parameters
    ----------
    stage : Stage
        The solver.
    takes : frozenset[str]
        The keyword options routed to it.
    reserve : Callable[[Any], int] | None
        Units held back from it for the steps after it, read from the
        problem; ``None`` holds nothing back.
    """

    stage: Stage
    takes: frozenset[str] = frozenset()
    reserve: Callable[[Any], int] | None = None


@dataclass(frozen=True)
class Then:
    """Steps run in order, each from the answer of the one before (issue #1077).

    Parameters
    ----------
    steps : tuple[Step, ...]
        At least one; one step is that stage alone, its spend and answer
        unchanged.
    handover : Callable[[Any], Any]
        Reads the answer out of a run: the next step's ``start``.

    A module-level ``handover`` and stages built from module-level functions
    and ``functools.partial`` keep a chain picklable, so it crosses a process
    boundary as a hand-written arm does.
    """

    steps: tuple[Step, ...]
    handover: Callable[[Any], Any]

    def __post_init__(self) -> None:
        if not self.steps:
            msg = "a chain has at least one step"
            raise ValueError(msg)

    @property
    def takes(self) -> frozenset[str]:
        """Every option some step takes."""
        return frozenset().union(*(step.takes for step in self.steps))

    def __call__(
        self,
        problem: Any,
        budget: Budget,
        rng: np.random.Generator,
        /,
        *,
        start: Any = None,
        **options: Any,
    ) -> Any:
        """Run the steps on ``problem`` within ``budget``, the first from ``start``.

        Raises
        ------
        ValueError
            If an option is taken by no step, or a step is left no budget.
        """
        unknown = set(options) - self.takes
        if unknown:
            msg = (
                f"no step takes {sorted(unknown)}; the chain takes {sorted(self.takes)}"
            )
            raise ValueError(msg)
        started = time.perf_counter()
        spent = 0
        run: Any = None
        for index, step in enumerate(self.steps):
            held = 0 if step.reserve is None else step.reserve(problem)
            size = budget.size - spent - held
            if size < 1:
                msg = (
                    f"step {index} is left {size} {budget.unit} of "
                    f"{budget.size}: {spent} spent, {held} held back"
                )
                raise ValueError(msg)
            chosen = {name: options[name] for name in step.takes if name in options}
            run = step.stage(
                problem, Budget(budget.unit, size), rng, start=start, **chosen
            )
            spent += run.spent
            start = self.handover(run)
        return dataclasses.replace(
            run, spent=spent, seconds=time.perf_counter() - started
        )


def then(stages: Sequence[Stage | Step], handover: Callable[[Any], Any]) -> Then:
    """A :class:`Then` over ``stages``, a bare stage read as a :class:`Step` taking nothing."""
    return Then(
        tuple(stage if isinstance(stage, Step) else Step(stage) for stage in stages),
        handover,
    )


@dataclass(frozen=True)
class BestOf:
    """Independent realizations from one start, the lowest value kept (issue #1077).

    Each branch runs on ``budget.size // len(branches)`` units, the equal
    split :func:`~sal.opt.budget.restarts` makes, from the same ``start``,
    on its own generator spawned from ``rng`` in branch order. The spawned
    streams are what make a branch's run independent of the others' and of
    their order, so a parallel run is the serial one bitwise. The run is the
    branch with the lowest ``value``, the first on a tie, with ``spent``
    summed over every branch and ``seconds`` the whole.

    Parameters
    ----------
    branches : tuple[Then, ...]
        At least two; a bare stage is a one-step :class:`Then`.
    value : Callable[[Any], float]
        Reads what is minimized out of a run.
    """

    branches: tuple[Then, ...]
    value: Callable[[Any], float]

    def __post_init__(self) -> None:
        if len(self.branches) < 2:
            msg = f"realizations are at least two, got {len(self.branches)}"
            raise ValueError(msg)

    @property
    def takes(self) -> frozenset[str]:
        """Every option some branch takes."""
        return frozenset().union(*(branch.takes for branch in self.branches))

    def __call__(
        self,
        problem: Any,
        budget: Budget,
        rng: np.random.Generator,
        /,
        *,
        start: Any = None,
        **options: Any,
    ) -> Any:
        """Run every branch on its share and keep the lowest.

        Raises
        ------
        ValueError
            If an option is taken by no branch, or a share is under one unit.
        """
        unknown = set(options) - self.takes
        if unknown:
            msg = (
                f"no branch takes {sorted(unknown)}; the branches take "
                f"{sorted(self.takes)}"
            )
            raise ValueError(msg)
        share = budget.size // len(self.branches)
        if share < 1:
            msg = (
                f"{len(self.branches)} realizations of {budget.size} "
                f"{budget.unit} leave each less than one"
            )
            raise ValueError(msg)
        started = time.perf_counter()
        streams = rng.spawn(len(self.branches))
        best: Any = None
        spent = 0
        for branch, stream in zip(self.branches, streams, strict=True):
            chosen = {name: options[name] for name in branch.takes if name in options}
            run = branch(
                problem, Budget(budget.unit, share), stream, start=start, **chosen
            )
            spent += run.spent
            if best is None or self.value(run) < self.value(best):
                best = run
        return dataclasses.replace(
            best, spent=spent, seconds=time.perf_counter() - started
        )
