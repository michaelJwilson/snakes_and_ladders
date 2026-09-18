"""Where a classical baseline first fails, which is where a gate can be argued.

Issue #596. `ROADMAP.md` Milestone 2.1 asks for a learned policy that beats
the classical baseline, and on every fixture in the tree today the baseline is
**1.000**: random-restart hill climbing reaches the enumerated maximum from
every start at the budget the comparison holds equal (`STATUS.md`, issue
#194). Nothing beats an exact answer, so the first work is not learning but
**sizing** --- growing an instance until the baseline measurably fails, and
recording the size at which it first does.

This module is that measurement and nothing else. It takes a
:data:`~snakes_and_ladders.opt.budget.Method`, a family of instances indexed by
size, and the target each size is scored against, and reports the fraction of
seeded starts that reached the target. It knows what an instance is no more
than `budget.py` does, which is why it sits beside it rather than in `qa/`:
`qa/CLAUDE.md`'s rule is that a script there *renders* what the application
computed, and a sizing sweep computes.

**The budget is the unit of comparison, and the clock is a ceiling on top of
it.** A baseline spends its budget on restarts and a policy on decisions, so
holding the budget equal is what makes the two comparable ---
:func:`snakes_and_ladders.opt.budget.restarts` derives the restart count from
the declared cost, and this module never counts a restart itself. A wall-clock
ceiling is separate: it bounds a sweep that would otherwise run for hours at
the largest size, and a probe that hits it reports the starts it managed
rather than the starts it was asked for, so a fraction is never read off a
sample the run did not take.

**A baseline that does not fail is a result.** If no size inside the ceiling
drops the baseline below the threshold, :attr:`FailureCurve.first_failure` is
``None`` and the curve is what gets reported --- the finding is then about the
problem and not about the work, and issue #596 says to bring the curve rather
than widen the budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from snakes_and_ladders.opt.budget import Budget, Method, OverspendError

InstanceT = TypeVar("InstanceT")

#: What a fraction has to fall below for a baseline to count as failing. Not
#: tuned: a baseline reaching the target from every start is the state issue
#: #596 records, so anything short of all of them is the first thing worth
#: reporting, and a threshold under 1 would hide the size where it starts.
DEFAULT_THRESHOLD = 1.0

#: How close a value must come to the target to count as reaching it. The
#: targets here are enumerated optima and planted truths, so a run either
#: lands on one or does not; the tolerance absorbs the ordering of a
#: floating-point reduction and nothing else.
DEFAULT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FailureProbe:
    """One reading: how often a method reached the target at one size.

    Parameters
    ----------
    size : int
        The instance's size, as the caller's family indexes it.
    starts : int
        Starts actually run, which is fewer than asked for when the wall-clock
        ceiling stopped the probe.
    asked : int
        Starts requested, so a truncated probe says so rather than looking
        like a smaller experiment.
    reached : int
        Starts whose value came within the tolerance of the target.
    best : float
        The best value any start reached. Lower is better, as in
        :class:`~snakes_and_ladders.opt.budget.Outcome`.
    target : float
        What the size was scored against.
    seconds : float
        Wall clock the probe spent.
    """

    size: int
    starts: int
    asked: int
    reached: int
    best: float
    target: float
    seconds: float

    @property
    def fraction(self) -> float:
        """Reached over run. Zero starts is refused, so this never divides by zero."""
        return self.reached / self.starts

    @property
    def truncated(self) -> bool:
        """Whether the clock stopped the probe before the starts were spent."""
        return self.starts < self.asked


@dataclass(frozen=True)
class FailureCurve(Generic[InstanceT]):
    """A sweep over sizes, and the first size at which the baseline failed.

    Parameters
    ----------
    probes : tuple[FailureProbe, ...]
        In the order the sizes were swept, which is the order they are read.
    threshold : float
        What a fraction had to fall below to count as a failure.
    """

    probes: tuple[FailureProbe, ...]
    threshold: float

    @property
    def first_failure(self) -> FailureProbe | None:
        """The first probe under the threshold, or ``None`` if none is.

        ``None`` is the reportable outcome issue #596 names: the baseline did
        not fail inside what was measured, so the gate cannot be argued on
        this problem at these sizes.
        """
        for probe in self.probes:
            if probe.fraction < self.threshold:
                return probe
        return None


def probe_failure(
    method: Method[InstanceT],
    instance: InstanceT,
    *,
    target: float,
    budget: Budget,
    starts: int,
    rng: np.random.Generator,
    size: int,
    tolerance: float = DEFAULT_TOLERANCE,
    relative: float | None = None,
    seconds: float | None = None,
) -> FailureProbe:
    """Run ``method`` from ``starts`` seeded starts and count what reached ``target``.

    Parameters
    ----------
    method : Method
        Called once per start with ``instance``, ``budget`` and the generator.
        A restart baseline is
        :func:`~snakes_and_ladders.opt.budget.restarts` of a single descent,
        so the restart count is derived from its declared cost and never from
        here.
    instance : InstanceT
        What the method takes. This module does not look inside it.
    target : float
        The value a start has to reach: an enumerated optimum, a planted
        truth, or the best value known at this size. Lower is better.
    budget : Budget
        Held equal across every method a caller compares.
    starts : int
        Starts to run, at least one.
    rng : np.random.Generator
        Advanced once per start, so the starts are independent and the sweep
        is reproducible from one generator.
    size : int
        Recorded on the probe. Passed rather than derived, since only the
        caller's family knows what size means.
    tolerance : float
        Absolute, against ``target``.
    relative : float | None
        A fractional allowance on ``abs(target)``, used instead of
        ``tolerance`` where the objective is continuous. An exact optimum is
        hit or missed and ``tolerance`` is the right instrument; an energy a
        method *approaches* is not, and on the Potts lattice the two
        instruments disagree completely --- every local method reads 0.00 by
        exact hit at 144 sites and lands within 0.04% of the target
        (issue #596). Which one a curve used belongs beside its numbers, so
        this is a parameter and not a default.
    seconds : float | None
        Wall-clock ceiling for this probe. ``None`` runs every start.

    Returns
    -------
    FailureProbe

    Raises
    ------
    ValueError
        If ``starts`` is under one, or ``seconds`` is not positive.
    OverspendError
        If a method reports spending more than the budget. "Wins by running
        longer" is the error a matched-budget comparison exists to make
        impossible, so it is refused here as in
        :mod:`~snakes_and_ladders.opt.budget`.
    """
    if starts < 1:
        msg = f"a probe runs at least one start, got {starts}"
        raise ValueError(msg)
    if seconds is not None and seconds <= 0.0:
        msg = f"a wall-clock ceiling is positive, got {seconds}"
        raise ValueError(msg)
    if relative is not None and relative < 0.0:
        msg = f"a relative allowance is non-negative, got {relative}"
        raise ValueError(msg)

    # One-sided, and that is a correction rather than a convenience: a target
    # may be the best value *known* at this size rather than an optimum, and a
    # start that beats it has not failed. Against an exact optimum the two
    # readings agree, since nothing goes below it by more than a reduction's
    # ordering.
    def allowance(value: float) -> float:
        return tolerance if relative is None else relative * abs(value)

    began = time.perf_counter()
    reached = 0
    best = np.inf
    run = 0
    for _ in range(starts):
        outcome = method(instance, budget, rng)
        if outcome.spent > budget.size:
            msg = (
                f"a start spent {outcome.spent} {budget.unit} against a budget "
                f"of {budget.size}"
            )
            raise OverspendError(msg)
        run += 1
        best = min(best, outcome.value)
        reached += int(outcome.value <= target + allowance(target))
        if seconds is not None and time.perf_counter() - began >= seconds:
            break

    return FailureProbe(
        size=size,
        starts=run,
        asked=starts,
        reached=reached,
        best=float(best),
        target=target,
        seconds=time.perf_counter() - began,
    )


def failure_curve(
    method: Method[InstanceT],
    family: Callable[[int], tuple[InstanceT, float]],
    sizes: Sequence[int],
    *,
    budget: Budget,
    starts: int,
    rng: np.random.Generator,
    threshold: float = DEFAULT_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
    relative: float | None = None,
    seconds: float | None = None,
    stop_at_failure: bool = True,
) -> FailureCurve[InstanceT]:
    """Sweep ``sizes`` and report where the baseline first fell below ``threshold``.

    Parameters
    ----------
    method : Method
        The baseline, at one budget for every size. A budget that grew with
        the size would measure two things at once.
    family : Callable[[int], tuple[InstanceT, float]]
        Size to ``(instance, target)``. The caller owns both: this module
        cannot know what the optimum of an instance is, and a family that
        returns the best value *observed* instead of an exact one is stating a
        weaker claim, which is the caller's to state.
    sizes : Sequence[int]
        Swept in the order given, which is read as ascending.
    budget : Budget
        Held equal across sizes and methods.
    starts : int
        Per probe.
    rng : np.random.Generator
        One generator for the sweep.
    threshold : float
        In ``(0, 1]``. See :data:`DEFAULT_THRESHOLD`.
    tolerance : float
        Absolute, against each size's target.
    seconds : float | None
        Wall-clock ceiling **per probe**, not for the sweep, so one slow size
        cannot silently truncate the next.
    stop_at_failure : bool
        Stop after the first size under the threshold, which is what a sizing
        sweep wants: the sizes past it cost more and say less. ``False`` runs
        every size, for the curve a report carries.

    Returns
    -------
    FailureCurve

    Raises
    ------
    ValueError
        If ``sizes`` is empty or ``threshold`` is outside ``(0, 1]``.
    """
    if not sizes:
        msg = "a sweep needs at least one size"
        raise ValueError(msg)
    if not 0.0 < threshold <= 1.0:
        msg = f"a threshold lies in (0, 1], got {threshold}"
        raise ValueError(msg)

    probes: list[FailureProbe] = []
    for size in sizes:
        instance, target = family(size)
        probe = probe_failure(
            method,
            instance,
            target=target,
            budget=budget,
            starts=starts,
            rng=rng,
            size=size,
            tolerance=tolerance,
            relative=relative,
            seconds=seconds,
        )
        probes.append(probe)
        if stop_at_failure and probe.fraction < threshold:
            break

    return FailureCurve(probes=tuple(probes), threshold=threshold)


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOLERANCE",
    "FailureCurve",
    "FailureProbe",
    "failure_curve",
    "probe_failure",
]
