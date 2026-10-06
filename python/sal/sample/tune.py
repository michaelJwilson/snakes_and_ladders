"""A step chosen by a pilot over a log grid of steps (issue #1219).

Dual averaging drives a step toward an acceptance target, and on #1195's HMM
starts that target was the distance that carried a start out of its basin:
adapted, the chain, annealing and tempering starts reached the truth's basin
in 3, 0 and 3 of 10 seeds against 10, 10 and 10 for a hand-run grid (#1216).
The grid chose by the outcome that mattered, so :func:`tune_step` runs the
grid itself: one short pilot per candidate step, each scored by a named
:class:`Criterion`, and the step the criterion ranks first is returned with
every candidate's evidence.

**The pilot is a cost, charged and reported.** The budget is the caller's,
in the unit the sampler spends; it is split evenly over the candidates, and
:attr:`TunedStep.spent` is what the pilots spent, never more than the
budget. A caller comparing at equal cost charges it beside the run it tuned.

**A pilot is a walk of the shared loop** (:func:`sal.sample.loop.anneal`):
the sampler's :data:`~sal.sample.loop.Step` at the candidate step, on a
schedule of the pilot's length --- one temperature for a stationary chain,
the run's own schedule compressed for an annealing run --- so any sampler
behind the loop is tuned by the same function. The walk records nothing
into the enclosing ``track`` run: its series would interleave with the run
it tunes.

**The two criteria answer two questions.** A stationary chain is judged by
the expected squared jump distance per unit spent (ESJD per gradient for a
Hamiltonian step): a rejection jumps zero, so the criterion penalizes a step
too large through its acceptance and a step too small through its distance.
A start or an annealing run is judged by the lowest energy it reached, ties
broken by the ESJD, then by the smaller step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Termination
from sal.sample import loop
from sal.sample.loop import Moved, Step
from sal.sample.schedule import LadderTempSchedule, TempSchedule
from sal.track import NULL_RUN, track

#: The candidate steps unless a caller names its own: nine from 1e-4 to 1 at
#: half a decade, the range #1195's HMM grid (1e-3 to 3e-2) and a unit
#: Gaussian's stability limit (2 for leapfrog) fall inside.
GRID: tuple[float, ...] = tuple(10.0 ** (k / 2.0) for k in range(-8, 1))


#: What a sampler's ``step_size`` reads as "choose it": a :class:`StepTuning`
#: beside it says how.
AUTO: Literal["auto"] = "auto"

#: A step size as a sampler takes it: a number, or :data:`AUTO`.
type StepSize = float | Literal["auto"]


class Criterion(StrEnum):
    """What a pilot ranks its candidate steps by."""

    ESJD = "expected squared jump distance per unit spent"
    """A stationary chain's: the mean squared distance a proposal moved the
    state, rejections moving it zero, summed and divided by the pilot's spend."""
    LOWEST_ENERGY = "lowest energy reached"
    """A start's or an annealing run's: the lowest energy the pilot visited,
    its start included."""


@dataclass(frozen=True)
class StepTuning:
    """How a sampler given ``step_size="auto"`` chooses its step.

    Required beside :data:`AUTO` rather than defaulted, for the reason a step
    is: the pilot's budget is right for one problem's scale and wrong for the
    next, so a default would be wrong silently.

    Parameters
    ----------
    budget : Budget
        The pilot's, in the sampler's unit; a sampler running several pilots
        (a ladder's rungs) splits it evenly between them.
    criterion : Criterion
        What ranks the candidates.
    grid : tuple[float, ...]
        The candidate steps, :data:`GRID` unless named.
    """

    budget: Budget
    criterion: Criterion
    grid: tuple[float, ...] = field(default=GRID)

    def split(self, parts: int) -> StepTuning:
        """The same tuning at ``budget.size // parts``, one pilot of ``parts``."""
        return StepTuning(
            Budget(self.budget.unit, self.budget.size // parts),
            self.criterion,
            self.grid,
        )


def compress(schedule: TempSchedule, n_steps: int) -> TempSchedule:
    """``schedule`` read at ``n_steps`` evenly spaced indices, its endpoints kept.

    An annealing run's pilot is its own schedule at the pilot's length, so a
    candidate step is judged from the hottest temperature to the coldest as
    the run will use it.
    """
    last = schedule.n_steps - 1
    if n_steps == 1:
        return LadderTempSchedule((schedule(last),))
    return LadderTempSchedule(
        tuple(schedule(round(i * last / (n_steps - 1))) for i in range(n_steps))
    )


@dataclass(frozen=True, kw_only=True)
class Pilot[S, C, R]:
    """One sampler, as :func:`tune_step` runs it at a candidate step.

    Parameters
    ----------
    step : Callable[[float], Step[S, C, R]]
        The sampler's move at a step size, built fresh per candidate so no
        candidate's counters reach the next.
    start : Moved[S, C]
        Where every candidate starts, with its energy and what it carries;
        its ``spent`` --- scoring it, ``grad U`` for a Hamiltonian step --- is
        charged once, since every candidate reads the same carried value.
    schedule : Callable[[int], TempSchedule]
        The temperatures of a pilot of the given number of proposals.
    per_proposal : int
        What every proposal costs, in ``unit``.
    unit : Cost
        What the step's charges count.
    keep : Callable[[S], S]
        A copy of a state the step may move in place.
    squared_jump : Callable[[S, S], float]
        The squared distance between two states.
    """

    step: Callable[[float], Step[S, C, R]]
    start: Moved[S, C]
    schedule: Callable[[int], TempSchedule]
    per_proposal: int
    unit: Cost
    keep: Callable[[S], S]
    squared_jump: Callable[[S, S], float]


@dataclass(frozen=True, kw_only=True)
class Candidate[S]:
    """One candidate step's pilot: what it measured, where it ended, what it cost.

    ``moved`` is the fraction of proposals that changed the state, which for
    a continuous Metropolis step is its acceptance rate.
    """

    step_size: float
    esjd: float
    lowest_energy: float
    moved: float
    spent: int
    best: S
    final: S


@dataclass(frozen=True, kw_only=True)
class TunedStep[S]:
    """The chosen step, the criterion that chose it, and every candidate's evidence.

    ``proposals`` is each pilot's length; ``spent`` sums the candidates' in
    ``unit``. The termination is the grid's length, never converged: the
    pilot runs every candidate.
    """

    step_size: float
    criterion: Criterion
    candidates: tuple[Candidate[S], ...]
    proposals: int
    spent: int
    unit: Cost
    termination: Termination

    @property
    def chosen(self) -> Candidate[S]:
        """The candidate whose step was returned."""
        return next(c for c in self.candidates if c.step_size == self.step_size)

    def table(self) -> str:
        """The evidence, one row per candidate, the chosen one marked."""
        rows = [f"{'step':>10}{'esjd':>12}{'lowest':>14}{'moved':>7}{'spent':>7}"]
        rows.extend(
            f"{c.step_size:>10.2e}{c.esjd:>12.3e}{c.lowest_energy:>14.4f}"
            f"{c.moved:>7.2f}{c.spent:>7}" + (" *" if c is self.chosen else "")
            for c in self.candidates
        )
        return "\n".join(rows)


def tune_step[S, C, R](
    sampler: Pilot[S, C, R],
    *,
    budget: Budget,
    criterion: Criterion,
    rng: R,
    grid: Sequence[float] = GRID,
) -> TunedStep[S]:
    """The step of ``grid`` that ``criterion`` ranks first, from one pilot per step on ``rng``.

    Each candidate walks ``((budget.size - start.spent) // len(grid)) //
    per_proposal`` proposals from ``sampler.start``, in grid order, every one
    drawing from ``rng`` in sequence so one seed reproduces the pilot. The
    start's charge is paid once, since it carries what every candidate reads.

    Returns
    -------
    TunedStep
        The chosen step and every candidate, ``spent`` at most ``budget.size``.

    Raises
    ------
    ValueError
        If the budget is in another unit than the sampler's, the grid is
        empty or holds a step that is not positive, or the budget leaves a
        candidate fewer than two proposals: one proposal's jump is a single
        accept-or-reject draw, not a rate.
    """
    if budget.unit is not sampler.unit:
        msg = (
            f"the pilot spends {sampler.unit.value!r}, and the budget is in "
            f"{budget.unit.value!r}"
        )
        raise ValueError(msg)
    steps = tuple(float(step) for step in grid)
    if not steps or min(steps) <= 0.0:
        msg = f"the grid must hold at least one positive step, got {steps}"
        raise ValueError(msg)
    proposals = (
        (budget.size - sampler.start.spent) // len(steps)
    ) // sampler.per_proposal
    if proposals < 2:
        msg = (
            f"a budget of {budget.size} {budget.unit.value} over {len(steps)} steps "
            f"leaves {proposals} proposals per step, and a pilot needs two"
        )
        raise ValueError(msg)
    candidates = []
    with track(NULL_RUN):
        for size in steps:
            candidates.append(_pilot(sampler, size, proposals, rng))
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: _rank(criterion, pair[1], pair[0]),
    )
    chosen = ranked[0][1]
    return TunedStep(
        step_size=chosen.step_size,
        criterion=criterion,
        candidates=tuple(candidates),
        proposals=proposals,
        spent=sampler.start.spent + sum(candidate.spent for candidate in candidates),
        unit=sampler.unit,
        termination=Termination.after(len(steps), converged=False),
    )


def _rank(
    criterion: Criterion, candidate: Candidate[object], index: int
) -> tuple[float, ...]:
    """The sort key: the criterion first, then the ESJD, then grid order."""
    if criterion is Criterion.ESJD:
        return (-candidate.esjd, index)
    return (candidate.lowest_energy, -candidate.esjd, index)


def _pilot[S, C, R](
    sampler: Pilot[S, C, R], step_size: float, proposals: int, rng: R
) -> Candidate[S]:
    """One candidate's walk of ``proposals`` on the shared loop, its jumps summed as it goes."""
    move = sampler.step(step_size)
    jumps: list[float] = []

    def step(
        state: S, energy: float, carried: C, temperature: float, r: R
    ) -> Moved[S, C]:
        moved = move(state, energy, carried, temperature, r)
        jumps.append(sampler.squared_jump(state, moved.state))
        return moved

    start = sampler.start._replace(spent=0)
    walked = loop.anneal(step, sampler.schedule(proposals), start, rng, sampler.keep)
    return Candidate(
        step_size=step_size,
        esjd=sum(jumps) / walked.spent if walked.spent else 0.0,
        lowest_energy=walked.energy,
        moved=sum(jump > 0.0 for jump in jumps) / len(jumps),
        spent=walked.spent,
        best=walked.best,
        final=walked.final,
    )


__all__ = [
    "AUTO",
    "GRID",
    "Candidate",
    "Criterion",
    "Pilot",
    "StepSize",
    "StepTuning",
    "TunedStep",
    "compress",
    "tune_step",
]
