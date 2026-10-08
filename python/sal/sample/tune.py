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

**The criteria answer three questions.** A stationary chain is judged by
the expected squared jump distance per unit spent (ESJD per gradient for a
Hamiltonian step): a rejection jumps zero, so the criterion penalizes a step
too large through its acceptance and a step too small through its distance.
A start or an annealing run is judged by the lowest energy it reached, ties
broken by the ESJD, then by the smaller step. A start handed to a polisher
is judged by what the polish reaches (issue #1251): each candidate's best
point is polished at a short budget and ranked by the polished objective,
ties broken as the lowest energy's are. That is #1195's grid, run per start:
a step that carries a start out of its basin descends furthest in a short
pilot and polishes into the other basin, so the lowest energy ranks it first
and the polished gap ranks it last.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import numpy as np

from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Termination
from sal.parallel import Pool, map_tasks
from sal.sample import loop
from sal.sample.loop import Moved, Step
from sal.sample.schedule import (
    LadderTempSchedule,
    ScheduleParams,
    ScheduleShape,
    TempSchedule,
)
from sal.track import NULL_RUN, track

if TYPE_CHECKING:
    from sal.opt.starts import Polished, Polisher
    from sal.sample.potts_mcmc import PottsMoves, Recolour
    from sal.sim.graph import PottsGraph

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
    POLISHED_GAP = "objective after a short polish"
    """A start's that a polisher takes over (issue #1251): the value the
    polish reaches from the pilot's best point, which is the gap to any
    reference up to a constant."""


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
    polish : Polisher | None
        What :attr:`Criterion.POLISHED_GAP` polishes each candidate's best
        point with, required with it and refused without it.
    polish_budget : Budget | None
        Each candidate's polish, in :attr:`~sal.cost.Cost.ITERATIONS`, as
        ``polish`` counts: a cap, so a pilot of ``g`` candidates polishes for
        at most ``g * polish_budget.size`` iterations beside ``budget``. A
        ladder splits it over its rungs as it splits ``budget``.

    Raises
    ------
    ValueError
        If ``polish`` and ``polish_budget`` are not both given with
        :attr:`Criterion.POLISHED_GAP` and both absent otherwise.
    """

    budget: Budget
    criterion: Criterion
    grid: tuple[float, ...] = field(default=GRID)
    polish: Polisher | None = None
    polish_budget: Budget | None = None

    def __post_init__(self) -> None:
        polishes = self.criterion is Criterion.POLISHED_GAP
        given = (self.polish is not None, self.polish_budget is not None)
        if given != (polishes, polishes):
            msg = (
                f"a polish and its budget come with {Criterion.POLISHED_GAP.name} "
                f"and only with it; the criterion is {self.criterion.name}"
            )
            raise ValueError(msg)

    def split(self, parts: int) -> StepTuning:
        """The same tuning at ``budget.size // parts`` and ``polish_budget.size // parts``, one pilot of ``parts``."""
        polish_budget = self.polish_budget
        if polish_budget is not None:
            polish_budget = Budget(polish_budget.unit, polish_budget.size // parts)
        return StepTuning(
            Budget(self.budget.unit, self.budget.size // parts),
            self.criterion,
            self.grid,
            self.polish,
            polish_budget,
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
    a continuous Metropolis step is its acceptance rate. ``polished`` is the
    value a polish reached from ``best`` and ``polish_iterations`` what it
    spent, under :attr:`Criterion.POLISHED_GAP`; ``None`` and 0 otherwise.
    """

    step_size: float
    esjd: float
    lowest_energy: float
    moved: float
    spent: int
    best: S
    final: S
    polished: float | None = None
    polish_iterations: int = 0


@dataclass(frozen=True, kw_only=True)
class TunedStep[S]:
    """The chosen step, the criterion that chose it, and every candidate's evidence.

    ``proposals`` is each pilot's length; ``spent`` sums the candidates' in
    ``unit``, and ``polish_iterations`` their polishes' in
    :attr:`~sal.cost.Cost.ITERATIONS`, 0 unless the criterion polishes. The
    termination is the grid's length, never converged: the pilot runs every
    candidate.
    """

    step_size: float
    criterion: Criterion
    candidates: tuple[Candidate[S], ...]
    proposals: int
    spent: int
    unit: Cost
    termination: Termination
    polish_iterations: int = 0

    @property
    def chosen(self) -> Candidate[S]:
        """The candidate whose step was returned."""
        return next(c for c in self.candidates if c.step_size == self.step_size)

    def table(self) -> str:
        """The evidence, one row per candidate, the chosen one marked."""
        rows = [
            f"{'step':>10}{'esjd':>12}{'lowest':>14}{'polished':>14}"
            f"{'moved':>7}{'spent':>7}"
        ]
        rows.extend(
            f"{c.step_size:>10.2e}{c.esjd:>12.3e}{c.lowest_energy:>14.4f}"
            + (f"{'-':>14}" if c.polished is None else f"{c.polished:>14.4f}")
            + f"{c.moved:>7.2f}{c.spent:>7}"
            + (" *" if c is self.chosen else "")
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
    polish: Callable[[S], Polished] | None = None,
) -> TunedStep[S]:
    """The step of ``grid`` that ``criterion`` ranks first, from one pilot per step on ``rng``.

    Each candidate walks ``((budget.size - start.spent) // len(grid)) //
    per_proposal`` proposals from ``sampler.start``, in grid order, every one
    drawing from ``rng`` in sequence so one seed reproduces the pilot. The
    start's charge is paid once, since it carries what every candidate reads.
    Under :attr:`Criterion.POLISHED_GAP`, ``polish`` takes each candidate's
    best point after its walk, in grid order, and draws nothing from ``rng``.

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
        accept-or-reject draw, not a rate; or if ``polish`` is not given with
        :attr:`Criterion.POLISHED_GAP` and only with it.
    """
    if budget.unit is not sampler.unit:
        msg = (
            f"the pilot spends {sampler.unit.value!r}, and the budget is in "
            f"{budget.unit.value!r}"
        )
        raise ValueError(msg)
    if (polish is not None) != (criterion is Criterion.POLISHED_GAP):
        msg = (
            f"a polish comes with {Criterion.POLISHED_GAP.name} and only with "
            f"it; the criterion is {criterion.name}"
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
            candidate = _pilot(sampler, size, proposals, rng)
            if polish is not None:
                polished = polish(candidate.best)
                candidate = replace(
                    candidate,
                    polished=polished.value,
                    polish_iterations=polished.iterations,
                )
            candidates.append(candidate)
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
        polish_iterations=sum(c.polish_iterations for c in candidates),
    )


def _rank(
    criterion: Criterion, candidate: Candidate[object], index: int
) -> tuple[float, ...]:
    """The sort key: the criterion first, then the lowest energy and the ESJD, then grid order."""
    if criterion is Criterion.ESJD:
        return (-candidate.esjd, index)
    if criterion is Criterion.POLISHED_GAP:
        assert candidate.polished is not None
        return (candidate.polished, candidate.lowest_energy, -candidate.esjd, index)
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


#: The candidate schedules unless a caller names its own (issue #1317): the
#: exponential, linear and cosine shapes from ``t_start`` in {2.0, 0.8} to
#: ``t_end`` in {0.05, 0.3}, unheld, 12 candidates. A caller's grid may hold
#: any :class:`~sal.sample.schedule.ScheduleShape` (#1333); the default stays
#: #1317's, so a default run's choice does not move. The endpoints bracket
#: :data:`~sal.search.ground_state.ANNEAL_SCHEDULE` (2.0 to 0.05) and
#: :data:`~sal.search.ground_state.SWENDSEN_WANG_SCHEDULE` (0.78 to 0.32).
SCHEDULE_GRID: tuple[ScheduleParams, ...] = tuple(
    ScheduleParams(shape, t_start, t_end)
    for shape in (ScheduleShape.EXPONENTIAL, ScheduleShape.LINEAR, ScheduleShape.COSINE)
    for t_start in (2.0, 0.8)
    for t_end in (0.05, 0.3)
)

#: What an annealed Potts entry point's ``schedule`` reads as "choose it": a
#: :class:`ScheduleTuning` beside it says how.
type Schedule = TempSchedule | Literal["auto"]


@dataclass(frozen=True)
class ScheduleTuning:
    """How an annealed Potts entry point given ``schedule="auto"`` chooses its schedule (issue #1317).

    Required beside ``"auto"`` rather than defaulted, as :class:`StepTuning`
    is beside ``step_size="auto"``: a pilot budget right for one instance is
    wrong for the next, silently.

    Parameters
    ----------
    budget : Budget
        The pilots', in :attr:`~sal.cost.Cost.SITE_VISITS`, split evenly over
        ``grid``: each candidate anneals for ``(budget.size // len(grid)) //
        n_nodes`` sweeps.
    criterion : Criterion
        What ranks the candidates: :attr:`Criterion.LOWEST_ENERGY` alone.
    n_steps : int
        The tuned run's step count, at which the chosen schedule is built.
    grid : tuple[ScheduleParams, ...]
        The candidate schedules, :data:`SCHEDULE_GRID` unless named.

    Raises
    ------
    ValueError
        If the budget is not in site visits, the criterion is not
        :attr:`Criterion.LOWEST_ENERGY`, ``n_steps`` is below one or the grid
        is empty.
    """

    budget: Budget
    criterion: Criterion
    n_steps: int
    grid: tuple[ScheduleParams, ...] = field(default=SCHEDULE_GRID)

    def __post_init__(self) -> None:
        if self.budget.unit is not Cost.SITE_VISITS:
            msg = (
                f"a schedule pilot spends {Cost.SITE_VISITS.value!r}, and the "
                f"budget is in {self.budget.unit.value!r}"
            )
            raise ValueError(msg)
        if self.criterion is not Criterion.LOWEST_ENERGY:
            msg = (
                f"a schedule is ranked by {Criterion.LOWEST_ENERGY.name}; "
                f"{self.criterion.name} has no schedule pilot yet (#1317)"
            )
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = f"the tuned run needs at least one step, got {self.n_steps}"
            raise ValueError(msg)
        if not self.grid:
            msg = "the schedule grid must hold at least one candidate"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScheduleCandidate:
    """One candidate schedule's pilot: its lowest energy and its spend."""

    params: ScheduleParams
    lowest_energy: float
    spent: int


@dataclass(frozen=True)
class TunedSchedule:
    """The schedule a pilot over a grid chose, with every candidate's evidence (issue #1317).

    Parameters
    ----------
    params : ScheduleParams
        The chosen candidate's.
    criterion : Criterion
    candidates : tuple[ScheduleCandidate, ...]
        In grid order.
    sweeps : int
        Each pilot's step count.
    spent : int
        Site visits of every pilot.
    unit : Cost
    termination : Termination
        After one pilot per candidate, never converged.
    """

    params: ScheduleParams
    criterion: Criterion
    candidates: tuple[ScheduleCandidate, ...]
    sweeps: int
    spent: int
    unit: Cost
    termination: Termination


def _schedule_pilot(
    params: ScheduleParams,
    rng: np.random.Generator,
    *,
    graph: PottsGraph,
    field: np.ndarray,
    move: PottsMoves,
    recolour: Recolour,
    sweeps: int,
) -> ScheduleCandidate:
    """One candidate's anneal on its own generator; thread-safe, it writes no shared state."""
    from sal.sample.potts_mcmc import anneal_potts

    run = anneal_potts(
        graph, field, params.build(sweeps), rng, move=move, recolour=recolour
    )
    return ScheduleCandidate(params, run.energy, run.spent)


def tune_schedule(
    graph: PottsGraph,
    field: np.ndarray,
    *,
    move: PottsMoves,
    recolour: Recolour,
    budget: Budget,
    criterion: Criterion,
    rng: np.random.Generator,
    grid: Sequence[ScheduleParams] = SCHEDULE_GRID,
    workers: int = 1,
    pool: Pool = "serial",
) -> TunedSchedule:
    """The schedule of ``grid`` that ``criterion`` ranks first, from one annealing pilot per candidate.

    ``qa.potts_schedule``'s question, asked by the package (issue #1317):
    every candidate is a :class:`~sal.sample.schedule.ScheduleParams` (shape,
    start, end and hold), built at ``(budget.size // len(grid)) //
    graph.n_nodes`` steps and annealed by
    :func:`~sal.sample.potts_mcmc.anneal_potts` under ``move`` and
    ``recolour``. Each pilot draws from its own generator spawned from
    ``rng`` by :func:`sal.parallel.map_tasks`, so a thread pool returns the
    serial result bitwise. Ranked by the lowest energy, ties to grid order.
    A pilot is charged its own spend, which for a Wolff move is below its
    share: Wolff's step visits one cluster, not a sweep.

    Raises
    ------
    ValueError
        As :class:`ScheduleTuning` does, or if the budget leaves a pilot
        fewer than two steps.
    """
    candidates_in = tuple(grid)
    tuning = ScheduleTuning(budget, criterion, 1, candidates_in)
    sweeps = (tuning.budget.size // len(candidates_in)) // graph.n_nodes
    if sweeps < 2:
        msg = (
            f"a budget of {budget.size} site visits over {len(candidates_in)} "
            f"schedules on {graph.n_nodes} sites leaves {sweeps} steps per "
            "pilot, and a pilot needs two"
        )
        raise ValueError(msg)
    body = functools.partial(
        _schedule_pilot,
        graph=graph,
        field=field,
        move=move,
        recolour=recolour,
        sweeps=sweeps,
    )
    with track(NULL_RUN):
        candidates = map_tasks(
            body, candidates_in, workers=workers, pool=pool, generator=rng
        )
    index = min(range(len(candidates)), key=lambda k: (candidates[k].lowest_energy, k))
    return TunedSchedule(
        params=candidates[index].params,
        criterion=criterion,
        candidates=tuple(candidates),
        sweeps=sweeps,
        spent=sum(candidate.spent for candidate in candidates),
        unit=Cost.SITE_VISITS,
        termination=Termination.after(len(candidates), converged=False),
    )


def resolve_schedule(
    schedule: Schedule,
    tuning: ScheduleTuning | None,
    *,
    graph: PottsGraph,
    field: np.ndarray,
    move: PottsMoves,
    recolour: Recolour,
    rng: np.random.Generator,
) -> tuple[TempSchedule, TunedSchedule | None]:
    """``schedule`` as given, or the one ``tuning``'s pilot chooses under ``"auto"``.

    The refusals of ``step_size="auto"`` (:mod:`sal.sample.hmc`), checked
    before any draw: ``"auto"`` without a tuning, or a tuning beside a given
    schedule it would not change. A given schedule draws nothing here, so
    its run is bitwise the run before #1317.

    Raises
    ------
    ValueError
        If ``"auto"`` comes without a :class:`ScheduleTuning`, or one comes
        with a given schedule.
    """
    if not isinstance(schedule, str):
        if tuning is not None:
            msg = (
                "a tuning chooses schedule='auto', and the schedule is given as "
                f"{schedule!r}"
            )
            raise ValueError(msg)
        return schedule, None
    if schedule != AUTO:
        msg = f"schedule is a TempSchedule or 'auto', got {schedule!r}"
        raise ValueError(msg)
    if tuning is None:
        msg = (
            "schedule='auto' needs a ScheduleTuning: the pilots' budget, "
            "criterion, step count and grid"
        )
        raise ValueError(msg)
    tuned = tune_schedule(
        graph,
        field,
        move=move,
        recolour=recolour,
        budget=tuning.budget,
        criterion=tuning.criterion,
        rng=rng,
        grid=tuning.grid,
    )
    return tuned.params.build(tuning.n_steps), tuned


__all__ = [
    "AUTO",
    "GRID",
    "SCHEDULE_GRID",
    "Candidate",
    "Criterion",
    "Pilot",
    "Schedule",
    "ScheduleCandidate",
    "ScheduleTuning",
    "StepSize",
    "StepTuning",
    "TunedSchedule",
    "TunedStep",
    "compress",
    "resolve_schedule",
    "tune_schedule",
    "tune_step",
]
