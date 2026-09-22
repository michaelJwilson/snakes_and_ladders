"""Initializers built from a sampler: a chain, an annealing run, a tempering run.

Three of :mod:`snakes_and_ladders.opt.initialize`'s starting points are a
sampler's output --- the last draw of an HMC chain, the best state of an
annealing run, the cold replica of a tempering run --- and they lived beside
the ones that are not, which put a sampler import in ``opt/`` against its
own sentence (issue #830). They implement the same
:class:`~snakes_and_ladders.opt.initialize.Initializer` seam and are reached
by the same callers; only the directory moved, and every start is bitwise
what it was.

The tempering start may calibrate its ladder first (issue #902):
:class:`LadderCalibration` states the warm-up, :class:`CalibratedLadder` what
it settled on and what it cost, and :class:`TemperingSpend` the whole start's
spend. They live here rather than beside :class:`~snakes_and_ladders.sample.hmc.Adaptation`
because :func:`~snakes_and_ladders.sample.hmc.parallel_tempering` takes a
fixed ladder and consumes none of them: the calibration is this start's
composition of that sampler with the adapters of
:mod:`snakes_and_ladders.sample.schedule`, as
:func:`~snakes_and_ladders.sample.potts_mcmc.adapt_ladder_potts` is the
lattice's.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import StrEnum

import torch

from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.initialize import Initializer
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.schedule import (
    AdaptedLadder,
    FeedbackLadder,
    Monotone,
    TempSchedule,
    adapt_ladder,
    adapt_ladder_by_round_trips,
    check_ladder,
)
from snakes_and_ladders.sample.tempered import up_fraction

#: The warm-up :class:`FromChain` runs unless told otherwise (issue #898): 300
#: proposals driving the acceptance to 0.65 with the step jittered by 0.4.
#: These are the values :data:`~snakes_and_ladders.sample.hmc.DUAL_AVERAGING_GAMMA`
#: was measured at --- the drawn chain's acceptance 0.643 against the 0.65
#: target over 10 seeds on the analytic Gaussian --- and 0.65 is Hoffman &
#: Gelman's target for HMC. :mod:`~snakes_and_ladders.sample.hmc` states no
#: default warm-up, since one right for one target is too short for the next,
#: so this one is the start's and is stated here.
CHAIN_ADAPTATION = hmc.Adaptation(warmup=300, target_acceptance=0.65, step_jitter=0.4)


class FromChain(Initializer):
    """A start drawn from a short Hamiltonian chain on the objective itself.

    The three above propose a point; this one samples the surface. Where the
    objective is a negative log-likelihood the chain targets its posterior
    under a flat prior, which `opt/CLAUDE.md` records is improper for most
    models --- so what is taken from it is a *start*, never a posterior
    summary.

    **The chain is warmed up by default.** Unless ``adaptation`` says
    otherwise the chain runs :data:`CHAIN_ADAPTATION` first: the step size
    and a diagonal mass are set by
    :class:`~snakes_and_ladders.sample.hmc.Adaptation`, ``step_size`` is the
    warm-up's starting point, and the ``burn_in`` and the draws run at the
    values it ended on. Without it the chain runs at the caller's step and
    unit mass whatever the objective's scale; ``adaptation=None`` is that
    fixed-parameter chain at unit mass, bitwise what this class drew before
    issue #898, and its cost is its proposals alone. The warm-up's gradients
    are in ``force_evaluations``, so a caller charging the chain charges them.

    **Label switching is the hazard, not the step size.** A mixture
    likelihood is invariant under permuting its components, so a chain that
    crosses between modes returns the same fit under arbitrary names.
    :meth:`starts` therefore returns the draws unchanged and the caller
    canonicalizes; :meth:`chain` hands back the diagnostics, because a seed
    drawn from a chain that has not mixed is a random restart with a longer
    bill.

    Parameters
    ----------
    n_samples : int
        Draws kept, at least 1. The last is the start.
    step_size : float
        Leapfrog step, positive.
    generator : torch.Generator
        The stream, passed in rather than seeded here (issue #337).
    n_steps : int
        Leapfrog steps per proposal.
    burn_in : int
        Proposals discarded before the kept draws.
    adaptation : hmc.Adaptation | None
        The warm-up run before the burn-in, :data:`CHAIN_ADAPTATION` by
        default; ``None`` runs no warm-up.

    Raises
    ------
    ValueError
        If fewer than one draw is asked for.
    """

    def __init__(
        self,
        n_samples: int,
        step_size: float,
        generator: torch.Generator,
        n_steps: int = hmc.DEFAULT_STEPS,
        burn_in: int = 0,
        adaptation: hmc.Adaptation | None = CHAIN_ADAPTATION,
    ) -> None:
        if n_samples < 1:
            msg = f"n_samples must be at least 1, got {n_samples}"
            raise ValueError(msg)
        self.n_samples = n_samples
        self.step_size = step_size
        self.generator = generator
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.adaptation = adaptation

    def chain(self, objective: Objective) -> hmc.HmcChain:
        """The chain itself: the draws, the acceptance rate, what it cost and what the warm-up set.

        Returns
        -------
        HmcChain
        """
        return hmc.sample(
            objective,
            self.generator,
            self.n_samples,
            step_size=self.step_size,
            n_steps=self.n_steps,
            burn_in=self.burn_in,
            adaptation=self.adaptation,
        )

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The chain's draws, in the order drawn.

        Returns
        -------
        list[torch.Tensor]
            ``n_samples`` points.
        """
        return list(self.chain(objective).theta)


class FromAnnealing(Initializer):
    """The best point a Hamiltonian chain visits while its temperature falls.

    The single-chain control :class:`FromTempering` is measured against:
    both spend gradients to leave a local optimum, and only a comparison at
    equal `force_evaluations` says whether the ladder earns its replicas.

    Parameters
    ----------
    schedule : TempSchedule
        Temperature per proposal; its length is the budget in proposals.
    step_size : float
        Leapfrog step, positive.
    generator : torch.Generator
        The stream, passed in.
    n_steps : int
        Leapfrog steps per proposal.
    """

    def __init__(
        self,
        schedule: TempSchedule,
        step_size: float,
        generator: torch.Generator,
        n_steps: int = hmc.DEFAULT_STEPS,
    ) -> None:
        self.schedule = schedule
        self.step_size = step_size
        self.generator = generator
        self.n_steps = n_steps

    def run(self, objective: Objective) -> hmc.Annealed:
        """The annealing run: its best point, the acceptance rate and its cost.

        Returns
        -------
        snakes_and_ladders.sample.hmc.Annealed
        """
        return hmc.anneal(
            objective,
            self.schedule,
            self.generator,
            step_size=self.step_size,
            n_steps=self.n_steps,
        )

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The lowest-valued point visited.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        return [self.run(objective).theta]


class FromTempering(Initializer):
    """The best point a temperature ladder visits, exchanging positions.

    The hot replicas cross barriers the cold one cannot, so this escapes a
    local optimum by construction where the starts above escape it by a
    spread. It is also the most expensive initializer here: its cost is
    ``n_rounds`` times the ladder's length in proposals, and a caller that
    does not report that against the fit it seeds has hidden the price.

    **The ladder may be calibrated first, and the calibration is a warm-up**
    (issue #902). With a :class:`LadderCalibration`, :meth:`run` measures
    candidate ladders by short runs of
    :func:`~snakes_and_ladders.sample.hmc.parallel_tempering` and revises
    them by :func:`~snakes_and_ladders.sample.schedule.adapt_ladder` or
    :func:`~snakes_and_ladders.sample.schedule.adapt_ladder_by_round_trips`,
    every measurement drawing from ``generator`` in sequence so one seed
    reproduces the warm-up and the run after it. Each measurement starts at
    the lowest-valued point the warm-up has visited, and so does the run;
    the warm-up's positions are discarded and its cost is not
    (:attr:`spent`). ``check_ladder`` guards the settled ladder as it guards
    every ladder.

    **A budget in seconds is a ceiling the rounds are predicted inside.** With
    ``n_rounds`` a :class:`~snakes_and_ladders.opt.budget.Budget` in
    :attr:`~snakes_and_ladders.cost.Cost.SECONDS`, the clock starts at the
    start's first transition --- the warm-up's, where there is one --- and a
    round after the first runs only if the longest round so far would end
    inside the budget. The first round runs regardless, since no round has
    been timed to predict it, and a round slower than every one before it
    can still pass the ceiling on a loaded host; :attr:`TemperingSpend.seconds`
    reports what was spent either way.

    Parameters
    ----------
    temperatures : tuple[float, ...]
        The ladder, coldest first, as
        :func:`snakes_and_ladders.sample.hmc.parallel_tempering` requires;
        with a calibration, the ladder it starts from, whose endpoints it
        keeps.
    n_rounds : int | Budget
        Transitions per replica, at least one; or a budget in
        :attr:`~snakes_and_ladders.cost.Cost.SECONDS` the rounds run inside.
    step_size : float
        Leapfrog step, positive.
    generator : torch.Generator
        The parent stream, passed in.
    n_steps : int
        Leapfrog steps per proposal.
    calibration : LadderCalibration | None
        The ladder's warm-up. ``None`` runs ``temperatures`` as given, every
        start bitwise what it was before the warm-up existed.

    Raises
    ------
    ValueError
        If ``n_rounds`` is a budget in any unit but seconds.
    """

    def __init__(
        self,
        temperatures: tuple[float, ...],
        n_rounds: int | Budget,
        step_size: float,
        generator: torch.Generator,
        n_steps: int = hmc.DEFAULT_STEPS,
        calibration: LadderCalibration | None = None,
    ) -> None:
        if isinstance(n_rounds, Budget) and n_rounds.unit is not Cost.SECONDS:
            msg = (
                f"a tempering start's budget is in {Cost.SECONDS.value!r}, got "
                f"{n_rounds.unit.value!r}: its rounds stop on the wall clock"
            )
            raise ValueError(msg)
        self.temperatures = temperatures
        self.n_rounds = n_rounds
        self.step_size = step_size
        self.generator = generator
        self.n_steps = n_steps
        self.calibration = calibration
        #: What the last :meth:`run` spent and, with a calibration, settled on;
        #: ``None`` before the first.
        self.spent: TemperingSpend | None = None

    def run(self, objective: Objective) -> hmc.Tempered:
        """The tempering run: its best point, both acceptance rates and its cost.

        Records what it spent, the warm-up included, on :attr:`spent`.

        Returns
        -------
        Tempered

        Raises
        ------
        ValueError
            If the calibration alone spent the budget in seconds, leaving no
            round of the run inside it.
        """
        began = time.perf_counter()
        calibrated: CalibratedLadder | None = None
        theta0: torch.Tensor | None = None
        temperatures = self.temperatures
        if self.calibration is not None:
            calibrated, theta0 = calibrate_ladder(
                objective,
                temperatures,
                self.calibration,
                self.generator,
                step_size=self.step_size,
                n_steps=self.n_steps,
            )
            temperatures = calibrated.ladder
        budget: Budget | None = None
        deadline: float | None = None
        rounds = self.n_rounds
        if isinstance(rounds, Budget):
            budget, deadline = rounds, began + rounds.size
            spent = time.perf_counter() - began
            if spent >= rounds.size:
                msg = (
                    f"the ladder's calibration spent {spent:.2f} s of a "
                    f"{rounds.size} s budget, leaving no round of the run inside it"
                )
                raise ValueError(msg)
            # Under a budget the deadline is the one bound on the rounds.
            rounds = sys.maxsize
        tempered = hmc.parallel_tempering(
            objective,
            temperatures,
            self.generator,
            rounds,
            step_size=self.step_size,
            n_steps=self.n_steps,
            theta0=theta0,
            deadline=deadline,
        )
        run_rounds = int(tempered.positions.shape[0])
        self.spent = TemperingSpend(
            calibration=calibrated,
            rounds=run_rounds,
            transitions=run_rounds * len(temperatures),
            force_evaluations=tempered.force_evaluations,
            seconds=time.perf_counter() - began,
            budget=budget,
        )
        return tempered

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The lowest-valued point visited at any temperature.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        return [self.run(objective).theta]


class LadderRule(StrEnum):
    """What a ladder's calibration drives: the exchange rate, or the circulation."""

    ACCEPTANCE = "acceptance"
    """Every neighbouring pair's exchange acceptance into a band, adding rungs:
    :func:`~snakes_and_ladders.sample.schedule.adapt_ladder`."""
    ROUND_TRIPS = "round_trips"
    """The rungs redistributed at fixed length by the walkers' up-fraction:
    :func:`~snakes_and_ladders.sample.schedule.adapt_ladder_by_round_trips`."""


@dataclass(frozen=True)
class LadderCalibration:
    """A warm-up that settles a tempering ladder before the run it serves.

    Each measurement is one
    :func:`~snakes_and_ladders.sample.hmc.parallel_tempering` run of
    ``rounds`` rounds on the candidate ladder, and costs ``rounds`` times the
    candidate's length in transitions. What it reads is the rule's:

    * :attr:`LadderRule.ACCEPTANCE` reads the per-pair exchange acceptance
      and hands it to :func:`~snakes_and_ladders.sample.schedule.adapt_ladder`
      with ``band`` and ``max_replicas``; each acceptance is a fraction of
      ``rounds`` proposals, so ``rounds`` sets what the band can resolve.
    * :attr:`LadderRule.ROUND_TRIPS` reads
      :func:`~snakes_and_ladders.sample.tempered.up_fraction` on the walker
      trace and hands it to
      :func:`~snakes_and_ladders.sample.schedule.adapt_ladder_by_round_trips`
      with ``tolerance``; a rung no labelled walker visited in ``rounds``
      rounds is refused there, and a warm-up too short to resolve the
      up-fraction places rungs where its noise fell (`sample/CLAUDE.md`).

    Parameters
    ----------
    rule : LadderRule
        What is driven; its string value is read into the member.
    rounds : int
        Rounds per measurement, at least 1.
    max_rounds : int
        Measurements before the warm-up stops, at least 1.
    band : tuple[float, float] | None
        ``(low, high)`` inside ``(0, 1)``; the acceptance rule's, and refused
        on the other.
    max_replicas : int | None
        The most rungs the acceptance rule may add up to; refused on the
        other.
    tolerance : float | None
        The largest relative move that stops the round-trip rule, positive;
        refused on the other.

    Raises
    ------
    ValueError
        If the rule is unknown, a count is below 1, or a setting is missing
        from its rule or given to the other.
    """

    rule: LadderRule
    rounds: int
    max_rounds: int
    band: tuple[float, float] | None = None
    max_replicas: int | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule", LadderRule(self.rule))
        if self.rounds < 1 or self.max_rounds < 1:
            msg = (
                f"rounds and max_rounds must be at least 1, got {self.rounds} "
                f"and {self.max_rounds}"
            )
            raise ValueError(msg)
        if self.rule is LadderRule.ACCEPTANCE:
            if self.band is None or self.max_replicas is None:
                msg = "the acceptance rule needs a band and max_replicas"
                raise ValueError(msg)
            if self.tolerance is not None:
                msg = "a tolerance is the round-trip rule's, not the acceptance rule's"
                raise ValueError(msg)
            return
        if self.tolerance is None:
            msg = "the round-trip rule needs a tolerance"
            raise ValueError(msg)
        if self.band is not None or self.max_replicas is not None:
            msg = (
                "a band and max_replicas are the acceptance rule's; the round-trip "
                "rule keeps the ladder's length"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class CalibratedLadder:
    """What a ladder's calibration settled on, and what it cost.

    Parameters
    ----------
    placement : AdaptedLadder | FeedbackLadder
        The adapter's own record: the ladder, the last measurement and
        whether it settled, in the rule's terms.
    rounds_run : int
        Tempering rounds over every measurement, ``placement.rounds``
        measurements of :attr:`LadderCalibration.rounds` each.
    transitions : int
        Hamiltonian transitions over every measurement:
        ``placement.replicas_measured * rounds``.
    force_evaluations : int
        Gradients those transitions spent.
    seconds : float
        Wall clock of the warm-up, on the host that ran it.
    """

    placement: AdaptedLadder | FeedbackLadder
    rounds_run: int
    transitions: int
    force_evaluations: int
    seconds: float

    @property
    def ladder(self) -> tuple[float, ...]:
        """The settled ladder, coldest first, its endpoints the starting ladder's."""
        return self.placement.temperatures

    @property
    def settled(self) -> bool:
        """``within_band`` for the acceptance rule, ``converged`` for the round-trip rule."""
        if isinstance(self.placement, AdaptedLadder):
            return self.placement.within_band
        return self.placement.converged


@dataclass(frozen=True)
class TemperingSpend:
    """What one tempering start spent, the ladder's warm-up included.

    Parameters
    ----------
    calibration : CalibratedLadder | None
        The warm-up, where there was one.
    rounds : int
        Rounds the run made on the settled ladder; under a budget, what the
        budget admitted.
    transitions : int
        The run's transitions, ``rounds`` times the settled ladder's length;
        the warm-up's are :attr:`CalibratedLadder.transitions`.
    force_evaluations : int
        The run's gradients.
    seconds : float
        Wall clock from the start's first transition to the run's end, the
        warm-up included.
    budget : Budget | None
        The ceiling in seconds the rounds ran inside, where one was given.
    """

    calibration: CalibratedLadder | None
    rounds: int
    transitions: int
    force_evaluations: int
    seconds: float
    budget: Budget | None

    @property
    def total_transitions(self) -> int:
        """The run's transitions and the warm-up's, which a comparison at equal cost charges."""
        warm_up = 0 if self.calibration is None else self.calibration.transitions
        return self.transitions + warm_up


def calibrate_ladder(
    objective: Objective,
    temperatures: TempSchedule | tuple[float, ...],
    calibration: LadderCalibration,
    generator: torch.Generator,
    *,
    step_size: float,
    n_steps: int = hmc.DEFAULT_STEPS,
) -> tuple[CalibratedLadder, torch.Tensor]:
    """A ladder for :func:`~snakes_and_ladders.sample.hmc.parallel_tempering`, from its own measurements.

    The Hamiltonian sibling of
    :func:`~snakes_and_ladders.sample.potts_mcmc.adapt_ladder_potts` and
    :func:`~snakes_and_ladders.sample.tempered.adapt_ladder_round_trips`:
    each measurement is a run of ``calibration.rounds`` rounds on the
    candidate, drawn from ``generator`` in sequence, started at the
    lowest-valued point any earlier measurement visited (the objective's
    initial point for the first).

    Returns
    -------
    tuple[CalibratedLadder, torch.Tensor]
        The settled ladder and its cost, and the lowest-valued point the
        warm-up visited --- where the run it serves starts.

    Raises
    ------
    ValueError
        As the adapter refuses, or if the settled ladder is not positive and
        strictly increasing.
    """
    began = time.perf_counter()
    best: torch.Tensor | None = None
    best_value = float("inf")
    transitions = 0
    force_evaluations = 0

    def measured(candidate: tuple[float, ...]) -> hmc.Tempered:
        nonlocal best, best_value, transitions, force_evaluations
        tempered = hmc.parallel_tempering(
            objective,
            candidate,
            generator,
            calibration.rounds,
            step_size=step_size,
            n_steps=n_steps,
            theta0=best,
        )
        transitions += calibration.rounds * len(candidate)
        force_evaluations += tempered.force_evaluations
        if tempered.value < best_value:
            best, best_value = tempered.theta, tempered.value
        return tempered

    def acceptance(candidate: tuple[float, ...]) -> list[float]:
        return [float(value) for value in measured(candidate).swap_acceptance]

    def circulation(candidate: tuple[float, ...]) -> list[float]:
        return [float(value) for value in up_fraction(measured(candidate).walkers)]

    placement: AdaptedLadder | FeedbackLadder
    if calibration.rule is LadderRule.ACCEPTANCE:
        # `LadderCalibration` has refused an acceptance rule without these.
        assert calibration.band is not None
        assert calibration.max_replicas is not None
        placement = adapt_ladder(
            acceptance,
            temperatures,
            calibration.band,
            calibration.max_rounds,
            calibration.max_replicas,
        )
    else:
        assert calibration.tolerance is not None
        placement = adapt_ladder_by_round_trips(
            circulation, temperatures, calibration.tolerance, calibration.max_rounds
        )
    check_ladder(
        placement.temperatures,
        needed_by="a calibrated tempering start",
        monotone=Monotone.INCREASING,
    )
    # Every measurement visits at least its start, so `best` is bound.
    assert best is not None
    return (
        CalibratedLadder(
            placement=placement,
            rounds_run=placement.rounds * calibration.rounds,
            transitions=transitions,
            force_evaluations=force_evaluations,
            seconds=time.perf_counter() - began,
        ),
        best,
    )
