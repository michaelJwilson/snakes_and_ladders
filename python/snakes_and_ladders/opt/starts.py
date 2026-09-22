"""Where a fit starts against what it reaches, measured through one loop (issue #894).

Four consumers wrote this loop by hand: `infra/seeding_sweep.py` (experiment
009), `tests/regression/opt/test_opt_mixture_seeding.py` (experiment 010),
`qa/mixture_seeding.py` and `docs/nb/spatio_sequential_starts.ipynb`. Each
ran every start over its seeds through :func:`~snakes_and_ladders.opt.budget.compare`,
polished what the start produced at one budget, and kept the columns the
comparison drops --- a seeding's cost, a recovery, a curve --- in a ledger of
its own beside it. :class:`StartsBenchmark` is the loop and
:class:`SolverComparison` the result.

**A start is an** :class:`~snakes_and_ladders.opt.initialize.Initializer`, **or
a factory of one from the cell's generator.** An initializer that draws holds
its generator (`KMeansPlusPlus(n, rng)`), and one instance pickled into every
cell of a process pool draws the same starts in each. A factory ---
``functools.partial(RandomRestart, 4, 0.5)`` --- is called with the generator
:func:`~snakes_and_ladders.opt.budget.compare` gives the cell, so each
``(instance, seed)`` draws its own stream and the run is the same at every
worker count. An initializer passed as itself draws from its own generator in
the order the cells run, which only ``workers=1`` fixes.

**Several starts share the polish budget.** An initializer offering ``n``
starts has each polished at ``size // n`` and the best kept: the arithmetic of
:func:`~snakes_and_ladders.opt.budget.restarts`, so a restart baseline is a
start like any other rather than a second loop. The seeding budget, in
:attr:`~snakes_and_ladders.cost.Cost.EVALUATIONS`, bounds the starts the seam
scores, one objective evaluation each; what an initializer spends inside
:meth:`~snakes_and_ladders.opt.initialize.Initializer.starts` is read in
seconds, since a chain and a quantile share no other unit.

**Every cell runs under** :func:`snakes_and_ladders.track.track`. The seam
records each start's value as it scores it and the polisher records its own
iterations, so a cell's ``objective`` series is its curve, the store's stamps
(:meth:`snakes_and_ladders.track.MemoryRun.stamps`) its seconds, and the entry
at which the seam handed over its hand-over index. What the initializer
records runs in a block of its own: a chain on a surrogate records a value of
another objective, which is no point of this curve. Seconds belong to the
host (:attr:`~snakes_and_ladders.cost.Cost.SECONDS`), so
:meth:`SolverComparison.table` holds none and :meth:`SolverComparison.readings`
holds them.

The renderers are in `snakes_and_ladders.qa.starts`: `qa/` imports `opt/`, and
the converse would be a second import cycle
(`tests/regression/test_directory_imports.py`). Model-agnostic, per
``opt/CLAUDE.md``: nothing here knows what an objective is, and the three
adapters name `opt`'s own optimizers.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any, NoReturn

import numpy as np
import torch

from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget, Comparison, Outcome, compare
from snakes_and_ladders.opt.emission_mixture import expectation_maximization
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.hmm import baum_welch_family
from snakes_and_ladders.opt.initialize import Initializer
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.termination import Termination
from snakes_and_ladders.track import MemoryRun, track

#: A start: an initializer, or a factory of one from the cell's generator.
type Start = Initializer | Callable[[np.random.Generator], Initializer]


@dataclass(frozen=True)
class Polished:
    """What a polisher reached from one start.

    Parameters
    ----------
    theta : torch.Tensor
        The polished point, in the objective's unconstrained coordinates.
    value : float
        The objective there, lower being better, as the optimizer reports it.
    termination : Termination
        How the optimizer's loop ended; its ``iterations`` are what the
        polish spent against the budget, one unit an iteration.
    """

    theta: torch.Tensor
    value: float
    termination: Termination


#: An optimizer run from one start at one budget. A type alias and not a
#: ``Protocol``: `opt.fit.fit` takes its start and its cap positionally beside
#: keywords a benchmark holds fixed, so no optimizer here matches a shared
#: signature unadapted, and an adapter of three positional arguments is a
#: plain callable. A function, a ``functools.partial`` of one or a dataclass
#: with ``__call__`` satisfies it, and each pickles for a process pool.
type Polisher = Callable[[Objective, torch.Tensor, Budget], Polished]


def refuse_start(name: str, objective: Objective, reason: str) -> NoReturn:
    """Refuse a start in one sentence, so a consumer prints it rather than dropping a row.

    Raises
    ------
    ValueError
        Always.
    """
    msg = f"the start {name!r} is refused on {type(objective).__name__}: {reason}"
    raise ValueError(msg)


def _ended(termination: Termination | None, iterations: int) -> Termination:
    """``termination``, or the budget branch where a producer left it unknown."""
    return (
        termination
        if termination is not None
        else Termination.after(iterations, converged=False)
    )


def polish_by_fit(
    objective: Objective, theta: torch.Tensor, budget: Budget
) -> Polished:
    """:func:`snakes_and_ladders.opt.fit.fit` from ``theta``, one L-BFGS iteration a unit.

    Returns
    -------
    Polished
    """
    result = fit(objective, theta, max_iterations=budget.size)
    return Polished(
        result.theta, result.value, _ended(result.termination, result.iterations)
    )


def _mixture_at(objective: Objective, theta: torch.Tensor) -> tuple[Any, Any, Any]:
    """The observations, the weights and the family ``theta`` encodes, or a refusal.

    Read by name --- ``observations``, ``components(theta)`` and the
    ``log_weight`` key of :meth:`~snakes_and_ladders.opt.objective.Objective.constrain`
    --- which :class:`~snakes_and_ladders.opt.mixture.GaussianMixtureObjective`
    carries, rather than through a protocol two implementers would not earn.
    """
    components = getattr(objective, "components", None)
    observations = getattr(objective, "observations", None)
    if not callable(components) or observations is None:
        refuse_start(
            "expectation-maximization",
            objective,
            "it carries no observations and no components(theta) to start a mixture from",
        )
    with torch.no_grad():
        weights = torch.exp(objective.constrain(theta)["log_weight"]).detach()
        family = components(theta.detach())
    return observations, weights, family


def polish_by_emission_em(
    objective: Objective, theta: torch.Tensor, budget: Budget
) -> Polished:
    """:func:`snakes_and_ladders.opt.emission_mixture.expectation_maximization` from ``theta``.

    One EM iteration a unit, at the entry point's own tolerance. The value is
    the log-likelihood EM reports, at the state its last M step was handed,
    and the point is where that step left it: the pair
    `test_opt_mixture_seeding` scored before the seam existed.

    Returns
    -------
    Polished
    """
    observations, weights, family = _mixture_at(objective, theta)
    fitted = expectation_maximization(
        observations, weights, family, max_iterations=budget.size
    )
    polished = objective.theta_from(
        {
            "log_weight": torch.log(fitted.weights),
            **fitted.components.named_parameters(),
        }
    )
    return Polished(
        polished,
        -fitted.log_likelihood,
        _ended(fitted.termination, fitted.iterations),
    )


def polish_by_baum_welch(
    objective: Objective, theta: torch.Tensor, budget: Budget
) -> Polished:
    """Baum-Welch from ``theta`` on an HMM objective, one EM iteration a unit.

    Through :func:`~snakes_and_ladders.opt.hmm.baum_welch_family`, which
    :func:`~snakes_and_ladders.opt.hmm.baum_welch` narrows to a categorical
    family and which alone reports its outer loop's termination (issue #865).

    Returns
    -------
    Polished
    """
    emissions = getattr(objective, "emissions", None)
    observations = getattr(objective, "observations", None)
    if not callable(emissions) or observations is None:
        refuse_start(
            "baum-welch",
            objective,
            "it is not an HMM objective carrying observations and emissions(theta)",
        )
    if getattr(objective, "covariate", None) is not None:
        refuse_start("baum-welch", objective, "a covariate is not passed through")
    with torch.no_grad():
        named = objective.constrain(theta.detach())
        family = emissions(theta.detach())
    fitted = baum_welch_family(
        observations.numpy(),
        named["log_initial"],
        named["log_transition"],
        family,
        max_iterations=budget.size,
    )
    polished = objective.theta_from(
        {
            "log_initial": fitted.log_initial,
            "log_transition": fitted.log_transition,
            **fitted.emissions.named_parameters(),
        }
    )
    return Polished(
        polished,
        -fitted.log_likelihood,
        _ended(fitted.termination, budget.size),
    )


@dataclass(frozen=True)
class StartTrial:
    """One ``(start, instance, seed)`` cell, as it rides back on the outcome's detail.

    Parameters
    ----------
    seeded_value : float
        The lowest value among the points the start offered.
    value : float
        The best polished value, the one the comparison scores.
    spent : int
        What the polishes spent together, in the polish budget's unit.
    termination : Termination
        The best polish's.
    theta : torch.Tensor
        The best polished point.
    values : tuple[float, ...]
        The cell's ``objective`` series: one entry per offered start, then
        the polisher's own, then the reported value.
    seconds : tuple[float, ...]
        Seconds since the cell opened, one per entry of ``values``.
    handover : int
        The index in ``values`` of the last seeding entry.
    state_bytes : int
        What the polisher recorded its state to hold, or the polished point's
        bytes where it recorded none.
    diagnostics : Mapping[str, float]
        The last value of every series the initializer recorded in its own
        block: a seeding's charge, a chain's acceptance.
    """

    seeded_value: float
    value: float
    spent: int
    termination: Termination
    theta: torch.Tensor
    values: tuple[float, ...]
    seconds: tuple[float, ...]
    handover: int
    state_bytes: int
    diagnostics: Mapping[str, float]


def _memory(run: object) -> MemoryRun:
    """The run :func:`track` was handed, as the store it is."""
    if not isinstance(run, MemoryRun):  # pragma: no cover - bound by the caller
        msg = "a cell records into the MemoryRun it opened"
        raise TypeError(msg)
    return run


@dataclass(frozen=True)
class _Cell:
    """One start as an :mod:`snakes_and_ladders.opt.budget` method; a dataclass so it pickles."""

    name: str
    start: Start
    polish: Polisher
    seeding_budget: Budget

    def __call__(
        self, objective: Objective, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        """Seed, score each start, polish each at its share, and keep the best.

        Returns
        -------
        Outcome
            The best polished value, the polishes' summed spend, and the
            :class:`StartTrial`.
        """
        with track(MemoryRun()) as cell:
            with track(MemoryRun()) as seeding:
                initializer = (
                    self.start
                    if isinstance(self.start, Initializer)
                    and not isinstance(self.start, type)
                    else self.start(rng)
                )
                thetas = initializer.starts(objective)
            if not thetas:
                refuse_start(self.name, objective, "it offered no starting point")
            if len(thetas) > self.seeding_budget.size:
                refuse_start(
                    self.name,
                    objective,
                    f"it offered {len(thetas)} starts against a seeding budget of "
                    f"{self.seeding_budget.size} {self.seeding_budget.unit}",
                )
            share = budget.size // len(thetas)
            if share < 1:
                refuse_start(
                    self.name,
                    objective,
                    f"{len(thetas)} starts share {budget.size} {budget.unit}, "
                    "under one each",
                )
            seeded: list[float] = []
            for theta in thetas:
                with torch.no_grad():
                    seeded.append(float(objective(theta.detach())))
                cell.record(len(seeded) - 1, objective=seeded[-1])
            run = _memory(cell.run)
            handover = len(run.series("objective")) - 1
            polished = [
                self.polish(objective, theta.detach(), Budget(budget.unit, share))
                for theta in thetas
            ]
            best = min(polished, key=lambda one: one.value)
            # The closing entry is the value reported, so the curve ends where
            # the comparison scores it: a restart's last polish need not be
            # its best, and an EM loop records nothing of its own.
            cell.record(len(run.series("objective")), objective=best.value)
        stamps = run.stamps("objective")
        memory = _memory(seeding.run)
        recorded = run.names()
        trial = StartTrial(
            seeded_value=min(seeded),
            value=best.value,
            spent=sum(one.termination.iterations for one in polished),
            termination=best.termination,
            theta=best.theta,
            values=tuple(float(value) for _, value in run.series("objective")),
            seconds=tuple(stamp - cell.started for stamp in stamps),
            handover=handover,
            state_bytes=int(run.last("state_bytes"))
            if "state_bytes" in recorded
            else int(best.theta.nbytes),
            diagnostics={name: float(memory.last(name)) for name in memory.names()},
        )
        return Outcome(best.value, trial.spent, trial)


@dataclass(frozen=True)
class StartsRow:
    """One start's host-stable row of :meth:`SolverComparison.table`.

    Parameters
    ----------
    start : str
        The start's name.
    trials : int
        Cells run: instances times seeds.
    seeded_value : float
        Mean over the trials of the best offered start's value.
    reached_value : float
        Mean over the instances of the best polished value over the seeds.
    gap : float
        Mean over the instances of ``reached - reference``:
        :meth:`~snakes_and_ladders.opt.budget.Comparison.mean_gap`.
    spent : int
        The largest polish spend of any trial, in the polish budget's unit.
    state_bytes : int
        The largest state any trial's polisher recorded.
    """

    start: str
    trials: int
    seeded_value: float
    reached_value: float
    gap: float
    spent: int
    state_bytes: int


@dataclass(frozen=True)
class StartsReading:
    """One start's host-dependent reading of :meth:`SolverComparison.readings`.

    Parameters
    ----------
    start : str
        The start's name.
    seeding_seconds : float
        Mean over the trials of the seconds to the hand-over.
    polish_seconds : float
        Mean over the trials of the seconds from the hand-over to the end.
    recovery : float | None
        Mean over the trials of the caller's recovery at the polished point,
        or ``None`` where no callable was given.
    error : float | None
        Mean over the trials of the caller's error at the polished point, or
        ``None``.
    """

    start: str
    seeding_seconds: float
    polish_seconds: float
    recovery: float | None
    error: float | None


@dataclass(frozen=True)
class Curve:
    """One trial's curve: the gap below the reference against seconds.

    Parameters
    ----------
    instance : int
        The objective's index in the benchmark's sequence.
    seed : int
        The seed the cell ran at.
    seconds : np.ndarray
        Seconds since the cell opened, one per tracked entry.
    values : np.ndarray
        The objective at each entry.
    gaps : np.ndarray
        ``values - reference[instance]``, in the objective's units.
    handover : int
        The index of the last seeding entry.
    """

    instance: int
    seed: int
    seconds: np.ndarray
    values: np.ndarray
    gaps: np.ndarray
    handover: int


@dataclass(frozen=True)
class StartDescription:
    """What a start is, read off the object the caller passed.

    Parameters
    ----------
    start : str
        The start's name.
    summary : str
        The first paragraph of the object's docstring, one line.
    hyperparameters : tuple[tuple[str, str], ...]
        ``(field, value)`` in the object's own field order.
    """

    start: str
    summary: str
    hyperparameters: tuple[tuple[str, str], ...]


def _shown(value: object) -> str:
    """A hyperparameter as a table cell: a number or a name as written, anything else by its type."""
    if isinstance(value, bool | int | float | str):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return repr(value)
    return type(value).__name__


def _described(name: str, start: Start) -> StartDescription:
    """A start's docstring summary and its hyperparameters, in field order."""
    target: object = start
    pairs: list[tuple[str, object]] = []
    if isinstance(start, functools.partial):
        target = start.func
        bound = inspect.signature(start.func).bind_partial(
            *start.args, **start.keywords
        )
        pairs = list(bound.arguments.items())
    elif dataclasses.is_dataclass(start) and not isinstance(start, type):
        pairs = [
            (field.name, getattr(start, field.name))
            for field in dataclasses.fields(start)
        ]
    else:
        pairs = [
            (key, value)
            for key, value in vars(start).items()
            if not key.startswith("_")
        ]
    document = inspect.getdoc(target) or ""
    summary = " ".join(document.split("\n\n", 1)[0].split())
    return StartDescription(
        name, summary, tuple((key, _shown(value)) for key, value in pairs)
    )


@dataclass(frozen=True)
class SolverComparison:
    """Every start on every instance over the seeds, polished at one budget.

    Parameters
    ----------
    comparison : Comparison
        The underlying :class:`~snakes_and_ladders.opt.budget.Comparison`,
        one method per start in the caller's order; each outcome's detail is
        its :class:`StartTrial`.
    starts : Mapping[str, Start]
        The objects the caller passed, which :meth:`describe` reads.
    objectives : tuple[Objective, ...]
        The instances, in order.
    seeds : tuple[int, ...]
        The seeds, in order.
    seeding_budget : Budget
        The starts' ceiling.
    """

    comparison: Comparison
    starts: Mapping[str, Start]
    objectives: tuple[Objective, ...]
    seeds: tuple[int, ...]
    seeding_budget: Budget

    @property
    def names(self) -> tuple[str, ...]:
        """The starts, in the caller's order."""
        return self.comparison.methods

    def trials(self, start: str) -> tuple[StartTrial, ...]:
        """One start's cells, in instance then seed order.

        Raises
        ------
        KeyError
            If ``start`` is not one of :attr:`names`.
        """
        if start not in self.names:
            msg = f"{start!r} is not one of the starts {self.names}"
            raise KeyError(msg)
        per = len(self.objectives) * len(self.seeds)
        row = self.names.index(start)
        cells = self.comparison.outcomes[row * per : (row + 1) * per]
        return tuple(_trial(outcome) for outcome in cells)

    def table(self) -> tuple[StartsRow, ...]:
        """The host-stable rows: every number here reproduces bitwise on a rerun."""
        gaps = self.comparison.mean_gap()
        rows = []
        for row, name in enumerate(self.names):
            trials = self.trials(name)
            rows.append(
                StartsRow(
                    start=name,
                    trials=len(trials),
                    seeded_value=float(np.mean([one.seeded_value for one in trials])),
                    reached_value=float(np.mean(self.comparison.best[row])),
                    gap=gaps[name],
                    spent=int(self.comparison.spent[row].max()),
                    state_bytes=max(one.state_bytes for one in trials),
                )
            )
        return tuple(rows)

    def readings(
        self,
        *,
        recovery: Callable[[Objective, torch.Tensor], float] | None = None,
        error: Callable[[Objective, torch.Tensor], float] | None = None,
    ) -> tuple[StartsReading, ...]:
        """The host-dependent readings: seconds per phase, and the caller's truth.

        Parameters
        ----------
        recovery, error : Callable[[Objective, torch.Tensor], float] | None
            Scored at each trial's polished point against its own objective;
            the truth is the caller's, since the seam holds none.
        """
        readings = []
        for name in self.names:
            trials = self.trials(name)
            objectives = [
                self.objectives[index // len(self.seeds)]
                for index in range(len(trials))
            ]
            readings.append(
                StartsReading(
                    start=name,
                    seeding_seconds=float(
                        np.mean([one.seconds[one.handover] for one in trials])
                    ),
                    polish_seconds=float(
                        np.mean(
                            [
                                one.seconds[-1] - one.seconds[one.handover]
                                for one in trials
                            ]
                        )
                    ),
                    recovery=None
                    if recovery is None
                    else float(
                        np.mean(
                            [
                                recovery(objective, one.theta)
                                for objective, one in zip(
                                    objectives, trials, strict=True
                                )
                            ]
                        )
                    ),
                    error=None
                    if error is None
                    else float(
                        np.mean(
                            [
                                error(objective, one.theta)
                                for objective, one in zip(
                                    objectives, trials, strict=True
                                )
                            ]
                        )
                    ),
                )
            )
        return tuple(readings)

    def curves(self) -> dict[str, tuple[Curve, ...]]:
        """Per start, every trial's gap against seconds, with its hand-over index."""
        out: dict[str, tuple[Curve, ...]] = {}
        for name in self.names:
            curves = []
            for index, one in enumerate(self.trials(name)):
                instance, seed = divmod(index, len(self.seeds))
                values = np.asarray(one.values, dtype=np.float64)
                curves.append(
                    Curve(
                        instance=instance,
                        seed=self.seeds[seed],
                        seconds=np.asarray(one.seconds, dtype=np.float64),
                        values=values,
                        gaps=values - float(self.comparison.reference[instance]),
                        handover=one.handover,
                    )
                )
            out[name] = tuple(curves)
        return out

    def describe(self) -> tuple[StartDescription, ...]:
        """Per start, its docstring's first paragraph and its hyperparameters, in the caller's order."""
        return tuple(_described(name, self.starts[name]) for name in self.names)


def _trial(outcome: Outcome) -> StartTrial:
    """The :class:`StartTrial` a cell carried back."""
    if not isinstance(outcome.detail, StartTrial):  # pragma: no cover - set by _Cell
        msg = "a cell of the seam carries its StartTrial on the outcome's detail"
        raise TypeError(msg)
    return outcome.detail


@dataclass(frozen=True)
class StartsBenchmark:
    """Every start through one seeding-then-polish loop, on every instance over the seeds.

    Under the seam rule: four consuming modules wrote this loop by hand ---
    `infra/seeding_sweep.py`, `tests/regression/opt/test_opt_mixture_seeding.py`,
    `snakes_and_ladders.qa.mixture_seeding` and the starts notebook
    `docs/nb/spatio_sequential_starts.ipynb`, which converts once issue
    #891's rewrite of it lands --- each with its own ledger beside
    :class:`~snakes_and_ladders.opt.budget.Comparison` (issue #894).

    Parameters
    ----------
    objective : Objective | Sequence[Objective]
        One instance, or several: a sequence runs every start on each, paired
        across starts as :func:`~snakes_and_ladders.opt.budget.compare` pairs
        instances.
    starts : Mapping[str, Start]
        Name to start, in the order every result reports them.
    polish : Polisher
        The optimizer every start hands over to.
    seeding_budget : Budget
        Starts the seam scores per cell, in
        :attr:`~snakes_and_ladders.cost.Cost.EVALUATIONS`.
    polish_budget : Budget
        Held equal across starts, and shared among a start's points.
    seeds : Sequence[int]
        At least one; cell ``(instance, seed)`` draws from
        ``np.random.default_rng([seed, instance])``.
    workers : int
        Cells at once, each pooled worker at one torch thread
        (:func:`~snakes_and_ladders.opt.budget.compare`).
    reference : Sequence[float] | None
        A known value per instance the gaps are read against; ``None`` reads
        them against the best any start reached.

    Raises
    ------
    ValueError
        If no start is given, or the seeding budget is not in evaluations.
    """

    objective: Objective | Sequence[Objective]
    starts: Mapping[str, Start]
    polish: Polisher
    _: KW_ONLY
    seeding_budget: Budget
    polish_budget: Budget
    seeds: Sequence[int]
    workers: int
    reference: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if not self.starts:
            msg = "a benchmark of starts needs at least one start"
            raise ValueError(msg)
        if self.seeding_budget.unit is not Cost.EVALUATIONS:
            msg = (
                f"the seeding budget counts the starts the seam scores, in "
                f"{Cost.EVALUATIONS}, not {self.seeding_budget.unit}"
            )
            raise ValueError(msg)

    @property
    def objectives(self) -> tuple[Objective, ...]:
        """The instances, a single objective being a sequence of one."""
        if isinstance(self.objective, Objective):
            return (self.objective,)
        return tuple(self.objective)

    def run(self) -> SolverComparison:
        """Run every ``(start, instance, seed)`` cell and collect the result.

        Returns
        -------
        SolverComparison

        Raises
        ------
        ValueError
            A start's refusal (:func:`refuse_start`), with the cell named.
        OverspendError
            If a polisher reports more than its share.
        """
        methods = {
            name: _Cell(name, start, self.polish, self.seeding_budget)
            for name, start in self.starts.items()
        }
        comparison = compare(
            methods,
            self.objectives,
            self.polish_budget,
            self.seeds,
            workers=self.workers,
            known=self.reference,
        )
        return SolverComparison(
            comparison=comparison,
            starts=dict(self.starts),
            objectives=self.objectives,
            seeds=tuple(self.seeds),
            seeding_budget=self.seeding_budget,
        )


__all__ = [
    "Curve",
    "Polished",
    "Polisher",
    "SolverComparison",
    "Start",
    "StartDescription",
    "StartTrial",
    "StartsBenchmark",
    "StartsReading",
    "StartsRow",
    "polish_by_baum_welch",
    "polish_by_emission_em",
    "polish_by_fit",
    "refuse_start",
]
