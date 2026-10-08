"""One long anneal, k restarts, or tempering at one site-visit budget, and the move mix (issue #1390, question 5).

**Budget split.** On `potts_reference`'s ``stress`` cell and its
``margin_0.1`` and ``q8`` variants, at a total of :data:`LEVELS` times
alpha-expansion+ICM's site visits on the cell (``B``), three arms:

* ``long`` --- one :func:`~sal.sample.potts_mcmc.anneal_potts` run of ``B``;
* ``restart-k`` --- ``k`` runs of ``B / k`` each, :data:`RESTARTS`, the
  lowest polished energy kept;
* ``tempering`` --- :func:`~sal.sample.potts_mcmc.cluster_tempering` on the
  ladder ``temperatures="auto"`` chooses once per cell from
  ``(T0 / END_RATIO, T0)`` on :data:`TUNING_SEED`; the pilot's visits are
  reported apart, since one measurement already exceeds ``B`` at x1 and x4.

Every anneal is single-site heat bath on the exponential schedule from
``T0 = J d / 3`` (experiment 032's constant) to ``T0 / END_RATIO``, with no
per-arm tuning, and ends with :attr:`~sal.sample.schedule.Polish.ICM_MERGE`;
the tempering's ``best`` is polished by the same pair (:func:`polish`).
Each arm is run twice per seed: with the polish **outside** ``B`` (the
schedule spends ``B``, the polish on top), and with it **counted** (the
schedule spends ``B`` less the polish visits the first run spent, same seed).
The referee is the cell's TRW-S bound; a gap is energy less the bound.

**Move mix.** On ``stress`` at x16 and at 4,000 sweeps' visits, the anneal
under :data:`MOVES` on the same schedule, unpolished and polished, on the
generators of experiment 029's workload arms (``[seed, 3]``), so the
4,000-sweep rows reproduce its 917.5 and 0.52 nats.

Run as ``python -m sal.qa.budget_split``; it prints the tables
``docs/experiments/035-one-long-anneal-restarts-or-tempering-at-one-budget.md``
reports.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sal.cost import Cost
from sal.opt.budget import Budget
from sal.qa.known_ground_states import expansion_icm
from sal.sample.potts_mcmc import (
    PottsMove,
    PottsMoves,
    Recolour,
    anneal_potts,
    cluster_tempering,
)
from sal.sample.potts_mcmc.chains import merge_labels
from sal.sample.schedule import Polish, ScheduleParams, ScheduleShape
from sal.sample.tune import LadderTuning
from sal.search.ground_state import Problem
from sal.search.icm import iterated_conditional_modes
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.potts import energy
from sal.sim.potts_cell import PottsReferenceParams

#: The cells: the reference and the two variants that move the gap most (032).
VARIANTS = ("stress", "margin_0.1", "q8")
#: Total budget, as multiples of alpha-expansion+ICM's site visits on the cell.
LEVELS = (1, 4, 16)
#: Restart counts at a fixed total.
RESTARTS = (4, 16)
#: ``T0 / (J d)``: experiment 032's stress-tuned constant.
START_PER_COUPLING = 1.0 / 3.0
#: ``T0 / t_end``, experiment 032's.
END_RATIO = 40.0
#: Seeds the reported runs draw from.
SEEDS = tuple(range(100, 116))
#: The seed the ladder pilot draws from, held out of every reported run.
TUNING_SEED = 0
#: The ladder pilot: at most 8 rungs, 20 sweeps per measurement, 4 measurements.
LADDER_RUNGS, LADDER_SWEEPS, LADDER_MEASUREMENTS = 8, 20, 4
#: The move mix's sets: a sequence runs as given, a bare cluster move with Gibbs (#1323).
MOVES: dict[str, PottsMoves] = {
    "wolff-hb alone": (PottsMove.WOLFF_HEAT_BATH,),
    "wolff + gibbs": PottsMove.WOLFF,
    "sw-hb alone": (PottsMove.SWENDSEN_WANG_HEAT_BATH,),
    "sw + gibbs": PottsMove.SWENDSEN_WANG,
    "glauber": (PottsMove.SINGLE_SITE,),
}
#: The move mix's second budget, in heat-bath sweeps: the downstream record's.
MOVE_SWEEPS = 4000
#: The move mix's seeds: experiment 029's workload seeds.
MOVE_SEEDS = tuple(range(8))


@dataclass(frozen=True)
class Cell:
    """One instance as the study runs it: its referee, its base budget and its schedule."""

    name: str
    problem: Problem
    bound: float
    base: int
    base_gap: float
    t0: float

    @property
    def per_sweep(self) -> int:
        """Site visits of one heat-bath sweep."""
        return self.problem.visits_per_sweep

    def schedule(self) -> ScheduleParams:
        """The exponential schedule from ``T0`` to ``T0 / END_RATIO``."""
        return ScheduleParams(ScheduleShape.EXPONENTIAL, self.t0, self.t0 / END_RATIO)


def cell(params: PottsReferenceParams, name: str) -> Cell:
    """``params`` drawn, with its TRW-S bound, alpha-expansion+ICM's spend and ``T0 = J d / 3``."""
    drawn = params.instance()
    problem = Problem(drawn.graph, drawn.field, drawn.n_states)
    reference = expansion_icm(problem)
    bound = float(trws(drawn.graph, drawn.field).bound)
    degree = 2.0 * len(drawn.graph.edges) / problem.n_nodes
    coupling = float(np.mean(np.asarray(drawn.graph.edge_coupling, dtype=float)))
    return Cell(
        name=name,
        problem=problem,
        bound=bound,
        base=reference.spent,
        base_gap=reference.energy - bound,
        t0=START_PER_COUPLING * coupling * degree,
    )


def polish(each: Cell, labelling: np.ndarray) -> tuple[float, int]:
    """``Polish.ICM_MERGE`` on ``labelling``: its energy and site visits, one sweep per ICM sweep and per merge round."""
    problem = each.problem
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        np.random.default_rng(0),
        start=labelling,
        n_states=problem.n_states,
    )
    merged, rounds = merge_labels(
        problem.graph, np.asarray(problem.field, dtype=float), settled.labelling
    )
    visits = (settled.sweeps + rounds) * each.per_sweep
    return energy(problem.graph, problem.field, merged), visits


@dataclass(frozen=True)
class Outcome:
    """One arm on one seed: gaps before and after the polish, and the visits each stage spent."""

    raw_gap: float
    gap: float
    spent: int
    polish_spent: int


def _anneal(each: Cell, budget: int, rng: np.random.Generator) -> Outcome:
    """One polished anneal of ``budget`` schedule visits."""
    budget = max(1, budget)
    run = anneal_potts(
        each.problem.graph,
        each.problem.field,
        each.schedule().build(max(2, budget // each.per_sweep)),
        rng,
        budget=Budget(Cost.SITE_VISITS, budget),
        polish=Polish.ICM_MERGE,
    )
    return Outcome(
        raw_gap=run.stages[0].energy - each.bound,
        gap=run.energy - each.bound,
        spent=int(run.spent),
        polish_spent=int(run.polish_spent),
    )


def restarts(
    each: Cell, budget: int, n_restarts: int, seed: int, counted: bool
) -> Outcome:
    """``n_restarts`` anneals of ``budget / n_restarts`` each, the best polished kept; ``counted`` takes the polish out of ``budget``."""
    runs = [
        _anneal(
            each, budget // n_restarts, np.random.default_rng([seed, n_restarts, r])
        )
        for r in range(n_restarts)
    ]
    if counted:
        left = budget - sum(run.polish_spent for run in runs)
        runs = [
            _anneal(
                each, left // n_restarts, np.random.default_rng([seed, n_restarts, r])
            )
            for r in range(n_restarts)
        ]
    best = min(runs, key=lambda run: run.gap)
    return Outcome(
        raw_gap=min(run.raw_gap for run in runs),
        gap=best.gap,
        spent=sum(run.spent for run in runs),
        polish_spent=sum(run.polish_spent for run in runs),
    )


@dataclass(frozen=True)
class Ladder:
    """The ladder ``"auto"`` chose on a cell, and the pilot's site visits."""

    temperatures: tuple[float, ...]
    spent: int
    within_band: bool


def ladder(each: Cell) -> Ladder:
    """``cluster_tempering``'s ``"auto"`` ladder from ``(T0 / END_RATIO, T0)``, on :data:`TUNING_SEED`."""
    tuning = LadderTuning(
        Budget(Cost.SWEEPS, LADDER_RUNGS * LADDER_SWEEPS * LADDER_MEASUREMENTS),
        LADDER_RUNGS,
        (each.t0 / END_RATIO, each.t0),
        n_sweeps=LADDER_SWEEPS,
    )
    run = cluster_tempering(
        each.problem.graph,
        each.problem.field,
        "auto",
        np.random.default_rng(TUNING_SEED),
        1,
        ladder_tuning=tuning,
    )
    assert run.tuned_ladder is not None
    return Ladder(
        run.temperatures, run.tuned_ladder.spent, run.tuned_ladder.within_band
    )


def tempering(
    each: Cell, rungs: Ladder, budget: int, seed: int, counted: bool
) -> Outcome:
    """Cluster tempering on ``rungs`` for ``budget`` visits, its ``best`` polished; ``counted`` as :func:`restarts`."""
    per_step = each.per_sweep * (len(rungs.temperatures) + 1)

    def run(visits: int) -> Outcome:
        tempered = cluster_tempering(
            each.problem.graph,
            each.problem.field,
            rungs.temperatures,
            np.random.default_rng([seed, 1]),
            max(1, visits // per_step),
        )
        polished, polish_spent = polish(each, tempered.best)
        return Outcome(
            raw_gap=tempered.energy - each.bound,
            gap=polished - each.bound,
            spent=int(tempered.spent) + polish_spent,
            polish_spent=polish_spent,
        )

    first = run(budget)
    return run(budget - first.polish_spent) if counted else first


def long_run(each: Cell, budget: int, seed: int, counted: bool) -> Outcome:
    """One anneal of ``budget``; ``counted`` as :func:`restarts`."""
    return restarts(each, budget, 1, seed, counted)


def split_arms(
    each: Cell, rungs: Ladder, budget: int, seed: int, counted: bool
) -> dict[str, Outcome]:
    """Every budget-split arm on one seed."""
    arms = {"long": long_run(each, budget, seed, counted)}
    for k in RESTARTS:
        arms[f"restart-{k}"] = restarts(each, budget, k, seed, counted)
    arms["tempering"] = tempering(each, rungs, budget, seed, counted)
    return arms


@dataclass(frozen=True)
class SplitRow:
    """One cell, level and arm: per seed, with the polish outside ``B`` and counted in it."""

    cell: str
    level: int
    budget: int
    arm: str
    outside: tuple[Outcome, ...]
    counted: tuple[Outcome, ...]


def split_study(
    names: Sequence[str] = VARIANTS,
    tier: str = "stress",
    levels: Sequence[int] = LEVELS,
    seeds: Sequence[int] = SEEDS,
) -> tuple[list[Cell], list[Ladder], list[SplitRow]]:
    """The budget-split rows on ``tier``'s cell and its variants ``names``."""
    params: PottsReferenceParams = fixture("potts_reference", tier).params
    cells = [
        cell(params if name == tier else params.variant(name), name) for name in names
    ]
    ladders = [ladder(each) for each in cells]
    rows = []
    for each, rungs in zip(cells, ladders, strict=True):
        for level in levels:
            budget = level * each.base
            outside = [split_arms(each, rungs, budget, s, False) for s in seeds]
            counted = [split_arms(each, rungs, budget, s, True) for s in seeds]
            for arm in outside[0]:
                rows.append(
                    SplitRow(
                        each.name,
                        level,
                        budget,
                        arm,
                        tuple(o[arm] for o in outside),
                        tuple(c[arm] for c in counted),
                    )
                )
    return cells, ladders, rows


def _mean_se(values: Sequence[float]) -> str:
    """Mean and standard error, two decimals."""
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.2f} ({se:.2f})"


def split_table(
    cells: Sequence[Cell], ladders: Sequence[Ladder], rows: Sequence[SplitRow]
) -> str:
    """The budget-split rows as Markdown: gaps in nats, spend as a multiple of ``B``."""
    lines = [
        "| cell | ae+icm visits (gap) | ladder | pilot visits |",
        "| --- | --- | --- | --- |",
    ]
    for each, rungs in zip(cells, ladders, strict=True):
        lines.append(
            f"| {each.name} | {each.base} ({each.base_gap:.2f}) | "
            f"{', '.join(f'{t:.3g}' for t in rungs.temperatures)} "
            f"(in band: {rungs.within_band}) | {rungs.spent} |"
        )
    lines += [
        "",
        "| cell | level | arm | raw gap | gap, polish outside B | spent / B "
        "| gap, polish counted | spent / B |",
        "|" + " --- |" * 8,
    ]
    for row in rows:
        lines.append(
            f"| {row.cell} | x{row.level} | {row.arm} | "
            f"{_mean_se([o.raw_gap for o in row.outside])} | "
            f"{_mean_se([o.gap for o in row.outside])} | "
            f"{statistics.fmean(o.spent for o in row.outside) / row.budget:.2f} | "
            f"{_mean_se([o.gap for o in row.counted])} | "
            f"{statistics.fmean(o.spent for o in row.counted) / row.budget:.2f} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class MoveRow:
    """One move set at one budget: per seed, gaps before and after the polish, and wall seconds."""

    moves: str
    budget: int
    raw: tuple[float, ...]
    polished: tuple[float, ...]
    seconds: float


def move_study(
    tier: str = "stress",
    sweeps: Sequence[int] | None = None,
    seeds: Sequence[int] = MOVE_SEEDS,
) -> tuple[Cell, list[MoveRow]]:
    """The move mix on ``tier``'s cell at x16 and :data:`MOVE_SWEEPS` sweeps' visits, or at ``sweeps``."""
    each = cell(fixture("potts_reference", tier).params, tier)
    budgets = (
        [16 * each.base, MOVE_SWEEPS * each.per_sweep]
        if sweeps is None
        else [s * each.per_sweep for s in sweeps]
    )
    rows = []
    for budget in budgets:
        steps = max(2, budget // each.per_sweep)
        for name, moves in MOVES.items():
            raw, polished = [], []
            started = time.perf_counter()
            for seed in seeds:
                run = anneal_potts(
                    each.problem.graph,
                    each.problem.field,
                    each.schedule().build(steps),
                    np.random.default_rng([seed, 3]),
                    move=moves,
                    recolour=Recolour.PER_MOVE,
                    budget=Budget(Cost.SITE_VISITS, budget),
                    polish=Polish.ICM_MERGE,
                )
                raw.append(run.stages[0].energy - each.bound)
                polished.append(run.energy - each.bound)
            rows.append(
                MoveRow(
                    name,
                    budget,
                    tuple(raw),
                    tuple(polished),
                    (time.perf_counter() - started) / len(seeds),
                )
            )
    return each, rows


def move_table(rows: Sequence[MoveRow]) -> str:
    """The move-mix rows as Markdown: median and mean gap, unpolished and polished."""
    lines = [
        "| moves | budget | raw gap median | raw gap mean (se) | polished median "
        "| polished mean (se) | s per run |",
        "|" + " --- |" * 7,
    ]
    for row in rows:
        lines.append(
            f"| {row.moves} | {row.budget} | {statistics.median(row.raw):.2f} | "
            f"{_mean_se(row.raw)} | {statistics.median(row.polished):.2f} | "
            f"{_mean_se(row.polished)} | {row.seconds:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run both studies and print their tables."""
    started = time.perf_counter()
    cells, ladders, rows = split_study()
    print(split_table(cells, ladders, rows))
    print(f"\nbudget split: {time.perf_counter() - started:.0f} s\n")
    clock = time.perf_counter()
    _, moved = move_study()
    print(move_table(moved))
    print(f"\nmove mix: {time.perf_counter() - clock:.0f} s")


if __name__ == "__main__":
    main()
