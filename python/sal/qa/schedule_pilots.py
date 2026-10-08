"""Whether a short schedule pilot ranks as the full run does, and what racing and paired streams buy (issue #1390, questions 2 and 3).

``tune_schedule`` ranks :data:`~sal.sample.tune.SCHEDULE_GRID` on pilots of
``budget // (len(grid) * n_nodes)`` steps, each the candidate's own schedule
at the pilot's length. This study measures that ranking against the one the
tuned run would see, on ``potts_reference`` at ``ci`` and ``stress`` and three
one-factor variants of ``stress``.

**The reference ranking.** Every candidate anneals for :data:`RUN_SWEEPS`
steps on each of :data:`HELD_OUT_SEEDS`, its best labelling is polished by
ICM, and the polished energy is averaged over the seeds. The full-length best
is the candidate of lowest mean.

**Question 2.** On each of :data:`TUNING_SEEDS`, ``tune_schedule`` under
``POLISHED_GAP`` runs pilots of ``RUN_SWEEPS * f`` steps for ``f`` in
:data:`FRACTIONS`; Kendall's tau-b between the pilots' polished energies and
the reference means is averaged over the seeds. The pilots' share of cost is
their site visits over those plus one tuned run's.

**Question 3.** On each of :data:`SELECTION_SEEDS`, the four combinations of
``racing`` and ``common`` run at one budget per fraction of
:data:`SELECTION_FRACTIONS`; each records whether it chose the full-length
best, its regret against it in reference energy, and its spend. The figure of
merit is P(best) per pilot site visit, read relative to both options off.

The polish is ICM from the pilot's best on ``default_rng(0)``, a callable as
``ScheduleTuning.polish`` takes it. The move is Swendsen-Wang with heat-bath
recolour, the anneal of ``profile_hotpaths.py --module anneal``.

Run as ``python -m sal.qa.schedule_pilots``; it prints the tables
``docs/experiments/033-schedule-pilot-transfer.md`` reports.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau

from sal.cost import Cost
from sal.opt.budget import Budget
from sal.parallel import Pool, map_tasks
from sal.sample.potts_mcmc import PottsMove, Recolour, anneal_potts
from sal.sample.tune import SCHEDULE_GRID, Criterion, TunedSchedule, tune_schedule
from sal.search.icm import iterated_conditional_modes
from sal.sim.fixtures import fixture
from sal.sim.potts import energy
from sal.sim.potts_cell import PlantedPotts, PottsReferenceParams

#: ``(tier, variant)``: the reference cell at two tiers and three variants of
#: ``stress``, a weak and a strong margin and twice the labels.
INSTANCES: tuple[tuple[str, str | None], ...] = (
    ("ci", None),
    ("stress", None),
    ("stress", "margin_0.1"),
    ("stress", "margin_3"),
    ("stress", "q8"),
)
#: The tuned run's length in steps, the profile's 200 rounded to divide by 16.
RUN_SWEEPS = 256
#: Pilot length over run length, question 2.
FRACTIONS = (1 / 16, 1 / 4, 1.0)
#: Pilot length over run length at which question 3 compares selections.
SELECTION_FRACTIONS = (1 / 16, 1 / 4)
#: Seeds of the full-length reference runs, held out of every pilot.
HELD_OUT_SEEDS = tuple(range(32))
#: Seeds of question 2's pilots.
TUNING_SEEDS = tuple(range(16))
#: Seeds of question 3's selections, per combination.
SELECTION_SEEDS = tuple(range(48))
#: The racing and common-random-number combinations, both off first.
COMBINATIONS = ((False, False), (True, False), (False, True), (True, True))
MOVE = PottsMove.SWENDSEN_WANG
RECOLOUR = Recolour.HEAT_BATH
#: Worker processes: two, the host's share.
WORKERS = 2

type Instance = tuple[str, str | None]


def name(instance: Instance) -> str:
    """``tier`` or ``tier/variant``."""
    tier, variant = instance
    return tier if variant is None else f"{tier}/{variant}"


@functools.cache
def cell(instance: Instance) -> PlantedPotts:
    """The drawn cell of ``instance``, once per process."""
    tier, variant = instance
    params: PottsReferenceParams = fixture("potts_reference", tier).params
    return (params if variant is None else params.variant(variant)).instance()


def _polish(labelling: np.ndarray, *, instance: Instance) -> np.ndarray:
    """ICM from ``labelling`` on ``default_rng(0)``: the callable polish ``ScheduleTuning`` takes."""
    drawn = cell(instance)
    return iterated_conditional_modes(
        drawn.graph, drawn.field, np.random.default_rng(0), start=labelling
    ).labelling


def full_run(task: tuple[Instance, int, int]) -> tuple[float, int]:
    """Candidate ``k`` at :data:`RUN_SWEEPS` on held-out seed ``s``: polished energy and spend."""
    instance, k, seed = task
    drawn = cell(instance)
    run = anneal_potts(
        drawn.graph,
        drawn.field,
        SCHEDULE_GRID[k].build(RUN_SWEEPS),
        np.random.default_rng([seed, 1]),
        move=MOVE,
        recolour=RECOLOUR,
    )
    polished = _polish(np.copy(run.best), instance=instance)
    return energy(drawn.graph, drawn.field, polished), run.spent


def pilot_budget(instance: Instance, fraction: float) -> Budget:
    """The budget at which ``tune_schedule`` runs ``RUN_SWEEPS * fraction`` steps per candidate."""
    steps = round(RUN_SWEEPS * fraction)
    return Budget(
        Cost.SITE_VISITS, len(SCHEDULE_GRID) * cell(instance).graph.n_nodes * steps
    )


def tune(
    instance: Instance, fraction: float, seed: int, racing: bool, common: bool
) -> TunedSchedule:
    """``tune_schedule`` under ``POLISHED_GAP`` at ``fraction`` of the run, on ``default_rng([seed, 2])``."""
    drawn = cell(instance)
    return tune_schedule(
        drawn.graph,
        drawn.field,
        move=MOVE,
        recolour=RECOLOUR,
        budget=pilot_budget(instance, fraction),
        criterion=Criterion.POLISHED_GAP,
        rng=np.random.default_rng([seed, 2]),
        racing=racing,
        common=common,
        polish=functools.partial(_polish, instance=instance),
    )


def pilot_scores(task: tuple[Instance, float, int]) -> tuple[tuple[float, ...], int]:
    """Question 2's pilot: every candidate's polished energy, and the pilots' spend."""
    instance, fraction, seed = task
    tuned = tune(instance, fraction, seed, racing=False, common=False)
    scores = tuple(candidate.score for candidate in tuned.candidates)
    return scores, tuned.spent


def selection(
    task: tuple[Instance, float, int, bool, bool],
) -> tuple[int, int]:
    """Question 3's pilot: the chosen candidate's grid index, and the pilots' spend."""
    instance, fraction, seed, racing, common = task
    tuned = tune(instance, fraction, seed, racing, common)
    return SCHEDULE_GRID.index(tuned.params), tuned.spent


@dataclass(frozen=True)
class Reference:
    """The full-length ranking of one instance."""

    means: tuple[float, ...]
    errors: tuple[float, ...]
    run_spent: float
    split_tau: float
    """Kendall's tau between the means over the even and the odd seeds: the reference's own noise."""

    @property
    def best(self) -> int:
        """The candidate of lowest mean polished energy, ties to grid order."""
        return min(range(len(self.means)), key=lambda k: (self.means[k], k))


def tau(scores: Sequence[float], reference: Sequence[float]) -> float:
    """Kendall's tau-b between a pilot's scores and the reference means; ``nan`` where either is constant."""
    if len(set(scores)) < 2 or len(set(reference)) < 2:
        return math.nan
    return float(kendalltau(scores, reference).statistic)


def references(
    instances: Sequence[Instance],
    seeds: Sequence[int] = HELD_OUT_SEEDS,
    workers: int = WORKERS,
) -> dict[Instance, Reference]:
    """Every instance's reference ranking from its full-length runs."""
    tasks = [
        (instance, k, seed)
        for instance in instances
        for k in range(len(SCHEDULE_GRID))
        for seed in seeds
    ]
    pool: Pool = "processes" if workers > 1 else "serial"
    runs = iter(map_tasks(full_run, tasks, workers=workers, pool=pool))
    out: dict[Instance, Reference] = {}
    for instance in instances:
        energies = np.empty((len(SCHEDULE_GRID), len(seeds)))
        spent = []
        for k in range(len(SCHEDULE_GRID)):
            for s in range(len(seeds)):
                energies[k, s], visits = next(runs)
                spent.append(visits)
        out[instance] = Reference(
            tuple(energies.mean(axis=1)),
            tuple(energies.std(axis=1, ddof=1) / math.sqrt(len(seeds))),
            float(np.mean(spent)),
            tau(
                energies[:, 0::2].mean(axis=1).tolist(),
                energies[:, 1::2].mean(axis=1).tolist(),
            ),
        )
    return out


@dataclass(frozen=True)
class Transfer:
    """Question 2 at one instance and fraction."""

    tau: float
    tau_error: float
    pilot_spent: float
    share: float


def transfer(
    instances: Sequence[Instance],
    refs: dict[Instance, Reference],
    fractions: Sequence[float] = FRACTIONS,
    seeds: Sequence[int] = TUNING_SEEDS,
    workers: int = WORKERS,
) -> dict[tuple[Instance, float], Transfer]:
    """Mean tau and the pilots' share of cost, per instance and fraction."""
    tasks = [
        (instance, fraction, seed)
        for instance in instances
        for fraction in fractions
        for seed in seeds
    ]
    pool: Pool = "processes" if workers > 1 else "serial"
    runs = iter(map_tasks(pilot_scores, tasks, workers=workers, pool=pool))
    out: dict[tuple[Instance, float], Transfer] = {}
    for instance in instances:
        ref = refs[instance]
        for fraction in fractions:
            taus, spent = [], []
            for _ in seeds:
                scores, visits = next(runs)
                taus.append(tau(scores, ref.means))
                spent.append(visits)
            kept = [t for t in taus if not math.isnan(t)]
            pilot = float(np.mean(spent))
            out[instance, fraction] = Transfer(
                float(np.mean(kept)) if kept else math.nan,
                float(np.std(kept, ddof=1) / math.sqrt(len(kept)))
                if len(kept) > 1
                else math.nan,
                pilot,
                pilot / (pilot + ref.run_spent),
            )
    return out


@dataclass(frozen=True)
class Selection:
    """Question 3 at one instance, fraction and combination."""

    p_best: float
    regret: float
    spent: float

    @property
    def per_visit(self) -> float:
        """P(best) per pilot site visit."""
        return self.p_best / self.spent


def selections(
    instances: Sequence[Instance],
    refs: dict[Instance, Reference],
    fractions: Sequence[float] = SELECTION_FRACTIONS,
    seeds: Sequence[int] = SELECTION_SEEDS,
    workers: int = WORKERS,
) -> dict[tuple[Instance, float, bool, bool], Selection]:
    """P(best), mean regret and mean spend, per instance, fraction and combination."""
    tasks = [
        (instance, fraction, seed, racing, common)
        for instance in instances
        for fraction in fractions
        for racing, common in COMBINATIONS
        for seed in seeds
    ]
    pool: Pool = "processes" if workers > 1 else "serial"
    runs = iter(map_tasks(selection, tasks, workers=workers, pool=pool))
    out: dict[tuple[Instance, float, bool, bool], Selection] = {}
    for instance in instances:
        ref = refs[instance]
        for fraction in fractions:
            for racing, common in COMBINATIONS:
                chosen, spent = zip(*(next(runs) for _ in seeds), strict=True)
                out[instance, fraction, racing, common] = Selection(
                    float(np.mean([k == ref.best for k in chosen])),
                    float(
                        np.mean([ref.means[k] - ref.means[ref.best] for k in chosen])
                    ),
                    float(np.mean(spent)),
                )
    return out


def _label(racing: bool, common: bool) -> str:
    return f"racing={'on' if racing else 'off'}, common={'on' if common else 'off'}"


def main() -> None:
    """Run both questions and print their tables."""
    opened = time.perf_counter()
    refs = references(INSTANCES)
    print("## reference: best candidate, its mean, the runner-up gap, its error")
    for instance in INSTANCES:
        ref = refs[instance]
        order = sorted(range(len(ref.means)), key=lambda k: (ref.means[k], k))
        print(
            f"{name(instance)}: best {SCHEDULE_GRID[ref.best]} "
            f"{ref.means[ref.best]:.2f} +- {ref.errors[ref.best]:.2f}; "
            f"gap to 2nd {ref.means[order[1]] - ref.means[order[0]]:.2f}; "
            f"spread {ref.means[order[-1]] - ref.means[order[0]]:.2f}; "
            f"split-half tau {ref.split_tau:.2f}; "
            f"run spent {ref.run_spent:,.0f}"
        )
    print(f"[{time.perf_counter() - opened:.0f} s]")
    moved = transfer(INSTANCES, refs)
    print("\n## question 2: mean Kendall tau (se), pilot share of cost")
    print("| instance | " + " | ".join(f"f = {f:g}" for f in FRACTIONS) + " |")
    for instance in INSTANCES:
        cells = [
            f"{moved[instance, f].tau:.2f} ({moved[instance, f].tau_error:.2f}); "
            f"{moved[instance, f].share:.0%}"
            for f in FRACTIONS
        ]
        print(f"| {name(instance)} | " + " | ".join(cells) + " |")
    print(f"[{time.perf_counter() - opened:.0f} s]")
    chosen = selections(INSTANCES, refs)
    print("\n## question 3: P(best); regret; spent; per-visit gain over both off")
    for fraction in SELECTION_FRACTIONS:
        for instance in INSTANCES:
            base = chosen[instance, fraction, False, False]
            for racing, common in COMBINATIONS:
                got = chosen[instance, fraction, racing, common]
                gain = got.per_visit / base.per_visit if base.p_best else math.nan
                print(
                    f"f={fraction:g} {name(instance)} {_label(racing, common)}: "
                    f"P {got.p_best:.2f}; regret {got.regret:.2f}; "
                    f"spent {got.spent:,.0f}; gain {gain:.2f}x"
                )
    print(f"[{time.perf_counter() - opened:.0f} s]")


if __name__ == "__main__":
    main()
