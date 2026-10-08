"""Does a start temperature in the instance's own energy units transfer across instances? (issue #1390, question 1).

:data:`~sal.sample.tune.SCHEDULE_GRID`'s temperatures are absolute: 2.0 and
0.8 to start, whatever the instance's energies are measured in. This script
asks whether ``T0 / s`` for an energy scale ``s`` read off the instance is
near-constant across `potts_reference`'s variants, so that one constant
tuned on the ``stress`` cell sets every variant's start without a pilot.

**The scales** (:func:`energy_scales`), each in nats:

* ``coupling`` --- ``J * d``, ``J`` the mean coupling and ``d = 2|E| / n``
  the mean degree: the energy a site pays to disagree with every neighbour.
* ``margin`` --- the median over sites of the field's top-two gap,
  ``h_i(1st) - h_i(2nd)`` over the allowed labels.
* ``proposal`` --- the median over sites of ``|e_i(k') - e_i(x_i)|``, where
  ``x`` is a uniform random labelling over each site's allowed labels,
  ``e_i(k) = -h_i(k) - sum_j J_ij [x_j = k]`` is site ``i``'s conditional
  energy and ``k'`` a uniform other allowed label: the energy change of a
  single-site heat-bath proposal at infinite temperature, from a random start.

**The protocol.** Per variant, the run is single-site heat-bath annealing at
:data:`VISITS_PER_SITE` site visits per site, then ICM from its best
labelling; its gap is that energy less the variant's TRW-S bound.

1. ``tune_schedule`` (racing, common random numbers, ``POLISHED_GAP`` under
   the ICM polish) chooses from :data:`LADDER`, an exponential schedule per
   ``T0`` with ``t_end = T0 / END_RATIO``, on each of :data:`TUNING_SEEDS`;
   the tuned ``T0`` is the median choice. ``SCHEDULE_GRID`` cannot answer the
   question --- it offers two starts --- so its choice is recorded beside it.
2. ``T0 / s`` per scale and variant, and its spread: max over min.
3. ``c = median T0 / s`` on ``stress`` alone, applied to every variant as
   ``T0 = c * s`` with no pilot, against the tuned ``T0``, each held out on
   :data:`HELD_OUT_SEEDS` at equal visits. Every ladder ``T0`` is held out
   too, so the best a grid could have done is read beside both.

Run as ``python -m sal.qa.temperature_scale``; it prints the tables
``docs/experiments/032-a-start-temperature-in-the-instances-units.md``
reports.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove, Recolour, anneal_potts
from sal.sample.schedule import ScheduleParams, ScheduleShape
from sal.sample.tune import SCHEDULE_GRID, Criterion, tune_schedule
from sal.search.icm import iterated_conditional_modes
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy, penalized
from sal.sim.potts_cell import PottsReferenceParams

#: The variants that move the energy scale: margin, q, graph, and n.
VARIANTS = (
    "stress",
    "margin_0.1",
    "margin_3",
    "q2",
    "q8",
    "q16",
    "square",
    "knn",
    "n_1e4",
)
#: The scales :func:`energy_scales` reads, in the order the tables print them.
SCALES = ("coupling", "margin", "proposal")
#: Start temperatures, a quarter-decade-ish ladder: ``2 ** (k / 2)`` from
#: 0.125 to 32, so a ratio is resolved to a factor of 1.41.
LADDER_T0 = tuple(2.0 ** (k / 2.0) for k in range(-6, 11))
#: ``T0 / t_end``: ``SCHEDULE_GRID``'s 2.0 to 0.05, held for every start.
END_RATIO = 40.0
#: The ladder as schedules.
LADDER = tuple(
    ScheduleParams(ScheduleShape.EXPONENTIAL, t0, t0 / END_RATIO) for t0 in LADDER_T0
)
#: The run's budget per site: 200 sweeps of the degree-6 stress cell, whose
#: sweep visits each site and both ends of each edge (``n + 2|E|``).
VISITS_PER_SITE = 1400
#: Seeds the pilots draw from; the held-out runs never use them.
TUNING_SEEDS = (0, 1, 2)
#: Seeds the reported gaps are averaged over.
HELD_OUT_SEEDS = tuple(range(100, 108))


def energy_scales(
    graph: PottsGraph, field: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """The three candidate energy scales of one instance, in nats, as the module defines them.

    ``rng`` draws the random labelling and the proposed labels of
    ``proposal``; ``-inf`` in ``field`` marks a forbidden label, never drawn.

    Returns
    -------
    dict[str, float]
        Keyed by :data:`SCALES`.
    """
    n, q = field.shape
    ends = np.asarray(graph.edge_index)
    coupling = np.asarray(graph.edge_coupling, dtype=float)
    degree = 2.0 * len(ends) / n
    allowed = np.isfinite(field)
    ordered = np.sort(np.where(allowed, field, -np.inf), axis=1)
    margin = float(np.median(ordered[:, -1] - ordered[:, -2]))
    # A uniform allowed label per site: the argmax of uniform scores over the
    # allowed entries.
    start = np.argmax(np.where(allowed, rng.random((n, q)), -1.0), axis=1)
    agree = np.zeros((n, q))
    np.add.at(agree, (ends[:, 0], start[ends[:, 1]]), coupling)
    np.add.at(agree, (ends[:, 1], start[ends[:, 0]]), coupling)
    conditional = -np.where(allowed, field, 0.0) - agree
    others = allowed.copy()
    others[np.arange(n), start] = False
    proposed = np.argmax(np.where(others, rng.random((n, q)), -1.0), axis=1)
    sites = np.arange(n)
    change = np.abs(conditional[sites, proposed] - conditional[sites, start])
    return {
        "coupling": float(np.mean(coupling)) * degree,
        "margin": margin,
        "proposal": float(np.median(change)),
    }


@dataclass(frozen=True)
class Cell:
    """One variant as the study runs it: the instance, its referee and its run length."""

    name: str
    graph: PottsGraph
    field: np.ndarray
    bound: float
    sweeps: int
    scales: Mapping[str, float]

    def polish(self) -> Callable[[np.ndarray], np.ndarray]:
        """ICM from a labelling to its fixed point: ``POLISHED_GAP``'s polish and the run's."""

        def polished(labelling: np.ndarray) -> np.ndarray:
            return iterated_conditional_modes(
                self.graph, self.field, np.random.default_rng(0), start=labelling
            ).labelling

        return polished

    def gaps(self, params: ScheduleParams) -> tuple[float, ...]:
        """The run's polished energy less the TRW-S bound, at :attr:`sweeps`, per :data:`HELD_OUT_SEEDS`."""
        polish = self.polish()
        found = []
        for seed in HELD_OUT_SEEDS:
            run = anneal_potts(
                self.graph,
                self.field,
                params.build(self.sweeps),
                np.random.default_rng(seed),
            )
            found.append(energy(self.graph, self.field, polish(run.best)) - self.bound)
        return tuple(found)

    def tune(self, grid: tuple[ScheduleParams, ...], seed: int) -> ScheduleParams:
        """``tune_schedule``'s choice from ``grid`` on ``seed``, its pilots at the run's length."""
        tuned = tune_schedule(
            self.graph,
            self.field,
            move=PottsMove.SINGLE_SITE,
            recolour=Recolour.PER_MOVE,
            budget=Budget(
                Cost.SITE_VISITS, len(grid) * self.graph.n_nodes * self.sweeps
            ),
            criterion=Criterion.POLISHED_GAP,
            rng=np.random.default_rng(seed),
            grid=grid,
            racing=True,
            common=True,
            polish=self.polish(),
        )
        return tuned.params


def cell(variant: PottsReferenceParams, name: str) -> Cell:
    """``variant`` drawn, with its TRW-S bound, its run length and its scales."""
    drawn = variant.instance()
    referee = (
        penalized(drawn.graph, drawn.field)
        if np.isneginf(drawn.field).any()
        else drawn.field
    )
    per_sweep = drawn.graph.n_nodes + 2 * len(drawn.graph.edges)
    return Cell(
        name=name,
        graph=drawn.graph,
        field=drawn.field,
        bound=float(trws(drawn.graph, referee).bound),
        sweeps=VISITS_PER_SITE * drawn.graph.n_nodes // per_sweep,
        scales=energy_scales(
            drawn.graph, drawn.field, np.random.default_rng(variant.seed)
        ),
    )


def relative(t0: float) -> ScheduleParams:
    """The study's schedule at start ``t0``: exponential to ``t0 / END_RATIO``."""
    return ScheduleParams(ScheduleShape.EXPONENTIAL, t0, t0 / END_RATIO)


@dataclass(frozen=True)
class Row:
    """One variant's result: its scales, its tuned starts, and its held-out gaps per ladder start."""

    name: str
    bound: float
    scales: Mapping[str, float]
    absolute: tuple[float, ...]
    tuned: float
    ladder: tuple[tuple[float, ...], ...]

    @property
    def flat(self) -> bool:
        """Whether every ladder start reaches one mean gap to 0.01 nats: ``T0`` is then undetermined."""
        means = [statistics.fmean(gaps) for gaps in self.ladder]
        return max(means) - min(means) < 0.01

    @property
    def tuned_gaps(self) -> tuple[float, ...]:
        """The held-out gaps at the tuned start."""
        return self.ladder[LADDER_T0.index(self.tuned)]


def tuned_row(each: Cell) -> Row:
    """``each`` tuned on both grids per :data:`TUNING_SEEDS`, and held out at every ladder start."""
    absolute = tuple(each.tune(SCHEDULE_GRID, seed).t_start for seed in TUNING_SEEDS)
    tuned = statistics.median_low(
        each.tune(LADDER, seed).t_start for seed in TUNING_SEEDS
    )
    return Row(
        name=each.name,
        bound=each.bound,
        scales=dict(each.scales),
        absolute=absolute,
        tuned=tuned,
        ladder=tuple(each.gaps(schedule) for schedule in LADDER),
    )


def transfer(
    cells: list[Cell], constants: Mapping[str, float]
) -> dict[str, list[tuple[float, ...]]]:
    """Per scale, each cell's held-out gaps at ``T0 = c * s``, no pilot run."""
    return {
        scale: [each.gaps(relative(constant * each.scales[scale])) for each in cells]
        for scale, constant in constants.items()
    }


def _sem(values: tuple[float, ...]) -> float:
    """The standard error of the mean."""
    return float(statistics.stdev(values)) / math.sqrt(len(values))


def report(
    rows: list[Row],
    constants: Mapping[str, float],
    transferred: Mapping[str, list[tuple[float, ...]]],
) -> str:
    """The tables the experiment file reports, as Markdown; a flat variant is left out of the spread."""
    lines = [
        "| variant | TRW-S bound | "
        + " | ".join(SCALES)
        + " | SCHEDULE_GRID T0 | tuned T0 | "
        + " | ".join(f"T0/{s}" for s in SCALES)
        + " | gap tuned | "
        + " | ".join(f"gap c*{s}" for s in SCALES)
        + " | best ladder T0 (held-out) | gap there |",
        "| " + " | ".join("---" for _ in range(9 + 3 * len(SCALES) - 2)) + " |",
    ]
    for index, row in enumerate(rows):
        means = [statistics.fmean(gaps) for gaps in row.ladder]
        best = int(np.argmin(means))
        lines.append(
            "| "
            + " | ".join(
                [
                    row.name + (" (flat)" if row.flat else ""),
                    f"{row.bound:.1f}",
                    *(f"{row.scales[s]:.2f}" for s in SCALES),
                    "/".join(f"{t:g}" for t in row.absolute),
                    f"{row.tuned:.3g}",
                    *(f"{row.tuned / row.scales[s]:.3g}" for s in SCALES),
                    f"{statistics.fmean(row.tuned_gaps):.2f} "
                    f"+- {_sem(row.tuned_gaps):.2f}",
                    *(f"{statistics.fmean(transferred[s][index]):.2f}" for s in SCALES),
                    f"{LADDER_T0[best]:.3g}",
                    f"{means[best]:.2f}",
                ]
            )
            + " |"
        )
    determined = [k for k, row in enumerate(rows) if not row.flat]
    tuned_total = sum(statistics.fmean(rows[k].tuned_gaps) for k in determined)
    for scale in ("absolute", *SCALES):
        ratios = [
            rows[k].tuned / (1.0 if scale == "absolute" else rows[k].scales[scale])
            for k in determined
        ]
        line = f"\n{scale}: T0/s spread (max/min) {max(ratios) / min(ratios):.3g}"
        if scale != "absolute":
            moved = sum(statistics.fmean(transferred[scale][k]) for k in determined)
            line += (
                f"; c = {constants[scale]:.3g}; transferred gap sum {moved:.2f} "
                f"against tuned {tuned_total:.2f} nats"
            )
        lines.append(line)
    return "\n".join(lines)


def study(
    tier: str = "stress", names: tuple[str, ...] = VARIANTS
) -> tuple[list[Row], dict[str, float], dict[str, list[tuple[float, ...]]]]:
    """The rows on ``tier``'s cell and its variants, ``c`` per scale read on the first, and the transferred gaps."""
    params: PottsReferenceParams = fixture("potts_reference", tier).params
    cells = [
        cell(params if name == tier else params.variant(name), name) for name in names
    ]
    rows = [tuned_row(each) for each in cells]
    source = rows[0]
    constants = {scale: source.tuned / source.scales[scale] for scale in SCALES}
    return rows, constants, transfer(cells, constants)


def main() -> None:
    """Run the study on the stress cell and its variants and print the tables."""
    began = time.perf_counter()
    rows, constants, transferred = study()
    print(report(rows, constants, transferred))
    print(
        f"\nwall {time.perf_counter() - began:.0f} s; held-out mean gap per ladder T0:"
    )
    print("T0 " + " ".join(f"{t:.3g}" for t in LADDER_T0))
    for row in rows:
        print(row.name, " ".join(f"{statistics.fmean(g):.2f}" for g in row.ladder))


if __name__ == "__main__":
    main()
