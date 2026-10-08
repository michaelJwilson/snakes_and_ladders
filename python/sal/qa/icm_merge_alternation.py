"""ICM and the whole-label merge alternated to a joint fixed point, against one pass (issue #1390, adaptation).

``Polish.ICM_MERGE`` runs index-order ICM to its fixed point, then
:func:`~sal.sample.potts_mcmc.chains.merge_labels` until no merge lowers the
energy, once (issue #1373). A merge moves every site of a label at once, so the
merged labelling need not be an ICM fixed point: a second ICM pass can lower it
further, and a second merge after that. This study measures what that
alternation recovers and what it costs.

**The passes.** From a start, round 1 is :func:`~sal.sample.potts_mcmc.chains._descend`
then :func:`~sal.sample.potts_mcmc.chains.merge_labels`, the NumPy oracles of the
Rust loop's polish; round ``r >= 2`` runs only if round ``r - 1``'s merge
changed the labelling, and the run stops at the first round whose ICM or merge
changes nothing: the joint fixed point. Each ICM sweep and each merge round is
charged a heat-bath sweep's site visits, as ``polish_spent`` charges them.

**The starts.** On ``potts_reference``'s ``stress`` cell and the variants
:data:`VARIANTS`:

* ``anneal xL`` --- the schedule's ``best`` of
  :func:`~sal.sample.potts_mcmc.anneal_potts` under a budget of ``L`` times
  alpha-expansion+ICM's site visits (``L`` in :data:`LEVELS`), single-site heat
  bath on the exponential schedule from ``T0 = J d / 3`` to ``T0 / 40``, run
  with ``Polish.ICM_MERGE``: its ``init`` stage, the ``best`` a ``polish=None``
  run of the same generator returns, is the start, and round 1 is checked
  bitwise against its ``merge`` stage (labelling, energy and
  ``polish_spent``);
* ``random`` --- a uniform labelling;
* ``ae`` --- alpha-expansion's labelling from the field's argmax.

The referee is the cell's TRW-S bound; a gap is energy less the bound.

Run as ``python -m sal.qa.icm_merge_alternation``; it prints the table
``docs/experiments/036-alternating-icm-and-the-label-merge.md`` reports.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.qa.known_ground_states import expansion_icm
from sal.sample.potts_mcmc import PottsMove, anneal_potts
from sal.sample.potts_mcmc.chains import _descend, merge_labels, step_visits
from sal.sample.schedule import Polish, ScheduleParams, ScheduleShape
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import Problem
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.potts import energy, site_field
from sal.sim.potts_cell import PottsReferenceParams

#: The cells: the reference and one variant per factor that moves the polish.
VARIANTS = ("stress", "margin_0.1", "q8", "imbalance_50", "equal", "forbidden")
#: Anneal budgets, as multiples of alpha-expansion+ICM's site visits on the cell.
LEVELS = (1, 4, 16)
#: ``T0 / (J d)``.
START_PER_COUPLING = 1.0 / 3.0
#: ``T0 / t_end``.
END_RATIO = 40.0
#: Seeds of the anneal and random starts.
SEEDS = tuple(range(32))
#: A gain from alternation at or above this, in nats, counts as one.
GAIN = 0.1
#: The share of runs with such a gain, at any start kind, that proposes iterating.
SHARE = 0.05


@dataclass(frozen=True)
class Cell:
    """One instance: its problem, field rows, TRW-S bound, base budget and schedule."""

    name: str
    problem: Problem
    rows: np.ndarray
    bound: float
    base: int
    t0: float

    @property
    def per_sweep(self) -> int:
        """Site visits of one heat-bath sweep, the unit ``polish_spent`` charges."""
        return step_visits(PottsMove.SINGLE_SITE, self.problem.graph)

    def schedule(self) -> ScheduleParams:
        """The exponential schedule from ``T0`` to ``T0 / END_RATIO``."""
        return ScheduleParams(ScheduleShape.EXPONENTIAL, self.t0, self.t0 / END_RATIO)


def cell(params: PottsReferenceParams, name: str) -> Cell:
    """``params`` drawn, with its TRW-S bound, alpha-expansion+ICM's spend and ``T0 = J d / 3``."""
    drawn = params.instance()
    problem = Problem(drawn.graph, drawn.field, drawn.n_states)
    degree = 2.0 * len(drawn.graph.edges) / problem.n_nodes
    coupling = float(np.mean(np.asarray(drawn.graph.edge_coupling, dtype=float)))
    return Cell(
        name=name,
        problem=problem,
        rows=site_field(np.asarray(drawn.field, dtype=float), problem.n_nodes),
        bound=float(trws(drawn.graph, drawn.field).bound),
        base=expansion_icm(problem).spent,
        t0=START_PER_COUPLING * coupling * degree,
    )


@dataclass(frozen=True)
class Round:
    """One ICM pass then one merge: energies after each, visits, and the merge's label count."""

    icm_energy: float
    merge_energy: float
    visits: int
    merged: bool
    labels_left: int


@dataclass(frozen=True)
class Alternation:
    """Every round from one start to the joint fixed point, and the wall seconds of the rounds after the first."""

    start: str
    rounds: tuple[Round, ...]
    seconds_after_first: float

    @property
    def one_pass(self) -> float:
        """Energy after round 1, ``Polish.ICM_MERGE``'s."""
        return self.rounds[0].merge_energy

    @property
    def final(self) -> float:
        """Energy at the joint fixed point."""
        return self.rounds[-1].merge_energy

    @property
    def gain(self) -> float:
        """Energy the rounds after the first recover, in nats."""
        return self.one_pass - self.final

    @property
    def extra_visits(self) -> int:
        """Site visits of the rounds after the first."""
        return sum(r.visits for r in self.rounds[1:])

    @property
    def collapsed(self) -> bool:
        """Whether round 1's merge changed the labelling and left one label."""
        return self.rounds[0].merged and self.rounds[0].labels_left == 1


def alternate(
    each: Cell, start: np.ndarray, name: str
) -> tuple[Alternation, np.ndarray]:
    """ICM then merge from ``start``, repeated while a merge changes the labelling and the next ICM does too."""
    graph, rows = each.problem.graph, each.rows
    labels = np.asarray(start, dtype=np.int64)
    rounds: list[Round] = []
    first = labels
    clock = time.perf_counter()
    while True:
        settled = _descend(graph, rows, labels, np.random.default_rng(0), 0)
        visits = settled.sweeps * each.per_sweep
        icm_moved = not np.array_equal(settled.labelling, labels)
        if rounds and not icm_moved:
            # The merge's last fixed point is unchanged: the joint fixed point.
            last = rounds[-1]
            rounds.append(
                Round(
                    last.merge_energy,
                    last.merge_energy,
                    visits,
                    False,
                    last.labels_left,
                )
            )
            break
        merged, n_rounds = merge_labels(graph, rows, settled.labelling)
        visits += n_rounds * each.per_sweep
        changed = n_rounds > 1
        rounds.append(
            Round(
                icm_energy=float(settled.energy),
                merge_energy=energy(graph, rows, merged),
                visits=visits,
                merged=changed,
                labels_left=int(np.unique(merged).size),
            )
        )
        labels = merged
        if len(rounds) == 1:
            first, clock = merged, time.perf_counter()
        if not changed:
            break
    return Alternation(name, tuple(rounds), time.perf_counter() - clock), first


def anneal_start(
    each: Cell, level: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """The ``Polish.ICM_MERGE`` anneal at ``level``: its ``init`` labelling, merged labelling and energy, and ``polish_spent``."""
    budget = level * each.base
    run = anneal_potts(
        each.problem.graph,
        each.problem.field,
        each.schedule().build(max(2, budget // each.per_sweep)),
        np.random.default_rng([seed, 4]),
        budget=Budget(Cost.SITE_VISITS, budget),
        polish=Polish.ICM_MERGE,
    )
    return (
        np.asarray(run.stages[0].best),
        np.asarray(run.best),
        float(run.energy),
        int(run.polish_spent),
    )


@dataclass(frozen=True)
class Row:
    """One cell and start kind, over its runs."""

    cell: str
    start: str
    bound: float
    runs: tuple[Alternation, ...]


def reference_cells(
    tier: str = "stress", names: Sequence[str] = VARIANTS
) -> list[tuple[str, PottsReferenceParams]]:
    """``potts_reference``'s ``tier`` cell and its declared variants ``names``, ``tier`` naming the cell itself."""
    params: PottsReferenceParams = fixture("potts_reference", tier).params
    return [(name, params if name == tier else params.variant(name)) for name in names]


def study(
    cells: Sequence[tuple[str, PottsReferenceParams]] | None = None,
    levels: Sequence[int] = LEVELS,
    seeds: Sequence[int] = SEEDS,
) -> list[Row]:
    """Alternation from every start kind on ``cells``, :func:`reference_cells` by default.

    Raises
    ------
    AssertionError
        If round 1 from an anneal's ``init`` differs from its ``Polish.ICM_MERGE``
        result in labelling, energy or ``polish_spent``.
    """
    rows = []
    for name, params in reference_cells() if cells is None else cells:
        each = cell(params, name)
        for level in levels:
            runs = []
            for seed in seeds:
                init, polished, polished_energy, spent = anneal_start(each, level, seed)
                run, first = alternate(each, init, f"anneal x{level}")
                assert np.array_equal(first, polished)
                assert run.one_pass == polished_energy
                assert run.rounds[0].visits == spent
                runs.append(run)
            rows.append(Row(name, f"anneal x{level}", each.bound, tuple(runs)))
        random = [
            alternate(
                each,
                np.random.default_rng([seed, 5]).integers(
                    0, each.problem.n_states, size=each.problem.n_nodes
                ),
                "random",
            )[0]
            for seed in seeds
        ]
        rows.append(Row(name, "random", each.bound, tuple(random)))
        expanded = alpha_expansion(
            each.problem.graph,
            each.problem.field,
            backend=Backend.RUST,
            n_states=each.problem.n_states,
        )
        ae = alternate(each, expanded.labelling, "ae")[0]
        rows.append(Row(name, "ae", each.bound, (ae,)))
    return rows


def _gain_share(runs: Sequence[Alternation]) -> float:
    """The share of ``runs`` whose alternation gains at least :data:`GAIN`."""
    return sum(run.gain >= GAIN for run in runs) / len(runs)


def table(rows: Sequence[Row]) -> str:
    """The rows as Markdown: merges, rounds, gains, gaps, visits and collapses."""
    lines = [
        "| cell | start | runs | round-1 merges | rounds (max) | gain >= 0.1 "
        "| max gain | one-pass gap | final gap | round-1 visits | extra visits "
        "| extra ms | collapses | collapsed: ICM gap, final gap |",
        "|" + " --- |" * 14,
    ]
    for row in rows:
        runs = row.runs
        collapsed = [run for run in runs if run.collapsed]
        lines.append(
            f"| {row.cell} | {row.start} | {len(runs)} | "
            f"{sum(r.rounds[0].merged for r in runs)} | "
            f"{statistics.fmean(len(r.rounds) for r in runs):.2f} "
            f"({max(len(r.rounds) for r in runs)}) | "
            f"{_gain_share(runs):.2f} | {max(r.gain for r in runs):.3f} | "
            f"{statistics.median(r.one_pass - row.bound for r in runs):.2f} | "
            f"{statistics.median(r.final - row.bound for r in runs):.2f} | "
            f"{statistics.fmean(r.rounds[0].visits for r in runs):.0f} | "
            f"{statistics.fmean(r.extra_visits for r in runs):.0f} | "
            f"{1e3 * statistics.fmean(r.seconds_after_first for r in runs):.2f} | "
            f"{len(collapsed)} | "
            + (
                f"{statistics.median(r.rounds[0].icm_energy - row.bound for r in collapsed):.2f}, "
                f"{statistics.median(r.final - row.bound for r in collapsed):.2f}"
                if collapsed
                else "-"
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the study and print its table and the decision."""
    started = time.perf_counter()
    rows = study()
    print(table(rows))
    worst = max(rows, key=lambda row: _gain_share(row.runs))
    print(
        f"\nlargest share gaining >= {GAIN} nats: {_gain_share(worst.runs):.2f} "
        f"({worst.cell}, {worst.start}); propose iterating: "
        f"{_gain_share(worst.runs) >= SHARE}"
    )
    print(f"{time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
