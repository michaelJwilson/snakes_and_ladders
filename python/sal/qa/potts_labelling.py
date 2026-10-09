"""The Potts labelling step, solver by solver, on one instance from one field and start (issue #1413).

A downstream package labels sites with a hidden Markov random field: a
per-site field built from emissions, single-site descent with a minimum-sites
floor, and a whole-label merge. This study runs sal's solvers for that step on
the fixture that mirrors the downstream instance (``potts_labelling``, a
:mod:`sal.sim.potts_cell` cell), every arm from the same field and the same
start, and reports what each arm reaches and what it costs.

**The arms.** :data:`ARMS`, each a call this package already holds:

- ``argmax``: the field's argmax, no coupling;
- ``icm``: :func:`~sal.search.icm.iterated_conditional_modes` in index
  order to a clean sweep, no floor;
- ``icm-floor``: the same with ``min_sites`` at the declared floor, the
  uniform floor (:attr:`~sal.search.icm.FloorPolicy.UNIFORM`);
- ``icm+floor-smallest``: ``icm``, then
  :func:`~sal.search.icm.merge_small_labels` under
  :attr:`~sal.search.icm.FloorPolicy.SMALLEST_FIRST_BEST_FIELD`;
- ``ae+icm``: :func:`~sal.search.alpha_expansion.alpha_expansion` from the
  start, then ``icm``;
- ``trws``: :func:`~sal.search.trws.trws`, its decoded labelling; its bound
  is the referee every gap is read against;
- ``icm+merge``: ``icm``, then the greedy whole-label merge
  (:func:`sal.sample.potts_mcmc.chains.merge_labels`, the NumPy oracle of
  :attr:`~sal.sample.schedule.Polish.ICM_MERGE`'s merge);
- ``anneal+icm_merge``: :func:`~sal.sample.potts_mcmc.anneal_potts`,
  single-site heat bath on :data:`~sal.search.ground_state.ANNEAL_SCHEDULE`
  over :data:`ANNEAL_SWEEPS` sweeps, polished by ``Polish.ICM_MERGE``.

**Per arm.** Energy and its gap to the TRW-S bound; sites unlike the planted
labelling under the best renaming
(:func:`~sal.search.spatio_sequential.label_accuracy`); the iterations and
termination the solver reports; wall time; the peak of Python-heap
allocations under :mod:`tracemalloc`, which sees NumPy's buffers and not the
Rust extension's own heap; and the Rust crossings, counted as calls into
``sal.oxisal`` functions and methods plus one per extension object built.
Peak and crossings are read on a run each, then wall time on a third run
without the tracer or the counter, after any compilation.

**The floor.** ``min_sites`` is :data:`FLOOR_SHARE` of the sites, rounded:
the downstream configuration's floor over the stream's site count.

**The instances.** ``potts_labelling`` at a tier, or one of its ``stress``
variants (``easy``, ``hard``, ``dev_6000``), which span the downstream
stream's problems.

Run as ``python -m sal.qa.potts_labelling [tier]`` (``stress`` by default); it
prints one row per arm and start, at ``stress`` for each variant too.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Any

import numpy as np

from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Termination
from sal.sample.potts_mcmc import PottsMove, Recolour, anneal_potts
from sal.sample.potts_mcmc.chains import merge_labels
from sal.sample.schedule import Polish
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import ANNEAL_SCHEDULE, Problem
from sal.search.icm import (
    FloorPolicy,
    iterated_conditional_modes,
    merge_small_labels,
)
from sal.search.spatio_sequential import label_accuracy
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.potts import energy
from sal.sim.potts_cell import PlantedPotts

#: The floor over the sites: the downstream configuration's 50 at 3,000.
FLOOR_SHARE = 50 / 3000
#: The anneal's sweeps: the downstream record's 4,000.
ANNEAL_SWEEPS = 4000
#: The starts every arm runs from: the field's argmax, and a uniform draw.
STARTS = ("argmax", "uniform")
#: The seed of the uniform start, the floor's uniforms and the anneal.
SEED = 0


@dataclass(frozen=True)
class Solved:
    """One arm's labelling, the iterations its solver reports, and why it stopped."""

    labelling: np.ndarray
    iterations: int
    termination: Termination


@dataclass(frozen=True)
class Row:
    """One arm from one start, as measured.

    Parameters
    ----------
    arm, start : str
        The arm and the start's name.
    energy : float
        :func:`~sal.sim.potts.energy` of the labelling.
    gap : float
        ``energy`` less the TRW-S bound, nats.
    unlike : int
        Sites unlike the planted labelling under the best renaming.
    labels : int
        Labels in use.
    iterations : int
        The solver's own iterations: sweeps, cycles, or TRW-S passes.
    termination : str
        The :class:`~sal.opt.termination.Stop` it left by.
    seconds : float
        Wall time, untraced.
    peak_bytes : int
        Peak Python-heap allocation under :mod:`tracemalloc`.
    crossings : int
        Rust calls and extension objects built.
    """

    arm: str
    start: str
    energy: float
    gap: float
    unlike: int
    labels: int
    iterations: int
    termination: str
    seconds: float
    peak_bytes: int
    crossings: int


@dataclass(frozen=True)
class Instance:
    """The fixture's instance, its floor and the TRW-S bound every gap is read against."""

    cell: PlantedPotts
    min_sites: int
    bound: float
    trws_seconds: float


def instance(tier: str = "stress", variant: str | None = None) -> Instance:
    """The ``potts_labelling`` cell at ``tier``, or its ``variant``, with its floor and TRW-S bound.

    Returns
    -------
    Instance
    """
    params = fixture("potts_labelling", tier).params
    cell = (params if variant is None else params.variant(variant)).instance()
    started = time.perf_counter()
    bound = trws(cell.graph, cell.field).bound
    seconds = time.perf_counter() - started
    floor = max(1, round(FLOOR_SHARE * cell.graph.n_nodes))
    return Instance(cell, floor, float(bound), seconds)


def start_of(cell: PlantedPotts, name: str) -> np.ndarray:
    """The start ``name`` names: the field's argmax, or a uniform draw from :data:`SEED`.

    Raises
    ------
    ValueError
        If ``name`` is not one of :data:`STARTS`.
    """
    if name == "argmax":
        return np.argmax(cell.field, axis=1).astype(np.int64)
    if name == "uniform":
        return np.random.default_rng([SEED, 1]).integers(
            0, cell.n_states, cell.graph.n_nodes
        )
    msg = f"start {name!r} is not one of {STARTS}"
    raise ValueError(msg)


def _icm(cell: PlantedPotts, start: np.ndarray, min_sites: int = 0) -> Solved:
    run = iterated_conditional_modes(
        cell.graph,
        cell.field,
        np.random.default_rng([SEED, 2]),
        start=start,
        min_sites=min_sites,
    )
    return Solved(np.asarray(run.labelling), run.sweeps, run.termination)


def _argmax(work: Instance, start: np.ndarray) -> Solved:
    del start
    return Solved(
        np.argmax(work.cell.field, axis=1).astype(np.int64),
        0,
        Termination.after(0, converged=True),
    )


def _plain(work: Instance, start: np.ndarray) -> Solved:
    return _icm(work.cell, start)


def _floored(work: Instance, start: np.ndarray) -> Solved:
    return _icm(work.cell, start, work.min_sites)


def _smallest_first(work: Instance, start: np.ndarray) -> Solved:
    descended = _icm(work.cell, start)
    run = merge_small_labels(
        work.cell.graph,
        work.cell.field,
        descended.labelling,
        np.random.default_rng([SEED, 3]),
        min_sites=work.min_sites,
        policy=FloorPolicy.SMALLEST_FIRST_BEST_FIELD,
    )
    return Solved(
        np.asarray(run.labelling), descended.iterations + run.sweeps, run.termination
    )


def _expansion(work: Instance, start: np.ndarray) -> Solved:
    expanded = alpha_expansion(work.cell.graph, work.cell.field, start=start)
    descended = _icm(work.cell, np.asarray(expanded.labelling))
    return Solved(
        descended.labelling,
        expanded.cycles + descended.iterations,
        descended.termination,
    )


def _trws(work: Instance, start: np.ndarray) -> Solved:
    del start
    run = trws(work.cell.graph, work.cell.field)
    return Solved(
        np.asarray(run.labelling), run.termination.iterations, run.termination
    )


def _merged(work: Instance, start: np.ndarray) -> Solved:
    descended = _icm(work.cell, start)
    labels, rounds = merge_labels(
        work.cell.graph, np.asarray(work.cell.field), descended.labelling
    )
    return Solved(
        labels,
        descended.iterations + rounds,
        Termination.after(rounds, converged=True),
    )


def _annealed(work: Instance, start: np.ndarray) -> Solved:
    graph = work.cell.graph
    visits = Problem(graph, work.cell.field, work.cell.n_states).visits_per_sweep
    run = anneal_potts(
        graph,
        work.cell.field,
        ANNEAL_SCHEDULE.build(ANNEAL_SWEEPS),
        np.random.default_rng([SEED, 4]),
        move=(PottsMove.SINGLE_SITE,),
        recolour=Recolour.PER_MOVE,
        start=start,
        budget=Budget(Cost.SITE_VISITS, ANNEAL_SWEEPS * visits),
        polish=Polish.ICM_MERGE,
    )
    return Solved(np.asarray(run.best), run.n_sweeps, run.termination)


#: Every arm by name, in the order the table prints them.
ARMS: dict[str, Callable[[Instance, np.ndarray], Solved]] = {
    "argmax": _argmax,
    "icm": _plain,
    "icm-floor": _floored,
    "icm+floor-smallest": _smallest_first,
    "ae+icm": _expansion,
    "trws": _trws,
    "icm+merge": _merged,
    "anneal+icm_merge": _annealed,
}


@contextmanager
def counted() -> Iterator[dict[str, int]]:
    """Count the calls into ``sal.oxisal`` made inside the block, and the extension objects they ran on.

    A profile hook sees each call of a compiled function or method; an
    extension object built by its constructor is counted once, the first time
    one of its methods is called.

    Yields
    ------
    dict[str, int]
        ``calls`` and ``objects``, filled as the block runs.
    """
    tally = {"calls": 0, "objects": 0}
    seen: set[int] = set()

    def hook(frame: FrameType, event: str, arg: Any) -> None:
        del frame
        if event != "c_call":
            return
        owner = getattr(arg, "__self__", None)
        if getattr(arg, "__module__", None) == "sal.oxisal":
            tally["calls"] += 1
        elif owner is not None and type(owner).__module__ == "sal.oxisal":
            tally["calls"] += 1
            if id(owner) not in seen:
                seen.add(id(owner))
                tally["objects"] += 1

    sys.setprofile(hook)
    try:
        yield tally
    finally:
        sys.setprofile(None)


def measure(work: Instance, arm: str, start: str) -> Row:
    """``arm`` from ``start`` on ``work``: one traced run, one counted, then one untraced for the wall.

    The first two compile whatever ``numba`` kernel the arm reaches, so no
    compilation lands in the wall.

    Returns
    -------
    Row
    """
    solve = ARMS[arm]
    begin = start_of(work.cell, start)
    tracemalloc.start()
    solve(work, begin.copy())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    with counted() as tally:
        solve(work, begin.copy())
    clock = time.perf_counter()
    solved = solve(work, begin.copy())
    seconds = time.perf_counter() - clock
    cell = work.cell
    value = energy(cell.graph, cell.field, solved.labelling)
    return Row(
        arm=arm,
        start=start,
        energy=float(value),
        gap=float(value - work.bound),
        unlike=round(
            (1.0 - label_accuracy(solved.labelling, cell.planted, cell.n_states))
            * cell.graph.n_nodes
        ),
        labels=int(np.unique(solved.labelling).size),
        iterations=int(solved.iterations),
        termination=str(solved.termination.reason),
        seconds=seconds,
        peak_bytes=int(peak),
        crossings=tally["calls"] + tally["objects"],
    )


def study(
    tier: str = "stress",
    arms: tuple[str, ...] = tuple(ARMS),
    starts: tuple[str, ...] = STARTS,
    variant: str | None = None,
) -> tuple[Instance, list[Row]]:
    """Every arm from every start on the ``tier`` cell or its ``variant``.

    Returns
    -------
    tuple[Instance, list[Row]]
    """
    work = instance(tier, variant)
    return work, [measure(work, arm, start) for start in starts for arm in arms]


def table(work: Instance, rows: list[Row]) -> str:
    """The rows as markdown, headed by the instance's size, bound and planted energy.

    Returns
    -------
    str
    """
    cell = work.cell
    lines = [
        f"{cell.graph.n_nodes} sites, {len(cell.graph.edges)} edges, q = "
        f"{cell.n_states}, floor {work.min_sites}; TRW-S bound {work.bound:.2f} "
        f"({work.trws_seconds:.2f} s); planted energy "
        f"{cell.planted_energy - work.bound:.2f} above it; Backend default "
        f"{Backend.RUST}",
        "",
        "| start | arm | E - bound | unlike | labels | iterations | stop | ms "
        "| peak MB | crossings |",
        "|" + " --- |" * 10,
    ]
    lines += [
        f"| {r.start} | {r.arm} | {r.gap:.2f} | {r.unlike} | {r.labels} | "
        f"{r.iterations} | {r.termination} | {1e3 * r.seconds:.1f} | "
        f"{r.peak_bytes / 1e6:.2f} | {r.crossings} |"
        for r in rows
    ]
    return "\n".join(lines)


def main() -> None:
    """Print the table at the tier ``argv[1]`` names, ``stress`` by default, then each ``stress`` variant."""
    tier = sys.argv[1] if len(sys.argv) > 1 else "stress"
    work, rows = study(tier)
    print(table(work, rows))
    if tier == "stress":
        for variant in fixture("potts_labelling", tier).params.variants:
            work, rows = study(tier, variant=variant)
            print(f"\n{variant}: " + table(work, rows))


if __name__ == "__main__":
    main()
