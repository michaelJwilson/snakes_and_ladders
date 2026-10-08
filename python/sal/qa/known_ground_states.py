"""The tuned Potts anneal against fixtures whose ground state is known (issue #1378, Part A).

Every anneal number on record before this study predates the Rust loop
(#1370), the polish from ``best`` with the merge (#1375), the Wolff budget
(#1357) and the ``POLISHED_GAP`` tuner (#1339). This script measures the anneal
as it now stands against an answer that does not come from a solver under
test, and reports where it misses.

**The fixtures and their referees.**

- zero field, ``q = 3``: the closed form ``-J|E|``
  (:func:`~sal.search.ground_state.uniform_ground_energy`), square lattices of
  144 to 4,096 sites at the critical coupling;
- ``q = 2`` in a field: the exact graph cut
  (:func:`~sal.search.maxflow.ising_ground_state`), the seeded sizing family
  at 144 to 4,096 sites and `spatio_only/release` at 5,041;
- the planted glass, `planted_glass/ci`: 18 sites, ``+/- J``, the enumerated
  minimum over all ``2^18`` labellings; the planted state scores above it and
  is reported as the bound it is;
- ``q = 3`` and ``q = 4`` in a field: the optimum
  :func:`sal.sandbox.potts_mip.mip` proves, at the sizes it proves within
  seconds (64 x 64 at ``q = 3`` in 3.5 s, 48 x 48 at ``q = 4`` in 3.5 s).

**The methods.** The anneal is
:func:`~sal.sample.potts_mcmc.anneal_potts` on a schedule
:func:`~sal.sample.tune.tune_schedule` chooses per instance and budget, racing,
ranked by :attr:`~sal.sample.tune.Criterion.POLISHED_GAP` (ICM then the label
merge), on :data:`TUNING_SEED`, which no reported run draws from; then run
under ``budget=`` with :attr:`~sal.sample.schedule.Polish.ICM_MERGE`. Its move
is heat-bath Swendsen-Wang composed with a Gibbs sweep, and in zero field a
heat-bath Wolff run beside it; on the glass, whose couplings carry both signs
and which no cluster move takes, the single-site heat bath. The baselines are
alpha-expansion from the field's argmax then ICM, and restart-ICM: index-order
descents from uniform draws until the budget is spent, the best kept.

**The budget.** Site visits, three levels :data:`LEVELS` apart from
alpha-expansion+ICM's own spend; on the glass, where alpha-expansion refuses a
negative coupling, from :data:`GLASS_BASE_SWEEPS` sweeps. The anneal's
``spent`` holds its polish, so it exceeds the level by ``polish_spent`` and
by at most one step; the pilots' spend is ``tuned`` and reported apart.

**The workload cell.** Beside the fixtures, :func:`workload` builds one
instance modelled on a downstream workload: 3,000 sites on a periodic
triangular lattice (degree 6), ``J = 1``, ``q = 4``, contiguous planted domains
of shares :data:`WORKLOAD_SHARES`, and a unary whose top-two margin is
``0.5 J`` times the degree with the argmax wrong on 30% of sites. It has no
closed form; it is read against the TRW-S bound (:func:`sal.search.trws.trws`),
the best energy any arm found, and the Hamming distance to the planting after
the best renaming, and it adds the downstream record's arms at 4,000 sweeps'
visits. Its instances stay here until #1390 makes canonical fixtures.

Run as ``python -m sal.qa.known_ground_states``; it prints one row per cell.
"""

from __future__ import annotations

import functools
import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove, Recolour, anneal_potts
from sal.sample.schedule import Polish, ScheduleParams
from sal.sample.tune import SCHEDULE_GRID, Criterion, tune_schedule
from sal.sandbox.potts_mip import mip
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import (
    SWEEP_SPREAD,
    Problem,
    lattice_rung,
    rung_field,
    uniform_ground_energy,
)
from sal.search.icm import iterated_conditional_modes, merge_labels
from sal.search.maxflow import ising_ground_state
from sal.sim.canonical import FrustratedLatticeParams
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import SpatioOnlyParams, critical_coupling, spatio_only_field

#: The seeds the reported runs draw from.
REPORTED_SEEDS = tuple(range(32))
#: The seed the schedule pilots draw from, held out of every reported run.
TUNING_SEED = 1000
#: The budget ladder, as multiples of the base level.
LEVELS = (1, 4, 16)
#: The glass's base level in heat-bath sweeps: alpha-expansion has none there.
GLASS_BASE_SWEEPS = 20
#: HiGHS's time limit per proof; every instance below proves inside it.
MIP_SECONDS = 60.0
#: Relative tolerance under which an energy is the known optimum.
EXACT = 1e-9
#: The second hit criterion: within 1% of ``|E*|``.
NEAR = 0.01

#: Zero-field sides: 144 to 4,096 sites.
ZERO_FIELD_SIDES = (12, 24, 48, 64)
#: ``q = 2`` sides of the seeded sizing family; `spatio_only/release` adds 5,041.
CUT_SIDES = (12, 24, 48, 64)
#: ``q = 3`` sides the MIP proves within seconds.
MIP_Q3_SIDES = (12, 24, 48, 64)
#: ``q = 4`` sides the MIP proves within seconds.
MIP_Q4_SIDES = (12, 24, 48)
#: The seed of every seeded field.
FIELD_SEED = 0


@dataclass(frozen=True)
class Instance:
    """One fixture, its known optimum, and the anneal moves run on it."""

    name: str
    family: str
    problem: Problem
    optimum: float
    moves: tuple[str, ...]


#: Anneal arms: move set and recolouring.
ANNEAL_MOVES: dict[str, tuple[PottsMove, Recolour]] = {
    "anneal-sw": (PottsMove.SWENDSEN_WANG, Recolour.HEAT_BATH),
    "anneal-wolff": (PottsMove.WOLFF, Recolour.HEAT_BATH),
    "anneal-gibbs": (PottsMove.SINGLE_SITE, Recolour.PER_MOVE),
}


def _square(side: int, n_states: int) -> Problem:
    """A square lattice at ``n_states``' critical coupling in the sizing family's tilt, at any ``q``."""
    graph = lattice_graph(
        (side, side), BoundaryCondition.OPEN, critical_coupling(n_states)
    )
    sizes = np.random.default_rng(FIELD_SEED).lognormal(
        0.0, SWEEP_SPREAD, size=side * side
    )
    field = spatio_only_field(np.linspace(-1.0, 1.0, n_states), sizes)
    return Problem(graph, field, n_states)


def _enumerated(problem: Problem) -> float:
    """The minimum energy over every two-label configuration, vectorized over all ``2^n``."""
    n = problem.n_nodes
    states = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    first, second = problem.graph.edge_index[:, 0], problem.graph.edge_index[:, 1]
    agree = states[:, first] == states[:, second]
    unary = problem.field[np.arange(n)[None, :], states].sum(axis=1)
    values = -unary - agree @ np.asarray(problem.graph.edge_coupling, dtype=float)
    return float(values.min())


def cut_instance(side: int) -> Instance:
    """The seeded ``q = 2`` sizing rung at ``side``, refereed by the exact graph cut."""
    problem = lattice_rung(side, 2, seed=FIELD_SEED).problem
    return Instance(
        f"cut-{side}x{side}-q2",
        "q2 cut",
        problem,
        float(ising_ground_state(problem.graph, problem.field).energy),
        ("anneal-sw",),
    )


def instances() -> list[Instance]:
    """Every fixture of the grid, with its optimum computed by its own referee."""
    out: list[Instance] = []
    for side in ZERO_FIELD_SIDES:
        rung = lattice_rung(side, 3)
        out.append(
            Instance(
                f"zero-{side}x{side}-q3",
                "zero field",
                rung.problem,
                uniform_ground_energy(rung),
                ("anneal-sw", "anneal-wolff"),
            )
        )
    out.extend(cut_instance(side) for side in CUT_SIDES)
    params: SpatioOnlyParams = fixture("spatio_only", "release").params
    field, _ = rung_field(params, 2)
    problem = Problem(params.graph, field, 2)
    out.append(
        Instance(
            "cut-spatio-release-q2",
            "q2 cut",
            problem,
            float(ising_ground_state(problem.graph, problem.field).energy),
            ("anneal-sw",),
        )
    )
    glass_params: FrustratedLatticeParams = fixture("planted_glass", "ci").params
    (frustration,) = glass_params.glass_frustrations
    glass = glass_params.glass(frustration, np.random.default_rng(glass_params.seed))
    problem = Problem(glass.graph, np.zeros((glass.graph.n_nodes, 2)), 2)
    out.append(
        Instance(
            "glass-ci-18",
            "planted glass",
            problem,
            _enumerated(problem),
            ("anneal-gibbs",),
        )
    )
    for n_states, sides in ((3, MIP_Q3_SIDES), (4, MIP_Q4_SIDES)):
        for side in sides:
            problem = _square(side, n_states)
            proof = mip(problem.graph, problem.field, time_limit=MIP_SECONDS)
            if not proof.proven:
                continue
            out.append(
                Instance(
                    f"mip-{side}x{side}-q{n_states}",
                    f"q{n_states} MIP",
                    problem,
                    proof.value,
                    ("anneal-sw",),
                )
            )
    return out


@dataclass(frozen=True)
class Run:
    """One method's run on one instance at one level and seed."""

    energy: float
    spent: int
    polish_spent: int
    seconds: float
    stages: tuple[tuple[str, float], ...] = ()


def expansion_icm(problem: Problem) -> Run:
    """Alpha-expansion from the field's argmax, then index-order ICM; it draws nothing."""
    started = time.perf_counter()
    expanded = alpha_expansion(
        problem.graph, problem.field, backend=Backend.RUST, n_states=problem.n_states
    )
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        np.random.default_rng(0),
        start=expanded.labelling,
        n_states=problem.n_states,
    )
    per_sweep = problem.visits_per_sweep
    spent = expanded.cycles * problem.n_states * per_sweep + settled.sweeps * per_sweep
    return Run(settled.energy, spent, 0, time.perf_counter() - started)


def restart_icm(problem: Problem, budget: int, rng: np.random.Generator) -> Run:
    """Index-order ICM from uniform draws until ``budget`` site visits are spent, the best kept."""
    per_sweep = problem.visits_per_sweep
    started = time.perf_counter()
    best = np.inf
    spent = 0
    while budget - spent >= per_sweep:
        settled = iterated_conditional_modes(
            problem.graph,
            problem.field,
            rng,
            start=rng.integers(0, problem.n_states, size=problem.n_nodes),
            max_iterations=(budget - spent) // per_sweep,
            n_states=problem.n_states,
        )
        spent += settled.sweeps * per_sweep
        best = min(best, settled.energy)
    return Run(float(best), spent, 0, time.perf_counter() - started)


def _polish(problem: Problem, labelling: np.ndarray) -> np.ndarray:
    """What ``POLISHED_GAP`` ranks a pilot by: ICM then the label merge, ``Polish.ICM_MERGE``'s pair."""
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        np.random.default_rng(0),
        start=labelling,
        n_states=problem.n_states,
    )
    return merge_labels(problem.graph, problem.field, settled.labelling).labelling


def tuned_schedule(
    problem: Problem, arm: str, budget: int
) -> tuple[ScheduleParams, int]:
    """The schedule the racing pilots choose at ``budget`` per candidate, and their spend."""
    move, recolour = ANNEAL_MOVES[arm]
    composed = 2 if move is not PottsMove.SINGLE_SITE else 1
    steps = max(2, budget // (composed * problem.visits_per_sweep))
    tuned = tune_schedule(
        problem.graph,
        problem.field,
        move=move,
        recolour=recolour,
        budget=Budget(Cost.SITE_VISITS, len(SCHEDULE_GRID) * problem.n_nodes * steps),
        criterion=Criterion.POLISHED_GAP,
        rng=np.random.default_rng(TUNING_SEED),
        racing=True,
        polish=functools.partial(_polish, problem),
    )
    return tuned.params, tuned.spent


def anneal(
    problem: Problem,
    arm: str,
    params: ScheduleParams,
    budget: int,
    rng: np.random.Generator,
) -> Run:
    """One anneal on ``params`` under a site-visit budget, polished by ``Polish.ICM_MERGE``."""
    move, recolour = ANNEAL_MOVES[arm]
    started = time.perf_counter()
    run = anneal_potts(
        problem.graph,
        problem.field,
        params.build(max(2, budget // problem.visits_per_sweep)),
        rng,
        move=move,
        recolour=recolour,
        budget=Budget(Cost.SITE_VISITS, budget),
        polish=Polish.ICM_MERGE,
    )
    return Run(
        float(run.energy),
        int(run.spent),
        int(run.polish_spent),
        time.perf_counter() - started,
        tuple((stage.name, float(stage.energy)) for stage in run.stages),
    )


@dataclass(frozen=True)
class Cell:
    """One method on one instance at one level, over the reported seeds."""

    instance: str
    family: str
    n_nodes: int
    optimum: float
    method: str
    level: int
    budget: int
    exact: float
    near: float
    median_gap: float
    spent: int
    polish_spent: int
    seconds: float
    stages: tuple[tuple[str, float], ...]
    schedule: str = ""
    tuning_spent: int = 0


def _gap(value: float, optimum: float) -> float:
    """``(E - E*) / |E*|``, the relative excess over the known optimum."""
    return (value - optimum) / abs(optimum)


def cell(
    instance: Instance,
    method: str,
    level: int,
    budget: int,
    runs: Sequence[Run],
    *,
    schedule: str = "",
    tuning_spent: int = 0,
) -> Cell:
    """The per-cell summary of ``runs``: hit fractions, median gap, spend, wall and stages."""
    gaps = np.array([_gap(run.energy, instance.optimum) for run in runs])
    stages: tuple[tuple[str, float], ...] = ()
    if runs[0].stages:
        names = [name for name, _ in runs[0].stages]
        stages = tuple(
            (name, float(np.median([run.stages[k][1] for run in runs])))
            for k, name in enumerate(names)
        )
    return Cell(
        instance=instance.name,
        family=instance.family,
        n_nodes=instance.problem.n_nodes,
        optimum=instance.optimum,
        method=method,
        level=level,
        budget=budget,
        exact=float(np.mean(gaps <= EXACT)),
        near=float(np.mean(gaps <= NEAR)),
        median_gap=float(np.median(gaps)),
        spent=int(np.median([run.spent for run in runs])),
        polish_spent=int(np.median([run.polish_spent for run in runs])),
        seconds=float(np.median([run.seconds for run in runs])),
        stages=stages,
        schedule=schedule,
        tuning_spent=tuning_spent,
    )


def measure_instance(
    instance: Instance,
    *,
    seeds: Sequence[int] = REPORTED_SEEDS,
    levels: Sequence[int] = LEVELS,
) -> list[Cell]:
    """Every method at every level on ``instance``; each run on its own seeded generator."""
    problem = instance.problem
    out: list[Cell] = []
    if instance.family == "planted glass":
        base = GLASS_BASE_SWEEPS * problem.visits_per_sweep
    else:
        reference = expansion_icm(problem)
        base = reference.spent
        out.append(cell(instance, "ae+icm", 1, base, [reference]))
    for level in levels:
        budget = level * base
        runs = [
            restart_icm(problem, budget, np.random.default_rng([seed, 1]))
            for seed in seeds
        ]
        out.append(cell(instance, "restart-icm", level, budget, runs))
        for arm in instance.moves:
            params, pilots = tuned_schedule(problem, arm, budget)
            runs = [
                anneal(problem, arm, params, budget, np.random.default_rng([seed, 2]))
                for seed in seeds
            ]
            label = f"{params.shape.value} {params.t_start:g}->{params.t_end:g}"
            out.append(
                cell(
                    instance,
                    arm,
                    level,
                    budget,
                    runs,
                    schedule=label,
                    tuning_spent=pilots,
                )
            )
    return out


def table(cells: Sequence[Cell]) -> str:
    """The cells as a markdown table, one row per cell."""
    header = (
        "| instance | n | E* | method | level | budget | exact | <=1% | median gap "
        "| spent | polish | wall ms | stages (median E) | schedule |"
    )
    rows = [header, "|" + " --- |" * 14]
    for c in cells:
        stages = ", ".join(f"{name} {value:.6g}" for name, value in c.stages)
        rows.append(
            f"| {c.instance} | {c.n_nodes} | {c.optimum:.10g} | {c.method} | "
            f"x{c.level} | {c.budget} | {c.exact:.2f} | {c.near:.2f} | "
            f"{c.median_gap:.3e} | {c.spent} | {c.polish_spent} | "
            f"{1e3 * c.seconds:.2f} | {stages} | {c.schedule} |"
        )
    return "\n".join(rows)


def misses(cells: Sequence[Cell], top: int = LEVELS[-1]) -> list[Cell]:
    """The anneal cells at the top level that do not reach the known optimum on every seed."""
    return [
        c
        for c in cells
        if c.method.startswith("anneal") and c.level == top and c.exact < 1.0
    ]


def measure(
    chosen: Callable[[Instance], bool] = lambda _: True,
) -> list[Cell]:
    """Every cell of the grid on the instances ``chosen`` admits."""
    return list(
        itertools.chain.from_iterable(
            measure_instance(each) for each in instances() if chosen(each)
        )
    )


# --- the downstream workload cell: no exact optimum, a bound and a planting ---

#: The workload's lattice: periodic triangular, every site of degree 6, 3,000 sites.
WORKLOAD_SHAPE = (50, 60)
#: ``q`` and the coupling of the workload.
WORKLOAD_STATES, WORKLOAD_COUPLING = 4, 1.0
#: Planted label shares: one label at 2%, the other three spanning 1:10.
WORKLOAD_SHARES = (0.02, 0.07, 0.21, 0.70)
#: The unary top-two margin, ``0.5 * J * degree``.
WORKLOAD_MARGIN = 0.5 * WORKLOAD_COUPLING * 6
#: Share of sites whose unary argmax is a wrong label, the planted one second.
WORKLOAD_MISLEAD = 0.30
#: Per-entry Gaussian jitter on the unary, so energies are continuous and ties
#: between labellings are measure zero; the margin stays 3.0 in mean.
WORKLOAD_JITTER = 0.25
#: The seed the workload instance is drawn from.
WORKLOAD_SEED = 7
#: The workload's reported seeds: fewer, its reference arms run 4,000 sweeps.
WORKLOAD_SEEDS = tuple(range(8))
#: The downstream reference runs: 4,000 sweeps' site visits.
WORKLOAD_SWEEPS = 4000


@dataclass(frozen=True)
class Workload:
    """The workload instance, its planting and its TRW-S bound."""

    problem: Problem
    planted: np.ndarray
    bound: float
    trws_energy: float
    trws_seconds: float


def workload() -> Workload:
    """The hex-lattice workload: contiguous planted domains, a misleading unary on 30% of sites.

    The planting thresholds a smooth random surface (eight random plane waves)
    at the quantiles :data:`WORKLOAD_SHARES` names, so domains are contiguous
    and their sizes exact. Each site's unary puts :data:`WORKLOAD_MARGIN` on
    its top label over the second and the same below the second on the rest;
    the top is the planted label except on :data:`WORKLOAD_MISLEAD` of sites,
    where a uniform wrong label is top and the planted one second;
    :data:`WORKLOAD_JITTER` is then added to every entry.
    """
    from sal.search.trws import trws
    from sal.sim.graph import triangular_lattice_graph

    rng = np.random.default_rng(WORKLOAD_SEED)
    rows, columns = WORKLOAD_SHAPE
    graph = triangular_lattice_graph(
        WORKLOAD_SHAPE, BoundaryCondition.PERIODIC, WORKLOAD_COUPLING
    )
    r, c = np.divmod(np.arange(rows * columns), columns)
    surface = np.zeros(rows * columns)
    for _ in range(8):
        kr, kc = rng.integers(1, 4), rng.integers(1, 4)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        surface += np.cos(2.0 * np.pi * (kr * r / rows + kc * c / columns) + phase)
    cuts = np.quantile(surface, np.cumsum(WORKLOAD_SHARES)[:-1])
    planted = np.searchsorted(cuts, surface).astype(np.int64)
    n = rows * columns
    field = np.full((n, WORKLOAD_STATES), -WORKLOAD_MARGIN)
    misled = rng.random(n) < WORKLOAD_MISLEAD
    wrong = (planted + rng.integers(1, WORKLOAD_STATES, size=n)) % WORKLOAD_STATES
    top = np.where(misled, wrong, planted)
    second = np.where(misled, planted, (planted + 1) % WORKLOAD_STATES)
    field[np.arange(n), second] = 0.0
    field[np.arange(n), top] = WORKLOAD_MARGIN
    field += rng.normal(0.0, WORKLOAD_JITTER, size=field.shape)
    problem = Problem(graph, field, WORKLOAD_STATES)
    started = time.perf_counter()
    bounded = trws(graph, field)
    return Workload(
        problem,
        planted,
        float(bounded.bound),
        float(bounded.energy),
        time.perf_counter() - started,
    )


@dataclass(frozen=True)
class WorkloadRow:
    """One arm on the workload, medians over :data:`WORKLOAD_SEEDS`."""

    method: str
    budget: int
    to_bound: float
    to_best: float
    accuracy: float
    spent: int
    polish_spent: int
    seconds: float
    best_energy: float


def _reference_anneal(
    problem: Problem,
    move: PottsMove | tuple[PottsMove, ...],
    recolour: Recolour,
    polish: Polish | None,
    rng: np.random.Generator,
) -> tuple[Run, np.ndarray]:
    """The downstream reference run: the record's 2.0 to 0.05 exponential over 4,000 sweeps' visits."""
    from sal.search.ground_state import ANNEAL_SCHEDULE

    budget = WORKLOAD_SWEEPS * problem.visits_per_sweep
    started = time.perf_counter()
    run = anneal_potts(
        problem.graph,
        problem.field,
        ANNEAL_SCHEDULE.build(WORKLOAD_SWEEPS),
        rng,
        move=move,
        recolour=recolour,
        budget=Budget(Cost.SITE_VISITS, budget),
        polish=polish,
    )
    return (
        Run(
            float(run.energy),
            int(run.spent),
            int(run.polish_spent),
            time.perf_counter() - started,
        ),
        np.asarray(run.best),
    )


def measure_workload(
    seeds: Sequence[int] = WORKLOAD_SEEDS,
) -> tuple[Workload, list[WorkloadRow]]:
    """Every arm on the workload: the grid's, and the downstream record's at 4,000 sweeps."""
    from sal.search.spatio_sequential import label_accuracy

    work = workload()
    problem = work.problem
    q = problem.n_states
    raw: list[tuple[str, int, list[Run], list[np.ndarray]]] = []

    started = time.perf_counter()
    expanded = alpha_expansion(
        problem.graph, problem.field, backend=Backend.RUST, n_states=q
    )
    ae_seconds = time.perf_counter() - started
    ae_spent = expanded.cycles * q * problem.visits_per_sweep
    raw.append(
        (
            "ae",
            ae_spent,
            [Run(float(expanded.energy), ae_spent, 0, ae_seconds)],
            [expanded.labelling],
        )
    )
    reference = expansion_icm(problem)
    base = reference.spent
    settled = iterated_conditional_modes(
        problem.graph,
        problem.field,
        np.random.default_rng(0),
        start=expanded.labelling,
        n_states=q,
    )
    raw.append(("ae+icm", base, [reference], [settled.labelling]))
    for level in LEVELS:
        budget = level * base
        runs = [
            restart_icm(problem, budget, np.random.default_rng([seed, 1]))
            for seed in seeds
        ]
        raw.append((f"restart-icm x{level}", budget, runs, []))
        for arm in ("anneal-sw", "anneal-wolff"):
            params, _ = tuned_schedule(problem, arm, budget)
            pairs = []
            for seed in seeds:
                move, recolour = ANNEAL_MOVES[arm]
                clock = time.perf_counter()
                run = anneal_potts(
                    problem.graph,
                    problem.field,
                    params.build(max(2, budget // problem.visits_per_sweep)),
                    np.random.default_rng([seed, 2]),
                    move=move,
                    recolour=recolour,
                    budget=Budget(Cost.SITE_VISITS, budget),
                    polish=Polish.ICM_MERGE,
                )
                pairs.append(
                    (
                        Run(
                            float(run.energy),
                            int(run.spent),
                            int(run.polish_spent),
                            time.perf_counter() - clock,
                        ),
                        np.asarray(run.best),
                    )
                )
            raw.append(
                (
                    f"{arm} x{level}",
                    budget,
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                )
            )
    references: dict[
        str, tuple[PottsMove | tuple[PottsMove, ...], Recolour, Polish | None]
    ] = {
        "glauber 4000 sweeps": ((PottsMove.SINGLE_SITE,), Recolour.PER_MOVE, None),
        "glauber 4000 + icm_merge": (
            (PottsMove.SINGLE_SITE,),
            Recolour.PER_MOVE,
            Polish.ICM_MERGE,
        ),
        "wolff-hb 4000 sweeps' visits": (
            (PottsMove.WOLFF_HEAT_BATH,),
            Recolour.PER_MOVE,
            None,
        ),
        "wolff+gibbs 4000 sweeps' visits": (
            PottsMove.WOLFF,
            Recolour.HEAT_BATH,
            None,
        ),
        "wolff-hb + icm_merge": (
            (PottsMove.WOLFF_HEAT_BATH,),
            Recolour.PER_MOVE,
            Polish.ICM_MERGE,
        ),
    }
    for name, (moves, recolouring, polish) in references.items():
        pairs = [
            _reference_anneal(
                problem, moves, recolouring, polish, np.random.default_rng([seed, 3])
            )
            for seed in seeds
        ]
        raw.append(
            (
                name,
                WORKLOAD_SWEEPS * problem.visits_per_sweep,
                [p[0] for p in pairs],
                [p[1] for p in pairs],
            )
        )
    best = min(
        [work.trws_energy] + [run.energy for _, _, runs, _ in raw for run in runs]
    )
    rows = [
        WorkloadRow(
            method=name,
            budget=budget,
            to_bound=float(np.median([run.energy - work.bound for run in runs])),
            to_best=float(np.median([run.energy - best for run in runs])),
            accuracy=float(
                np.median([label_accuracy(lab, work.planted, q) for lab in labels])
            )
            if labels
            else float("nan"),
            spent=int(np.median([run.spent for run in runs])),
            polish_spent=int(np.median([run.polish_spent for run in runs])),
            seconds=float(np.median([run.seconds for run in runs])),
            best_energy=float(min(run.energy for run in runs)),
        )
        for name, budget, runs, labels in raw
    ]
    return work, rows


def workload_table(work: Workload, rows: Sequence[WorkloadRow]) -> str:
    """The workload's rows as markdown, with the bound and TRW-S's own decode."""
    n = work.problem.n_nodes
    argmax = work.problem.field.argmax(axis=1)
    lines = [
        f"workload: {n} sites, triangular periodic, q = {work.problem.n_states}, "
        f"J = {WORKLOAD_COUPLING}; TRW-S bound {work.bound:.4f}, its decode "
        f"{work.trws_energy:.4f} ({1e3 * work.trws_seconds:.0f} ms); unary argmax "
        f"misses {float(np.mean(argmax != work.planted)):.3f} of sites",
        "",
        "| method | budget | E - bound | E - best | Hamming (aligned) | spent "
        "| polish | wall ms |",
        "|" + " --- |" * 8,
    ]
    for row in rows:
        hamming = "" if np.isnan(row.accuracy) else f"{(1.0 - row.accuracy) * n:.0f}"
        lines.append(
            f"| {row.method} | {row.budget} | {row.to_bound:.4f} | "
            f"{row.to_best:.4f} | {hamming} | {row.spent} | {row.polish_spent} | "
            f"{1e3 * row.seconds:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Print the grid and the top-level misses."""
    started = time.perf_counter()
    cells = measure()
    print(table(cells))
    print()
    print("top-level misses:")
    for c in misses(cells):
        print(
            f"  {c.instance} {c.method}: exact {c.exact:.2f}, <=1% {c.near:.2f}, "
            f"median gap {c.median_gap:.3e} at {c.budget} site visits"
        )
    work, rows = measure_workload()
    print()
    print(workload_table(work, rows))
    print(f"wall {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
