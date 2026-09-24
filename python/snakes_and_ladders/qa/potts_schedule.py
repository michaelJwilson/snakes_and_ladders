"""Swendsen-Wang's and Wolff's annealing schedules, tuned on seeds the notebook never reports (issue #1038).

`docs/nb/potts_starts.ipynb` runs both cluster moves on the single-site
annealer's schedule, exponential from 2.0 to 0.05, at 1,000 sweeps' worth of
site visits on `spatio_only/release` at ten states. This script asks whether a
schedule of their own does better, and writes what it finds to
``docs/nb/data/potts_schedule.json`` for the notebook to read, so the search
runs once here and never inside the notebook's execution budget.

**The search.** Per move and per :class:`~snakes_and_ladders.sample.schedule.ScheduleShape`,
Nelder-Mead over ``(log t_start, log t_end, hold)`` minimises the mean energy
the move hands over on :data:`TUNING_SEEDS`, starting at the current
schedule, ``(log 2.0, log 0.05, 0)``, and capped at :data:`EVALUATIONS`
evaluations. The seeds are fixed, so the objective is a deterministic
function of the schedule and the simplex compares like with like. Every
evaluation is written, not only the minimum.

**Wolff at matched spend.** Wolff's step count is the budget's sweep count, so
it spends a fraction of the budget (`search.ground_state`). The matched count
is the one at which Wolff's mean spend on :data:`TUNING_SEEDS` meets the
budget, found by bisection before any reported run and then fixed. It is not
``budget / mean visits per step`` read once: the mean cluster grows with the
step count, from 1.75 sites at 1,000 steps to 277 at 100,000 on seed 5, so a
ratio read at one count misses the spend at another.

**To convergence (issue #1046).** The 40-evaluation cap above ended
Swendsen-Wang's searches at evaluations 39 and 40. :func:`nelder_mead` stops
on a rule instead: the simplex's objective range below one standard error of
the objective on :data:`TUNING_SEEDS`, and every vertex within
:data:`VERTEX_TOLERANCE` of the best per coordinate, with
:data:`CONVERGE_CAP` evaluations as a guard. :func:`match_steps` is #1041's
step-count match, bracketing from either side, and :func:`joint_rounds`
alternates it with a schedule search for a move whose spend depends on its
schedule. `qa.potts_converge` runs them; the search above and its file are
left as #1038 recorded them.

**Threads.** The script pins BLAS to one thread per process, as the suite does.
At this size `sim.potts.energies`' one matrix-vector product took 10 ms under
the host's default threading against 5 us pinned, which was 95% of an
annealed run's wall clock; the pin changes energies by at most 7e-13.

Run as ``uv run --no-sync python -m snakes_and_ladders.qa.potts_schedule``.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.parallel import map_tasks
from snakes_and_ladders.sample.potts_mcmc import PottsMove
from snakes_and_ladders.sample.schedule import ScheduleParams, ScheduleShape
from snakes_and_ladders.search.ground_state import (
    ANNEAL_SCHEDULE,
    Rung,
    run_annealed,
)
from snakes_and_ladders.search.potts_starts import spatio_rung
from snakes_and_ladders.sim.fixtures import REPO_ROOT, fixture

#: Where the result is written and where the notebook reads it.
OUTPUT = REPO_ROOT / "docs" / "nb" / "data" / "potts_schedule.json"
#: The seeds the search reads. The notebook reports seeds 0 to 4, and no
#: schedule is chosen on them.
TUNING_SEEDS = (5, 6, 7, 8, 9)
#: The seeds the notebook reports, held out of the search.
REPORTED_SEEDS = (0, 1, 2, 3, 4)
#: The notebook's budget, in heat-bath sweeps' worth of site visits.
SWEEPS = 1000
#: Evaluations Nelder-Mead may spend per move and shape.
EVALUATIONS = 40
#: The moves whose schedules are tuned.
MOVES = (PottsMove.SWENDSEN_WANG, PottsMove.WOLFF)
#: Worker processes. Two, so the host keeps two cores for the other agent
#: and the CI queue.
WORKERS = 2
#: Bounds on ``(log t_start, log t_end, hold)``: temperatures a decade either
#: side of the current endpoints, and at least a tenth of the steps a ramp.
BOUNDS = (
    (math.log(0.05), math.log(20.0)),
    (math.log(0.005), math.log(1.0)),
    (0.0, 0.9),
)
#: The simplex's first steps from the warm start: halve ``t_start``, double
#: ``t_end``, hold a fifth of the steps.
SIMPLEX_STEPS = (-math.log(2.0), math.log(2.0), 0.2)
#: Relative width at which the bisection for Wolff's matched step count stops.
BISECTION_TOLERANCE = 0.01
#: The thread-count variables pinned to one in every worker.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@functools.cache
def release_rung() -> Rung:
    """``release-q10``: the notebook's rung, from the declared fixture."""
    return spatio_rung(fixture("spatio_only", "release").params, "release-q10")


def solver_budget(rung: Rung) -> Budget:
    """:data:`SWEEPS` heat-bath sweeps' worth of site visits, the notebook's ``SOLVER_BUDGET``."""
    return Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)


def params_of(
    point: Sequence[float] | np.ndarray, shape: ScheduleShape
) -> ScheduleParams:
    """The schedule at a search point ``(log t_start, log t_end, hold)``."""
    return ScheduleParams(
        shape, math.exp(point[0]), math.exp(point[1]), float(point[2])
    )


@dataclass(frozen=True)
class Evaluation:
    """One schedule on the tuning seeds.

    Parameters
    ----------
    params : ScheduleParams
    steps : int | None
        The step count, ``None`` for the budget's rule.
    energies : tuple[float, ...]
        The handed-over energy per seed of :data:`TUNING_SEEDS`.
    spent : tuple[int, ...]
        Site visits per seed.
    threshold : float | None
        Niedermayer's ``E_0``, ``None`` for the graph's own (issue #1046).
    """

    params: ScheduleParams
    steps: int | None
    energies: tuple[float, ...]
    spent: tuple[int, ...]
    threshold: float | None = None

    @property
    def mean(self) -> float:
        """Mean handed-over energy, the objective."""
        return float(np.mean(self.energies))

    @property
    def standard_error(self) -> float:
        """The objective's standard error over the seeds, ``sd / sqrt(n)``."""
        return float(np.std(self.energies, ddof=1) / math.sqrt(len(self.energies)))

    def record(self) -> dict[str, Any]:
        """The evaluation as JSON-ready data; ``threshold`` only where one was set."""
        record = {
            **asdict(self.params),
            "steps": self.steps,
            "energies": list(self.energies),
            "spent": list(self.spent),
            "mean": self.mean,
        }
        if self.threshold is not None:
            record["threshold"] = self.threshold
        return record


def _cell_generator(seed: int) -> np.random.Generator:
    """``default_rng([seed, 0])``: the generator :class:`~snakes_and_ladders.opt.starts.StartsBenchmark` hands cell ``(instance 0, seed)``.

    A run here on it is therefore the notebook's run at that seed.
    """
    return np.random.default_rng([seed, 0])


def handover(
    move: PottsMove,
    params: ScheduleParams,
    steps: int | None,
    rng: np.random.Generator,
    threshold: float | None = None,
) -> tuple[float, int]:
    """One annealed run on the notebook's rung and budget: its energy and spend.

    ``threshold`` is Niedermayer's ``E_0``, ``None`` for the graph's own.
    """
    rung = release_rung()
    run = run_annealed(
        rung,
        solver_budget(rung),
        rng,
        move,
        schedule=params,
        steps=steps,
        threshold=threshold,
    )
    return run.energy, run.spent


def evaluate(
    move: PottsMove,
    params: ScheduleParams,
    steps: int | None = None,
    seeds: Sequence[int] = TUNING_SEEDS,
    threshold: float | None = None,
) -> Evaluation:
    """``params`` on every seed of ``seeds``, serially.

    Raises
    ------
    ValueError
        If a reported seed is among ``seeds``.
    """
    if set(seeds) & set(REPORTED_SEEDS):
        msg = f"seeds {sorted(set(seeds) & set(REPORTED_SEEDS))} are reported, not tuned on"
        raise ValueError(msg)
    runs = [
        handover(move, params, steps, _cell_generator(seed), threshold)
        for seed in seeds
    ]
    return Evaluation(
        params,
        steps,
        tuple(energy for energy, _ in runs),
        tuple(spent for _, spent in runs),
        threshold,
    )


class _CapReached(Exception):
    """Raised by the objective when the evaluation cap is spent."""


def tune(task: tuple[PottsMove, ScheduleShape]) -> list[Evaluation]:
    """Nelder-Mead on one move's schedule of one shape, every evaluation in order.

    The first evaluation is the warm start, the current endpoints with no
    hold; for the exponential shape it is the current schedule.
    """
    move, shape = task
    evaluations: list[Evaluation] = []

    def objective(point: np.ndarray) -> float:
        # SciPy checks `maxfev` between iterations and an iteration may
        # evaluate several points, so the cap is enforced here.
        if len(evaluations) == EVALUATIONS:
            raise _CapReached
        # Each point is scored on the tuning seeds and recorded.
        evaluation = evaluate(move, params_of(point, shape))
        evaluations.append(evaluation)
        return evaluation.mean

    start = np.array(
        [math.log(ANNEAL_SCHEDULE.t_start), math.log(ANNEAL_SCHEDULE.t_end), 0.0]
    )
    simplex = np.vstack([start, start + np.diag(SIMPLEX_STEPS)])
    with contextlib.suppress(_CapReached):
        minimize(
            objective,
            start,
            method="Nelder-Mead",
            bounds=BOUNDS,
            options={"maxfev": EVALUATIONS, "initial_simplex": simplex},
        )
    return evaluations


def _wolff_spend(task: tuple[int, int]) -> tuple[float, int]:
    """Wolff on the current schedule at ``steps`` steps for one tuning seed."""
    steps, seed = task
    return handover(PottsMove.WOLFF, ANNEAL_SCHEDULE, steps, _cell_generator(seed))


def matched_steps(budget: Budget, lower: int) -> tuple[int, list[dict[str, Any]]]:
    """Wolff's step count at which its mean spend on the tuning seeds meets ``budget``.

    Doubles from ``lower`` until the mean spend reaches the budget, then
    bisects to :data:`BISECTION_TOLERANCE`, and returns the probed count whose
    mean spend is nearest the budget, with every probe.
    """
    probes: list[dict[str, Any]] = []

    def spend(steps: int) -> float:
        # One probe: the tuning seeds at `steps`, on the worker pool.
        runs = map_tasks(
            _wolff_spend,
            [(steps, seed) for seed in TUNING_SEEDS],
            workers=WORKERS,
            backend="processes",
            intra_op_threads=1,
        )
        spent = [visits for _, visits in runs]
        probes.append(
            {
                "steps": steps,
                "spent": spent,
                "mean_spent": float(np.mean(spent)),
                "energies": [energy for energy, _ in runs],
            }
        )
        return float(np.mean(spent))

    low, high = lower, lower
    while spend(high) < budget.size:
        low, high = high, 2 * high
    while (high - low) > BISECTION_TOLERANCE * low:
        middle = (low + high) // 2
        if spend(middle) < budget.size:
            low = middle
        else:
            high = middle
    nearest = min(probes, key=lambda probe: abs(probe["mean_spent"] - budget.size))
    return int(nearest["steps"]), probes


def main() -> None:
    """Tune, bisect, and write :data:`OUTPUT`."""
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    rung = release_rung()
    budget = solver_budget(rung)
    load_before = os.getloadavg()
    opened = time.perf_counter()
    tasks = [(move, shape) for move in MOVES for shape in ScheduleShape]
    traces = map_tasks(
        tune, tasks, workers=WORKERS, backend="processes", intra_op_threads=1
    )
    tuned_seconds = time.perf_counter() - opened
    moves: dict[str, Any] = {}
    for move in MOVES:
        shapes = {
            str(shape): [evaluation.record() for evaluation in trace]
            for (task_move, shape), trace in zip(tasks, traces, strict=True)
            if task_move is move
        }
        best = min(
            (
                evaluation
                for (task_move, _), trace in zip(tasks, traces, strict=True)
                if task_move is move
                for evaluation in trace
            ),
            key=lambda evaluation: evaluation.mean,
        )
        moves[str(move)] = {
            "chosen": best.record(),
            "shapes": shapes,
        }
    steps, probes = matched_steps(budget, max(1, budget.size // rung.visits_per_sweep))
    result = {
        "issue": 1038,
        "instance": f"spatio_only/release, {rung.name}",
        "sweeps": SWEEPS,
        "budget": budget.size,
        "tuning_seeds": list(TUNING_SEEDS),
        "evaluations_cap": EVALUATIONS,
        "bounds": [list(bound) for bound in BOUNDS],
        "current": asdict(ANNEAL_SCHEDULE),
        "moves": moves,
        "matched_wolff": {
            "schedule": asdict(ANNEAL_SCHEDULE),
            "steps": steps,
            "probes": probes,
        },
        "host": {
            "cores": os.cpu_count(),
            "workers": WORKERS,
            "threads": {
                variable: os.environ[variable] for variable in THREAD_VARIABLES
            },
            "load_before": list(load_before),
            "load_after": list(os.getloadavg()),
            "tuning_seconds": tuned_seconds,
            "total_seconds": time.perf_counter() - opened,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=1) + "\n")


@dataclass(frozen=True)
class Tuned:
    """What :data:`OUTPUT` holds, read back for the notebook.

    Parameters
    ----------
    chosen : dict[str, ScheduleParams]
        Per move name, the schedule of lowest mean energy on the tuning seeds.
    matched_steps : int
        Wolff's step count at matched spend.
    raw : dict[str, Any]
        The file as parsed, for the trace.
    """

    chosen: dict[str, ScheduleParams]
    matched_steps: int
    raw: dict[str, Any]


def load(path: Path = OUTPUT) -> Tuned:
    """Read a tuning result written by :func:`main`.

    Returns
    -------
    Tuned
    """
    raw = json.loads(path.read_text())
    chosen = {
        name: ScheduleParams(
            ScheduleShape(entry["chosen"]["shape"]),
            entry["chosen"]["t_start"],
            entry["chosen"]["t_end"],
            entry["chosen"]["hold"],
        )
        for name, entry in raw["moves"].items()
    }
    return Tuned(chosen, int(raw["matched_wolff"]["steps"]), raw)


# --- Issue #1046: every cluster move's schedule, searched to a stopping rule.

#: Evaluations a converging search may spend: a guard, not the rule that
#: ends it (:func:`nelder_mead`).
CONVERGE_CAP = 200
#: The largest distance per coordinate from the best vertex at which a search
#: may stop: 1% in a temperature (the coordinates are logarithms), 1% of the
#: steps in the hold, and 1% of the coupling in Niedermayer's threshold.
VERTEX_TOLERANCE = 0.01
#: Wall clock after which a search stops, writes its state and is reported
#: as not converged: three hours.
WALL_LIMIT = 3 * 3600.0
#: Outer rounds of a joint search of step count and schedule, and the relative
#: change of the count between two rounds below which it stops.
ROUNDS = 5
COUNT_TOLERANCE = 0.01
#: Relative distance from the budget at which a step-count match stops, and
#: the factor its bracket grows or shrinks by: #1041's values.
MATCH_TOLERANCE = 0.02
MATCH_GROWTH = 1.5
#: The first steps of a simplex started at a tuned point: a quarter of
#: :data:`SIMPLEX_STEPS`' factor in each temperature, a tenth of the steps
#: held, and a tenth of the coupling in the threshold.
WARM_STEPS = (math.log(1.25), -math.log(1.25), 0.1, 0.1)
#: Bounds on the threshold coordinate, ``E_0`` over the coupling: from
#: Wolff's value, 0, to twice the coupling.
THRESHOLD_BOUNDS = (0.0, 2.0)
#: Where the #1046 searches write their progress after every evaluation, so a
#: search stopped by the host is not lost. Under the build directory, which
#: git ignores.
STATE_DIR = REPO_ROOT / "target" / "potts_converge"
#: The stop conditions a search records.
CONVERGED, CAPPED, WALL, COUNT_FIXED, ROUNDS_SPENT = (
    "converged",
    "cap",
    "wall",
    "count fixed",
    "rounds",
)


@dataclass(frozen=True)
class Simplex:
    """How one Nelder-Mead search ended.

    Parameters
    ----------
    point : np.ndarray
        The best vertex evaluated.
    value, error : float
        Its objective and the objective's standard error there.
    stop : str
        :data:`CONVERGED`, :data:`CAPPED` or :data:`WALL`.
    evaluations : int
        Distinct points evaluated.
    vertices : np.ndarray
        The simplex when the search stopped, best vertex first, each vertex
        at the point it was scored at.
    """

    point: np.ndarray
    value: float
    error: float
    stop: str
    evaluations: int
    vertices: np.ndarray


class _WallReached(Exception):
    """Raised by the objective when the search's wall clock is spent."""


def fold(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """``point`` mirrored at each bound it crosses until it lies inside ``[lower, upper]``.

    Returns
    -------
    np.ndarray
    """
    width = upper - lower
    offset = np.mod(point - lower, 2.0 * width)
    folded: np.ndarray = lower + np.where(offset > width, 2.0 * width - offset, offset)
    return folded


def nelder_mead(
    objective: Callable[[np.ndarray], tuple[float, float]],
    simplex: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    *,
    cap: int = CONVERGE_CAP,
    tolerance: float = VERTEX_TOLERANCE,
    deadline: float = math.inf,
    clock: Callable[[], float] = time.monotonic,
) -> Simplex:
    """Minimise ``objective`` from ``simplex`` until the objective cannot tell its vertices apart.

    Nelder-Mead with SciPy's coefficients (reflection 1, expansion 2,
    contraction and shrink 1/2). A vertex may leave ``bounds`` and is scored
    at its mirror image inside them (:func:`fold`). SciPy clips the vertex
    itself, and a simplex started on a bound then loses that coordinate; a
    vertex scored at its clipped point sees a flat objective outside and
    does not come back. On a quadratic with its minimum at a hold of 0.4,
    both searches started at hold 0 ended at 0, and the folded one ends at
    0.4. ``objective`` returns the mean and
    its standard error. The search stops when the simplex's objective range
    is below the best vertex's standard error **and** every vertex is within
    ``tolerance`` of the best in every coordinate: past that, a step
    compares seeds' noise, and the vertices name one schedule to the
    tolerance, both read at the folded vertices. It also stops after
    ``cap`` distinct evaluations, a guard, or when ``clock()`` passes
    ``deadline``, and says which. A point already evaluated is not evaluated
    again: the objective is deterministic.

    Returns
    -------
    Simplex
    """
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    seen: dict[tuple[float, ...], tuple[float, float]] = {}

    def key_of(point: np.ndarray) -> tuple[float, ...]:
        # The schedule a vertex names: its point folded into the bounds.
        return tuple(fold(point, lower, upper).tolist())

    def score(point: np.ndarray) -> tuple[np.ndarray, float]:
        # Reuse a point already scored, and enforce the cap and the clock.
        key = key_of(point)
        if key not in seen:
            if len(seen) >= cap:
                raise _CapReached
            if clock() >= deadline:
                raise _WallReached
            seen[key] = objective(np.array(key))
        return point, seen[key][0]

    vertices = np.array(simplex, dtype=float)
    values = np.empty(len(vertices))
    stop = CONVERGED
    try:
        for index, vertex in enumerate(vertices):
            vertices[index], values[index] = score(vertex)
        while True:
            order = np.argsort(values, kind="stable")
            vertices, values = vertices[order], values[order]
            error = seen[key_of(vertices[0])][1]
            named = fold(vertices, lower, upper)
            spread = float(np.max(np.abs(named[1:] - named[0])))
            if values[-1] - values[0] < error and spread <= tolerance:
                break
            centroid = vertices[:-1].mean(axis=0)
            reflected, reflected_value = score(2.0 * centroid - vertices[-1])
            if reflected_value < values[0]:
                expanded, expanded_value = score(3.0 * centroid - 2.0 * vertices[-1])
                if expanded_value < reflected_value:
                    vertices[-1], values[-1] = expanded, expanded_value
                else:
                    vertices[-1], values[-1] = reflected, reflected_value
                continue
            if reflected_value < values[-2]:
                vertices[-1], values[-1] = reflected, reflected_value
                continue
            if reflected_value < values[-1]:
                # Outside contraction, toward the reflected point.
                contracted, contracted_value = score(
                    1.5 * centroid - 0.5 * vertices[-1]
                )
                accept = contracted_value <= reflected_value
            else:
                # Inside contraction, toward the worst vertex.
                contracted, contracted_value = score(
                    0.5 * centroid + 0.5 * vertices[-1]
                )
                accept = contracted_value < values[-1]
            if accept:
                vertices[-1], values[-1] = contracted, contracted_value
                continue
            # Shrink every vertex halfway to the best.
            for index in range(1, len(vertices)):
                vertices[index], values[index] = score(
                    vertices[0] + 0.5 * (vertices[index] - vertices[0])
                )
    except _CapReached:
        stop = CAPPED
    except _WallReached:
        stop = WALL
    order = np.argsort(values, kind="stable")
    best_key = min(seen, key=lambda key: seen[key][0])
    return Simplex(
        np.array(best_key),
        seen[best_key][0],
        seen[best_key][1],
        stop,
        len(seen),
        fold(vertices[order], lower, upper),
    )


def match_steps(
    spend: Callable[[int], float],
    budget: float,
    start: int,
    tolerance: float = MATCH_TOLERANCE,
    growth: float = MATCH_GROWTH,
) -> tuple[int, list[tuple[int, float]]]:
    """The probed step count whose mean spend is nearest ``budget``, and every probe.

    #1041's search, run both ways: from ``start`` the count grows (or
    shrinks) by ``growth`` until the spend crosses the budget, then the
    bracket is bisected, until a probe is within ``tolerance`` of the budget
    or the bracket is narrower than 1% of its lower end. The spend need not
    be monotone in the count --- a Niedermayer run spent 28.9 million visits
    at 45,761 steps and 40.7 million at 46,406 (#1041) --- so the answer is
    the nearest *probe*, not a root of the spend.

    Returns
    -------
    tuple[int, list[tuple[int, float]]]
        The count, and ``(count, mean spend)`` per probe in order.
    """
    probes: list[tuple[int, float]] = []

    def probe(steps: int) -> float:
        # One count's mean spend, recorded.
        mean = spend(steps)
        probes.append((steps, mean))
        return mean

    def close(mean: float) -> bool:
        return abs(mean - budget) <= tolerance * budget

    mean = probe(start)
    low = high = start
    if mean < budget:
        while mean < budget and not close(mean):
            low, high = high, max(high + 1, round(growth * high))
            mean = probe(high)
    else:
        while mean >= budget and not close(mean) and low > 1:
            high, low = low, max(1, round(low / growth))
            mean = probe(low)
    while not close(mean) and (high - low) > max(1.0, 0.01 * low):
        middle = (low + high) // 2
        mean = probe(middle)
        if mean < budget:
            low = middle
        else:
            high = middle
    nearest = min(probes, key=lambda entry: abs(entry[1] - budget))
    return nearest[0], probes


def joint_rounds(
    match: Callable[[np.ndarray, int], int],
    search: Callable[[np.ndarray, int], tuple[np.ndarray, str]],
    point: np.ndarray,
    count: int,
    *,
    rounds: int = ROUNDS,
    tolerance: float = COUNT_TOLERANCE,
    deadline: float = math.inf,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[np.ndarray, int, str, list[dict[str, Any]]]:
    """A step count and a schedule that fix each other, in alternating rounds.

    A single-cluster move's spend depends on its schedule, so neither the
    count nor the schedule can be fixed first. The count is matched to the
    budget at the start, ``match(point, count)``; each round then searches
    the schedule at that count, ``search(point, count)``, and matches the
    count again at the point found. The rounds stop when that match moves
    the count by less than ``tolerance`` (:data:`COUNT_FIXED`), after
    ``rounds`` searches (:data:`ROUNDS_SPENT`), or when a search stops on
    the wall clock or ``deadline`` passes before one starts (:data:`WALL`).
    The point is reported at the count last matched to it, except where a
    search stopped on the wall clock, at the count it was searched at.

    Returns
    -------
    tuple[np.ndarray, int, str, list[dict[str, Any]]]
        The point, the count it is reported at, the stop condition, and per
        round the count matched and the search's stop.
    """
    history: list[dict[str, Any]] = [{"count": match(point, count)}]
    for _ in range(rounds):
        matched = int(history[-1]["count"])
        if clock() >= deadline:
            return point, matched, WALL, history
        point, stop = search(point, matched)
        history[-1]["search"] = stop
        if stop == WALL:
            return point, matched, WALL, history
        history.append({"count": match(point, matched)})
        if abs(history[-1]["count"] - matched) < tolerance * matched:
            return point, int(history[-1]["count"]), COUNT_FIXED, history
    return point, int(history[-1]["count"]), ROUNDS_SPENT, history


if __name__ == "__main__":
    main()
