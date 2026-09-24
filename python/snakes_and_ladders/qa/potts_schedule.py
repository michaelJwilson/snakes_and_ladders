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
from collections.abc import Sequence
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
    """

    params: ScheduleParams
    steps: int | None
    energies: tuple[float, ...]
    spent: tuple[int, ...]

    @property
    def mean(self) -> float:
        """Mean handed-over energy, the objective."""
        return float(np.mean(self.energies))

    def record(self) -> dict[str, Any]:
        """The evaluation as JSON-ready data."""
        return {
            **asdict(self.params),
            "steps": self.steps,
            "energies": list(self.energies),
            "spent": list(self.spent),
            "mean": self.mean,
        }


def handover(
    move: PottsMove, params: ScheduleParams, steps: int | None, seed: int
) -> tuple[float, int]:
    """One annealed run on the notebook's generator for ``seed``: its energy and spend.

    The generator is ``default_rng([seed, 0])``, the one
    :class:`~snakes_and_ladders.opt.starts.StartsBenchmark` hands cell
    ``(instance 0, seed)``, so a run here is the notebook's run at that seed.
    """
    rung = release_rung()
    run = run_annealed(
        rung,
        solver_budget(rung),
        np.random.default_rng([seed, 0]),
        move,
        schedule=params,
        steps=steps,
    )
    return run.energy, run.spent


def evaluate(
    move: PottsMove,
    params: ScheduleParams,
    steps: int | None = None,
    seeds: Sequence[int] = TUNING_SEEDS,
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
    runs = [handover(move, params, steps, seed) for seed in seeds]
    return Evaluation(
        params,
        steps,
        tuple(energy for energy, _ in runs),
        tuple(spent for _, spent in runs),
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
    return handover(PottsMove.WOLFF, ANNEAL_SCHEDULE, steps, seed)


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


if __name__ == "__main__":
    main()
