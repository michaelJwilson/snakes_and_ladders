"""Every cluster move's annealing schedule, searched until the tuning seeds cannot tell it from its neighbours (issue #1046).

`docs/nb/potts_starts.ipynb` runs five cluster moves on `spatio_only/release`
at ten states. #1038 tuned Swendsen-Wang's schedule and Wolff's with 40
evaluations per shape, and Swendsen-Wang's search found its best at
evaluations 39 and 40; Wolff was tuned at the budget's 1,000 steps, where it
spends 0.04% of the budget. #1041 ran the ghost-spin, label-directed and
Niedermayer moves on Swendsen-Wang's tuned schedule, and Niedermayer at
Wolff's threshold, 0. This script searches each move's schedule until
:func:`~snakes_and_ladders.qa.potts_schedule.nelder_mead`'s stopping rule
holds, and writes every evaluation to ``docs/nb/data/potts_converge.json``
for the notebook to read. #1038's and #1041's files are left as they are.

**The searches.** Each runs on
:data:`~snakes_and_ladders.qa.potts_schedule.TUNING_SEEDS` over
``(log t_start, log t_end, hold)`` on one shape, and stops when the
simplex's objective range is below one standard error of the objective and
every vertex is within 1% of the best per coordinate, or at
:data:`~snakes_and_ladders.qa.potts_schedule.CONVERGE_CAP` evaluations, or
at three hours; the stop is recorded.

- Swendsen-Wang, ghost-spin and label-directed start at #1038's tuned
  Swendsen-Wang schedule (linear, 0.784 to 0.324), which all three ran on.
  Swendsen-Wang's other shapes are searched only if its linear search stops
  at the cap.
- Wolff and Niedermayer build one cluster a step, so their spend depends on
  the schedule and the step count is matched to the budget in rounds with
  the schedule (:func:`~snakes_and_ladders.qa.potts_schedule.joint_rounds`).
  Wolff starts at the current schedule and #1038's matched 41,250 steps, on
  the exponential shape #1038 chose for it; Niedermayer at Swendsen-Wang's
  tuned schedule and #1041's matched 46,083 steps. Niedermayer also searches
  its threshold ``E_0``, in units of the coupling, from Wolff's value 0.

**The arms.** Per move, the default schedule, the #1038/#1041 schedule and
the converged one run on
:data:`~snakes_and_ladders.qa.potts_schedule.REPORTED_SEEDS` through
:class:`~snakes_and_ladders.opt.starts.StartsBenchmark`, polished as the
notebook polishes, so a row here is the row the notebook would compute.

**Threads.** BLAS is pinned to one thread per process, as in
:mod:`~snakes_and_ladders.qa.potts_schedule` (issue #1044).

Run as ``uv run --no-sync python -m snakes_and_ladders.qa.potts_converge
[search] [arms]``; with no part named, both run, in that order.
"""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.starts import StartsBenchmark
from snakes_and_ladders.parallel import map_tasks
from snakes_and_ladders.qa.potts_schedule import (
    BOUNDS,
    CAPPED,
    CONVERGE_CAP,
    COUNT_TOLERANCE,
    REPORTED_SEEDS,
    ROUNDS,
    SIMPLEX_STEPS,
    STATE_DIR,
    THREAD_VARIABLES,
    TUNING_SEEDS,
    VERTEX_TOLERANCE,
    WALL_LIMIT,
    WARM_STEPS,
    WORKERS,
    Evaluation,
    evaluate,
    joint_rounds,
    match_steps,
    nelder_mead,
    params_of,
    release_rung,
    solver_budget,
)
from snakes_and_ladders.qa.potts_schedule import load as load_tuned
from snakes_and_ladders.sample.potts_mcmc import PottsMove
from snakes_and_ladders.sample.schedule import ScheduleParams, ScheduleShape
from snakes_and_ladders.search.ground_state import (
    ANNEAL_SCHEDULE,
    MethodRun,
    Rung,
    run_alpha_expansion,
    run_annealed,
)
from snakes_and_ladders.search.potts_starts import (
    PottsObjective,
    RunStart,
    polish_by_icm,
)
from snakes_and_ladders.sim.fixtures import REPO_ROOT

#: Where the result is written and where the notebook reads it.
OUTPUT = REPO_ROOT / "docs" / "nb" / "data" / "potts_converge.json"
#: #1041's file, which holds Niedermayer's matched step count.
CLUSTERS = REPO_ROOT / "docs" / "nb" / "data" / "potts_clusters.json"
#: ICM's cap in sweeps and the seeding budget, the notebook's.
POLISH = Budget(Cost.SWEEPS, 200)
SEEDING = Budget(Cost.EVALUATIONS, 1)
#: Niedermayer's thresholds, ``E_0`` over the coupling, each scored at its
#: own matched step count. The first plan searched ``E_0`` beside the
#: schedule at a fixed count, and its first ``E_0`` vertex, 0.1, ran one
#: evaluation for more than 25 minutes against 3 for ``E_0 = 0``: above 0 the
#: clusters grow and the spend at a count fixed for 0 runs past the budget.
THRESHOLD_SCAN = (0.0, 0.02, 0.1)
#: Where a threshold's step-count match starts: low, so the bracket grows to
#: the budget rather than starting far past it.
SCAN_START = 2000
#: The searches in the order the pool takes them: the longest first, so the
#: short ones fill the other worker beside it.
ORDER = ("niedermayer", "swendsen-wang", "ghost-spin", "label-directed", "wolff")


@dataclass(frozen=True)
class Plan:
    """One move's search: where it starts and what it varies.

    Parameters
    ----------
    move : PottsMove
    shape : ScheduleShape
    start : tuple[float, ...]
        The warm start, ``(log t_start, log t_end, hold)`` and, for
        Niedermayer, ``E_0`` over the coupling.
    steps : tuple[float, ...]
        The first simplex's step per coordinate.
    previous : ScheduleParams
        The schedule #1038 or #1041 ran the move on.
    count : int | None
        The step count it ran at, ``None`` for the budget's rule; a joint
        search's first count.
    joint : bool
        Whether the step count is matched in rounds with the schedule.
    thresholds : tuple[float, ...]
        Niedermayer's ``E_0`` values, in units of the coupling, each scored
        at its own matched count before the schedule search; empty for every
        other move.
    """

    move: PottsMove
    shape: ScheduleShape
    start: tuple[float, ...]
    steps: tuple[float, ...]
    previous: ScheduleParams
    count: int | None
    joint: bool
    thresholds: tuple[float, ...] = ()


def point_of(params: ScheduleParams) -> tuple[float, float, float]:
    """``(log t_start, log t_end, hold)`` of a schedule."""
    return (math.log(params.t_start), math.log(params.t_end), params.hold)


def plans() -> dict[str, Plan]:
    """Every move's search, by the names the notebook uses.

    Returns
    -------
    dict[str, Plan]
    """
    tuned = load_tuned()
    sw = tuned.chosen["swendsen-wang"]
    niedermayer_count = int(
        json.loads(CLUSTERS.read_text())["matched_niedermayer"]["steps"]
    )
    warm = WARM_STEPS[:3]
    table = {
        name: Plan(move, sw.shape, point_of(sw), warm, sw, None, False)
        for name, move in (
            ("swendsen-wang", PottsMove.SWENDSEN_WANG),
            ("ghost-spin", PottsMove.GHOST_SPIN),
            ("label-directed", PottsMove.LABEL_DIRECTED),
        )
    }
    # Wolff from the current schedule, the one it ran on at the matched count,
    # with #1038's simplex from that point.
    table["wolff"] = Plan(
        PottsMove.WOLFF,
        tuned.chosen["wolff"].shape,
        point_of(ANNEAL_SCHEDULE),
        SIMPLEX_STEPS,
        ANNEAL_SCHEDULE,
        tuned.matched_steps,
        True,
    )
    table["niedermayer"] = Plan(
        PottsMove.NIEDERMAYER,
        sw.shape,
        point_of(sw),
        warm,
        sw,
        niedermayer_count,
        True,
        THRESHOLD_SCAN,
    )
    return table


@functools.cache
def coupling() -> float:
    """The rung's coupling, the unit of Niedermayer's threshold coordinate."""
    _, _, couplings = release_rung().graph.compressed_adjacency()
    return float(np.max(couplings))


def replayed(name: str) -> tuple[dict[tuple[Any, ...], dict[str, Any]], float]:
    """Every evaluation a stopped run of ``name``'s search checkpointed, by :func:`replay_key`, and its seconds.

    The objective is deterministic, so a search restarted with these
    replays its own path to where it stopped without a run; the seconds
    count against its wall clock.
    """
    path = STATE_DIR / f"{name}.json"
    if not path.exists():
        return {}, 0.0
    state = json.loads(path.read_text())
    records = [record for search in state["searches"] for record in search["trace"]] + [
        record for match in state["matches"] for record in match["probes"]
    ]
    records += [entry["evaluation"] for entry in state.get("threshold_scan", [])]
    return {replay_key(record): record for record in records}, float(state["seconds"])


def replay_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """What names an evaluation: its schedule, step count and threshold."""
    return (
        str(record["shape"]),
        record["t_start"],
        record["t_end"],
        record["hold"],
        record["steps"],
        record.get("threshold"),
    )


def converge(name: str) -> dict[str, Any]:
    """One move's search, every evaluation in order, checkpointed to :data:`STATE_DIR`.

    A checkpoint of a stopped run is replayed (:func:`replayed`), and its
    seconds count against :data:`WALL_LIMIT`. For Niedermayer each
    threshold of the plan is first scored at its own matched count on the
    warm start, and the schedule is searched at the best one.

    Returns
    -------
    dict[str, Any]
        The plan, every search and match, the chosen evaluation, the stop,
        and the wall clock.
    """
    plan = plans()[name]
    rung = release_rung()
    budget = solver_budget(rung)
    replay, before = replayed(name)
    opened = time.perf_counter() - before
    deadline = time.monotonic() + WALL_LIMIT - before
    state: dict[str, Any] = {
        "move": str(plan.move),
        "shape": str(plan.shape),
        "start": list(plan.start),
        "simplex_steps": list(plan.steps),
        "previous": {**asdict(plan.previous), "steps": plan.count},
        "joint": plan.joint,
        "resumed_after": before,
        "replayed": 0,
        "searches": [],
        "matches": [],
    }
    # Every evaluation by (point, count, threshold), so the chosen one can be
    # read back.
    scored: dict[tuple[tuple[float, ...], int | None, float | None], Evaluation] = {}
    threshold: float | None = None

    def checkpoint() -> None:
        # The state so far, so a search the host stops is not lost.
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state["seconds"] = time.perf_counter() - opened
        (STATE_DIR / f"{name}.json").write_text(json.dumps(state, indent=1) + "\n")

    def run(point: Sequence[float] | np.ndarray, count: int | None) -> Evaluation:
        # One point on the tuning seeds, serially, or its checkpointed record.
        params = params_of(point, plan.shape)
        key = (
            str(params.shape),
            params.t_start,
            params.t_end,
            params.hold,
            count,
            threshold,
        )
        if key in replay:
            record = replay[key]
            evaluation = Evaluation(
                params,
                count,
                tuple(record["energies"]),
                tuple(record["spent"]),
                threshold,
            )
            state["replayed"] += 1
        else:
            evaluation = evaluate(plan.move, params, count, threshold=threshold)
        scored[(tuple(float(value) for value in point), count, threshold)] = evaluation
        return evaluation

    def search(point: np.ndarray, count: int | None) -> tuple[np.ndarray, str]:
        # One Nelder-Mead search at a fixed count, from `point`.
        steps = plan.steps if not state["searches"] else WARM_STEPS[: len(point)]
        trace: list[dict[str, Any]] = []
        state["searches"].append({"count": count, "trace": trace})

        def objective(candidate: np.ndarray) -> tuple[float, float]:
            evaluation = run(candidate, count)
            trace.append(evaluation.record())
            checkpoint()
            return evaluation.mean, evaluation.standard_error

        simplex = np.vstack([point, point + np.diag(steps)])
        result = nelder_mead(objective, simplex, BOUNDS, deadline=deadline)
        state["searches"][-1].update(
            {
                "stop": result.stop,
                "evaluations": result.evaluations,
                "best": result.point.tolist(),
                "vertices": result.vertices.tolist(),
            }
        )
        checkpoint()
        return result.point, result.stop

    def match(point: np.ndarray, count: int) -> int:
        # The count whose mean spend on the tuning seeds is nearest the budget.
        probes: list[dict[str, Any]] = []
        state["matches"].append(
            {"point": point.tolist(), "threshold": threshold, "probes": probes}
        )

        def spend(steps: int) -> float:
            evaluation = run(point, steps)
            probes.append(evaluation.record())
            checkpoint()
            return float(np.mean(evaluation.spent))

        steps, _ = match_steps(spend, budget.size, count)
        state["matches"][-1]["steps"] = steps
        return steps

    start = np.array(plan.start, dtype=float)
    count = plan.count
    if plan.thresholds:
        # Each threshold at its own matched count on the warm start; the
        # schedule is then searched at the one of lowest mean energy.
        scan: list[dict[str, Any]] = []
        state["threshold_scan"] = scan
        for units in plan.thresholds:
            threshold = units * coupling()
            assert plan.count is not None
            steps = match(start, plan.count if units == 0.0 else SCAN_START)
            scan.append(
                {
                    "threshold": threshold,
                    "steps": steps,
                    "evaluation": scored[
                        (tuple(float(value) for value in start), steps, threshold)
                    ].record(),
                }
            )
            checkpoint()
        best = min(scan, key=lambda entry: entry["evaluation"]["mean"])
        threshold, count = best["threshold"], best["steps"]
    if plan.joint:
        assert count is not None
        point, count, stop, rounds = joint_rounds(
            match, search, start, count, deadline=deadline
        )
        state["rounds"] = rounds
    else:
        point, stop = search(start, None)
        count = None
    key = (tuple(float(value) for value in point), count, threshold)
    state.update(
        {
            "stop": stop,
            "count": count,
            "threshold": threshold,
            "evaluations": len(scored),
            "chosen": scored[key].record(),
            "standard_error": scored[key].standard_error,
        }
    )
    checkpoint()
    return state


def schedule_of(entry: dict[str, Any]) -> ScheduleParams:
    """The schedule a record names."""
    return ScheduleParams(
        ScheduleShape(entry["shape"]), entry["t_start"], entry["t_end"], entry["hold"]
    )


def _run_start(
    run: Callable[[Rung, Budget, np.random.Generator], MethodRun],
) -> Callable[[np.random.Generator], RunStart]:
    """A start built per cell from its generator, charged at the notebook's budget."""
    return functools.partial(RunStart, run, solver_budget(release_rung()))


def arms(
    result: dict[str, Any],
) -> dict[str, Callable[[np.random.Generator], RunStart]]:
    """Per move, the default, the #1038/#1041 and the converged schedule, by ``"<move> <arm>"``.

    Returns
    -------
    dict[str, Callable[[np.random.Generator], RunStart]]
    """
    table: dict[str, Callable[[Rung, Budget, np.random.Generator], MethodRun]] = {}
    for name, plan in plans().items():
        entry = result["moves"][name]
        chosen = entry["chosen"]
        table[f"{name} default"] = functools.partial(run_annealed, move=plan.move)
        table[f"{name} previous"] = functools.partial(
            run_annealed, move=plan.move, schedule=plan.previous, steps=plan.count
        )
        table[f"{name} converged"] = functools.partial(
            run_annealed,
            move=plan.move,
            schedule=schedule_of(chosen),
            steps=entry["count"],
            threshold=chosen.get("threshold"),
        )
    return {name: _run_start(run) for name, run in table.items()}


def run_arms(
    result: dict[str, Any], names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Every arm of :func:`arms` (or those named) on the reported seeds, polished as the notebook polishes.

    Returns
    -------
    dict[str, Any]
        Per arm, per seed: the handed-over and polished energies, the site
        visits, the cell's seconds, and the ICM sweeps.
    """
    rung = release_rung()
    budget = solver_budget(rung)
    reference = run_alpha_expansion(rung, budget, np.random.default_rng(0)).energy
    table = arms(result)
    chosen = {name: table[name] for name in (names or list(table))}
    benchmark = StartsBenchmark(
        PottsObjective(rung),
        chosen,
        polish_by_icm,
        seeding_budget=SEEDING,
        polish_budget=POLISH,
        seeds=list(REPORTED_SEEDS),
        workers=WORKERS,
        reference=[reference],
    ).run()
    rows: dict[str, Any] = {}
    for name in chosen:
        trials = benchmark.trials(name)
        rows[name] = {
            "handed": [float(trial.seeded_value) for trial in trials],
            "polished": [float(trial.value) for trial in trials],
            "visits": [float(trial.diagnostics["site_visits"]) for trial in trials],
            "seconds": [float(trial.seconds[-1]) for trial in trials],
            "icm_sweeps": [int(trial.termination.iterations) for trial in trials],
        }
    return {"reference": reference, "arms": rows}


#: The parts :func:`main` runs, in the order each needs the last.
PARTS = ("search", "arms")


def main(parts: Sequence[str] = PARTS) -> None:
    """Run the parts named and merge them into :data:`OUTPUT`."""
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    rung = release_rung()
    budget = solver_budget(rung)
    result: dict[str, Any] = (
        json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {"issue": 1046}
    )
    result.update(
        {
            "instance": f"spatio_only/release, {rung.name}",
            "budget": budget.size,
            "tuning_seeds": list(TUNING_SEEDS),
            "cap": CONVERGE_CAP,
            "vertex_tolerance": VERTEX_TOLERANCE,
            "wall_limit": WALL_LIMIT,
            "rounds": ROUNDS,
            "count_tolerance": COUNT_TOLERANCE,
            "bounds": [list(bound) for bound in BOUNDS],
            "threshold_scan": list(THRESHOLD_SCAN),
            "coupling": coupling(),
        }
    )
    for part in parts:
        load_before = os.getloadavg()
        opened = time.perf_counter()
        if part == "search":
            # A move whose checkpoint records a stop is read back, not rerun.
            done = {
                name: json.loads((STATE_DIR / f"{name}.json").read_text())
                for name in ORDER
                if (STATE_DIR / f"{name}.json").exists()
                and "stop" in json.loads((STATE_DIR / f"{name}.json").read_text())
            }
            due = [name for name in ORDER if name not in done]
            searches = map_tasks(
                converge,
                due,
                workers=WORKERS,
                backend="processes",
                intra_op_threads=1,
            )
            done.update(zip(due, searches, strict=True))
            result["moves"] = {name: done[name] for name in ORDER}
            if result["moves"]["swendsen-wang"]["stop"] == CAPPED:
                # The plan's condition for searching the other shapes.
                print("swendsen-wang stopped at the cap: its other shapes are due")
        elif part == "arms":
            result["arms"] = run_arms(result)
        else:
            msg = f"no part {part!r}; the parts are {PARTS}"
            raise ValueError(msg)
        result.setdefault("host", {}).setdefault(part, []).append(
            {
                "cores": os.cpu_count(),
                "workers": WORKERS,
                "threads": {
                    variable: os.environ[variable] for variable in THREAD_VARIABLES
                },
                "load_before": list(load_before),
                "load_after": list(os.getloadavg()),
                "seconds": time.perf_counter() - opened,
            }
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=1) + "\n")


def load(path: Path = OUTPUT) -> dict[str, Any]:
    """The file :func:`main` wrote, parsed.

    Returns
    -------
    dict[str, Any]
    """
    return dict(json.loads(path.read_text()))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or PARTS)
