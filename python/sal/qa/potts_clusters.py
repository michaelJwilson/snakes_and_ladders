"""What the cluster moves' clusters span, and every cluster arm against the graph cuts (issue #1041).

On `spatio_only/release` at ten states the graph cuts hand over their own
labelling on every seed, and Swendsen-Wang on its tuned schedule (#1038) hands
over 299.6 above it. The ticket's hypothesis: a Fortuin-Kasteleyn bond reads
the coupling and not the field, so a cluster built by it spans sites whose own
field points to different labels, and its recolouring is then refused or
accepted against part of its members. This script measures that first and
runs the arms second, and writes both to ``docs/nb/data/potts_clusters.json``
for the notebook, so neither runs inside the notebook's budget.

**The diagnosis.** On :data:`TUNING_SEEDS`, Swendsen-Wang on its tuned
schedule and Wolff at its matched step count on the current one, each on the
oracle's Python pass so every cluster's members are read. Per schedule step:
the mean and largest cluster, the accept rate, the sites relabelled, the
fraction of relabelled sites whose field-preferred label is not the label the
cluster drew, the fraction of relabelled sites whose preference differs from
their cluster's majority preference (*mixed*), the same over every cluster of
two or more sites, the energy change and the visits. The Python pass is the
compiled pass's law on another order of draws (`search.ground_state`), so the
chains are not the notebook's chains bitwise.

**The rule fixed before the run.** The ghost-spin arm exists to keep a
cluster off sites whose field disagrees with it. If fewer than
:data:`MIXED_FLOOR` of the sites Swendsen-Wang relabels over its run are off
their cluster's majority preference, there is nothing for it to separate and
the arm is dropped.

**Niedermayer's matched count.** Niedermayer's step is one cluster, as
Wolff's is, so its step count is matched to the budget on
:data:`TUNING_SEEDS` before any reported run (:func:`matched_niedermayer`),
on Swendsen-Wang's tuned schedule.

**The arms.** Every arm runs through
:class:`~sal.opt.starts.StartsBenchmark` with the notebook's
polish, seeds, generator per cell and reference, so a row here is the row the
notebook would compute. Each is charged every part's site visits.

**Threads.** BLAS is pinned to one thread per process, as in
:mod:`~sal.qa.potts_schedule` (issue #1044).

Run as ``uv run --no-sync python -m sal.qa.potts_clusters
[diagnosis] [matched] [arms]``; with no part named, all three run, in that
order, since the arms read the matched count.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np

from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.starts import StartsBenchmark
from sal.parallel import map_tasks
from sal.qa.potts_schedule import (
    REPORTED_SEEDS,
    THREAD_VARIABLES,
    TUNING_SEEDS,
    WORKERS,
    release_rung,
    solver_budget,
)
from sal.qa.potts_schedule import load as load_tuned
from sal.sample.potts_mcmc import (
    ClusterCounter,
    PottsMove,
    Recolour,
    adjacency_lists,
    swendsen_wang_sweep,
    wolff_sweep,
)
from sal.search.cluster_moves import (
    CLUSTER_LADDER_REPLICAS,
    run_cluster_tempering,
)
from sal.search.ground_state import (
    ANNEAL_SCHEDULE,
    EXPANSION_RESERVE_CYCLES,
    EXPANSION_SW_SCHEDULE,
    MethodRun,
    Rung,
    run_alpha_expansion,
    run_annealed,
    run_expansion_then_swendsen_wang,
    run_swendsen_wang_then_expansion,
)
from sal.search.potts_starts import (
    LabellingEnergy,
    RunStart,
    polish_by_icm,
)
from sal.sim.fixtures import REPO_ROOT
from sal.sim.graph import PottsGraph
from sal.sim.potts import energies, energy

#: Where the result is written and where the notebook reads it.
OUTPUT = REPO_ROOT / "docs" / "nb" / "data" / "potts_clusters.json"
#: The fraction of Swendsen-Wang's relabelled sites in a cluster of mixed
#: field preference below which the ghost-spin arm is dropped.
MIXED_FLOOR = 0.05
#: Schedule steps per recorded row of the diagnosis: Wolff's matched run is
#: 41,250 steps, so its rows are bins of steps, and Swendsen-Wang's 1,000 are
#: one step a row.
DIAGNOSIS_ROWS = 1000
#: Relative distance from the budget at which the search for Niedermayer's
#: matched step count stops, and the factor its bracket grows by.
MATCH_TOLERANCE = 0.02
MATCH_GROWTH = 1.5
#: The budget's step count at one heat-bath sweep a step.
SOLVER_STEPS = 1000
#: ICM's cap in sweeps and the seeding budget, the notebook's.
POLISH = Budget(Cost.SWEEPS, 200)
SEEDING = Budget(Cost.EVALUATIONS, 1)


@dataclass
class DiagnosedClusters(ClusterCounter):
    """:class:`~sal.sample.potts_mcmc.ClusterCounter` that also reads each cluster against the field.

    The sweep calls :meth:`record` after it recolours a cluster, so the
    cluster's label in ``state`` is the drawn one where the move was accepted.

    Attributes
    ----------
    state : np.ndarray
        The chain's labelling, the array the sweep mutates.
    preferred : np.ndarray
        Each site's field-preferred label, ``argmax`` of its row.
    relabelled, disagree, mixed : int
        Sites in accepted clusters; of those, sites whose preferred label is
        not the drawn one; and sites whose preference is not their cluster's
        majority preference.
    multi_sites, multi_mixed : int
        Sites in clusters of two or more, accepted or not, and of those the
        sites not of their cluster's majority preference.
    """

    state: np.ndarray = dataclass_field(default_factory=lambda: np.empty(0, int))
    preferred: np.ndarray = dataclass_field(default_factory=lambda: np.empty(0, int))
    relabelled: int = 0
    disagree: int = 0
    mixed: int = 0
    multi_sites: int = 0
    multi_mixed: int = 0

    def record(
        self, members: np.ndarray, outcome: Recolour, graph: PottsGraph | None
    ) -> None:
        """Add one cluster, and read its members' preferences."""
        super().record(members, outcome, graph)
        preferences = self.preferred[members]
        # The members not of the cluster's most common preference.
        off_majority = int(members.size - np.bincount(preferences).max())
        if members.size > 1:
            self.multi_sites += int(members.size)
            self.multi_mixed += off_majority
        if outcome.accepted:
            self.relabelled += int(members.size)
            self.disagree += int((preferences != self.state[members[0]]).sum())
            self.mixed += off_majority


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, ``nan`` over zero."""
    return numerator / denominator if denominator else float("nan")


def diagnose(task: tuple[str, int]) -> dict[str, Any]:
    """One diagnosed run on the notebook's rung and budget, binned to :data:`DIAGNOSIS_ROWS` rows.

    ``task`` is ``(move name, seed)``: ``"swendsen-wang"`` on its tuned
    schedule at the budget's step count, or ``"wolff"`` on the current
    schedule at the matched count.

    Returns
    -------
    dict[str, Any]
        Per row, the quantities the module docstring lists, and the run's
        totals.
    """
    name, seed = task
    rung = release_rung()
    tuned = load_tuned()
    if name == "swendsen-wang":
        params, steps = tuned.chosen[name], SOLVER_STEPS
    else:
        params, steps = ANNEAL_SCHEDULE, tuned.matched_steps
    schedule = params.build(steps)
    rng = np.random.default_rng([seed, 0])
    rows = np.asarray(rung.field, dtype=float)
    state = np.ascontiguousarray(
        rng.integers(0, rung.n_states, size=rung.n_nodes), dtype=np.int64
    )
    preferred = rows.argmax(axis=1)
    offsets, neighbours, couplings = rung.graph.compressed_adjacency()
    lists = adjacency_lists(offsets, neighbours, couplings)
    per_member = 1 + 2 * len(rung.graph.edges) // rung.n_nodes
    edges = len(rung.graph.edges)
    bins = np.minimum(np.arange(steps) * DIAGNOSIS_ROWS // steps, DIAGNOSIS_ROWS - 1)
    keys = (
        "temperature",
        "clusters",
        "size_sum",
        "size_max",
        "proposals",
        "accepts",
        "relabelled",
        "disagree",
        "mixed",
        "multi_sites",
        "multi_mixed",
        "energy_change",
        "visits",
    )
    table = {key: np.zeros(DIAGNOSIS_ROWS) for key in keys}
    counts = np.bincount(bins, minlength=DIAGNOSIS_ROWS)
    before = float(energies(rung.graph, rows, state[None])[0])
    first = before
    opened = time.perf_counter()
    for step in range(steps):
        temperature = schedule(step)
        beta = 1.0 / temperature
        counter = DiagnosedClusters(state=state, preferred=preferred)
        if name == "swendsen-wang":
            # The oracle's pass: the compiled one reads no cluster's members.
            swendsen_wang_sweep(state, rung.graph, rows, rng, counter, beta)
            visits = rung.n_nodes + 2 * edges
        else:
            wolff_sweep(
                state,
                rows,
                offsets,
                neighbours,
                couplings,
                rng,
                counter,
                rung.graph,
                beta,
                lists=lists,
            )
            visits = sum(counter.sizes) * per_member
        after = float(energies(rung.graph, rows, state[None])[0])
        row = int(bins[step])
        table["temperature"][row] += temperature
        table["clusters"][row] += len(counter.sizes)
        table["size_sum"][row] += sum(counter.sizes)
        table["size_max"][row] = max(table["size_max"][row], counter.max_size)
        table["proposals"][row] += counter.proposals
        table["accepts"][row] += counter.accepts
        table["relabelled"][row] += counter.relabelled
        table["disagree"][row] += counter.disagree
        table["mixed"][row] += counter.mixed
        table["multi_sites"][row] += counter.multi_sites
        table["multi_mixed"][row] += counter.multi_mixed
        table["energy_change"][row] += after - before
        table["visits"][row] += visits
        before = after
    table["temperature"] /= counts
    return {
        "move": name,
        "seed": seed,
        "steps": steps,
        "schedule": asdict(params),
        "seconds": time.perf_counter() - opened,
        "start_energy": first,
        "final_energy": before,
        "rows": {key: values.tolist() for key, values in table.items()},
    }


def _rounded(values: Sequence[float]) -> list[float]:
    """``values`` at six significant digits, so the file stays a size to commit."""
    return [float(f"{value:.6g}") for value in values]


def summarise(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per move, the fractions over the whole run and per row, summed over seeds.

    Returns
    -------
    dict[str, Any]
    """
    summary: dict[str, Any] = {}
    for name in sorted({run["move"] for run in runs}):
        mine = [run for run in runs if run["move"] == name]
        total = {
            key: np.sum([run["rows"][key] for run in mine], axis=0)
            for key in mine[0]["rows"]
        }
        largest = np.max([run["rows"]["size_max"] for run in mine], axis=0)
        summary[name] = {
            "seeds": [run["seed"] for run in mine],
            "steps": mine[0]["steps"],
            "seconds": [run["seconds"] for run in mine],
            "whole_run": {
                "mean_cluster": _ratio(
                    total["size_sum"].sum(), total["clusters"].sum()
                ),
                "max_cluster": float(largest.max()),
                "accept_rate": _ratio(total["accepts"].sum(), total["proposals"].sum()),
                "relabelled_per_seed": float(total["relabelled"].sum() / len(mine)),
                "disagree_fraction": _ratio(
                    total["disagree"].sum(), total["relabelled"].sum()
                ),
                "mixed_fraction": _ratio(
                    total["mixed"].sum(), total["relabelled"].sum()
                ),
                "multi_mixed_fraction": _ratio(
                    total["multi_mixed"].sum(), total["multi_sites"].sum()
                ),
                "visits_per_seed": float(total["visits"].sum() / len(mine)),
                "energy_change_per_seed": float(
                    total["energy_change"].sum() / len(mine)
                ),
            },
            "per_row": {
                "temperature": (total["temperature"] / len(mine)).tolist(),
                "mean_cluster": (total["size_sum"] / total["clusters"]).tolist(),
                "max_cluster": largest.tolist(),
                "accept_rate": np.divide(
                    total["accepts"],
                    total["proposals"],
                    out=np.full(DIAGNOSIS_ROWS, np.nan),
                    where=total["proposals"] > 0,
                ).tolist(),
                "relabelled": (total["relabelled"] / len(mine)).tolist(),
                "disagree_fraction": np.divide(
                    total["disagree"],
                    total["relabelled"],
                    out=np.full(DIAGNOSIS_ROWS, np.nan),
                    where=total["relabelled"] > 0,
                ).tolist(),
                "mixed_fraction": np.divide(
                    total["mixed"],
                    total["relabelled"],
                    out=np.full(DIAGNOSIS_ROWS, np.nan),
                    where=total["relabelled"] > 0,
                ).tolist(),
                "multi_mixed_fraction": np.divide(
                    total["multi_mixed"],
                    total["multi_sites"],
                    out=np.full(DIAGNOSIS_ROWS, np.nan),
                    where=total["multi_sites"] > 0,
                ).tolist(),
                "energy_change": (total["energy_change"] / len(mine)).tolist(),
                "visits": (total["visits"] / len(mine)).tolist(),
            },
        }
        rows = summary[name]["per_row"]
        summary[name]["per_row"] = {key: _rounded(row) for key, row in rows.items()}
    return summary


def _niedermayer_spend(task: tuple[int, int]) -> tuple[float, int]:
    """Niedermayer on Swendsen-Wang's tuned schedule at ``steps`` steps for one tuning seed: its energy and spend."""
    steps, seed = task
    rung = release_rung()
    run = run_annealed(
        rung,
        solver_budget(rung),
        np.random.default_rng([seed, 0]),
        PottsMove.NIEDERMAYER,
        schedule=load_tuned().chosen["swendsen-wang"],
        steps=steps,
    )
    return run.energy, run.spent


def matched_niedermayer(budget: Budget, start: int) -> tuple[int, list[dict[str, Any]]]:
    """Niedermayer's step count at which its mean spend on :data:`TUNING_SEEDS` meets ``budget``.

    Grows the count from ``start`` by :data:`MATCH_GROWTH` until the mean
    spend reaches the budget, then bisects, until a probe's mean spend is
    within :data:`MATCH_TOLERANCE` of the budget or the bracket is narrower
    than 1% of its lower end; the probe nearest the budget is returned, with
    every probe. Not a rescaling by ``budget / spend``: the spend grows
    faster than the count, since a longer anneal orders the lattice and the
    clusters grow with it, and a first search by rescaling from Wolff's
    41,250 probed 41,250, 64,577, 31,714 and 94,335 steps at 22.2, 70.7, 11.7
    and 164.5 million visits against the budget's 34.7 million.
    """
    probes: list[dict[str, Any]] = []

    def spend(steps: int) -> float:
        # One probe: the tuning seeds at `steps`, on the worker pool.
        runs = map_tasks(
            _niedermayer_spend,
            [(steps, seed) for seed in TUNING_SEEDS],
            workers=WORKERS,
            pool="processes",
            intra_op_threads=1,
        )
        spent = [visits for _, visits in runs]
        probes.append(
            {
                "steps": steps,
                "spent": spent,
                "mean_spent": float(np.mean(spent)),
                "energies": [value for value, _ in runs],
            }
        )
        return float(np.mean(spent))

    def close(mean: float) -> bool:
        return abs(mean - budget.size) <= MATCH_TOLERANCE * budget.size

    low, high = start, start
    mean = spend(high)
    while mean < budget.size and not close(mean):
        low, high = high, round(MATCH_GROWTH * high)
        mean = spend(high)
    while not close(mean) and (high - low) > 0.01 * low:
        middle = (low + high) // 2
        mean = spend(middle)
        if mean < budget.size:
            low = middle
        else:
            high = middle
    nearest = min(probes, key=lambda probe: abs(probe["mean_spent"] - budget.size))
    return int(nearest["steps"]), probes


def _run_start(
    run: Callable[[Rung, Budget, np.random.Generator], MethodRun],
) -> Callable[[np.random.Generator], RunStart]:
    """A start built per cell from its generator, charged at the notebook's budget."""
    return functools.partial(RunStart, run, solver_budget(release_rung()))


def arms() -> dict[str, Callable[[np.random.Generator], RunStart]]:
    """Every arm of the ticket that the diagnosis did not rule out, by name.

    Returns
    -------
    dict[str, Callable[[np.random.Generator], RunStart]]
    """
    sw = load_tuned().chosen["swendsen-wang"]
    matched = int(load()["matched_niedermayer"]["steps"])
    table: dict[str, Callable[[Rung, Budget, np.random.Generator], MethodRun]] = {
        # The reference arm: #1038's tuned schedule, rerun here.
        "swendsen-wang tuned": functools.partial(
            run_annealed, move=PottsMove.SWENDSEN_WANG, schedule=sw
        ),
        # Niedermayer's rule at its threshold on a ferromagnet: Wolff's bonds,
        # a transposition for a recolouring. Swendsen-Wang's tuned schedule,
        # at the step count matched on the tuning seeds.
        "niedermayer matched": functools.partial(
            run_annealed, move=PottsMove.NIEDERMAYER, schedule=sw, steps=matched
        ),
        "swendsen-wang>expansion": functools.partial(
            run_swendsen_wang_then_expansion, schedule=sw
        ),
        "expansion>swendsen-wang": functools.partial(
            run_expansion_then_swendsen_wang, schedule=EXPANSION_SW_SCHEDULE
        ),
        "ghost-spin": functools.partial(
            run_annealed, move=PottsMove.GHOST_SPIN, schedule=sw
        ),
        "label-directed": functools.partial(
            run_annealed, move=PottsMove.LABEL_DIRECTED, schedule=sw
        ),
        "tempering + houdayer": functools.partial(
            run_cluster_tempering, t_hot=sw.t_start, t_cold=sw.t_end
        ),
    }
    return {name: _run_start(run) for name, run in table.items()}


def run_arms(names: Sequence[str] | None = None) -> dict[str, Any]:
    """Every arm of :func:`arms` (or those named) on :data:`REPORTED_SEEDS`, polished as the notebook polishes.

    Returns
    -------
    dict[str, Any]
        Per arm, per seed: the handed-over and polished energies, the site
        visits, the start's and the cell's seconds, and the ICM sweeps.
    """
    rung = release_rung()
    budget = solver_budget(rung)
    reference = run_alpha_expansion(rung, budget, np.random.default_rng(0)).energy
    table = arms()
    chosen = {name: table[name] for name in (names or list(table))}
    result = StartsBenchmark(
        LabellingEnergy(rung),
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
        trials = result.trials(name)
        rows[name] = {
            "handed": [float(trial.seeded_value) for trial in trials],
            "polished": [float(trial.value) for trial in trials],
            "visits": [float(trial.diagnostics["site_visits"]) for trial in trials],
            "seconds": [float(trial.seconds[-1]) for trial in trials],
            "icm_sweeps": [int(trial.termination.iterations) for trial in trials],
            # Houdayer's cluster, for the one arm that records one.
            "cluster_mean": [
                float(trial.diagnostics.get("cluster_mean", float("nan")))
                for trial in trials
            ],
            "cluster_accept": [
                float(trial.diagnostics.get("cluster_accept", float("nan")))
                for trial in trials
            ],
            "below_reference": [
                {"seed": seed, "energy": energy(rung.graph, rung.field, labelling)}
                for seed, labelling, value in zip(
                    REPORTED_SEEDS,
                    (trial.theta.numpy() for trial in trials),
                    (trial.value for trial in trials),
                    strict=True,
                )
                if value < reference - 1e-9
            ],
        }
    return {"reference": reference, "arms": rows}


#: The parts :func:`main` runs, in the order each needs the last.
PARTS = ("diagnosis", "matched", "arms")


def main(parts: Sequence[str] = PARTS) -> None:
    """Run the parts named and merge them into :data:`OUTPUT`."""
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    rung = release_rung()
    budget = solver_budget(rung)
    result: dict[str, Any] = (
        json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {"issue": 1041}
    )
    result.update(
        {
            "instance": f"spatio_only/release, {rung.name}",
            "budget": budget.size,
            "mixed_floor": MIXED_FLOOR,
        }
    )
    for part in parts:
        load_before = os.getloadavg()
        opened = time.perf_counter()
        if part == "diagnosis":
            tasks = [
                (name, seed)
                for name in ("swendsen-wang", "wolff")
                for seed in TUNING_SEEDS
            ]
            runs = map_tasks(
                diagnose,
                tasks,
                workers=WORKERS,
                pool="processes",
                intra_op_threads=1,
            )
            result["diagnosis"] = {
                "tuning_seeds": list(TUNING_SEEDS),
                "rows": DIAGNOSIS_ROWS,
                "moves": summarise(runs),
            }
        elif part == "matched":
            steps, probes = matched_niedermayer(budget, load_tuned().matched_steps)
            result["matched_niedermayer"] = {
                "schedule": asdict(load_tuned().chosen["swendsen-wang"]),
                "steps": steps,
                "probes": probes,
            }
        elif part == "arms":
            result["arms"] = run_arms()
            result["arms"]["reserve_cycles"] = EXPANSION_RESERVE_CYCLES
            result["arms"]["replicas"] = CLUSTER_LADDER_REPLICAS
            result["arms"]["expansion_sw_schedule"] = asdict(EXPANSION_SW_SCHEDULE)
        else:
            msg = f"no part {part!r}; the parts are {PARTS}"
            raise ValueError(msg)
        result.setdefault("host", {})[part] = {
            "cores": os.cpu_count(),
            "workers": WORKERS,
            "threads": {
                variable: os.environ[variable] for variable in THREAD_VARIABLES
            },
            "load_before": list(load_before),
            "load_after": list(os.getloadavg()),
            "seconds": time.perf_counter() - opened,
        }
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
