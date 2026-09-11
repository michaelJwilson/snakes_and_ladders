"""The seeding sweep of issue #541, run once and reported.

Not collected by ``pytest``: it is the experiment's own run, over more
instances than the release-tier test reproduces at. Writes one JSON that
``docs/experiments/009-projection-emission-seedings.md``, ``STATUS.md`` and the
pull-request body quote.

    python infra/seeding_sweep.py --out <path> [--workers N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from snakes_and_ladders.opt.budget import Budget, Method, compare, restarts
from snakes_and_ladders.opt.mixture import mixture_log_likelihood, responsibilities
from snakes_and_ladders.parallel import Backend, map_tasks
from snakes_and_ladders.search.projection import (
    SEEDINGS,
    CountPairAt,
    ProjectedCounts,
    SeededFit,
    fit_projection,
    flatten,
    project,
)
from snakes_and_ladders.sim.count_pairs import binned_model
from snakes_and_ladders.sim.fixtures import KEY, fixture
from snakes_and_ladders.sim.spatio_sequential import SpatioSequentialParams

PROBLEM = "spatio_sequential_counts"

#: Observations per projected instance: 40 per component at the key model's
#: 100, and ten seconds an expectation-maximization iteration.
N_SAMPLES = 4000

#: Instances the candidates are paired on. McNemar reads the discordant ones,
#: so this count is the evidence a winner is declared against.
N_INSTANCES = 6

#: Iterations of the projected fit every candidate is held to. Six, because a
#: seeding is a claim about where a fit starts and the key model's 100
#: components cost ten seconds an iteration.
BUDGET = Budget("passes", 6)

#: Restarts the baseline spends its budget on.
N_RESTARTS = 2

#: Relative tolerance on the projected log-likelihood at which an instance
#: counts as reached. A summed log-likelihood, so relative (``DEV.md``).
TOLERANCE = 1.0e-4


def instances(
    factor: int,
) -> tuple[list[ProjectedCounts], SpatioSequentialParams]:
    """The projected instances, and the model they are drawn from.

    Returns
    -------
    tuple[list[ProjectedCounts], SpatioSequentialParams]
    """
    declared = fixture(PROBLEM, KEY).params
    params = binned_model(declared.model, factor)
    drawn = [
        project(params, N_SAMPLES, np.random.default_rng([541, index]))
        for index in range(N_INSTANCES)
    ]
    return drawn, params


def seam(instance: ProjectedCounts, params: SpatioSequentialParams) -> CountPairAt:
    """The ``ComponentsAt`` every candidate ends at, its shapes declared once.

    Returns
    -------
    CountPairAt
    """
    truth = flatten(params)
    return CountPairAt(
        float(truth.total.dispersion.mean()),
        float(truth.successes.concentration.mean()),
        instance.trials,
    )


def bayes(instance: ProjectedCounts) -> dict[str, float]:
    """What the generating parameters reach: the reference every fit falls short of.

    Returns
    -------
    dict[str, float]
    """
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    log_weight = torch.log(torch.as_tensor(instance.weights, dtype=torch.float64))
    posterior = responsibilities(values, log_weight, instance.truth)
    return {
        "log_likelihood": float(
            mixture_log_likelihood(values, log_weight, instance.truth)
        ),
        "recovery": float(
            np.mean(posterior.argmax(dim=1).numpy() == instance.components)
        ),
    }


def _detail(task: tuple[ProjectedCounts, str, CountPairAt]) -> dict[str, object]:
    """One candidate's fit on one instance, with its curve and its recovery.

    Returns
    -------
    dict[str, object]
    """
    instance, name, at = task
    fitted = fit_projection(instance, name, at, BUDGET, np.random.default_rng([0, 0]))
    return {
        "name": name,
        "log_likelihoods": [float(value) for value in fitted.log_likelihoods],
        "iterations": fitted.iterations,
        "recovery": fitted.recovery,
        "mean_error": fitted.mean_error,
        "passes": fitted.seeding.passes,
        "diagnostics": fitted.seeding.diagnostics,
    }


def main() -> None:
    """Run the sweep and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()

    declared = fixture(PROBLEM, KEY).params
    drawn, params = instances(declared.key_factor)
    at = seam(drawn[0], params)

    workers = int(arguments.workers)
    methods: dict[str, Method[ProjectedCounts]] = {
        name: SeededFit(name, at) for name in SEEDINGS
    }
    methods["restart"] = restarts(SeededFit("prior", at), BUDGET.size // N_RESTARTS)

    comparison = compare(methods, drawn, BUDGET, [0], workers=workers)
    backend: Backend = "processes"
    detail = map_tasks(
        _detail,
        [(drawn[0], name, at) for name in methods if name != "restart"],
        workers=workers,
        backend=backend,
        intra_op_threads=None,
    )

    gaps = comparison.mean_gap()
    winner = min(gaps, key=lambda name: gaps[name])
    result = {
        "fixture": str(fixture(PROBLEM, KEY).path),
        "bin_factor": declared.key_factor,
        "n_components": drawn[0].n_components,
        "n_samples": N_SAMPLES,
        "n_instances": N_INSTANCES,
        "budget": {"unit": BUDGET.unit, "size": BUDGET.size},
        "methods": list(comparison.methods),
        "best": comparison.best.tolist(),
        "spent": comparison.spent.tolist(),
        "reference": comparison.reference.tolist(),
        "hits": comparison.hits(TOLERANCE, relative=True),
        "mean_gap": gaps,
        "table": comparison.table(),
        "bayes": bayes(drawn[0]),
        "detail": list(detail),
        "winner": winner,
        "paired_p": {
            name: comparison.paired_p(winner, name, TOLERANCE, relative=True)
            for name in comparison.methods
            if name != winner
        },
    }
    arguments.out.write_text(json.dumps(result, indent=1))
    print(comparison.table())
    print("winner", winner, result["paired_p"])


if __name__ == "__main__":
    main()
