"""Cluster moves against the graph cuts: the hybrids and the cluster tempering of issue #1041.

`search.ground_state` runs each move set alone, on one schedule, charged in
site visits. Issue #1041 asks whether a cluster move that reads the field ---
or one handed the graph cut's labelling, or one handing its labelling to the
cut --- reaches what alpha-expansion reaches on `spatio_only/release` at ten
states. The single-move arms are
:func:`~snakes_and_ladders.search.ground_state.run_annealed` under another
:class:`~snakes_and_ladders.sample.potts_mcmc.PottsMove`; this module holds
the arms built from two parts, each charged both parts' visits.

Every function here has the shape of a
:data:`~snakes_and_ladders.search.ground_state.METHODS` entry once its
keywords are bound by ``functools.partial``, and none is in that table:
`docs/nb/potts_starts.ipynb` reports ``METHODS`` as issue #906's solvers, and
an arm added there would add a row to tables the notebook reproduces.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.sample.potts_mcmc import (
    ClusterCounter,
    PottsMove,
    cluster_tempering,
)
from snakes_and_ladders.sample.schedule import ScheduleParams
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.ground_state import (
    MethodRun,
    Rung,
    run_alpha_expansion,
    run_annealed,
)

#: Cycles of the expansion :func:`run_swendsen_wang_then_expansion` holds
#: back from Swendsen-Wang's share. From a uniform start the expansion ends in
#: 3 cycles on `spatio_only/release` at ten states; 10 leaves it room from a
#: labelling that is not its own.
EXPANSION_RESERVE_CYCLES = 10
#: Replicas in :func:`run_cluster_tempering`'s ladder, the count
#: `search.ground_state.run_tempering` uses.
CLUSTER_LADDER_REPLICAS = 6


def run_swendsen_wang_then_expansion(
    rung: Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    schedule: ScheduleParams,
    reserve_cycles: int = EXPANSION_RESERVE_CYCLES,
) -> MethodRun:
    """Swendsen-Wang on ``schedule``, then alpha-expansion from its labelling.

    The anneal runs on ``budget`` less ``reserve_cycles`` expansion cycles,
    by :func:`~snakes_and_ladders.search.ground_state.run_annealed`'s rule;
    the expansion starts from the anneal's best labelling with what is left
    as its cap, and is charged the cycles it ran.

    Returns
    -------
    MethodRun
        The expansion's labelling and energy, ``spent`` both parts summed.
    """
    started = time.perf_counter()
    per_cycle = rung.n_states * rung.visits_per_sweep
    anneal = run_annealed(
        rung,
        Budget(budget.unit, budget.size - reserve_cycles * per_cycle),
        rng,
        PottsMove.SWENDSEN_WANG,
        schedule=schedule,
    )
    left = budget.size - anneal.spent
    expansion = alpha_expansion(
        rung.graph,
        rung.field,
        rung.n_states,
        start=anneal.labelling,
        max_cycles=max(1, left // per_cycle),
        backend=Backend.RUST,
    )
    return MethodRun(
        labelling=expansion.labelling,
        energy=expansion.energy,
        spent=anneal.spent + expansion.cycles * per_cycle,
        seconds=time.perf_counter() - started,
        termination=expansion.termination,
    )


def run_expansion_then_swendsen_wang(
    rung: Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    schedule: ScheduleParams,
) -> MethodRun:
    """Alpha-expansion, then Swendsen-Wang on ``schedule`` from its labelling.

    The expansion runs as
    :func:`~snakes_and_ladders.search.ground_state.run_alpha_expansion` does
    and is charged its cycles; the anneal gets the rest of ``budget`` by
    :func:`~snakes_and_ladders.search.ground_state.run_annealed`'s rule. The
    anneal returns the lowest energy it visited, its start included, so the
    arm hands over the expansion's energy or lower.

    Returns
    -------
    MethodRun
        The anneal's labelling and energy, ``spent`` both parts summed.
    """
    started = time.perf_counter()
    expansion = run_alpha_expansion(rung, budget, rng)
    anneal = run_annealed(
        rung,
        Budget(budget.unit, budget.size - expansion.spent),
        rng,
        PottsMove.SWENDSEN_WANG,
        schedule=schedule,
        initial=expansion.labelling,
    )
    return replace(
        anneal,
        spent=expansion.spent + anneal.spent,
        seconds=time.perf_counter() - started,
    )


def run_cluster_tempering(
    rung: Rung,
    budget: Budget,
    rng: np.random.Generator,
    *,
    t_hot: float,
    t_cold: float,
    n_replicas: int = CLUSTER_LADDER_REPLICAS,
    houdayer_pairs: int = 1,
) -> MethodRun:
    """Swendsen-Wang replicas on a geometric ladder, Houdayer moves at its cold end.

    :func:`~snakes_and_ladders.sample.potts_mcmc.cluster_tempering` over
    ``n_replicas`` temperatures from ``t_cold`` to ``t_hot``. A step costs
    one sweep's visits per replica and one per Houdayer pair, so the budget
    buys ``budget // ((n_replicas + houdayer_pairs) * visits_per_sweep)``
    steps, fixed before the run.

    Returns
    -------
    MethodRun
        The lowest-energy labelling seen at any temperature.
    """
    steps = max(
        1, budget.size // ((n_replicas + houdayer_pairs) * rung.visits_per_sweep)
    )
    ladder = tuple(float(value) for value in np.geomspace(t_cold, t_hot, n_replicas))
    started = time.perf_counter()
    run = cluster_tempering(
        rung.graph,
        rung.field,
        ladder,
        rng,
        steps,
        houdayer_pairs=houdayer_pairs,
    )
    # The Houdayer moves as one counter, so the run reports its cluster.
    counter = ClusterCounter(
        sizes=list(run.houdayer_sizes),
        proposals=len(run.houdayer_sizes),
        accepts=run.houdayer_accepts,
    )
    return MethodRun(
        labelling=run.best,
        energy=run.best_energy,
        spent=run.site_visits,
        seconds=time.perf_counter() - started,
        trace=(counter,),
    )
