"""Cluster moves against the graph cuts: the cluster tempering of issue #1041.

`search.ground_state` runs each move set alone, on one schedule, charged in
site visits. Issue #1041 asks whether a cluster move that reads the field ---
or one handed the graph cut's labelling, or one handing its labelling to the
cut --- reaches what alpha-expansion reaches on `spatio_only/release` at ten
states. The single-move arms are
:func:`~sal.search.ground_state.run_annealed` under another
:class:`~sal.sample.potts_mcmc.PottsMove`, and the two
hybrids, each charged both parts' visits, are
the ``swendsen-wang>expansion`` and ``expansion>swendsen-wang`` chains,
in :data:`~sal.search.ground_state.ARMS` since issue #1052.
This module holds the replica ladder.

The function here has the shape of a
:data:`~sal.search.ground_state.METHODS` entry once its
keywords are bound by ``functools.partial``, and is in no table:
`docs/nb/potts_starts.ipynb` reports ``METHODS`` as issue #906's solvers, and
an arm added there would add a row to tables the notebook reproduces.
"""

from __future__ import annotations

import time

import numpy as np

from sal.opt.budget import Budget
from sal.opt.termination import Termination
from sal.sample.potts_mcmc import (
    ClusterCounter,
    cluster_tempering,
)
from sal.search.ground_state import MethodRun, Rung

#: Replicas in :func:`run_cluster_tempering`'s ladder, the count
#: `search.ground_state.run_tempering` uses.
CLUSTER_LADDER_REPLICAS = 6


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

    :func:`~sal.sample.potts_mcmc.cluster_tempering` over
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
        # A step count fixed before the run: it ends on its budget (#1085).
        termination=Termination.after(steps, converged=False),
    )
