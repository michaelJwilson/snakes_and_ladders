"""Compiled kernels for the deterministic Potts loops, beside their oracles.

Issue #264's audit ranked the Python-level loops of this package by self time
and found one that a compiled kernel removes outright: the single-site descent
sweep in :mod:`snakes_and_ladders.search.alpha_expansion`. It is integer arithmetic over the
compressed-row adjacency (:meth:`snakes_and_ladders.sim.graph.PottsGraph.compressed_adjacency`)
and deterministic given its inputs, so the oracle reproduces it **bitwise**:
the first minimum on a tie is the first minimum in both. The energy it
reports stays with the NumPy oracle, whose pairwise summation a sequential
loop would match only to the last place.

That exactness is why these are ``numba`` rather than Rust. Root
``CLAUDE.md``'s backend rule admits one compiled path per measurement; the
Rust extension already carries the *sampling* sweep, whose agreement with its
oracle can only be distributional (:mod:`snakes_and_ladders.search.potts_mcmc_rust`), and
a second copy of that reasoning is what the rule exists to refuse. A kernel
whose pin is exact carries no such cost and lives here.

The kernels take arrays, never the graph: the caller flattens once and the
kernel walks strides (the layout rule), and ``cache=True`` writes the compiled
object beside the source so the first call in a process pays once.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def icm_sweeps(
    labelling: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    max_sweeps: int,
) -> int:
    """Single-site descent in place; returns the sweeps run.

    The same update :func:`snakes_and_ladders.search.alpha_expansion.iterated_conditional_modes`
    states: each site takes the label minimizing ``-values[node, label] -
    sum_j J_ij [label == labelling[j]]`` given its neighbours, in index order,
    until a sweep changes nothing or ``max_sweeps`` is reached.
    """
    n_nodes = labelling.shape[0]
    n_states = values.shape[1]
    local = np.empty(n_states, dtype=np.float64)
    sweeps = 0
    for _ in range(max_sweeps):
        sweeps += 1
        changed = False
        for node in range(n_nodes):
            for state in range(n_states):
                local[state] = -values[node, state]
            for position in range(offsets[node], offsets[node + 1]):
                local[labelling[neighbours[position]]] -= couplings[position]
            best = 0
            for state in range(1, n_states):
                if local[state] < local[best]:
                    best = state
            if best != labelling[node]:
                labelling[node] = best
                changed = True
        if not changed:
            break
    return sweeps
