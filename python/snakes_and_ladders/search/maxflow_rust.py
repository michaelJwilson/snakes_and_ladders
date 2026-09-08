"""Rust Dinic max-flow (`snakes_and_ladders.oxi_snakes_and_ladders`), pinned against `snakes_and_ladders.search.maxflow`.

The NumPy/Python implementation stays as the oracle, per root ``CLAUDE.md``
("Every accelerated kernel keeps its pure Python/NumPy implementation as an
oracle") and the same rule ``likelihood/CLAUDE.md`` states for pruning.

**Why this one is a Rust port and the samplers are not.** Root ``CLAUDE.md``
reserves the Rust backend for CPU-bound hot paths built from control flow and
irregular memory access, which is exactly a level graph and a blocking flow
over an adjacency structure: there is no array arithmetic here for NumPy to
vectorize, so the reference pays full Python interpreter cost per arc.

Measured on square lattices with a random per-node field, **two numbers, and
both belong in any claim made here**: the kernel alone is 26-32x the Python
reference, while a caller of this wrapper sees 6.3-10.7x. Issue #220 attributed
the difference to the Python lists that crossed the boundary by copy, and
issue #336 replaced them with `rust-numpy` buffers, the fix issue #202
applied to the categorical sampler. Measured, that copy was 0.03-0.2 ms of a
0.7-17 ms call at extents 16-64, and removing it moved the caller-visible
number by under 3%. The term that remained was
:func:`snakes_and_ladders.search.maxflow.energy`, which scored the returned
configuration edge by edge in Python: 0.6, 2.6 and 9.9 ms at extents 16, 32
and 64, against 0.14, 0.84 and 6.9 ms for the kernel. Issue #341 vectorized
it to 0.08, 0.28 and 1.0 ms, below the kernel at every extent. `STATUS.md`
carries both tables.

A caller that needs only the configuration can call the extension directly
with the arrays this wrapper builds and skip that term.

The port also removes a fragility rather than only a cost. The reference
recurses to the depth of the level graph, so a lattice past a few thousand
nodes needs ``sys.setrecursionlimit`` raised and a deep one is a stack
overflow rather than a slow answer. The Rust blocking flow uses an explicit
stack and has no such bound.

Agreement is **exact**, not a tolerance: a ground state is a combinatorial
minimum, so the two implementations must report the same energy. The
*configuration* may legitimately differ where the minimum is degenerate,
which is why the tests compare energies.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.search.maxflow import energy, site_field
from snakes_and_ladders.sim.graph import PottsGraph


def ising_ground_state(
    graph: PottsGraph, field_values: np.ndarray
) -> tuple[np.ndarray, float]:
    """The exact two-state ferromagnetic ground state, computed in Rust.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling must be non-negative.
    field_values : np.ndarray
        ``(2,)`` or ``(n_nodes, 2)``, as :func:`snakes_and_ladders.search.maxflow.site_field`.

    Returns
    -------
    tuple[np.ndarray, float]
        The ground-state configuration and its energy, the latter evaluated
        in Python on the returned configuration rather than read back from
        the cut. That keeps the reduction's arithmetic and the energy
        function independent, so a construction that were wrong could not
        also report itself as right.
    """
    values = site_field(graph, field_values)
    # `as_slice` on the Rust side succeeds only for a C-contiguous array, so
    # every argument is normalized here; `ascontiguousarray` is free when the
    # array already is one, and `site_field` already returns `float64`.
    states = oxi_snakes_and_ladders.ising_ground_state(
        graph.n_nodes,
        np.ascontiguousarray(values, dtype=np.float64).reshape(-1),
        np.asarray(graph.edges, dtype=np.int64).reshape(-1),
        np.asarray(graph.coupling, dtype=np.float64),
    )
    configuration = np.asarray(states, dtype=np.int64)
    return configuration, float(energy(graph, values, configuration))


def max_flow(
    n_nodes: int,
    arcs: list[tuple[int, int]],
    capacity: list[float],
    source: int,
    sink: int,
) -> float:
    """Maximum flow on an explicit directed network, computed in Rust.

    Back arcs are added with zero capacity, so an undirected edge is passed
    twice. The Python side deliberately keeps no network object: a structure
    mirrored on both sides of the boundary is a structure that can fall out
    of step, and the only thing a caller needs back is a number.
    """
    return float(
        oxi_snakes_and_ladders.max_flow(
            n_nodes,
            np.asarray(arcs, dtype=np.int64).reshape(-1),
            np.asarray(capacity, dtype=np.float64),
            source,
            sink,
        )
    )
