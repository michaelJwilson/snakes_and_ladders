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
both belong in any claim made here**: the kernel alone is 28-34x the Python
reference, while a caller of this wrapper sees 6.6-10.6x. The difference is
the marshalling below --- flattening the edge list and the field into Python
lists that cross the boundary by copy. That is the same FFI copy gap issue
#199 measured for the categorical sampler and #202 closed with `rust-numpy`;
the same fix applies here and is deferred to it rather than solved twice.

A caller that solves the *same graph* repeatedly, which is what alpha
expansion (issue #207) does once per label per cycle, can hoist the edge and
coupling lists out of its loop and recover most of that gap without waiting.

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
    flat_edges = [node for edge in graph.edges for node in edge]
    states = oxi_snakes_and_ladders.ising_ground_state(
        graph.n_nodes,
        [float(value) for value in values.reshape(-1)],
        flat_edges,
        [float(coupling) for coupling in graph.coupling],
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
    flat = [node for arc in arcs for node in arc]
    return float(oxi_snakes_and_ladders.max_flow(n_nodes, flat, capacity, source, sink))
