"""Rust Boykov--Kolmogorov max flow (`snakes_and_ladders.oxi_snakes_and_ladders`), pinned against `snakes_and_ladders.search.maxflow`.

The NumPy/Python implementation --- Dinic, kept readable --- stays as the
oracle, per root ``CLAUDE.md`` ("Every accelerated kernel keeps its pure
Python/NumPy implementation as an oracle") and the same rule
``likelihood/CLAUDE.md`` states for pruning.

**Why this one is a Rust port and the samplers are not.** Root ``CLAUDE.md``
reserves the Rust backend for CPU-bound hot paths built from control flow and
irregular memory access, which is exactly a search tree or a level graph over
an adjacency structure: there is no array arithmetic here for NumPy to
vectorize, so the reference pays full Python interpreter cost per arc.

**Why Boykov--Kolmogorov.** Issue #715 built four kernels behind this seam
--- Dinic (the port issue #220 made), highest-label push-relabel,
Boykov--Kolmogorov and a synchronous parallel push-relabel on rayon ---
pinned each to the Python Dinic arc for arc, and timed them on square
lattices with a random per-node field and as the inner solver of alpha
expansion. Boykov--Kolmogorov won both tables: **61 ms** on the 256x256 cut
against Dinic's 234, push-relabel's 1,680 and the parallel kernel's 339, at
a fitted exponent of 1.03 in the site count against Dinic's 1.15; **352 ms**
on the 64x64 x 10-label expansion against Dinic's 637. The other three are
conserved in :mod:`snakes_and_ladders.sandbox.maxflow_declined` behind the
``sandbox`` Cargo feature with the tests that measured them, so the
comparison stays re-runnable. `STATUS.md` carries both tables.

**Two numbers, and both belong in any claim made here.** The kernel alone
against a caller of this wrapper: issue #336 measured the boundary copy at
0.03-0.2 ms of a 0.7-17 ms call and the term that remained was
:func:`snakes_and_ladders.sim.potts.energy`, which issue #341 vectorized
to 0.08, 0.28 and 1.0 ms at extents 16, 32 and 64.

A caller that needs only the configuration can call the extension directly
with the arrays this wrapper builds and skip that term; a caller with a
batch of fields calls :func:`ising_ground_states`, which is where threads pay.

Agreement is **exact**, not a tolerance: a ground state is a combinatorial
minimum, so the two implementations must report the same energy. The
configuration is read off the minimal minimum cut, which every maximum flow
shares, so it too is compared element for element.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.search.maxflow import FlowNetwork, MinCut
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import energy, site_field


def ising_ground_state(
    graph: PottsGraph, field_values: np.ndarray
) -> tuple[np.ndarray, float]:
    """The exact two-state ferromagnetic ground state, computed in Rust.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling must be non-negative.
    field_values : np.ndarray
        ``(2,)`` or ``(n_nodes, 2)``, as :func:`snakes_and_ladders.search.maxflow.ising_ground_state`.

    Returns
    -------
    tuple[np.ndarray, float]
        The ground-state configuration and its energy, the latter evaluated
        in Python on the returned configuration rather than read back from
        the cut. That keeps the reduction's arithmetic and the energy
        function independent, so a construction that were wrong could not
        also report itself as right.
    """
    values = site_field(
        np.asarray(field_values, dtype=float), graph.n_nodes, n_states=2
    )
    # `as_slice` on the Rust side succeeds only for a C-contiguous array, so
    # every argument is normalized here; `ascontiguousarray` is free when the
    # array already is one, and `site_field` already returns `float64`.
    states = oxi_snakes_and_ladders.ising_ground_state(
        graph.n_nodes,
        np.ascontiguousarray(values, dtype=np.float64).reshape(-1),
        graph.edge_index.reshape(-1),
        graph.edge_coupling,
    )
    configuration = np.asarray(states, dtype=np.int64)
    return configuration, energy(graph, values, configuration)


def ising_ground_states(
    graph: PottsGraph, fields: np.ndarray, threads: int | None = None
) -> np.ndarray:
    """The ground states of a batch of per-node fields on one graph.

    The cuts are independent, so the batch is the axis rayon takes: one
    network per field, solved on the pool, the GIL released for the whole
    call (root ``CLAUDE.md``, parallel over independent tasks). This is the
    entry point a sweep over instances or seeded starts calls, and the one
    place threads pay: eight 256x256 cuts took 2.16 s on one thread and 0.90 s
    on four of this 4-core host, and eight 64x64 cuts 77 ms and 21 (issue
    #715). Inside one cut nothing parallelizes --- the synchronous parallel
    push-relabel the ticket measured was no faster on four threads than on
    one, and is conserved in ``sandbox/maxflow_declined.py`` with the number.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling must be non-negative.
    fields : np.ndarray
        ``(batch, n_nodes, 2)``: one per-node field per instance.
    threads : int | None
        The pool's size; ``None`` is the global pool.

    Returns
    -------
    np.ndarray
        ``(batch, n_nodes)`` ``int64`` configurations.
    """
    batch = np.ascontiguousarray(fields, dtype=np.float64)
    if batch.ndim != 3 or batch.shape[1:] != (graph.n_nodes, 2):
        msg = f"fields must be (batch, {graph.n_nodes}, 2), got {batch.shape}"
        raise ValueError(msg)
    states = oxi_snakes_and_ladders.ising_ground_states(
        graph.n_nodes,
        batch.reshape(-1),
        graph.edge_index.reshape(-1),
        graph.edge_coupling,
        threads,
    )
    return np.asarray(states, dtype=np.int64).reshape(batch.shape[0], graph.n_nodes)


def min_cut(network: FlowNetwork, source: int, sink: int) -> MinCut:
    """Maximum flow and the minimum cut it certifies, computed in Rust.

    The same signature and the same return as
    :func:`snakes_and_ladders.search.maxflow.max_flow`, so a caller chooses
    the solver and nothing else. The network crosses once, as the three
    contiguous arrays :meth:`~snakes_and_ladders.search.maxflow.FlowNetwork.as_arrays`
    builds; back-arc capacities cross with them, since an undirected edge
    carries a capacity in both directions and a caller that had to pass such
    an edge twice would be building a second network.

    **The network is not mutated.** The pure implementation turns its
    capacities into residual capacities in place, and this one leaves them as
    they were: the residual graph lives in Rust and is dropped with the call.
    Only :attr:`MinCut.source_side` reports on it, which is what the callers
    here read. A caller that needs the residual capacities back needs the
    pure implementation.

    Until issue #528 the binding returned the flow value and discarded the
    side, so :mod:`snakes_and_ladders.search.alpha_expansion`, which needs
    the cut rather than the value, could not use it at all.
    """
    arcs, capacity, reverse = network.as_arrays()
    value, side = oxi_snakes_and_ladders.max_flow(
        network.n_nodes, arcs, capacity, source, sink, reverse
    )
    return MinCut(value=float(value), source_side=np.asarray(side, dtype=bool))
