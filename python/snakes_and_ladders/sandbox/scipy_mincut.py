"""``scipy.sparse.csgraph.maximum_flow`` fronting the ground-state minimum cut (issues #322, #388).

The front that was measured and declined.
:func:`~snakes_and_ladders.search.maxflow.ising_ground_state` reduces a
two-state ferromagnet to a minimum cut and solves it with Dinic;
:func:`ising_ground_state` here builds the same network and hands it to
``scipy.sparse.csgraph.maximum_flow``, reading the cut back off the residual
graph because scipy returns a flow and not a partition.

**Declined on the clock against the implementation already fronting this
path**, recorded in `STATUS.md` and
`docs/experiments/008-frameworks-on-three-hot-paths.md`. Per
``ising_ground_state`` on a square lattice with a random per-node field: 2.15,
7.26 and 33.02 ms at extents 16, 32 and 64, against 7.59, 36.86 and 290.33 ms
for the Python Dinic and 0.51, 2.37 and 14.12 ms for the Rust Dinic of issue
#336. That is 3.5x to 8.8x the Python reference and 2.3x to 4.2x *slower* than
the compiled path a caller already gets, so adopting it would add a third
implementation of one algorithm and slow the caller down. One thread on a
shared four-core Linux x86-64 host under the exclusive lock, best of 5 (3 at
extent 64).

**The capacity scaling is the part with content, and it has a floor.** scipy's
``maximum_flow`` takes ``int32`` capacities and the reduction is real-valued,
so every capacity is multiplied by ``scale`` and rounded, and the rounded
problem can be minimised by a cut the real-valued one is not. Measured on 20
random fields on the extent-16 lattice at ``J = 0.6``: at 1e2 that happens on
3 of them, the chosen cut landing 7.18e-4, 1.92e-3 and 4.31e-3 above the
minimum; from 1e3 to 1e8 all 20 agree with both Dinic implementations at every
digit. Issue #388 swept the same range on one field and saw no difference,
which is why the range it recorded ran from 1e2.

**No bound is derived** relating the rounding error to the energy gap between
the best and second-best cut, so nothing here claims the energy is unchanged
at a scale, a lattice or a field that was not run.
`tests/regression/search/test_maxflow_scipy.py` pins what was run --- the
agreement above 1e3 and the three failures at 1e2 --- and says the same.

Imports ``scipy`` at module scope, which is a core dependency rather than the
``frameworks`` extra (issue #383): this module skips nowhere and is declined
on its numbers alone.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from snakes_and_ladders.search.maxflow import energy, site_field
from snakes_and_ladders.sim.graph import PottsGraph

#: Default capacity multiplier before rounding to ``int32``. 1e6 is inside the
#: range the energy was measured unchanged over --- three decimal orders above
#: the 1e2 that was not, and three below the ``int32`` ceiling for a capacity
#: of order one.
DEFAULT_SCALE = 1.0e6


def _network(
    graph: PottsGraph, values: np.ndarray, scale: float
) -> tuple[csr_matrix, int, int]:
    """The reduction's flow network as an ``int32`` CSR capacity matrix.

    The construction is
    :func:`~snakes_and_ladders.search.maxflow.ising_ground_state`'s, arc for
    arc: ``source -> i`` at ``D_i(1)``, ``i -> sink`` at ``D_i(0)``, the
    per-node minimum subtracted into a constant so every capacity is
    non-negative, and each undirected coupling as a pair of opposed arcs at
    ``J``. Only the arithmetic differs --- scaled and rounded to ``int32``.

    Returns
    -------
    tuple[csr_matrix, int, int]
        The capacities, the source and the sink.

    Raises
    ------
    ValueError
        If a scaled capacity does not fit ``int32``. Raised rather than
        wrapped: an overflowed capacity is a different flow problem, and its
        answer would be a lattice-shaped wrong one.
    """
    n_nodes = graph.n_nodes
    source, sink = n_nodes, n_nodes + 1
    nodes = np.arange(n_nodes)
    cost = -values
    offsets = cost.min(axis=1)
    ends = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    coupling = np.asarray(graph.coupling, dtype=float)

    rows = np.concatenate([np.full(n_nodes, source), nodes, ends[:, 0], ends[:, 1]])
    columns = np.concatenate([nodes, np.full(n_nodes, sink), ends[:, 1], ends[:, 0]])
    capacities = np.concatenate(
        [cost[:, 1] - offsets, cost[:, 0] - offsets, coupling, coupling]
    )

    scaled = np.rint(capacities * scale)
    if scaled.size and scaled.max() > np.iinfo(np.int32).max:
        msg = (
            f"capacity {scaled.max()} overflows int32 at scale {scale}: "
            "scipy's maximum_flow takes int32 capacities, and a scale that "
            "does not fit poses a different flow problem rather than a "
            "coarser one"
        )
        raise ValueError(msg)
    return (
        csr_matrix(
            (scaled.astype(np.int32), (rows, columns)),
            shape=(n_nodes + 2, n_nodes + 2),
        ),
        source,
        sink,
    )


def _source_side(capacities: csr_matrix, flow: csr_matrix, source: int) -> np.ndarray:
    """The nodes reachable from ``source`` on positive residual capacity.

    scipy returns a maximum *flow* and no partition, so the cut is recovered
    the way the max-flow min-cut theorem defines it. The residual is taken as
    a sparse difference so that a back arc carrying flow appears with positive
    residual even where its forward capacity is structurally absent.

    Parameters
    ----------
    capacities : csr_matrix
        The network handed to ``maximum_flow``.
    flow : csr_matrix
        Its ``flow`` attribute, antisymmetric.
    source : int
        The terminal to search from.

    Returns
    -------
    np.ndarray
        Boolean per node.
    """
    residual = (capacities - flow).tocsr()
    indptr, indices, data = residual.indptr, residual.indices, residual.data
    seen = np.zeros(residual.shape[0], dtype=bool)
    seen[source] = True
    stack = [source]
    while stack:
        node = stack.pop()
        for position in range(indptr[node], indptr[node + 1]):
            neighbour = int(indices[position])
            if data[position] > 0 and not seen[neighbour]:
                seen[neighbour] = True
                stack.append(neighbour)
    return seen


def ising_ground_state(
    graph: PottsGraph, field_values: np.ndarray, *, scale: float = DEFAULT_SCALE
) -> tuple[np.ndarray, float]:
    """The exact two-state ferromagnetic ground state, through scipy's max flow.

    Returns what
    :func:`~snakes_and_ladders.search.maxflow.ising_ground_state` returns and
    refuses what it refuses, so the two can be pinned against each other and
    timed against each other with no adapter in between. The energy is scored
    by :func:`~snakes_and_ladders.search.maxflow.energy` on the returned
    configuration, in the reduction's own real arithmetic, so the rounding
    below reaches which cut is chosen and never the number reported for it.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling must be non-negative.
    field_values : np.ndarray
        ``(2,)`` or ``(n_nodes, 2)``, as
        :func:`~snakes_and_ladders.search.maxflow.site_field` takes it.
    scale : float
        Capacity multiplier before rounding to ``int32``. See this module's
        docstring for what is and is not claimed about it.

    Returns
    -------
    tuple[np.ndarray, float]
        The ground-state configuration and its energy. Where the minimum is
        degenerate the configuration may differ from Dinic's; the energy may
        not.

    Raises
    ------
    ValueError
        If any coupling is negative --- the submodularity boundary, where the
        problem is NP-hard and no cut computes it --- if ``scale`` is not
        positive, or if a scaled capacity overflows ``int32``.
    """
    values = site_field(graph, field_values)
    couplings = np.asarray(graph.coupling, dtype=float)
    if couplings.size and couplings.min() < 0.0:
        msg = (
            f"every coupling must be non-negative, got {couplings.min()}: a "
            "negative coupling makes the energy non-submodular, the ground "
            "state NP-hard, and this construction inapplicable rather than slow"
        )
        raise ValueError(msg)
    if not scale > 0.0:
        msg = f"scale must be positive, got {scale}"
        raise ValueError(msg)

    capacities, source, sink = _network(graph, values, scale)
    result = maximum_flow(capacities, source, sink)
    reachable = _source_side(capacities, result.flow, source)
    configuration = (~reachable[: graph.n_nodes]).astype(np.int64)
    return configuration, float(energy(graph, values, configuration))


__all__ = ["DEFAULT_SCALE", "ising_ground_state"]
