"""The local-polytope bound on a Potts ground state, by sequential tree-reweighted message passing (issue #1060).

TRW-S (Kolmogorov 2006) maximizes the dual of the local-polytope linear
relaxation of ``min_x E(x)``, the relaxation
:func:`sal.search.tightening.dual_bound` maximizes with no plaquettes, by
message passing in a fixed site order. :func:`trws` returns the dual value, a
**lower** bound on the minimum energy, and the lowest-energy labelling decoded
on the way, an upper bound; their difference is the gap.

**The decomposition.** Order the sites by index and orient every edge from its
lower end to its higher. At site ``s`` link the ``i``-th edge arriving from a
lower site to the ``i``-th edge leaving to a higher one, in edge order. The
links cut the edges into monotone chains, each edge in exactly one, and site
``s`` in ``n_s = max(in_s, out_s, 1)`` of them. Give each chain the energy
``theta_bar_s / n_s`` at its sites and ``theta_bar_st`` on its edges, where
``theta_bar`` is the energy reparametrized by the messages. The chains sum to
``theta_bar``, which is ``E`` at every labelling, so ``sum_T min_x E_T(x)`` is
at most ``min_x E(x)`` **for any messages at all**. The bound is computed that
way, explicitly, after every iteration; its validity rests on the sum and not
on the updates.

**The updates.** A forward pass visits the sites ascending; at site ``s`` it
forms ``theta_hat_s = theta_s + sum_u M_us`` from every incoming message and
sends each later neighbour ``t``

``M_st(x_t) = min_(x_s) [theta_hat_s(x_s) / n_s - M_ts(x_s) + theta_st(x_s, x_t)]``,

shifted to minimum zero. A backward pass does the same descending. With
``gamma_st = 1 / n_s`` this is Kolmogorov's update for the chains above, and
under it the bound does not decrease; the tests assert that at every
iteration rather than rely on it. The Potts table is ``-J`` on the diagonal
and zero off it, so the minimum over ``x_s`` is the smallest entry, or the
diagonal, per ``x_t``: ``O(q)`` a message rather than ``O(q^2)``.

**The decode.** After each iteration the sites are labelled in index order,
each taking the first state minimizing its field, the couplings to the
already labelled lower neighbours, and the messages from the higher ones,
as Kolmogorov (2006) decodes. The lowest-energy labelling over the
iterations is kept.

**Sign.** ``sim.potts.energy`` is ``-sum h - sum J [x_s = x_t]``, which a
ground state minimizes; this module works in that sign throughout, so
``theta_s = -h_s`` and the bound is a lower bound on the energy.
:func:`~sal.search.tightening.dual_bound` works on the log-weight, its
negative, and turns the sign once at its return.

**What the bound is not.** It is the value of the local-polytope LP where the
passes reach that LP's maximum, which coordinate ascent does not guarantee in
general (Kolmogorov 2006, weak tree agreement). The trace is monotone and
every value on it is a bound; its limit is measured against
``dual_bound``'s, not assumed equal to it. On two frustrated 3x3 triangular
lattices it converges 0.016 and 0.0067 below ``dual_bound``'s value, and
agrees with it to 1e-9 relative on the six other small lattices and the two
CI fixtures tested (``tests/regression/search/test_trws.py``). Against
HiGHS's solution of the explicit LP (``tests/validation/test_highs.py``,
issue #1063) it agrees to 1e-8 relative wherever it converges to it, and on
``spatio_tiling/release`` it stops 0.0496 below it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sal.backend import Backend, refuse_backend
from sal.incidence import SparseIncidence
from sal.opt.termination import Termination, check_cap
from sal.search.alpha_expansion import BoundedLabelling
from sal.search.trws.numba import trws_iterations_checked
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, energy, log_weight_of, site_field

#: Iterations :func:`trws` runs at most.
MAX_ITERATIONS = 5000
#: The relative rise of the bound per iteration below which the run has converged.
TOLERANCE = 1e-12


@dataclass(frozen=True, kw_only=True)
class TrwsResult(BoundedLabelling):
    """TRW-S's :class:`~sal.search.alpha_expansion.BoundedLabelling`, and its bound's trace.

    ``bound`` is the largest chain bound over the iterations, ``labelling``
    the lowest-energy one decoded over them, ``int64``, and ``termination``
    :attr:`~sal.opt.termination.Stop.CONVERGED` where the bound rose by at
    most ``tolerance * max(1, |bound|)`` in an iteration or the gap closed to
    that, :attr:`~sal.opt.termination.Stop.BUDGET` where ``max_iterations``
    ran out.

    Parameters
    ----------
    trace : np.ndarray
        The chain bound after each iteration, ``float64``, one per iteration
        taken.
    """

    trace: np.ndarray

    @property
    def iterations(self) -> int:
        """Iterations taken: a forward and a backward pass each."""
        return self.termination.iterations


@dataclass(frozen=True)
class ChainLayout:
    """A graph as TRW-S reads it: entries, rows, chain weights and the chains.

    Edge ``e`` joins ``ends[e, 0] < ends[e, 1]``. Entry ``2e`` sits at the
    lower end and names the higher, entry ``2e + 1`` the reverse, so an
    entry's reverse is ``k ^ 1``.

    Parameters
    ----------
    ends : np.ndarray
        ``(n_edges, 2)`` ``int64``, lower end first.
    coupling : np.ndarray
        ``J`` per edge, ``float64``.
    neighbours : np.ndarray
        The site each entry names, ``(2 * n_edges,)``.
    offsets, slots : np.ndarray
        Site ``s``'s entries are ``slots[offsets[s]:offsets[s + 1]]``, in edge
        order.
    weight : np.ndarray
        ``1 / n_s`` per site.
    chain_offsets, chain_heads, chain_edges : np.ndarray
        Chain ``c`` starts at site ``chain_heads[c]`` and walks the edges
        ``chain_edges[chain_offsets[c]:chain_offsets[c + 1]]``, lower end to
        higher; a site with no edge is a chain of no edges.
    """

    ends: np.ndarray
    coupling: np.ndarray
    neighbours: np.ndarray
    offsets: np.ndarray
    slots: np.ndarray
    weight: np.ndarray
    chain_offsets: np.ndarray
    chain_heads: np.ndarray
    chain_edges: np.ndarray


def chain_layout(graph: PottsGraph) -> ChainLayout:
    """The entries, rows, weights and monotone chains of ``graph`` in index order.

    Raises
    ------
    ValueError
        If an edge joins a site to itself: its term is a constant, and it has
        no lower and higher end to orient.
    """
    n_nodes = graph.n_nodes
    index = graph.edge_index
    if bool((index[:, 0] == index[:, 1]).any()):
        loop = int(np.flatnonzero(index[:, 0] == index[:, 1])[0])
        msg = f"edge {loop} joins site {int(index[loop, 0])} to itself"
        raise ValueError(msg)
    ends = np.ascontiguousarray(np.sort(index, axis=1))
    n_edges = ends.shape[0]
    owners = np.empty(2 * n_edges, dtype=np.int64)
    owners[0::2], owners[1::2] = ends[:, 0], ends[:, 1]
    neighbours = np.empty(2 * n_edges, dtype=np.int64)
    neighbours[0::2], neighbours[1::2] = ends[:, 1], ends[:, 0]
    # Each site's entries in edge order: the package's one row builder (#277).
    rows = SparseIncidence.from_pairs(n_nodes, n_nodes, owners, neighbours)
    leaving = np.bincount(ends[:, 0], minlength=n_nodes)
    arriving = np.bincount(ends[:, 1], minlength=n_nodes)
    weight = 1.0 / np.maximum(np.maximum(leaving, arriving), 1).astype(np.float64)

    # The i-th edge arriving at a site continues as the i-th edge leaving it.
    outgoing: list[list[int]] = [[] for _ in range(n_nodes)]
    rank = np.empty(n_edges, dtype=np.int64)
    seen = [0] * n_nodes
    for edge, (low, high) in enumerate(ends.tolist()):
        outgoing[low].append(edge)
        rank[edge] = seen[high]
        seen[high] += 1
    heads: list[int] = []
    walked: list[int] = []
    bounds = [0]
    for node in range(n_nodes):
        if not outgoing[node] and not arriving[node]:
            heads.append(node)
            bounds.append(len(walked))
        # Edges leaving past the arrivals start a chain here.
        for first in outgoing[node][int(arriving[node]) :]:
            heads.append(node)
            edge = first
            while edge >= 0:
                walked.append(edge)
                high = int(ends[edge, 1])
                step = int(rank[edge])
                edge = outgoing[high][step] if step < len(outgoing[high]) else -1
            bounds.append(len(walked))
    return ChainLayout(
        ends=ends,
        coupling=np.ascontiguousarray(graph.edge_coupling, dtype=np.float64),
        neighbours=neighbours,
        offsets=np.ascontiguousarray(rows.offsets, dtype=np.int64),
        slots=np.ascontiguousarray(rows.order, dtype=np.int64),
        weight=weight,
        chain_offsets=np.asarray(bounds, dtype=np.int64),
        chain_heads=np.asarray(heads, dtype=np.int64),
        chain_edges=np.asarray(walked, dtype=np.int64),
    )


def trws(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    *,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
    backend: Backend = Backend.NUMBA,
) -> TrwsResult:
    """Bound the Potts ground-state energy from below by TRW-S, and decode under it.

    ``backend`` chooses the implementation and nothing else. The
    :data:`~sal.backend.Backend.NUMBA` kernel,
    :func:`~sal.search.trws.numba.trws_iterations`, runs the reference's
    arithmetic in the reference's order and returns its trace and labelling
    **bitwise**; :data:`~sal.backend.Backend.PYTHON` is the reference, and
    the oracle that pins it.

    Parameters
    ----------
    graph : PottsGraph
        The graph and its per-edge couplings, of either sign. No edge may join
        a site to itself.
    field : SiteField | np.ndarray
        External field as a log-weight, ``(n_states,)`` or
        ``(n_nodes, n_states)``, or a :class:`~sal.sim.potts.SiteField`.
    max_iterations : int
        Iterations allowed, each a forward and a backward pass.
    tolerance : float
        The run stops once the bound rises by at most
        ``tolerance * max(1, |bound|)`` in an iteration, or the gap closes to
        that.
    backend : Backend
        :data:`~sal.backend.Backend.NUMBA` or
        :data:`~sal.backend.Backend.PYTHON`.

    Returns
    -------
    TrwsResult
        The bound, the labelling, its energy, the trace and the termination.

    Raises
    ------
    ValueError
        If the field has neither shape, an edge is a self-loop,
        ``max_iterations < 1``, ``tolerance < 0``, or ``backend`` is neither
        of the two.
    """
    refuse_backend("TRW-S", backend, (Backend.NUMBA, Backend.PYTHON))
    values = site_field(
        np.asarray(log_weight_of(field), dtype=np.float64), graph.n_nodes
    )
    unary = np.ascontiguousarray(-values)
    layout = chain_layout(graph)
    check_cap("max_iterations", max_iterations)
    if backend is Backend.NUMBA:
        _, labelling, trace, _, taken, converged = trws_iterations_checked(
            unary,
            layout.offsets,
            layout.slots,
            layout.neighbours,
            layout.ends,
            layout.coupling,
            layout.weight,
            layout.chain_offsets,
            layout.chain_heads,
            layout.chain_edges,
            int(max_iterations),
            float(tolerance),
        )
    else:
        if not tolerance >= 0.0:
            msg = f"tolerance must be >= 0, got {tolerance}"
            raise ValueError(msg)
        labelling, trace, taken, converged = _reference(
            unary, layout, max_iterations, tolerance
        )
    return TrwsResult(
        bound=float(trace.max()),
        labelling=labelling,
        energy=energy(graph, values, labelling),
        trace=trace,
        termination=Termination.after(taken, converged=converged),
    )


def _share(
    node: int, unary: np.ndarray, layout: ChainLayout, messages: np.ndarray
) -> np.ndarray:
    """``theta_s + sum_k M_k``, the incoming messages added in row order."""
    share: np.ndarray = unary[node].copy()
    for position in range(layout.offsets[node], layout.offsets[node + 1]):
        share += messages[layout.slots[position]]
    return share


def _potts_min(values: np.ndarray, coupling: float) -> np.ndarray:
    """``min_a [values[a] - coupling [a == b]]`` per ``b``: the kernel's ``_potts_min``."""
    first = int(np.argmin(values))
    others = np.delete(values, first)
    other = np.full(values.shape[0], values[first])
    other[first] = others.min() if others.size else np.inf
    smallest: np.ndarray = np.minimum(values - coupling, other)
    return smallest


def _reference(
    unary: np.ndarray, layout: ChainLayout, max_iterations: int, tolerance: float
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """The Python reference: the kernel's passes, bound and decode, in its order."""
    n_nodes, n_states = unary.shape
    offsets, slots = layout.offsets.tolist(), layout.slots.tolist()
    neighbours, weight = layout.neighbours.tolist(), layout.weight.tolist()
    coupling = layout.coupling.tolist()
    messages = np.zeros((layout.slots.shape[0], n_states))
    labelling = np.zeros(n_nodes, dtype=np.int64)
    trace: list[float] = []
    best_energy = np.inf
    for _ in range(max_iterations):
        for parity, order in ((0, range(n_nodes)), (1, range(n_nodes - 1, -1, -1))):
            for node in order:
                share = _share(node, unary, layout, messages)
                for position in range(offsets[node], offsets[node + 1]):
                    entry = slots[position]
                    if entry % 2 != parity:
                        continue
                    lent = weight[node] * share - messages[entry]
                    sent = _potts_min(lent, coupling[entry // 2])
                    messages[entry ^ 1] = sent - sent.min()
        bound = _bound(unary, layout, messages)
        trace.append(bound)
        labels, current = _decode(unary, offsets, slots, neighbours, coupling, messages)
        if current < best_energy:
            best_energy = current
            labelling = labels
        scale = max(1.0, abs(bound))
        if best_energy - bound <= tolerance * scale:
            return labelling, np.asarray(trace), len(trace), True
        if len(trace) > 1 and bound - trace[-2] <= tolerance * scale:
            return labelling, np.asarray(trace), len(trace), True
    return labelling, np.asarray(trace), len(trace), False


def _bound(unary: np.ndarray, layout: ChainLayout, messages: np.ndarray) -> float:
    """``sum_T min_x E_T(x)`` over the chains: the kernel's ``_chain_bound``."""
    shares = [_share(node, unary, layout, messages) for node in range(unary.shape[0])]
    weight = layout.weight
    total = 0.0
    edges = layout.chain_edges.tolist()
    bounds = layout.chain_offsets.tolist()
    for chain, head in enumerate(layout.chain_heads.tolist()):
        value = weight[head] * shares[head]
        for edge in edges[bounds[chain] : bounds[chain + 1]]:
            high = int(layout.ends[edge, 1])
            sent = _potts_min(value - messages[2 * edge], float(layout.coupling[edge]))
            value = sent - messages[2 * edge + 1] + weight[high] * shares[high]
        total += float(value.min())
    return total


def _decode(
    unary: np.ndarray,
    offsets: list[int],
    slots: list[int],
    neighbours: list[int],
    coupling: list[float],
    messages: np.ndarray,
) -> tuple[np.ndarray, float]:
    """The kernel's ``_decode``: the labelling, and its energy summed in the kernel's order."""
    n_nodes = unary.shape[0]
    labels = [0] * n_nodes
    for node in range(n_nodes):
        local = unary[node].copy()
        for position in range(offsets[node], offsets[node + 1]):
            entry = slots[position]
            other = neighbours[entry]
            if other < node:
                local[labels[other]] -= coupling[entry // 2]
            else:
                local += messages[entry]
        labels[node] = int(np.argmin(local))
    total = 0.0
    for node in range(n_nodes):
        total += float(unary[node, labels[node]])
    for edge, value in enumerate(coupling):
        if labels[neighbours[2 * edge + 1]] == labels[neighbours[2 * edge]]:
            total -= value
    return np.asarray(labels, dtype=np.int64), total
