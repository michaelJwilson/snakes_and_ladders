"""Max flow, and the exact ground state of a ferromagnetic Ising model.

Two states, every coupling non-negative, an arbitrary external field: that
energy is submodular, and a minimum cut gives its exact global minimum in
polynomial time. It is the only case in this repository where a discrete
optimum is *proved* rather than enumerated, which is what makes it worth
having --- enumeration stops at about twenty sites, and a heuristic past that
point has had nothing to be checked against.

**Where this stops, and why it is refused rather than approximated.** A
negative coupling makes the energy non-submodular; finding the ground state is
then NP-hard and no cut computes it. More than two states is not a cut problem
at all --- that is alpha expansion (issue #207), which uses this as its inner
solver and calls it once per label.

Dinic rather than push-relabel. Push-relabel has the better worst case and
Boykov-Kolmogorov is faster still on grid graphs, but this exists to be an
*oracle*, and the property that matters first is that it is checkable: level
graph, blocking flow, and the max-flow min-cut theorem as a self-check on
termination. Replacing the inner solver behind this interface is a later
change with these tests already standing.

See Cormen et al. ch. 26; Boykov, Veksler & Zabih (2001) for the reduction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.incidence import SparseIncidence
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import energy, site_field


@dataclass
class FlowNetwork:
    """A directed graph with capacities, held as paired residual arcs.

    Arc ``2 * e`` and arc ``2 * e + 1`` are the two directions of one edge, so
    the reverse of an arc is its index with the low bit flipped. Pushing flow
    subtracts from one and adds to the other, which is what makes the residual
    graph implicit rather than a second structure to keep in step.

    **Why ``outgoing`` is a list of lists.** Root ``CLAUDE.md`` asks for
    neighbour lists as offsets into one array, and the survey issue #586 ran
    reported this class for not being one. It was measured rather than
    converted, and the measurement says not to: over 16,384 rows of degree
    six, walking every row costs 1.95 ms as a list of lists, 3.93 ms as a
    flat Python list with offsets, and 25.63 ms as a NumPy array sliced per
    row. The rule is about a consumer that walks the layout in NumPy or in a
    compiled kernel; Dinic here is a pure-Python inner loop, where a row is
    one list index and a NumPy slice is an object allocation. The compiled
    consumer has the contiguous form already ---
    :meth:`as_arrays` builds it for
    :mod:`snakes_and_ladders.search.maxflow_rust`, and that is the boundary
    the layout rule is about.

    ``add_edge`` also appends, which offsets cannot do without knowing the
    degrees first. :meth:`from_arcs` is the case where they *are* known --- a
    caller holding every arc at once --- and it groups them with the same
    counting sort :class:`snakes_and_ladders.incidence.SparseIncidence` does,
    then hands back the list of lists Dinic wants (issue #598).
    """

    n_nodes: int
    target: list[int] = field(default_factory=list)
    capacity: list[float] = field(default_factory=list)
    outgoing: list[list[int]] = field(default_factory=list)

    #: The contiguous form :meth:`as_arrays` hands the compiled consumer, kept
    #: when it is already known rather than derived again (issue #642).
    #: :meth:`from_arcs` builds these arrays on its way to the list store and
    #: used to discard them, so a network built from arcs and solved in Rust
    #: made the round trip NumPy -> list -> NumPy for nothing. ``None`` means
    #: *not known*, never *empty*: it is what every mutation sets, and
    #: :meth:`as_arrays` derives the form from the lists whenever it is unset,
    #: so the lists remain the single store and this is only ever a shortcut
    #: to a value they would have produced.
    _arcs: np.ndarray | None = field(default=None, repr=False, compare=False)
    _forward: np.ndarray | None = field(default=None, repr=False, compare=False)
    _backward: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.outgoing:
            self.outgoing = [[] for _ in range(self.n_nodes)]

    def _forget_arrays(self) -> None:
        """Drop the contiguous form, because the lists it was derived from moved.

        Every write to ``target``, ``capacity`` or ``outgoing`` calls this.
        A cache that outlives its store is the defect this class would be
        trading for the one it fixes, and the two writers --- :meth:`add_edge`
        and :func:`max_flow`'s in-place push --- are the whole set.
        """
        self._arcs = self._forward = self._backward = None

    @classmethod
    def from_arcs(
        cls,
        n_nodes: int,
        tail: np.ndarray,
        head: np.ndarray,
        capacity: np.ndarray,
        reverse: np.ndarray,
    ) -> FlowNetwork:
        """Every arc at once, in the order :meth:`add_edge` would have appended them.

        The same network as calling :meth:`add_edge` once per row, built
        without a Python call per arc.
        :func:`snakes_and_ladders.search.alpha_expansion.expand` builds one
        network per label per cycle and made 76,928 of those calls on a 32x32
        lattice at four labels --- 53.0 per cent of the run against the Rust
        cut kernel's 12.1 (issue #598).

        The arc order is the contract, not an implementation detail: a
        minimum cut need not be unique, and the tests pin the Python and Rust
        solvers to the same labelling, so a different order could return a
        different cut of the same capacity.

        Parameters
        ----------
        n_nodes : int
            Nodes the network spans.
        tail, head : np.ndarray
            One row per edge: the arc runs ``tail -> head``.
        capacity, reverse : np.ndarray
            The forward and back arc capacity of each edge.

        Returns
        -------
        FlowNetwork

        Raises
        ------
        ValueError
            If the four arrays differ in length, or any capacity is negative
            --- the same refusal :meth:`add_edge` makes, and for the same
            reason.
        """
        tail = np.asarray(tail, dtype=np.int64)
        head = np.asarray(head, dtype=np.int64)
        forward = np.asarray(capacity, dtype=np.float64)
        backward = np.asarray(reverse, dtype=np.float64)
        if not (tail.shape == head.shape == forward.shape == backward.shape):
            msg = (
                f"tail, head, capacity and reverse must agree in shape, got "
                f"{tail.shape}, {head.shape}, {forward.shape}, {backward.shape}"
            )
            raise ValueError(msg)
        if bool(np.any(forward < 0.0)) or bool(np.any(backward < 0.0)):
            msg = "capacities must be non-negative"
            raise ValueError(msg)

        n_arcs = 2 * tail.size
        target = np.empty(n_arcs, dtype=np.int64)
        target[0::2], target[1::2] = head, tail
        capacities = np.empty(n_arcs, dtype=np.float64)
        capacities[0::2], capacities[1::2] = forward, backward
        # Arc `2 * e` leaves the tail and arc `2 * e + 1` the head, which is
        # the order `add_edge` appends them in.
        leaves = np.empty(n_arcs, dtype=np.int64)
        leaves[0::2], leaves[1::2] = tail, head
        grouped = SparseIncidence.from_pairs(
            n_nodes, n_arcs, leaves, np.arange(n_arcs, dtype=np.int64)
        )
        # The contiguous form `as_arrays` would rebuild, kept rather than
        # derived again (issue #642). One `(from, to)` pair per *edge*, not per
        # arc: `as_arrays` reshapes `target` to `(n_edges, 2)` and reverses each
        # row, and edge `e` reversed is `(tail, head)` -- which is this method's
        # own arguments, as the two capacity arrays are.
        arcs = np.empty(n_arcs, dtype=np.int64)
        arcs[0::2], arcs[1::2] = tail, head
        return cls(
            n_nodes=n_nodes,
            target=target.tolist(),
            capacity=capacities.tolist(),
            outgoing=[
                grouped.indices[start:stop].tolist()
                for start, stop in zip(
                    grouped.offsets[:-1], grouped.offsets[1:], strict=True
                )
            ],
            _arcs=arcs,
            _forward=np.ascontiguousarray(forward),
            _backward=np.ascontiguousarray(backward),
        )

    def add_edge(
        self, source: int, sink: int, capacity: float, reverse: float = 0.0
    ) -> None:
        """Add ``source -> sink``, with ``reverse`` capacity on the back arc.

        Raises
        ------
        ValueError
            If either capacity is negative. A negative capacity is not a flow
            network, and Dinic would loop rather than report the problem.
        """
        if capacity < 0.0 or reverse < 0.0:
            msg = f"capacities must be non-negative, got {capacity} and {reverse}"
            raise ValueError(msg)
        self._forget_arrays()
        self.outgoing[source].append(len(self.target))
        self.target.append(sink)
        self.capacity.append(capacity)
        self.outgoing[sink].append(len(self.target))
        self.target.append(source)
        self.capacity.append(reverse)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The paired arcs as ``(arcs, capacity, reverse)``, one row per edge.

        ``arcs`` is ``2 * n_edges`` flattened ``(from, to)`` pairs, and the
        two capacity arrays are the forward and back arc of each. This is the
        layout :mod:`snakes_and_ladders.search.maxflow_rust` crosses the
        boundary with, and it is written here rather than there so the
        paired-arc convention above has one reading: a second one, kept in
        the module that consumes it, is a convention that can fall out of
        step with the ``add_edge`` that produces it.

        Residual capacities, after :func:`max_flow` has run, are what a
        second call would see --- the arrays are read from the current state
        and not from the network as it was built.
        """
        if self._arcs is not None and self._forward is not None:
            # The three are set and cleared together, never singly.
            assert self._backward is not None
            return self._arcs, self._forward, self._backward
        target = np.asarray(self.target, dtype=np.int64).reshape(-1, 2)
        capacity = np.asarray(self.capacity, dtype=np.float64).reshape(-1, 2)
        # Column 0 of a row is the head (`add_edge` appends the sink first),
        # column 1 the tail, so the `(from, to)` order the binding takes is
        # the reversed row.
        return (
            np.ascontiguousarray(target[:, ::-1]).reshape(-1),
            np.ascontiguousarray(capacity[:, 0]),
            np.ascontiguousarray(capacity[:, 1]),
        )


@dataclass(frozen=True)
class MinCut:
    """The value of a maximum flow, and the partition it certifies.

    Parameters
    ----------
    value : float
        The maximum flow, which equals the minimum cut's capacity.
    source_side : np.ndarray
        Boolean per node: reachable from the source in the residual graph on
        termination. That set *is* the minimum cut, by the theorem, which is
        why nothing here searches for a cut separately.
    """

    value: float
    source_side: np.ndarray


def max_flow(network: FlowNetwork, source: int, sink: int) -> MinCut:
    """Dinic's algorithm: repeated level graphs and blocking flows.

    Parameters
    ----------
    network : FlowNetwork
        Mutated in place --- its capacities become residual capacities.
    source, sink : int
        Terminals.

    Returns
    -------
    MinCut
        The flow value and the source side of the minimum cut.

    Raises
    ------
    ValueError
        If the terminals coincide, where "the" flow is unbounded and every
        answer is as good as any other.
    """
    if source == sink:
        msg = f"source and sink must differ, both are {source}"
        raise ValueError(msg)

    # This function turns capacities into residual capacities, so the
    # contiguous form a `from_arcs` network carries is about to go stale
    # (issue #642). Dropped here, once per solve, rather than at the push that
    # invalidates it: that push is Dinic's inner loop, where a Python call per
    # arc would cost more than the derivation this whole change avoids.
    network._forget_arrays()

    total = 0.0
    while True:
        level = _levels(network, source)
        if level[sink] < 0:
            break
        progress = [0] * network.n_nodes
        while True:
            pushed = _augment(network, source, sink, float("inf"), level, progress)
            if pushed <= 0.0:
                break
            total += pushed

    return MinCut(value=total, source_side=_levels(network, source) >= 0)


def _levels(network: FlowNetwork, source: int) -> np.ndarray:
    """Breadth-first distances in the residual graph; ``-1`` where unreachable."""
    level = np.full(network.n_nodes, -1, dtype=np.int64)
    level[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for arc in network.outgoing[node]:
            neighbour = network.target[arc]
            if network.capacity[arc] > 0.0 and level[neighbour] < 0:
                level[neighbour] = level[node] + 1
                queue.append(neighbour)
    return level


def _augment(
    network: FlowNetwork,
    node: int,
    sink: int,
    limit: float,
    level: np.ndarray,
    progress: list[int],
) -> float:
    """Push along one level-respecting path, remembering exhausted arcs.

    ``progress`` is what makes the blocking flow linear rather than quadratic:
    an arc that cannot carry more flow in this phase is never revisited, so
    each is examined once per level graph.
    """
    if node == sink:
        return limit
    while progress[node] < len(network.outgoing[node]):
        arc = network.outgoing[node][progress[node]]
        neighbour = network.target[arc]
        if network.capacity[arc] > 0.0 and level[neighbour] == level[node] + 1:
            pushed = _augment(
                network,
                neighbour,
                sink,
                min(limit, network.capacity[arc]),
                level,
                progress,
            )
            if pushed > 0.0:
                network.capacity[arc] -= pushed
                network.capacity[arc ^ 1] += pushed
                return pushed
        progress[node] += 1
    return 0.0


def check_non_negative_couplings(graph: PottsGraph, reason: str) -> None:
    """Refuse a negative coupling, saying which construction the sign holds up.

    Three constructions rest on non-negativity for three reasons --- the
    expansion's metric bound, the swap's submodular sub-problem, this
    module's submodular energy --- and each stated it in its own copy of the
    refusal (issue #858). The reason is the caller's; the refusal and the
    number in it are not.

    Parameters
    ----------
    graph : PottsGraph
        The graph whose couplings are read.
    reason : str
        What non-negativity buys the caller, which follows the colon.

    Raises
    ------
    ValueError
        If any coupling is negative, naming the smallest.
    """
    couplings = graph.edge_coupling
    if couplings.size and couplings.min() < 0.0:
        msg = f"every coupling must be non-negative, got {couplings.min()}: {reason}"
        raise ValueError(msg)


def ising_ground_state(
    graph: PottsGraph, field_values: np.ndarray, *, backend: Backend = Backend.PYTHON
) -> tuple[np.ndarray, float]:
    """The exact minimum-energy configuration of a two-state ferromagnet.

    **The construction.** Writing agreement as ``1 - disagreement`` turns the
    coupling term into a constant plus a cut, so minimizing the energy becomes

        min_s  sum_i D_i(s_i) + sum_(ij) J_ij [s_i != s_j]

    with ``D_i(a) = -h_i[a]``. A node on the source side of the cut takes state
    0 and on the sink side state 1, so cutting ``source -> i`` costs ``D_i(1)``
    and cutting ``i -> sink`` costs ``D_i(0)``; an edge contributes ``J_ij``
    exactly when its endpoints land on opposite sides. The per-node minimum is
    subtracted into a constant so every capacity is non-negative, which a flow
    network requires and which is where a negative coupling breaks the
    construction rather than merely slowing it.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling must be non-negative.
    field_values : np.ndarray
        ``(2,)`` or ``(n_nodes, 2)``. A **uniform** field makes the
        ferromagnetic ground state trivial: every coupling favours agreement
        and every site prefers the same state, so the answer is
        ``argmax(h)`` everywhere and a cut is an expensive way to say so. The
        problem has content when the field varies by site, which is also the
        shape alpha expansion (issue #207) needs. More than two states is
        refused: a cut solves ``k = 2`` and alpha expansion covers the rest.
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.PYTHON` is the push-relabel
        cut here, the oracle; :data:`~snakes_and_ladders.backend.Backend.RUST`
        is :mod:`snakes_and_ladders.search.maxflow_rust`, pinned to it bitwise.
        Chosen here so a caller names the kernel rather than the module, as
        ``bcjr`` and ``sum_product`` already ask (#813).

    Returns
    -------
    tuple[np.ndarray, float]
        The ground-state configuration and its energy.

    Raises
    ------
    ValueError
        If the field shape is wrong, or any coupling is negative. The second
        is the submodularity boundary: the problem is NP-hard there and this
        returns nothing rather than a lattice-shaped wrong answer.
    """
    if backend not in (Backend.PYTHON, Backend.RUST):
        msg = f"ising_ground_state runs on {Backend.PYTHON} or {Backend.RUST}, not {backend}"
        raise ValueError(msg)
    if backend is Backend.RUST:
        # Local, because the twin imports `FlowNetwork` and `MinCut` from
        # here: a module-level import is the cycle.
        from snakes_and_ladders.search import maxflow_rust

        return maxflow_rust.ising_ground_state(graph, field_values)
    values = site_field(
        np.asarray(field_values, dtype=float), graph.n_nodes, n_states=2
    )
    check_non_negative_couplings(
        graph,
        "a negative coupling makes the energy non-submodular, the ground "
        "state NP-hard, and this construction inapplicable rather than slow",
    )

    source, sink = graph.n_nodes, graph.n_nodes + 1
    network = FlowNetwork(n_nodes=graph.n_nodes + 2)

    cost = -values
    offsets = cost.min(axis=1)
    for node in range(graph.n_nodes):
        network.add_edge(source, node, float(cost[node, 1] - offsets[node]))
        network.add_edge(node, sink, float(cost[node, 0] - offsets[node]))

    for (first, second), coupling in graph.weighted_edges():
        network.add_edge(first, second, coupling, reverse=coupling)

    cut = max_flow(network, source, sink)
    configuration = (~cut.source_side[: graph.n_nodes]).astype(np.int64)
    return configuration, energy(graph, values, configuration)


def cut_energy(graph: PottsGraph, field_values: np.ndarray, cut_value: float) -> float:
    """The energy a cut of capacity ``cut_value`` corresponds to.

    Separated so a test can check the reduction's arithmetic against the
    energy evaluated directly on the returned configuration. The two must
    agree; if they do not, the construction is wrong in a way that reading the
    configuration back and scoring it would hide.
    """
    values = site_field(
        np.asarray(field_values, dtype=float), graph.n_nodes, n_states=2
    )
    offsets = (-values).min(axis=1)
    return cut_value + float(offsets.sum()) - float(graph.edge_coupling.sum())
