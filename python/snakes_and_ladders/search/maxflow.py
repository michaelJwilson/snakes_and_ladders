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

from snakes_and_ladders.sim.graph import PottsGraph


@dataclass
class FlowNetwork:
    """A directed graph with capacities, held as paired residual arcs.

    Arc ``2 * e`` and arc ``2 * e + 1`` are the two directions of one edge, so
    the reverse of an arc is its index with the low bit flipped. Pushing flow
    subtracts from one and adds to the other, which is what makes the residual
    graph implicit rather than a second structure to keep in step.
    """

    n_nodes: int
    target: list[int] = field(default_factory=list)
    capacity: list[float] = field(default_factory=list)
    outgoing: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.outgoing:
            self.outgoing = [[] for _ in range(self.n_nodes)]

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
        self.outgoing[source].append(len(self.target))
        self.target.append(sink)
        self.capacity.append(capacity)
        self.outgoing[sink].append(len(self.target))
        self.target.append(source)
        self.capacity.append(reverse)


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


def site_field(graph: PottsGraph, field_values: np.ndarray) -> np.ndarray:
    """Broadcast a shared field to one per node, or pass a per-node one through.

    A **uniform** field makes the ferromagnetic ground state trivial: every
    coupling favours agreement and every site prefers the same state, so the
    answer is `argmax(h)` everywhere and a cut is an expensive way to say so.
    The problem only has content when the field varies by site, which is also
    the shape alpha expansion (issue #207) needs --- its binary sub-problem
    has a per-node data term read off the current labelling. So both are
    accepted and the per-node case is the one worth solving.

    Parameters
    ----------
    graph : PottsGraph
        Supplies the node count.
    field_values : np.ndarray
        Either ``(2,)``, shared by every node, or ``(n_nodes, 2)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_nodes, 2)``.

    Raises
    ------
    ValueError
        If the shape is neither, or the state count is not 2.
    """
    values = np.asarray(field_values, dtype=float)
    if values.ndim == 1:
        if values.shape[0] != 2:
            msg = (
                f"a cut solves the two-state case only, got {values.shape[0]} "
                "states; alpha expansion (issue #207) covers more"
            )
            raise ValueError(msg)
        return np.tile(values, (graph.n_nodes, 1))
    if values.shape != (graph.n_nodes, 2):
        msg = f"a per-node field must be ({graph.n_nodes}, 2), got {values.shape}"
        raise ValueError(msg)
    return values


def energy(
    graph: PottsGraph, field_values: np.ndarray, configurations: np.ndarray
) -> np.ndarray:
    """``E(s) = -sum_i h_i[s_i] - sum_(ij) J_ij [s_i == s_j]``.

    The negated log weight :func:`snakes_and_ladders.likelihood.potts.log_weights` computes,
    generalized to a per-node field. With a shared field the two agree
    exactly, which a test pins --- so the generalization cannot drift from the
    model the rest of the repository fits.
    """
    values = site_field(graph, field_values)
    total = values[np.arange(graph.n_nodes), configurations].sum(axis=-1)
    for (first, second), coupling in graph.weighted_edges():
        total = total + coupling * (
            configurations[..., first] == configurations[..., second]
        )
    return -np.asarray(total)


def ising_ground_state(
    graph: PottsGraph, field_values: np.ndarray
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
        ``(2,)`` or ``(n_nodes, 2)``. See :func:`site_field` for why the
        second is the case with content.

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
    values = site_field(graph, field_values)
    couplings = np.asarray(graph.coupling, dtype=float)
    if couplings.size and couplings.min() < 0.0:
        msg = (
            f"every coupling must be non-negative, got {couplings.min()}: a "
            "negative coupling makes the energy non-submodular, the ground "
            "state NP-hard, and this construction inapplicable rather than slow"
        )
        raise ValueError(msg)

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
    return configuration, float(energy(graph, values, configuration))


def cut_energy(graph: PottsGraph, field_values: np.ndarray, cut_value: float) -> float:
    """The energy a cut of capacity ``cut_value`` corresponds to.

    Separated so a test can check the reduction's arithmetic against the
    energy evaluated directly on the returned configuration. The two must
    agree; if they do not, the construction is wrong in a way that reading the
    configuration back and scoring it would hide.
    """
    values = site_field(graph, field_values)
    offsets = (-values).min(axis=1)
    return cut_value + float(offsets.sum()) - float(np.asarray(graph.coupling).sum())
