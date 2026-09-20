"""Alpha expansion: approximate MAP on a Potts MRF, with a proved bound.

`k`-state MAP is NP-hard, so :mod:`snakes_and_ladders.search.maxflow`'s exact minimum cut
stops at two states. Alpha expansion recovers the general case as a sequence
of binary cuts: for each label ``alpha`` in turn, every site is offered the
choice of keeping its current label or switching to ``alpha``, and *that*
binary problem is submodular for a Potts pairwise term, so the exact solver
handles it unchanged.

**The guarantee is the point.** The Potts pairwise term ``V(a, b) = J [a != b]``
is a metric for ``J >= 0``, and for a metric the algorithm's local minimum is
within ``2 * c_max / c_min`` of the *global* one --- exactly **2** for a
uniform coupling (Boykov, Veksler & Zabih 2001). No other method here states a
bound: belief propagation (issue #172) reports a measured deviation, the
samplers (#174) a distribution, and enumeration stops at nine sites. A bound
holds at every size, so a result can be checked where the algorithm runs.

It is a local minimum with respect to moves that change *arbitrarily many*
sites at once, which is what makes it beat single-site descent: no sequence of
single flips crosses a barrier that one expansion crosses in a step.

See Boykov, Veksler & Zabih (2001); Kolmogorov & Zabih (2004) for which
energies a cut can represent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search.maxflow import FlowNetwork, max_flow
from snakes_and_ladders.search.maxflow_rust import min_cut
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import energy, site_field

# The bound is `2 * c_max / c_min` for a metric pairwise term; with a uniform
# coupling the ratio is 1 and the factor is exactly 2.
UNIFORM_POTTS_BOUND = 2.0

DEFAULT_MAX_CYCLES = 50


@dataclass(frozen=True)
class ExpansionResult:
    """A labelling, and what reaching it cost.

    Parameters
    ----------
    labelling : np.ndarray
        One state per node.
    energy : float
        Its energy under :func:`energy`.
    cycles : int
        Complete sweeps over the label set. The loop stops on the first sweep
        that lowers nothing, so this is one more than the number that helped.
    moves : int
        Expansions that strictly lowered the energy. Zero means the starting
        labelling was already expansion-optimal, which is information rather
        than a failure.
    """

    labelling: np.ndarray
    energy: float
    cycles: int
    moves: int


def _expansion_network(
    graph: PottsGraph, values: np.ndarray, labelling: np.ndarray, alpha: int
) -> FlowNetwork:
    """The expansion network of :func:`expand`, built without a call per arc.

    Arc for arc and in the same order as the ``add_edge`` loop it replaces,
    which `tests/regression/search/test_alpha_expansion.py` asserts against a
    transcription of that loop: the order is the contract, since a minimum cut
    need not be unique and the two solvers are pinned to one labelling.

    Issue #598 measured the loop at 76,928 calls and 53.0 per cent of a 32x32
    run at four labels, against the Rust cut kernel's 12.1 --- the network was
    built in Python and solved in Rust.

    Parameters
    ----------
    graph : PottsGraph
        The graph.
    values : np.ndarray
        Per-site field, shape ``(n_nodes, n_states)``, as
        :func:`snakes_and_ladders.sim.potts.site_field` widens it.
    labelling : np.ndarray
        The labelling to expand, one label per node.
    alpha : int
        The label every node may move to.

    Returns
    -------
    FlowNetwork
        Spanning ``n_nodes + 2`` nodes plus one auxiliary per disagreeing
        edge, with the source at ``n_nodes`` and the sink at ``n_nodes + 1``.
    """
    n_nodes = graph.n_nodes
    source, sink = n_nodes, n_nodes + 1
    first, second, coupling = graph.endpoints
    labels = np.asarray(labelling)
    disagree = labels[first] != labels[second]
    n_auxiliary = int(np.count_nonzero(disagree))

    # The keep branch of a node already labelled alpha is unaffordable rather
    # than absent, so the node stays in the network (see :func:`expand`).
    switch = -values[:, alpha].astype(np.float64)
    keep = np.where(
        labels == alpha,
        _infinite_capacity(graph, values),
        -values[np.arange(n_nodes), labels].astype(np.float64),
    )
    offset = np.minimum(keep, switch)

    node_tail = np.empty(2 * n_nodes, dtype=np.int64)
    node_head = np.empty(2 * n_nodes, dtype=np.int64)
    node_capacity = np.empty(2 * n_nodes, dtype=np.float64)
    nodes = np.arange(n_nodes, dtype=np.int64)
    node_tail[0::2], node_tail[1::2] = source, nodes
    node_head[0::2], node_head[1::2] = nodes, sink
    node_capacity[0::2], node_capacity[1::2] = switch - offset, keep - offset

    # One arc per agreeing edge and three per disagreeing one, laid out in
    # edge order so the auxiliaries are numbered as the loop numbered them.
    width = np.where(disagree, 3, 1)
    start = np.concatenate(([0], np.cumsum(width)[:-1]))
    edge_tail = np.empty(int(width.sum()), dtype=np.int64)
    edge_head = np.empty_like(edge_tail)
    edge_capacity = np.zeros(edge_tail.size, dtype=np.float64)
    edge_reverse = np.zeros(edge_tail.size, dtype=np.float64)

    first_differs = np.where(labels[first] != alpha, coupling, 0.0)
    second_differs = np.where(labels[second] != alpha, coupling, 0.0)

    agreeing = start[~disagree]
    edge_tail[agreeing], edge_head[agreeing] = first[~disagree], second[~disagree]
    edge_capacity[agreeing] = edge_reverse[agreeing] = first_differs[~disagree]

    split = np.flatnonzero(disagree)
    at = start[split]
    auxiliary = np.arange(n_auxiliary, dtype=np.int64) + n_nodes + 2
    edge_tail[at], edge_head[at] = first[split], auxiliary
    edge_capacity[at] = edge_reverse[at] = first_differs[split]
    edge_tail[at + 1], edge_head[at + 1] = second[split], auxiliary
    edge_capacity[at + 1] = edge_reverse[at + 1] = second_differs[split]
    edge_tail[at + 2], edge_head[at + 2] = auxiliary, sink
    edge_capacity[at + 2] = coupling[split]

    return FlowNetwork.from_arcs(
        n_nodes + 2 + n_auxiliary,
        np.concatenate((node_tail, edge_tail)),
        np.concatenate((node_head, edge_head)),
        np.concatenate((node_capacity, edge_capacity)),
        np.concatenate((np.zeros(2 * n_nodes), edge_reverse)),
    )


def expand(
    graph: PottsGraph,
    field_values: np.ndarray,
    labelling: np.ndarray,
    alpha: int,
    *,
    backend: Backend = Backend.PYTHON,
) -> tuple[np.ndarray, float]:
    """The optimal ``alpha``-expansion of ``labelling``, by one minimum cut.

    Every node chooses between keeping its current label and taking ``alpha``.
    The orientation is the half of this that is easy to get backwards, so it
    is stated rather than implied: **the source side keeps, the sink side
    takes ``alpha``**. An edge from the source to a node is therefore cut
    exactly when that node switches, and carries the data cost of switching.

    The pairwise term needs an auxiliary node wherever the two endpoints
    currently disagree, because three distinct costs have to be representable
    on one edge --- ``V(f_p, alpha)``, ``V(alpha, f_q)`` and ``V(f_p, f_q)``
    --- and a single arc can express only two. With ``V(a, b) = J [a != b]``:

    * endpoints agreeing: one arc of capacity ``V(f_p, alpha)``, which is
      ``J`` unless they are already ``alpha``, in which case it is zero;
    * endpoints disagreeing: an auxiliary ``a`` with ``p--a`` at
      ``V(f_p, alpha)``, ``q--a`` at ``V(f_q, alpha)``, and ``a -> sink`` at
      ``V(f_p, f_q) = J``.

    Getting those capacities wrong does not break loudly. It produces a
    labelling that is merely *worse*, which is indistinguishable from the
    algorithm working on a problem where it does badly --- which is why the
    `k = 2` reduction to :mod:`snakes_and_ladders.search.maxflow` is the test that matters.

    A node already labelled ``alpha`` cannot move away, since an expansion
    moves labels *to* ``alpha`` only. It is pinned by making its keep branch
    unaffordable rather than by dropping it, so the edge terms around it stay
    in the same network.

    ``backend`` chooses the minimum-cut solver and nothing else; the network
    is built here either way. :data:`~snakes_and_ladders.backend.Backend.RUST`
    runs :func:`snakes_and_ladders.search.maxflow_rust.min_cut`, which issue
    #528 measured at 49.0% of this function's caller by `cProfile` self time.
    It is **opt-in**, unlike the `numba` sweep of
    :func:`iterated_conditional_modes`: a minimum cut is a combinatorial
    minimum whose *value* both solvers must report exactly, but the cut
    attaining it need not be unique, and a degenerate network could hand back
    a different labelling of the same energy. The tests pin both routes to the
    same labelling on the seeded fixtures; the default does not move on the
    strength of that.

    Returns
    -------
    tuple[np.ndarray, float]
        The expanded labelling and its energy. When no expansion helps, the
        input is returned unchanged.

    Raises
    ------
    ValueError
        If ``backend`` names an implementation this function does not have.
    """
    if backend not in (Backend.PYTHON, Backend.RUST):
        msg = f"alpha expansion has no {backend} minimum-cut backend"
        raise ValueError(msg)

    values = site_field(np.asarray(field_values, dtype=float), graph.n_nodes)
    network = _expansion_network(graph, values, labelling, alpha)
    source, sink = graph.n_nodes, graph.n_nodes + 1

    cut = (
        min_cut(network, source, sink)
        if backend is Backend.RUST
        else max_flow(network, source, sink)
    )
    switched = ~cut.source_side[: graph.n_nodes]
    proposed = np.where(switched, alpha, labelling)

    current, candidate = (
        energy(graph, values, labelling),
        energy(graph, values, proposed),
    )
    if candidate < current:
        return proposed, candidate
    return labelling, current


def alpha_expansion(
    graph: PottsGraph,
    field_values: np.ndarray,
    n_states: int,
    *,
    start: np.ndarray | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    backend: Backend = Backend.PYTHON,
) -> ExpansionResult:
    """Cycle over labels until a full sweep lowers nothing.

    Two invariants make this checkable without an oracle, and both are
    asserted by the tests: the energy is **monotonically non-increasing**
    across every move, which a sign error in :func:`expand` breaks
    immediately; and the loop **terminates**, which follows from monotonicity
    over a finite state space.

    Parameters
    ----------
    graph : PottsGraph
        Every coupling must be non-negative --- the metric condition the
        bound rests on.
    field_values : np.ndarray
        ``(n_states,)`` or ``(n_nodes, n_states)``.
    n_states : int
        Label count.
    start : np.ndarray | None
        Initial labelling; the per-node data optimum when omitted, which is
        the labelling ignoring every coupling.
    max_cycles : int
        Refuse past this many sweeps rather than looping. Monotonicity makes
        exceeding it impossible on a correct implementation, so reaching it
        is a bug report rather than a tuning knob.
    backend : Backend
        Which minimum-cut solver each :func:`expand` runs, and nothing else.
        See :func:`expand` for why the Rust one is opt-in.

    Raises
    ------
    ValueError
        If a coupling is negative, or the cap is reached.
    """
    couplings = graph.edge_coupling
    if couplings.size and couplings.min() < 0.0:
        msg = (
            f"every coupling must be non-negative, got {couplings.min()}: the "
            "Potts pairwise term is a metric only then, and the factor-2 bound "
            "rests on it"
        )
        raise ValueError(msg)

    values = site_field(
        np.asarray(field_values, dtype=float), graph.n_nodes, n_states=n_states
    )
    labelling = (
        values.argmax(axis=1).astype(np.int64) if start is None else start.copy()
    )
    current = energy(graph, values, labelling)

    moves = 0
    for cycle in range(1, max_cycles + 1):
        improved = False
        for alpha in range(n_states):
            labelling, candidate = expand(
                graph, values, labelling, alpha, backend=backend
            )
            if candidate < current - 1e-12:
                current = candidate
                improved = True
                moves += 1
        if not improved:
            return ExpansionResult(
                labelling=labelling, energy=current, cycles=cycle, moves=moves
            )

    msg = (
        f"alpha expansion did not settle in {max_cycles} cycles. The energy is "
        "non-increasing over a finite state space, so this cannot happen on a "
        "correct implementation and is a defect rather than a budget"
    )
    raise ValueError(msg)


def iterated_conditional_modes(
    graph: PottsGraph,
    field_values: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    *,
    max_sweeps: int = 200,
    backend: Backend = Backend.NUMBA,
) -> tuple[np.ndarray, float]:
    """Single-site descent: the baseline alpha expansion has to beat.

    Each site takes the label minimizing the energy given its neighbours, in
    index order, until a sweep changes nothing. This is the natural point of
    comparison because it is the *same objective* under a move set of one
    site at a time --- so a difference between the two is a statement about
    the move set rather than about the model or the code path.

    Local deltas rather than a full energy per candidate: only the site's own
    field term and its incident edges change, so a sweep costs
    ``O(n_nodes * k * degree)`` rather than ``O(n_nodes * k * n_edges)``. The
    distinction matters because this is meant to be a *fair* baseline, and a
    baseline made slow by its implementation is not one.

    ``backend`` chooses the sweep's implementation and nothing else. The
    :data:`~snakes_and_ladders.backend.Backend.NUMBA` kernel in
    :mod:`snakes_and_ladders.sample.kernels` walks the compressed-row adjacency and
    returns the labelling the Python loop returns **bitwise** -- same update,
    same index order, same first-minimum tie rule -- which is what lets it be
    the default: the audit behind it (#264) measured the Python sweep at
    the top of the descent's self time, and the labelling a caller sees does
    not move. :data:`~snakes_and_ladders.backend.Backend.PYTHON` is the oracle
    that pins it.

    Returns
    -------
    tuple[np.ndarray, float]
        The labelling it settles on, and its energy.
    """
    values = site_field(np.asarray(field_values, dtype=float), graph.n_nodes)
    labelling = rng.integers(0, n_states, size=graph.n_nodes)

    if backend is Backend.NUMBA:
        from snakes_and_ladders.sample.kernels import icm_sweeps

        offsets, neighbour_index, couplings = graph.compressed_adjacency()
        icm_sweeps(
            labelling,
            np.ascontiguousarray(values, dtype=np.float64),
            offsets,
            neighbour_index,
            couplings,
            max_sweeps,
        )
        return labelling, energy(graph, values, labelling)
    if backend is not Backend.PYTHON:
        msg = f"iterated conditional modes has no {backend} backend"
        raise ValueError(msg)

    # The compressed rows as Python sequences, converted once rather than
    # sliced per site: a NumPy slice and gather per site measured a third of
    # this sweep (issue #277, `sim.potts.heat_bath_log_weights`). The
    # `numba` kernel above takes the arrays themselves.
    offsets, neighbour_index, edge_couplings = graph.compressed_adjacency()
    bounds = offsets.tolist()
    neighbours, couplings = neighbour_index.tolist(), edge_couplings.tolist()

    # The labels as a Python list for the duration: the sweep reads a
    # neighbour's label once per incident edge, and a list read is 0.18 us
    # cheaper than a NumPy scalar one. Same reads, same order, same writes.
    labels = labelling.tolist()
    for _ in range(max_sweeps):
        changed = False
        for node in range(graph.n_nodes):
            local = -values[node].copy()
            for position in range(bounds[node], bounds[node + 1]):
                local[labels[neighbours[position]]] -= couplings[position]
            best = int(np.argmin(local))
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break
    labelling[:] = labels

    return labelling, energy(graph, values, labelling)


def _infinite_capacity(graph: PottsGraph, values: np.ndarray) -> float:
    """A capacity no cut would ever pay, scaled to this problem.

    ``float('inf')`` would work arithmetically and destroy the flow
    bookkeeping, since subtracting it leaves a residual of ``nan``. This is
    larger than every alternative cut by construction and stays finite.
    """
    return 1.0 + float(np.abs(values).sum()) + float(graph.edge_coupling.sum())


def swap(
    graph: PottsGraph,
    field_values: np.ndarray,
    labelling: np.ndarray,
    alpha: int,
    beta: int,
    *,
    backend: Backend = Backend.PYTHON,
) -> tuple[np.ndarray, float]:
    """The optimal ``alpha``-``beta`` swap of ``labelling``, by one minimum cut.

    Only the sites currently labelled ``alpha`` or ``beta`` move, and each
    chooses between the two; every other site is held. The binary problem is
    therefore over that subset alone, and it needs **no auxiliary node**: both
    endpoints of an edge inside the subset end at ``alpha`` or ``beta``, so
    the Potts term takes two values and one arc carries it. That is the whole
    difference from :func:`expand` --- a smaller network, a cheaper cut, and
    **no bound**, since the factor-2 guarantee of Boykov, Veksler & Zabih is a
    property of the expansion move and not of this one.

    The source side takes ``alpha`` and the sink side ``beta``. A held
    neighbour carries neither label --- those are exactly the ones that move
    --- so it agrees with the moving site under neither choice and drops out
    of the data term entirely.

    Returns
    -------
    tuple[np.ndarray, float]
        The swapped labelling and its energy; the input unchanged where no
        swap lowers it.

    Raises
    ------
    ValueError
        If ``backend`` names an implementation this function does not have,
        or ``alpha`` and ``beta`` are the same label, where the move is the
        identity and a caller asking for it has a bug rather than a no-op.
    """
    if backend not in (Backend.PYTHON, Backend.RUST):
        msg = f"the alpha-beta swap has no {backend} minimum-cut backend"
        raise ValueError(msg)
    if alpha == beta:
        msg = f"a swap needs two distinct labels, got {alpha} twice"
        raise ValueError(msg)

    values = site_field(np.asarray(field_values, dtype=float), graph.n_nodes)
    moving = np.flatnonzero((labelling == alpha) | (labelling == beta))
    if moving.size == 0:
        return labelling, energy(graph, values, labelling)
    position = {int(node): index for index, node in enumerate(moving)}

    source, sink = moving.size, moving.size + 1
    network = FlowNetwork(n_nodes=moving.size + 2)

    # The data term of a moving site is its own field and nothing else. A
    # *held* neighbour carries neither alpha nor beta --- those are exactly
    # the labels that move --- so it agrees with the moving site under
    # neither choice and contributes the same constant to both. That is why
    # this move needs no auxiliary node and why the expansion does: there,
    # a held neighbour can already be alpha.
    to_alpha = -values[moving, alpha].astype(float)
    to_beta = -values[moving, beta].astype(float)
    offsets = np.minimum(to_alpha, to_beta)
    for index in range(moving.size):
        # Cut source -> index when the site lands on the sink side, taking
        # beta, so that arc carries the cost of beta.
        network.add_edge(source, index, float(to_beta[index] - offsets[index]))
        network.add_edge(index, sink, float(to_alpha[index] - offsets[index]))

    for (first, second), coupling in graph.weighted_edges():
        if first in position and second in position:
            network.add_edge(
                position[first], position[second], coupling, reverse=coupling
            )

    cut = (
        min_cut(network, source, sink)
        if backend is Backend.RUST
        else max_flow(network, source, sink)
    )
    proposed = labelling.copy()
    proposed[moving] = np.where(cut.source_side[: moving.size], alpha, beta)

    current, candidate = (
        energy(graph, values, labelling),
        energy(graph, values, proposed),
    )
    if candidate < current:
        return proposed, candidate
    return labelling, current


def alpha_beta_swap(
    graph: PottsGraph,
    field_values: np.ndarray,
    n_states: int,
    *,
    start: np.ndarray | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    backend: Backend = Backend.PYTHON,
) -> ExpansionResult:
    """Cycle over every label pair until a full sweep lowers nothing.

    The same loop as :func:`alpha_expansion` over a different move set, so a
    difference between the two is a statement about the move and not about
    the model or the code path. It is the cheaper move --- one cut over the
    sites carrying two labels rather than over the whole lattice, and no
    auxiliary node --- and it carries **no bound**, which is the trade issue
    #551 measures. A cycle is ``n_states * (n_states - 1) / 2`` cuts against
    the expansion's ``n_states``, so "cheaper per move" is not "cheaper per
    cycle" and the comparison is run at equal budget rather than equal cycles.

    Raises
    ------
    ValueError
        If a coupling is negative, or the cap is reached. Monotonicity over a
        finite state space makes the second impossible on a correct
        implementation, as it is for :func:`alpha_expansion`.
    """
    couplings = graph.edge_coupling
    if couplings.size and couplings.min() < 0.0:
        msg = (
            f"every coupling must be non-negative, got {couplings.min()}: the "
            "swap's binary sub-problem is submodular only then"
        )
        raise ValueError(msg)

    values = site_field(
        np.asarray(field_values, dtype=float), graph.n_nodes, n_states=n_states
    )
    labelling = (
        values.argmax(axis=1).astype(np.int64) if start is None else start.copy()
    )
    current = energy(graph, values, labelling)

    moves = 0
    for cycle in range(1, max_cycles + 1):
        improved = False
        for alpha in range(n_states):
            for beta in range(alpha + 1, n_states):
                labelling, candidate = swap(
                    graph, values, labelling, alpha, beta, backend=backend
                )
                if candidate < current - 1e-12:
                    current = candidate
                    improved = True
                    moves += 1
        if not improved:
            return ExpansionResult(
                labelling=labelling, energy=current, cycles=cycle, moves=moves
            )

    msg = (
        f"the alpha-beta swap did not settle in {max_cycles} cycles. The energy "
        "is non-increasing over a finite state space, so this cannot happen on "
        "a correct implementation and is a defect rather than a budget"
    )
    raise ValueError(msg)
