"""The ``numba`` kernel of :mod:`sal.search.trws`: sequential tree-reweighted message passing.

:func:`trws_iterations` runs the forward and backward passes, the chain bound
and the decode of :func:`sal.search.trws.trws` in the order the Python
reference runs them, operation for operation, so the two return the same
bound trace and labelling bitwise. Messages, the reparametrized unaries and
every scratch row are allocated once per call. It touches no Python object
and so is ``nogil=True`` (#604). :func:`trws_iterations_checked` checks the
shapes with the Rust entries' messages (#571), because a compiled kernel
indexes without bounds checks.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _node_share(
    node: int,
    unary: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    messages: np.ndarray,
    out: np.ndarray,
) -> None:
    """``theta_s + sum_k M_k``, the incoming messages added in row order, into ``out``."""
    n_states = unary.shape[1]
    for label in range(n_states):
        out[label] = unary[node, label]
    for position in range(offsets[node], offsets[node + 1]):
        entry = slots[position]
        for label in range(n_states):
            out[label] += messages[entry, label]


@njit(cache=True, nogil=True)
def _potts_min(values: np.ndarray, coupling: float, out: np.ndarray) -> None:
    """``out[b] = min_a [values[a] - coupling [a == b]]``, in ``O(n_states)``.

    The Potts table is ``-J`` on the diagonal and zero off it, so the minimum
    is ``values[b] - J`` against the smallest ``values[a]`` over ``a != b``:
    the first minimum everywhere but at its own index, the second there.
    """
    n_states = values.shape[0]
    first = 0
    for label in range(1, n_states):
        if values[label] < values[first]:
            first = label
    second = np.inf
    for label in range(n_states):
        if label != first and values[label] < second:
            second = values[label]
    for label in range(n_states):
        other = second if label == first else values[first]
        diagonal = values[label] - coupling
        out[label] = diagonal if diagonal < other else other


@njit(cache=True, nogil=True)
def _pass(
    forward: bool,
    unary: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    coupling: np.ndarray,
    weight: np.ndarray,
    messages: np.ndarray,
    share: np.ndarray,
    lent: np.ndarray,
    sent: np.ndarray,
) -> None:
    """One monotone pass: each site in order sends to every later neighbour.

    ``forward`` visits the sites ascending and sends along even entries (the
    site is the edge's lower end); otherwise descending, along odd ones.
    """
    n_nodes, n_states = unary.shape
    parity = 0 if forward else 1
    for step in range(n_nodes):
        node = step if forward else n_nodes - 1 - step
        _node_share(node, unary, offsets, slots, messages, share)
        for position in range(offsets[node], offsets[node + 1]):
            entry = slots[position]
            if entry % 2 != parity:
                continue
            # `messages[entry]` is what the neighbour sent this site; its
            # reverse, `entry ^ 1`, is what this site sends the neighbour.
            for label in range(n_states):
                lent[label] = weight[node] * share[label] - messages[entry, label]
            _potts_min(lent, coupling[entry // 2], sent)
            lowest = sent[0]
            for label in range(1, n_states):
                lowest = min(lowest, sent[label])
            reverse = entry ^ 1
            for label in range(n_states):
                messages[reverse, label] = sent[label] - lowest


@njit(cache=True, nogil=True)
def _chain_bound(
    unary: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    ends: np.ndarray,
    coupling: np.ndarray,
    weight: np.ndarray,
    chain_offsets: np.ndarray,
    chain_heads: np.ndarray,
    chain_edges: np.ndarray,
    messages: np.ndarray,
    shares: np.ndarray,
    value: np.ndarray,
    lent: np.ndarray,
    sent: np.ndarray,
) -> float:
    """``sum_T min_x E_T(x)`` over the monotone chains, at the current messages.

    Each chain's energy carries ``weight[s] * theta_bar_s`` at a site and
    ``theta_bar_st`` on an edge; the chains sum to the reparametrized energy,
    which is the energy, so the sum of their minima bounds its minimum.
    """
    n_nodes, n_states = unary.shape
    for node in range(n_nodes):
        _node_share(node, unary, offsets, slots, messages, shares[node])
    total = 0.0
    for chain in range(chain_heads.shape[0]):
        head = chain_heads[chain]
        for label in range(n_states):
            value[label] = weight[head] * shares[head, label]
        for position in range(chain_offsets[chain], chain_offsets[chain + 1]):
            edge = chain_edges[position]
            high = ends[edge, 1]
            # theta_bar_st = theta_st - M_{t->s}(x_s) - M_{s->t}(x_t): the
            # first is entry 2e (owner s), the second entry 2e + 1 (owner t).
            for label in range(n_states):
                lent[label] = value[label] - messages[2 * edge, label]
            _potts_min(lent, coupling[edge], sent)
            for label in range(n_states):
                value[label] = (
                    sent[label]
                    - messages[2 * edge + 1, label]
                    + weight[high] * shares[high, label]
                )
        lowest = value[0]
        for label in range(1, n_states):
            lowest = min(lowest, value[label])
        total += lowest
    return total


@njit(cache=True, nogil=True)
def _decode(
    unary: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    neighbours: np.ndarray,
    coupling: np.ndarray,
    messages: np.ndarray,
    labels: np.ndarray,
    local: np.ndarray,
) -> float:
    """Label the sites in index order against the earlier labels and the later messages; its energy."""
    n_nodes, n_states = unary.shape
    for node in range(n_nodes):
        for label in range(n_states):
            local[label] = unary[node, label]
        for position in range(offsets[node], offsets[node + 1]):
            entry = slots[position]
            other = neighbours[entry]
            if other < node:
                local[labels[other]] -= coupling[entry // 2]
            else:
                for label in range(n_states):
                    local[label] += messages[entry, label]
        best = 0
        for label in range(1, n_states):
            if local[label] < local[best]:
                best = label
        labels[node] = best
    total = 0.0
    for node in range(n_nodes):
        total += unary[node, labels[node]]
    for edge in range(coupling.shape[0]):
        if labels[neighbours[2 * edge + 1]] == labels[neighbours[2 * edge]]:
            total -= coupling[edge]
    return total


@njit(cache=True, nogil=True)
def trws_iterations(
    unary: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    neighbours: np.ndarray,
    ends: np.ndarray,
    coupling: np.ndarray,
    weight: np.ndarray,
    chain_offsets: np.ndarray,
    chain_heads: np.ndarray,
    chain_edges: np.ndarray,
    max_iterations: int,
    tolerance: float,
    messages: np.ndarray,
    labelling: np.ndarray,
    trace: np.ndarray,
    energies: np.ndarray,
) -> int:
    """TRW-S iterations in place; returns the count taken, negated where the run converged.

    One iteration is a forward pass, a backward pass, the chain bound into
    ``trace`` and a decode into ``energies``; the lowest-energy labelling so
    far is kept in ``labelling`` (the first at a tie). The run converges when
    the bound rose by at most ``tolerance * max(1, |bound|)`` or the gap to
    the kept labelling closed to that.

    Unchecked: :func:`trws_iterations_checked` validates the shapes first.
    ``unary`` is the energy's site term, ``-h``; ``coupling`` is ``J`` per
    edge; entry ``2e`` of ``neighbours`` sits at edge ``e``'s lower end and
    names the higher, ``2e + 1`` the reverse; ``messages[k]`` is what entry
    ``k``'s neighbour sends its owner.
    """
    n_nodes, n_states = unary.shape
    share = np.empty(n_states, dtype=np.float64)
    lent = np.empty(n_states, dtype=np.float64)
    sent = np.empty(n_states, dtype=np.float64)
    value = np.empty(n_states, dtype=np.float64)
    shares = np.empty((n_nodes, n_states), dtype=np.float64)
    labels = np.empty(n_nodes, dtype=np.int64)
    best_energy = np.inf
    taken = 0
    for iteration in range(max_iterations):
        taken += 1
        _pass(
            True, unary, offsets, slots, coupling, weight, messages, share, lent, sent
        )
        _pass(
            False, unary, offsets, slots, coupling, weight, messages, share, lent, sent
        )
        bound = _chain_bound(
            unary,
            offsets,
            slots,
            ends,
            coupling,
            weight,
            chain_offsets,
            chain_heads,
            chain_edges,
            messages,
            shares,
            value,
            lent,
            sent,
        )
        trace[iteration] = bound
        current = _decode(
            unary, offsets, slots, neighbours, coupling, messages, labels, share
        )
        energies[iteration] = current
        if current < best_energy:
            best_energy = current
            for node in range(n_nodes):
                labelling[node] = labels[node]
        scale = max(1.0, abs(bound))
        if best_energy - bound <= tolerance * scale:
            return -taken
        if iteration > 0 and bound - trace[iteration - 1] <= tolerance * scale:
            return -taken
    return taken


def trws_iterations_checked(
    unary: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    neighbours: np.ndarray,
    ends: np.ndarray,
    coupling: np.ndarray,
    weight: np.ndarray,
    chain_offsets: np.ndarray,
    chain_heads: np.ndarray,
    chain_edges: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """:func:`trws_iterations` behind the shape checks the Rust entries make (#571), messages theirs.

    Allocates the messages at zero, the labelling and the two traces, and
    runs the kernel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]
        The messages, the kept labelling, the bound trace and the decoded
        energy trace (each cut to the iterations taken), the iterations, and
        whether the run converged.

    Raises
    ------
    ValueError
        Naming what was wanted and what was given, where a shape, a range or
        a count is off.
    """
    unary = np.ascontiguousarray(unary, dtype=np.float64)
    if unary.ndim != 2 or unary.shape[1] == 0:
        msg = (
            f"unary must be 2-D, (n_nodes, n_states) with n_states >= 1; got "
            f"shape {unary.shape}"
        )
        raise ValueError(msg)
    n_nodes = unary.shape[0]
    ends = np.ascontiguousarray(ends, dtype=np.int64)
    coupling = np.ascontiguousarray(coupling, dtype=np.float64)
    if ends.ndim != 2 or ends.shape[1] != 2:
        msg = f"ends must be (n_edges, 2); got shape {ends.shape}"
        raise ValueError(msg)
    n_edges = ends.shape[0]
    if coupling.shape != (n_edges,):
        msg = (
            f"coupling has shape {coupling.shape} and ends {n_edges} rows; they "
            f"index in step"
        )
        raise ValueError(msg)
    if n_edges and not (
        int(ends.min()) >= 0
        and int(ends.max()) < n_nodes
        and bool((ends[:, 0] < ends[:, 1]).all())
    ):
        msg = f"ends must be (lower, higher) pairs of sites in [0, {n_nodes})"
        raise ValueError(msg)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    slots = np.ascontiguousarray(slots, dtype=np.int64)
    neighbours = np.ascontiguousarray(neighbours, dtype=np.int64)
    if offsets.shape != (n_nodes + 1,):
        msg = (
            f"offsets has shape {offsets.shape}, expected ({n_nodes + 1},) "
            f"(one per node, plus the end)"
        )
        raise ValueError(msg)
    if bool((np.diff(offsets) < 0).any()) or int(offsets[0]) != 0:
        msg = "offsets must start at 0 and be non-decreasing"
        raise ValueError(msg)
    if slots.shape != (2 * n_edges,) or int(offsets[n_nodes]) != 2 * n_edges:
        msg = (
            f"slots has shape {slots.shape} and offsets ends at "
            f"{int(offsets[n_nodes])}; both must be 2 * n_edges = {2 * n_edges}"
        )
        raise ValueError(msg)
    if neighbours.shape != (2 * n_edges,):
        msg = f"neighbours has shape {neighbours.shape}, expected ({2 * n_edges},)"
        raise ValueError(msg)
    if n_edges and not (int(slots.min()) >= 0 and int(slots.max()) < 2 * n_edges):
        msg = f"slots must index entries in [0, {2 * n_edges})"
        raise ValueError(msg)
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    if weight.shape != (n_nodes,):
        msg = f"weight has shape {weight.shape}, expected ({n_nodes},)"
        raise ValueError(msg)
    chain_offsets = np.ascontiguousarray(chain_offsets, dtype=np.int64)
    chain_heads = np.ascontiguousarray(chain_heads, dtype=np.int64)
    chain_edges = np.ascontiguousarray(chain_edges, dtype=np.int64)
    n_chains = chain_heads.shape[0]
    if (
        chain_offsets.shape != (n_chains + 1,)
        or int(chain_offsets[-1]) != (chain_edges.shape[0])
    ):
        msg = (
            f"chain_offsets has shape {chain_offsets.shape} for {n_chains} chains "
            f"of {chain_edges.shape[0]} edges; expected ({n_chains + 1},) ending "
            f"at the edge count"
        )
        raise ValueError(msg)
    if chain_edges.size and not (
        int(chain_edges.min()) >= 0 and int(chain_edges.max()) < n_edges
    ):
        msg = f"chain_edges must index edges in [0, {n_edges})"
        raise ValueError(msg)
    if n_chains and not (
        int(chain_heads.min()) >= 0 and int(chain_heads.max()) < n_nodes
    ):
        msg = f"chain_heads must index sites in [0, {n_nodes})"
        raise ValueError(msg)
    if max_iterations < 1:
        msg = f"max_iterations must be >= 1, got {max_iterations}"
        raise ValueError(msg)
    if not tolerance >= 0.0:
        msg = f"tolerance must be >= 0, got {tolerance}"
        raise ValueError(msg)
    messages = np.zeros((2 * n_edges, unary.shape[1]), dtype=np.float64)
    labelling = np.zeros(n_nodes, dtype=np.int64)
    trace = np.empty(max_iterations, dtype=np.float64)
    energies = np.empty(max_iterations, dtype=np.float64)
    signed = int(
        trws_iterations(
            unary,
            offsets,
            slots,
            neighbours,
            ends,
            coupling,
            weight,
            chain_offsets,
            chain_heads,
            chain_edges,
            max_iterations,
            tolerance,
            messages,
            labelling,
            trace,
            energies,
        )
    )
    taken = abs(signed)
    return messages, labelling, trace[:taken], energies[:taken], taken, signed < 0
