"""Iterated conditional modes: single-site descent on a Potts energy, with a minimum-sites floor.

Each site takes the state minimizing the energy given its neighbours, until a
sweep changes nothing. It is the *same objective* alpha expansion
(:mod:`snakes_and_ladders.search.alpha_expansion`) minimizes under a move set
of one site at a time, so a difference between the two is a statement about
the move set. It was written there as the baseline the expansion has to beat
(#858) and moved here as a solver of its own (issue #1055), returning the
expansion's :class:`~snakes_and_ladders.search.alpha_expansion.Labelling`.

**The floor.** With ``min_sites > 0``, after each sweep's site updates every
state holding ``0 < count < min_sites`` sites is dissolved: each of its sites,
in index order, takes ``surviving[floor(u * m)]``, where ``surviving`` is the
ascending list of states holding at least ``min_sites`` sites, ``m`` its
length, the counts read after the sweep and before any recolouring, and ``u``
the site's uniform for that sweep. An empty state is not below the floor and
may be joined. A recolouring is a change, so the descent continues past it.
``min_sites = 0`` is the descent before the floor existed, bitwise.

**The draws.** Both backends read the same randomness, drawn up front in one
order: the start where none is given (``rng.integers``), then one
permutation per sweep for a random order that runs every sweep, then
``rng.random(max_sweeps * n_nodes)`` for the floor, only where
``min_sites > 0``. So an unfloored descent spends the generator exactly as it
did before the floor, and a floored one spends ``max_sweeps * n_nodes``
uniforms however early it stops. A random order that stops on a clean sweep
draws its permutations per sweep, which only
:data:`~snakes_and_ladders.backend.Backend.PYTHON` runs; its floor's uniforms
precede them.

**The kernel.** :func:`icm_sweeps` is the ``numba`` descent on the Rust
sweeps' signature and convention: ``state`` in place, ``field`` the 2-D
log-weight, the compressed adjacency, and the orders and uniforms above.
:func:`icm_sweeps_checked` checks its shapes with the Rust entries' messages
(#571), because a compiled kernel indexes without bounds checks. It touches
no Python object and so is ``nogil=True`` (#604). It lives beside its solver
rather than in :mod:`snakes_and_ladders.sample.kernels` (issue #1059).
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numba import njit

from snakes_and_ladders.backend import Backend, refuse_backend
from snakes_and_ladders.search.alpha_expansion import Labelling
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import SiteField, energy, log_weight_of, site_field


@njit(cache=True, nogil=True)
def icm_sweeps(
    state: np.ndarray,
    field: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    orders: np.ndarray,
    draws: np.ndarray,
    n_sweeps: int,
    stop_when_clean: bool,
    min_sites: int,
) -> int:
    """Single-site descent in place, with the minimum-sites floor; returns the sweeps run.

    The update :func:`snakes_and_ladders.search.icm.iterated_conditional_modes`
    states, on the Rust sweeps' signature (issue #1055): each site takes the
    first state minimizing ``-field[node, s] - sum_j J_ij [s == state[j]]``,
    ``field`` the log-weight, row ``orders[sweep]`` giving the visiting order,
    or index order where ``orders`` has no rows.

    After each sweep's site updates, with ``min_sites > 0``, every state
    holding ``0 < count < min_sites`` sites is dissolved: each of its sites,
    in index order, takes ``surviving[floor(draws[sweep * n_nodes + node] *
    m)]``, ``surviving`` the ascending states holding at least ``min_sites``
    and ``m`` their number, the counts read before any site is recoloured. A
    recolouring is a change. ``min_sites == 0`` reads no draw.

    Unchecked: :func:`icm_sweeps_checked` validates the shapes first. Holds no
    Python object, so the GIL is released for the call.

    Returns
    -------
    int
        The sweeps run, or ``-sweep`` (1-based) where a sweep dissolved a
        state with no state at the floor to take its sites; ``state`` is then
        as that sweep's site updates left it.
    """
    n_nodes = state.shape[0]
    n_states = field.shape[1]
    local = np.empty(n_states, dtype=np.float64)
    counts = np.zeros(n_states, dtype=np.int64)
    surviving = np.empty(n_states, dtype=np.int64)
    ordered = orders.shape[0] > 0
    sweeps = 0
    for sweep in range(n_sweeps):
        sweeps += 1
        changed = False
        for position in range(n_nodes):
            node = orders[sweep, position] if ordered else position
            for label in range(n_states):
                local[label] = -field[node, label]
            for edge in range(offsets[node], offsets[node + 1]):
                local[state[neighbours[edge]]] -= couplings[edge]
            best = 0
            for label in range(1, n_states):
                if local[label] < local[best]:
                    best = label
            if best != state[node]:
                state[node] = best
                changed = True
        if min_sites > 0:
            for label in range(n_states):
                counts[label] = 0
            for node in range(n_nodes):
                counts[state[node]] += 1
            m = 0
            below = False
            for label in range(n_states):
                if counts[label] >= min_sites:
                    surviving[m] = label
                    m += 1
                elif counts[label] > 0:
                    below = True
            if below:
                if m == 0:
                    return -sweeps
                base = sweep * n_nodes
                for node in range(n_nodes):
                    if counts[state[node]] < min_sites:
                        pick = int(draws[base + node] * m)
                        state[node] = surviving[min(pick, m - 1)]
                        changed = True
        if stop_when_clean and not changed:
            break
    return sweeps


def no_survivor(sweep: int, min_sites: int) -> str:
    """The message both backends raise where a floor leaves no state to recolour into."""
    return (
        f"sweep {sweep} left no state holding min_sites={min_sites} sites, so "
        f"the states below it have nowhere to dissolve into; a floor of at most "
        f"ceil(n_nodes / n_states) always leaves one"
    )


def icm_sweeps_checked(
    state: np.ndarray,
    field: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    orders: np.ndarray,
    draws: np.ndarray,
    n_sweeps: int,
    stop_when_clean: bool,
    min_sites: int,
) -> int:
    """:func:`icm_sweeps` behind the shape checks the Rust entries make (#571), messages theirs.

    ``state`` is updated in place, so it must already be a 1-D C-contiguous
    ``int64`` array; the rest are made contiguous in the kernel's dtypes here,
    as the Rust wrappers' readonly views require.

    Returns
    -------
    int
        The sweeps run.

    Raises
    ------
    ValueError
        Naming what was wanted and what was given, where a shape, a range or
        a count is off; or :func:`no_survivor`'s message where a sweep leaves
        no state at the floor.
    """
    if (
        not isinstance(state, np.ndarray)
        or state.ndim != 1
        or state.dtype != np.int64
        or not state.flags.c_contiguous
    ):
        given = (
            f"{state.dtype}, shape {state.shape}"
            if isinstance(state, np.ndarray)
            else type(state).__name__
        )
        msg = (
            f"state must be a 1-D C-contiguous int64 array, updated in place; "
            f"got {given}"
        )
        raise ValueError(msg)
    n_nodes = state.shape[0]
    field = np.ascontiguousarray(field, dtype=np.float64)
    if field.ndim != 2:
        msg = (
            f"field must be 2-D, (n_nodes, n_states), one row per site; got "
            f"shape {field.shape}"
        )
        raise ValueError(msg)
    n_rows, n_states = field.shape
    if n_rows != n_nodes:
        msg = (
            f"field has {n_rows} rows and state has {n_nodes} sites; the field "
            f"carries one row per site"
        )
        raise ValueError(msg)
    if n_states == 0:
        msg = "field is empty, so there are no states to draw from"
        raise ValueError(msg)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    neighbours = np.ascontiguousarray(neighbours, dtype=np.int64)
    couplings = np.ascontiguousarray(couplings, dtype=np.float64)
    if offsets.ndim != 1 or offsets.shape[0] != n_nodes + 1:
        msg = (
            f"offsets has length {offsets.shape[0] if offsets.ndim == 1 else offsets.shape}, "
            f"expected {n_nodes + 1} (one per node, plus the end)"
        )
        raise ValueError(msg)
    if bool((offsets < 0).any()):
        msg = "offsets must be >= 0"
        raise ValueError(msg)
    if bool((np.diff(offsets) < 0).any()):
        msg = "offsets must be non-decreasing, one row of the adjacency per node"
        raise ValueError(msg)
    if neighbours.shape != couplings.shape or neighbours.ndim != 1:
        msg = (
            f"neighbours has length {neighbours.shape} and couplings "
            f"{couplings.shape}; they index in step"
        )
        raise ValueError(msg)
    if int(offsets[n_nodes]) != neighbours.shape[0]:
        msg = (
            f"offsets ends at {int(offsets[n_nodes])} but the adjacency has "
            f"{neighbours.shape[0]} entries"
        )
        raise ValueError(msg)
    if neighbours.size and not (
        int(neighbours.min()) >= 0 and int(neighbours.max()) < n_nodes
    ):
        msg = f"neighbours must index sites in [0, {n_nodes})"
        raise ValueError(msg)
    if n_sweeps < 0:
        msg = f"n_sweeps must be >= 0, got {n_sweeps}"
        raise ValueError(msg)
    if min_sites < 0:
        msg = f"min_sites must be >= 0, got {min_sites}"
        raise ValueError(msg)
    orders = np.ascontiguousarray(orders, dtype=np.int64)
    if orders.size == 0:
        orders = np.empty((0, n_nodes), dtype=np.int64)
    elif orders.shape != (n_sweeps, n_nodes):
        msg = (
            f"orders has shape {orders.shape}, expected ({n_sweeps}, {n_nodes}) "
            f"(one visiting order per sweep) or empty (index order)"
        )
        raise ValueError(msg)
    elif not (int(orders.min()) >= 0 and int(orders.max()) < n_nodes):
        msg = f"orders must index sites in [0, {n_nodes})"
        raise ValueError(msg)
    draws = np.ascontiguousarray(draws, dtype=np.float64).reshape(-1)
    if min_sites > 0 and draws.shape[0] != n_sweeps * n_nodes:
        msg = f"draws has length {draws.shape[0]}, expected {n_sweeps} * {n_nodes}"
        raise ValueError(msg)
    outside = np.flatnonzero((state < 0) | (state >= n_states))
    if outside.size:
        node = int(outside[0])
        msg = f"state at node {node} is {int(state[node])}, expected [0, {n_states})"
        raise ValueError(msg)
    sweeps = int(
        icm_sweeps(
            state,
            field,
            offsets,
            neighbours,
            couplings,
            orders,
            draws,
            n_sweeps,
            stop_when_clean,
            min_sites,
        )
    )
    if sweeps < 0:
        raise ValueError(no_survivor(-sweeps, min_sites))
    return sweeps


class SweepOrder(StrEnum):
    """The order one sweep visits the sites in --- a parameter, not a second method.

    A sweep order is first-class here for the reason
    ``likelihood/schedule.py`` gives for message orders: being unable to ask
    for a different one hides what the default buys. The update is the same
    argmin either way, so the pair's spread *is* the order's effect
    (issue #858).
    """

    INDEX = "index"
    """``range(n_nodes)``: the sites in index order, every sweep."""
    RANDOM = "random"
    """A fresh ``rng.permutation(n_nodes)`` per sweep, which is Gibbs at T = 0."""


def check_min_sites(min_sites: int, n_nodes: int) -> None:
    """Refuse a floor that is negative or that no state can reach.

    Raises
    ------
    ValueError
        If ``min_sites < 0`` or ``min_sites > n_nodes``.
    """
    if min_sites < 0:
        msg = f"min_sites must be >= 0, got {min_sites}"
        raise ValueError(msg)
    if min_sites > n_nodes:
        msg = (
            f"min_sites={min_sites} exceeds the {n_nodes} sites, so no state can "
            f"hold it and every state would dissolve"
        )
        raise ValueError(msg)


def iterated_conditional_modes(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    *,
    start: np.ndarray | None = None,
    max_sweeps: int = 200,
    sweep_order: SweepOrder = SweepOrder.INDEX,
    stop_when_clean: bool = True,
    min_sites: int = 0,
    backend: Backend = Backend.NUMBA,
) -> Labelling:
    """Single-site descent to a local minimum, dissolving states below ``min_sites`` after each sweep.

    Three parameters say what a caller varies and the sweep does not
    (issue #858): where it starts, the order it visits sites in, and whether
    a clean sweep ends it. Gibbs at ``T = 0`` is this descent under
    :data:`SweepOrder.RANDOM` with ``stop_when_clean=False``, and the label
    block of :mod:`snakes_and_ladders.search.spatio_sequential` is this
    descent from a given ``start``; neither is a second implementation of the
    update. The floor and its draws are the module's.

    Local deltas rather than a full energy per candidate: only the site's own
    field term and its incident edges change, so a sweep costs
    ``O(n_nodes * k * degree)`` rather than ``O(n_nodes * k * n_edges)``.

    ``backend`` chooses the sweep's implementation and nothing else. The
    :data:`~snakes_and_ladders.backend.Backend.NUMBA` kernel,
    :func:`icm_sweeps`, returns the
    labelling the Python loop returns **bitwise** --- same update, same
    order, same first-minimum tie rule, same floor from the same draws ---
    which is what lets it be the default (#264).
    :data:`~snakes_and_ladders.backend.Backend.PYTHON` is the oracle that
    pins it.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, read through its compressed adjacency.
    field : SiteField | np.ndarray
        External field as a log-weight, ``(n_states,)`` or
        ``(n_nodes, n_states)``, or a :class:`~snakes_and_ladders.sim.potts.SiteField`.
    n_states : int
        Labels available at each site.
    rng : np.random.Generator
        Draws the start where ``start`` is ``None``, one permutation per
        sweep under :data:`SweepOrder.RANDOM`, and the floor's uniforms where
        ``min_sites > 0``, in that order.
    start : np.ndarray | None
        The labelling to descend from, or ``None`` to draw one uniformly.
    max_sweeps : int
        Sweeps the descent is allowed.
    sweep_order : SweepOrder
        The order sites are visited in; index order by default.
    stop_when_clean : bool
        Whether a sweep that changes nothing, recolouring included, ends the
        descent. ``False`` runs every sweep of ``max_sweeps``.
    min_sites : int
        The floor: after each sweep a state holding at least one and fewer
        than this many sites is dissolved into the states at or above it.
        ``0``, the default, dissolves nothing and draws nothing.
    backend : Backend
        The sweep's implementation. A random order that stops on a clean
        sweep draws each permutation as the sweep starts, which the compiled
        kernel, taking every order up front, would spend past the stop; it
        needs :data:`~snakes_and_ladders.backend.Backend.PYTHON` and is
        refused. The floor's uniforms are drawn up front on both backends,
        so a floored index-order descent that stops early runs on either.

    Returns
    -------
    Labelling
        The labelling it settles on, and its energy.

    Raises
    ------
    ValueError
        If ``min_sites`` is negative or exceeds ``n_nodes``; if a sweep
        leaves no state at the floor to dissolve into (never where
        ``min_sites <= ceil(n_nodes / n_states)``); if ``backend`` names no
        sweep, or names the compiled one for a descent it does not implement.
    """
    n_nodes = graph.n_nodes
    check_min_sites(min_sites, n_nodes)
    values = site_field(np.asarray(log_weight_of(field), dtype=float), n_nodes)
    labelling = (
        rng.integers(0, n_states, size=n_nodes)
        if start is None
        else np.asarray(start, dtype=np.int64).copy()
    )
    lazy = sweep_order is SweepOrder.RANDOM and stop_when_clean
    if backend is Backend.NUMBA and lazy:
        msg = (
            f"the compiled sweep takes every sweep's order drawn up front, "
            f"which spends the generator past a clean sweep the Python sweep "
            f"stops at; {sweep_order} order with stop_when_clean=True needs "
            f"{Backend.PYTHON}"
        )
        raise ValueError(msg)
    refuse_backend(
        "iterated conditional modes", backend, (Backend.NUMBA, Backend.PYTHON)
    )
    # The permutations the sweep visits in, one per sweep and in the order a
    # per-sweep draw would take them (issue #923): with every sweep run, the
    # same stream. No rows is index order.
    orders = (
        np.stack([rng.permutation(n_nodes) for _ in range(max_sweeps)]).astype(np.int64)
        if sweep_order is SweepOrder.RANDOM and not lazy and max_sweeps > 0
        else np.empty((0, n_nodes), dtype=np.int64)
    )
    draws = (
        rng.random(max_sweeps * n_nodes) if min_sites > 0 else np.empty(0, np.float64)
    )
    offsets, neighbour_index, edge_couplings = graph.compressed_adjacency()

    if backend is Backend.NUMBA:
        labelling = np.ascontiguousarray(labelling, dtype=np.int64)
        icm_sweeps_checked(
            labelling,
            values,
            offsets,
            neighbour_index,
            edge_couplings,
            orders,
            draws,
            max_sweeps,
            stop_when_clean,
            min_sites,
        )
        return Labelling(labelling, energy(graph, values, labelling))

    # The compressed rows as Python sequences, converted once rather than
    # sliced per site: a NumPy slice and gather per site measured a third of
    # this sweep (issue #277, `sim.potts.heat_bath_log_weights`). The
    # `numba` kernel takes the arrays themselves.
    bounds = offsets.tolist()
    neighbours, couplings = neighbour_index.tolist(), edge_couplings.tolist()
    uniforms = draws.tolist()

    # The labels as a Python list for the duration: the sweep reads a
    # neighbour's label once per incident edge, and a list read is 0.18 us
    # cheaper than a NumPy scalar one. Same reads, same order, same writes.
    labels = labelling.tolist()
    for sweep in range(max_sweeps):
        order = (
            rng.permutation(n_nodes)
            if lazy
            else (orders[sweep] if orders.shape[0] else range(n_nodes))
        )
        changed = False
        for node in order:
            local = -values[node].copy()
            for position in range(bounds[node], bounds[node + 1]):
                local[labels[neighbours[position]]] -= couplings[position]
            best = int(np.argmin(local))
            if best != labels[node]:
                labels[node] = best
                changed = True
        if min_sites > 0 and _dissolve(labels, uniforms, sweep, n_states, min_sites):
            changed = True
        if stop_when_clean and not changed:
            break
    labelling[:] = labels

    return Labelling(labelling, energy(graph, values, labelling))


def _dissolve(
    labels: list[int],
    uniforms: list[float],
    sweep: int,
    n_states: int,
    min_sites: int,
) -> bool:
    """The floor after one sweep, in place: the oracle of the kernel's; whether a site moved.

    Raises
    ------
    ValueError
        If a state is below the floor and none is at it.
    """
    counts = [0] * n_states
    for label in labels:
        counts[label] += 1
    if not any(0 < count < min_sites for count in counts):
        return False
    surviving = [state for state in range(n_states) if counts[state] >= min_sites]
    if not surviving:
        raise ValueError(no_survivor(sweep + 1, min_sites))
    m = len(surviving)
    base = sweep * len(labels)
    for node, label in enumerate(labels):
        if counts[label] < min_sites:
            labels[node] = surviving[min(int(uniforms[base + node] * m), m - 1)]
    return True
