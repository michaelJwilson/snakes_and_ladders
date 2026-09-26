"""The ``numba`` descent sweep of :mod:`sal.search.icm`, on the Rust sweeps' signature.

:func:`icm_sweeps` takes ``state`` in place, ``field`` the 2-D log-weight,
the compressed adjacency, and its randomness --- the visiting orders and the
floor's uniforms --- drawn up front, so the Python oracle in
:func:`sal.search.icm.iterated_conditional_modes` reads the
same draws and reproduces it bitwise. :func:`icm_sweeps_checked` checks its
shapes with the Rust entries' messages (#571), because a compiled kernel
indexes without bounds checks. It touches no Python object and so is
``nogil=True`` (#604). A backend's twin lives at
``<subpackage>.<solver>.<backend>`` (issue #1059).
"""

from __future__ import annotations

import numpy as np
from numba import njit


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

    The update :func:`sal.search.icm.iterated_conditional_modes`
    states, on the Rust sweeps' signature (issue #1055): each site takes the
    first state minimizing ``-field[node, s] - sum_j J_ij [s == state[j]]``,
    ``field`` the log-weight, row ``orders[sweep % rows]`` giving the visiting
    order --- one row per sweep, or one row every sweep repeats --- or index
    order where ``orders`` has no rows.

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
    rows = max(orders.shape[0], 1)
    sweeps = 0
    for sweep in range(n_sweeps):
        sweeps += 1
        changed = False
        for position in range(n_nodes):
            node = orders[sweep % rows, position] if ordered else position
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
    elif orders.shape not in {(n_sweeps, n_nodes), (1, n_nodes)}:
        msg = (
            f"orders has shape {orders.shape}, expected ({n_sweeps}, {n_nodes}) "
            f"(one visiting order per sweep), (1, {n_nodes}) (one order every "
            f"sweep repeats) or empty (index order)"
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


@njit(cache=True, nogil=True)
def greedy_colouring(offsets: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """:func:`sal.search.icm.colouring`'s greedy rule, compiled (issue #1073).

    Each site in index order takes the smallest colour none of its earlier
    neighbours holds. The Python loop is the oracle it is pinned against;
    it cost 10 ms of a 16 ms checkerboard descent at 75 x 75.
    """
    n_nodes = offsets.size - 1
    colour = np.full(n_nodes, -1, dtype=np.int64)
    taken = np.full(n_nodes + 1, -1, dtype=np.int64)
    for node in range(n_nodes):
        for position in range(offsets[node], offsets[node + 1]):
            neighbour_colour = colour[neighbours[position]]
            if neighbour_colour >= 0:
                taken[neighbour_colour] = node
        chosen = 0
        while taken[chosen] == node:
            chosen += 1
        colour[node] = chosen
    return colour
