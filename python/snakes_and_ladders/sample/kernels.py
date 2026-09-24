"""Compiled kernels whose oracles reproduce them bitwise, beside those oracles.

Issue #264's audit ranked the Python-level loops of this package by self time
and found one that a compiled kernel removes outright: the single-site descent
sweep in :mod:`snakes_and_ladders.search.icm`. It is integer arithmetic over the
compressed-row adjacency (:meth:`snakes_and_ladders.sim.graph.PottsGraph.compressed_adjacency`)
and deterministic given its inputs, so the oracle reproduces it **bitwise**:
the first minimum on a tie is the first minimum in both. The energy it
reports stays with the NumPy oracle, whose pairwise summation a sequential
loop would match only to the last place.

That exactness is why these are ``numba`` rather than Rust. Root
``CLAUDE.md``'s backend rule admits one compiled path per measurement; the
Rust extension already carries the Potts *sampling* sweep
(:func:`snakes_and_ladders.sample.potts_mcmc.sample_potts` on the extension), and a second copy of it
is what the rule exists to refuse. A kernel whose pin is exact carries no
such cost and lives here.

Issue #561's sweep over the factor graph is here on that same test rather
than on being deterministic. A heat-bath draw exponentiates, and ``libm``'s
``exp`` and NumPy's disagree in the last place on 4.6% of ``float64`` inputs,
so the arithmetic alone would make it a distributional port. What keeps the
pin exact is that :func:`gibbs_sweep_sites` decides a site only where the
draw clears every cumulative boundary by more than the two exponentials can
move it, and hands the rest back; a bound, not an assumption about rounding.

Issue #563's density is here for the reason the descent sweep is: it takes no
exponential, so summing the same terms in the same order is bitwise
reproduction outright, and the pin needs no bound. What it does need is that
order -- floating-point addition is not associative, so the kernel sums
factors left to right in graph order as the oracle does, and a vectorized sum
would be a different number.

Issue #1055 put the descent on the Rust sweeps' signature and convention:
``state`` in place, ``field`` the 2-D log-weight, the compressed adjacency,
and its randomness --- the visiting orders and the floor's uniforms ---
drawn up front, so the oracle reads the same draws. Its shapes are checked
by :func:`icm_sweeps_checked` in Python, with the Rust entries' messages
(#571), because a compiled kernel indexes without bounds checks. Every
kernel here touches no Python object and so is ``nogil=True``: a thread
backend runs them concurrently, the rule root ``CLAUDE.md`` states (#604).

The kernels take arrays, never the graph: the caller flattens once and the
kernel walks strides (the layout rule), and ``cache=True`` writes the compiled
object beside the source so the first call in a process pays once.
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


@njit(cache=True, nogil=True)
def gibbs_sweep_sites(
    state: np.ndarray,
    draws: np.ndarray,
    cardinality: np.ndarray,
    tables: np.ndarray,
    entry_offsets: np.ndarray,
    entry_start: np.ndarray,
    entry_stride: np.ndarray,
    term_offsets: np.ndarray,
    term_column: np.ndarray,
    term_stride: np.ndarray,
    local: np.ndarray,
    beta: float,
    guard: float,
    first: int,
) -> int:
    """Heat-bath updates from ``first`` on, in place; returns where it stopped.

    The update :func:`snakes_and_ladders.sample.gibbs.gibbs_sweep` states, over the
    edge layout :class:`snakes_and_ladders.sample.gibbs._EdgeLayout` builds: the
    variable's conditional is gathered from ``tables`` by the entries that
    touch it, scaled by ``beta``, shifted by its maximum, exponentiated into a
    running cumulative sum, and searched for ``draws[position]`` times that
    sum's last entry.

    Every step but one is the arithmetic NumPy performs, operation for
    operation, so it is reproduced bitwise. The exception is ``exp``, which
    NumPy computes by its own SIMD polynomial and this kernel by ``libm``:
    the two disagree in the last place on 4.6% of ``float64`` inputs
    (measured, 2e6 draws), and one draw across a moved threshold sends two
    chains apart. So the kernel decides a site only where the decision cannot
    depend on that last place -- where the draw clears every cumulative
    boundary by ``slack``, which bounds the difference the two exponentials
    can make to the boundary and to the draw together. Where it does not,
    the kernel returns that position without touching ``state``, and the
    NumPy path decides that one site. That is what makes the compiled sweep
    the *same chain* rather than a chain of the same law, and it is why this
    kernel is the default. Issue #599 has since carried the same
    construction into the Rust sampling sweep, which is the default too
    (:mod:`snakes_and_ladders.backend`).

    ``guard`` sets that boundary in units of the last place, per state:
    :func:`snakes_and_ladders.sample.gibbs.gibbs_sweep` passes the bound it derives,
    and a test raises it to drive every site onto the NumPy path and pin the
    handing back itself, which no realistic draw reaches.

    Returns
    -------
    int
        ``state.shape[0]`` if every remaining site was decided, otherwise the
        position of the first site it declined to decide.
    """
    n_variables = int(state.shape[0])
    for position in range(first, n_variables):
        n_states = cardinality[position]
        for label in range(n_states):
            local[label] = 0.0
        for entry in range(entry_offsets[position], entry_offsets[position + 1]):
            base = entry_start[entry]
            for term in range(term_offsets[entry], term_offsets[entry + 1]):
                base += state[term_column[term]] * term_stride[term]
            stride = entry_stride[entry]
            for label in range(n_states):
                local[label] += tables[base + label * stride]
        top = local[0] * beta
        for label in range(n_states):
            local[label] = local[label] * beta
            top = max(top, local[label])
        running = 0.0
        for label in range(n_states):
            running += np.exp(local[label] - top)
            local[label] = running
        target = draws[position] * running
        # 2**-52, the largest relative gap between neighbouring float64 values.
        slack = guard * n_states * 2.220446049250313e-16 * running
        chosen = -1
        for label in range(n_states):
            gap = local[label] - target
            if not (gap > slack or -gap > slack):
                return position
            if chosen < 0 and gap >= 0.0:
                chosen = label
        if chosen < 0:
            return position
        state[position] = chosen
    return n_variables


@njit(cache=True, nogil=True)
def factor_graph_log_density(
    state: np.ndarray,
    tables: np.ndarray,
    factor_start: np.ndarray,
    factor_offsets: np.ndarray,
    factor_column: np.ndarray,
    factor_stride: np.ndarray,
) -> float:
    """``sum_f log psi_f`` at one full assignment, over the edge layout.

    The density :meth:`snakes_and_ladders.sim.factor_graph.FactorGraph.log_density`
    defines, read from the ``tables`` of
    :class:`snakes_and_ladders.sample.gibbs._EdgeLayout` rather than from a table
    and a tuple key per factor: each factor's element is the offset its axes
    fix, and the sum runs left to right over factors in graph order.

    That order is the whole of the pin. Floating-point addition is not
    associative, so a pairwise or vectorized sum may move the last place of a
    recorded log-density; this accumulates the same terms in the same sequence
    as the oracle and reproduces it **bitwise**. No exponential is taken, so
    nothing here depends on which ``exp`` :func:`gibbs_sweep_sites` had to
    bound.

    **The pin is not a cost, which #651 established by measuring it.** Gathering
    the offsets and calling ``np.sum`` is *slower* at every size from a thousand
    factors to a million --- 0.60x to 0.77x, the extra array and its second pass
    costing more than fusing the accumulate saves --- so the ordered form is
    also the fast one and nothing is traded for the reproduction.
    ``tests/benchmarks/test_factor_density_sum_bench.py`` holds both.
    """
    total = 0.0
    for factor in range(factor_start.shape[0]):
        offset = factor_start[factor]
        for axis in range(factor_offsets[factor], factor_offsets[factor + 1]):
            offset += state[factor_column[axis]] * factor_stride[axis]
        total += tables[offset]
    return total
