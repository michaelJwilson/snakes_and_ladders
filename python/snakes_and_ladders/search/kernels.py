"""Compiled kernels whose oracles reproduce them bitwise, beside those oracles.

Issue #264's audit ranked the Python-level loops of this package by self time
and found one that a compiled kernel removes outright: the single-site descent
sweep in :mod:`snakes_and_ladders.search.alpha_expansion`. It is integer arithmetic over the
compressed-row adjacency (:meth:`snakes_and_ladders.sim.graph.PottsGraph.compressed_adjacency`)
and deterministic given its inputs, so the oracle reproduces it **bitwise**:
the first minimum on a tie is the first minimum in both. The energy it
reports stays with the NumPy oracle, whose pairwise summation a sequential
loop would match only to the last place.

That exactness is why these are ``numba`` rather than Rust. Root
``CLAUDE.md``'s backend rule admits one compiled path per measurement; the
Rust extension already carries the Potts *sampling* sweep, whose agreement
with its oracle can only be distributional
(:mod:`snakes_and_ladders.search.potts_mcmc_rust`), and a second copy of that
reasoning is what the rule exists to refuse. A kernel whose pin is exact
carries no such cost and lives here.

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

The kernels take arrays, never the graph: the caller flattens once and the
kernel walks strides (the layout rule), and ``cache=True`` writes the compiled
object beside the source so the first call in a process pays once.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def icm_sweeps(
    labelling: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    max_sweeps: int,
) -> int:
    """Single-site descent in place; returns the sweeps run.

    The same update :func:`snakes_and_ladders.search.alpha_expansion.iterated_conditional_modes`
    states: each site takes the label minimizing ``-values[node, label] -
    sum_j J_ij [label == labelling[j]]`` given its neighbours, in index order,
    until a sweep changes nothing or ``max_sweeps`` is reached.
    """
    n_nodes = labelling.shape[0]
    n_states = values.shape[1]
    local = np.empty(n_states, dtype=np.float64)
    sweeps = 0
    for _ in range(max_sweeps):
        sweeps += 1
        changed = False
        for node in range(n_nodes):
            for state in range(n_states):
                local[state] = -values[node, state]
            for position in range(offsets[node], offsets[node + 1]):
                local[labelling[neighbours[position]]] -= couplings[position]
            best = 0
            for state in range(1, n_states):
                if local[state] < local[best]:
                    best = state
            if best != labelling[node]:
                labelling[node] = best
                changed = True
        if not changed:
            break
    return sweeps


@njit(cache=True)
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

    The update :func:`snakes_and_ladders.search.gibbs.gibbs_sweep` states, over the
    edge layout :class:`snakes_and_ladders.search.gibbs._EdgeLayout` builds: the
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
    kernel is the default where the Rust sampling sweep is opt-in
    (:mod:`snakes_and_ladders.search.backend`).

    ``guard`` sets that boundary in units of the last place, per state:
    :func:`snakes_and_ladders.search.gibbs.gibbs_sweep` passes the bound it derives,
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


@njit(cache=True)
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
    :class:`snakes_and_ladders.search.gibbs._EdgeLayout` rather than from a table
    and a tuple key per factor: each factor's element is the offset its axes
    fix, and the sum runs left to right over factors in graph order.

    That order is the whole of the pin. Floating-point addition is not
    associative, so a pairwise or vectorized sum would move the last place of
    every recorded log-density; this accumulates the same terms in the same
    sequence as the oracle and reproduces it **bitwise**. No exponential is
    taken, so nothing here depends on which ``exp``
    :func:`gibbs_sweep_sites` had to bound.
    """
    total = 0.0
    for factor in range(factor_start.shape[0]):
        offset = factor_start[factor]
        for axis in range(factor_offsets[factor], factor_offsets[factor + 1]):
            offset += state[factor_column[axis]] * factor_stride[axis]
        total += tables[offset]
    return total
