"""Iterated conditional modes: single-site descent on a Potts energy, with a minimum-sites floor.

Each site takes the state minimizing the energy given its neighbours, until a
sweep changes nothing. It is the *same objective* alpha expansion
(:mod:`sal.search.alpha_expansion`) minimizes under a move set
of one site at a time, so a difference between the two is a statement about
the move set. It was written there as the baseline the expansion has to beat
(#858) and moved here as a solver of its own (issue #1055), returning the
expansion's :class:`~sal.search.alpha_expansion.Labelling`.

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
:data:`~sal.backend.Backend.PYTHON` runs; its floor's uniforms
precede them.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from sal.backend import Backend, refuse_backend
from sal.search.alpha_expansion import Labelling
from sal.search.numba.icm import icm_sweeps_checked, no_survivor
from sal.sim.graph import PottsGraph
from sal.sim.potts import (
    SiteField,
    check_labelling,
    energy,
    log_weight_of,
    site_field,
)


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
    block of :mod:`sal.search.spatio_sequential` is this
    descent from a given ``start``; neither is a second implementation of the
    update. The floor and its draws are the module's.

    Local deltas rather than a full energy per candidate: only the site's own
    field term and its incident edges change, so a sweep costs
    ``O(n_nodes * k * degree)`` rather than ``O(n_nodes * k * n_edges)``.

    ``backend`` chooses the sweep's implementation and nothing else. The
    :data:`~sal.backend.Backend.NUMBA` kernel,
    :func:`~sal.search.numba.icm.icm_sweeps`, returns the
    labelling the Python loop returns **bitwise** --- same update, same
    order, same first-minimum tie rule, same floor from the same draws ---
    which is what lets it be the default (#264).
    :data:`~sal.backend.Backend.PYTHON` is the oracle that
    pins it.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, read through its compressed adjacency.
    field : SiteField | np.ndarray
        External field as a log-weight, ``(n_states,)`` or
        ``(n_nodes, n_states)``, or a :class:`~sal.sim.potts.SiteField`.
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
        needs :data:`~sal.backend.Backend.PYTHON` and is
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
        sweep, or names the compiled one for a descent it does not implement;
        if ``start`` is not one integer state in range per node
        (:func:`~sal.sim.potts.check_labelling`).
    """
    n_nodes = graph.n_nodes
    check_min_sites(min_sites, n_nodes)
    values = site_field(np.asarray(log_weight_of(field), dtype=float), n_nodes)
    labelling = (
        rng.integers(0, n_states, size=n_nodes)
        if start is None
        else check_labelling(start, n_nodes, n_states)
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
