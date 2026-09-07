"""Exact evaluation of a Potts MRF: enumeration, and the strip transfer matrix.

Both are oracles. :func:`enumerate_potts` sums over every configuration and so
is correct on any graph at a size where ``k ** n_nodes`` is affordable;
:func:`strip_log_partition` is exact on a 2-D open strip at widths where
``k ** M`` columns are affordable, which reaches lattices enumeration cannot.
Belief propagation (:mod:`snakes_and_ladders.likelihood.belief_propagation`) is checked
against the first where it is exact and *measured* against the second where it
is not.

The model is the one :func:`snakes_and_ladders.sim.potts.simulate_potts` samples from, and
the convention is taken from there rather than restated independently: a
configuration ``s`` carries unnormalized log weight

    sum_i h[s_i] + sum_(i,j) in edges J_ij [s_i == s_j]

so ``J > 0`` favours agreement. See ``docs/tex/main.tex``, "Potts Models in an
External Field" (Mezard & Montanari, ch. 2).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.enumeration import (
    MAX_ENUMERABLE_CONFIGURATIONS,
    refuse_oversized,
)
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph


@dataclass(frozen=True)
class ExactPotts:
    """What enumeration establishes about a Potts MRF.

    Parameters
    ----------
    log_partition : float
        ``log Z``.
    single_site : np.ndarray
        Marginal ``p(s_i = a)``, shape ``(n_nodes, n_states)``.
    pairwise : np.ndarray
        Marginal ``p(s_i = a, s_j = b)`` for each edge in the graph's own
        edge order, shape ``(n_edges, n_states, n_states)``.
    """

    log_partition: float
    single_site: np.ndarray
    pairwise: np.ndarray


def log_weights(
    graph: PottsGraph, field: np.ndarray, configurations: np.ndarray
) -> np.ndarray:
    """Unnormalized log weight of each configuration.

    Parameters
    ----------
    graph : PottsGraph
        The graph. Its edge order fixes which coupling applies where.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``.
    configurations : np.ndarray
        Integer states, shape ``(n_configurations, n_nodes)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_configurations,)``.

    Raises
    ------
    ValueError
        If ``configurations`` does not carry one column per node. A silently
        broadcast mismatch would return weights for a different model.
    """
    if configurations.shape[1] != graph.n_nodes:
        msg = (
            f"configurations has {configurations.shape[1]} columns for a graph "
            f"of {graph.n_nodes} nodes"
        )
        raise ValueError(msg)
    total = field[configurations].sum(axis=1)
    for (first, second), coupling in graph.weighted_edges():
        agree = configurations[:, first] == configurations[:, second]
        total = total + coupling * agree
    return np.asarray(total)


def enumerate_potts(
    graph: PottsGraph, field: np.ndarray, *, max_configurations: int | None = None
) -> ExactPotts:
    """Sum over every configuration: the oracle, exponential and deliberately so.

    Shares no recursion, message or factorization code with
    :mod:`snakes_and_ladders.likelihood.belief_propagation` -- only the model convention,
    through :func:`log_weights`. Per ``likelihood/CLAUDE.md``, correctness
    comes from brute force rather than from a second approximate backend.

    Parameters
    ----------
    graph : PottsGraph
        The graph.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``.
    max_configurations : int | None
        Refuse above this many configurations. ``None`` uses
        :data:`snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`.

    Returns
    -------
    ExactPotts
        ``log Z`` and the exact marginals.

    Raises
    ------
    ValueError
        If the enumeration would exceed the cap. Refused rather than
        attempted: the failure would otherwise be a machine running out of
        memory inside a test, which reads as infrastructure rather than as
        the caller asking for a size this cannot do.
    """
    n_states = int(field.shape[0])
    limit = (
        MAX_ENUMERABLE_CONFIGURATIONS
        if max_configurations is None
        else max_configurations
    )
    refuse_oversized(
        n_states**graph.n_nodes,
        what=f"{n_states}**{graph.n_nodes} spin configurations",
        limit=limit,
    )

    configurations = np.array(
        list(itertools.product(range(n_states), repeat=graph.n_nodes)), dtype=np.int64
    )
    weights = log_weights(graph, field, configurations)
    peak = weights.max()
    unnormalized = np.exp(weights - peak)
    total = unnormalized.sum()
    probability = unnormalized / total

    single_site = np.zeros((graph.n_nodes, n_states))
    for node in range(graph.n_nodes):
        np.add.at(single_site[node], configurations[:, node], probability)

    pairwise = np.zeros((len(graph.edges), n_states, n_states))
    for position, (first, second) in enumerate(graph.edges):
        np.add.at(
            pairwise[position],
            (configurations[:, first], configurations[:, second]),
            probability,
        )

    return ExactPotts(
        log_partition=float(np.log(total) + peak),
        single_site=single_site,
        pairwise=pairwise,
    )


def strip_log_partition(
    shape: tuple[int, int],
    boundary: BoundaryCondition,
    coupling: float,
    field: np.ndarray,
) -> float:
    """Exact ``log Z`` for an ``N x M`` strip, by transferring a whole column.

    A column of ``M`` sites has ``k ** M`` joint states, and the recursion
    across ``N`` columns costs ``k ** (2 * M)`` per step. That exponent in
    ``M`` alone, rather than in ``N * M``, is the point: it reaches strips
    that :func:`enumerate_potts` cannot, which is what lets the loopy regime
    be measured against an exact number rather than against another
    approximation.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(N, M)`` -- ``N`` columns of ``M`` sites, matching the row-major
        node indexing :func:`snakes_and_ladders.sim.graph.lattice_graph` uses.
    boundary : BoundaryCondition
        ``OPEN`` only.
    coupling : float
        Uniform ``J``.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)``.

    Returns
    -------
    float
        ``log Z``.

    Raises
    ------
    ValueError
        If ``shape`` is not two-dimensional, or the boundary is periodic. A
        periodic strip closes the recursion into a cycle, which is a trace
        over the transfer operator rather than this forward pass; returning
        this open-boundary number for it would be silently wrong, so it is
        refused instead.
    """
    if len(shape) != 2:
        msg = f"strip_log_partition takes a 2-D shape, got {shape}"
        raise ValueError(msg)
    if boundary is not BoundaryCondition.OPEN:
        msg = (
            f"strip_log_partition is exact for {BoundaryCondition.OPEN} only; "
            f"{boundary} closes the lattice into a cycle, which needs a trace "
            "over the transfer operator rather than this forward recursion"
        )
        raise ValueError(msg)

    n_columns, width = shape
    n_states = int(field.shape[0])
    columns = np.array(
        list(itertools.product(range(n_states), repeat=width)), dtype=np.int64
    )

    # Everything internal to one column: its own field, and the bonds running
    # along it. Identical for every column, so it is built once.
    internal = field[columns].sum(axis=1)
    for position in range(width - 1):
        internal = internal + coupling * (
            columns[:, position] == columns[:, position + 1]
        )

    # The bond between adjacent columns, site by site: `between[u, v]` is the
    # coupling contribution of column state `u` sitting beside column state
    # `v`.
    agreements = (columns[:, np.newaxis, :] == columns[np.newaxis, :, :]).sum(axis=2)
    between = coupling * agreements

    alpha = internal
    for _ in range(n_columns - 1):
        alpha = logsumexp(alpha[:, np.newaxis] + between, axis=0) + internal
    return float(logsumexp(alpha, axis=0))
