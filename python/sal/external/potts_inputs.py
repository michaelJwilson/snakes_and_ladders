"""The bytes gco, PyMaxflow, HiGHS and OpenGM receive for a Potts model (issues #1282, #1279).

One construction per framework, read by :func:`sal.external.potts.ground_state`
and :func:`sal.external.potts.lower_bound`, and by the adapters
:mod:`sal.validation.gco`, :mod:`sal.validation.pymaxflow` and
:mod:`sal.validation.highs`, so an answer from either path is the other's
bitwise. Apart from :mod:`sal.external.potts` so an adapter imports
NumPy and ``sim`` alone, not the ground-state ladder ``MethodRun`` lives in.

:func:`stand_in` and :func:`allowed_by` are #1274's forbidden-label rule
(#1139), moved here from ``tests/validation/_forbidden.py`` (#1284).
"""

from __future__ import annotations

import numpy as np

from sal.sim.graph import PottsGraph
from sal.sim.potts import site_field

#: HiGHS's status, from `linprog` and `milp` alike, for an optimal solution.
OPTIMAL = 0

#: The index ``infra/opengm/sal_opengm.cxx`` reads for each OpenGM algorithm,
#: by the name its :class:`~sal.external.solvers.Solver` carries after
#: ``opengm_`` (#1279).
OPENGM_ALGORITHMS = {
    "icm": 0,
    "lbp": 1,
    "astar": 2,
    "trws": 3,
    "dd": 4,
    "expansion": 5,
    "swap": 6,
}

#: Distance from 0 or 1 within which a node marginal reads as integral:
#: HiGHS's default primal feasibility tolerance, 1e-7.
INTEGRALITY = 1e-7


def stand_in(graph: PottsGraph, finite: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """``finite`` with each forbidden entry at its site's large finite penalty (#1274).

    Site ``i``'s forbidden entries sit at its lowest allowed log-weight less
    ``1 + sum_{e at i} |J_e|``. Moving ``i`` to any allowed label then lowers
    the energy by more than ``i``'s couplings can raise it, so no optimum and
    no expansion or swap fixed point holds a forbidden label. Moved here from
    ``tests/validation/_forbidden.py`` (#1284), which reads it from here.
    """
    incident = np.zeros(graph.n_nodes)
    edges, coupling = graph.edge_index, np.abs(graph.edge_coupling)
    np.add.at(incident, edges[:, 0], coupling)
    np.add.at(incident, edges[:, 1], coupling)
    floor = np.where(allowed, finite, np.inf).min(axis=1) - (1.0 + incident)
    return np.where(allowed, finite, floor[:, None])


def allowed_by(allowed: np.ndarray, labelling: np.ndarray) -> bool:
    """Whether every site of ``labelling`` holds an allowed label."""
    return bool(allowed[np.arange(allowed.shape[0]), labelling].all())


def expansion_inputs(
    graph: PottsGraph,
    field: np.ndarray,
    *,
    start: np.ndarray | None = None,
    swap: bool = False,
) -> dict[str, np.ndarray]:
    """gco's script inputs for ``field``, finite, shape ``(n_nodes, n_states)``.

    The data cost is ``-field`` less its row minimum, the edges in ascending
    order within each pair; ``swap`` runs the alpha-beta swap in place of the
    expansion (#997).
    """
    cost = -field
    unary = cost - cost.min(axis=1, keepdims=True)
    edges = np.sort(graph.edge_index, axis=1)
    inputs = {
        "unary": np.ascontiguousarray(unary, dtype=np.float64),
        "first": np.ascontiguousarray(edges[:, 0], dtype=np.int64),
        "second": np.ascontiguousarray(edges[:, 1], dtype=np.int64),
        "weight": np.ascontiguousarray(graph.edge_coupling, dtype=np.float64),
    }
    if start is not None:
        inputs["start"] = np.ascontiguousarray(start, dtype=np.int64)
    if swap:
        inputs["swap"] = np.asarray(True)
    return inputs


def cut_inputs(
    n_nodes: int,
    tail: np.ndarray,
    head: np.ndarray,
    capacity: np.ndarray,
    reverse: np.ndarray,
    source: np.ndarray,
    sink: np.ndarray,
) -> dict[str, np.ndarray]:
    """PyMaxflow's script inputs for non-terminal nodes ``0 .. n_nodes - 1``."""
    return {
        "n_nodes": np.asarray(n_nodes, dtype=np.int64),
        "tail": np.ascontiguousarray(tail, dtype=np.int64),
        "head": np.ascontiguousarray(head, dtype=np.int64),
        "capacity": np.ascontiguousarray(capacity, dtype=np.float64),
        "reverse": np.ascontiguousarray(reverse, dtype=np.float64),
        "source": np.ascontiguousarray(source, dtype=np.float64),
        "sink": np.ascontiguousarray(sink, dtype=np.float64),
    }


def ising_inputs(graph: PottsGraph, field: np.ndarray) -> dict[str, np.ndarray]:
    """The two-state ferromagnet's :func:`cut_inputs`, as ``search.maxflow`` builds the cut.

    ``source -> i`` costs ``D_i(1)`` and ``i -> sink`` costs ``D_i(0)`` above
    the per-node minimum, and each coupling is a capacity both ways; a node
    on the sink side takes state 1.
    """
    values = site_field(np.asarray(field, dtype=float), graph.n_nodes, n_states=2)
    cost = -values
    offsets = cost.min(axis=1)
    edges = graph.edge_index
    return cut_inputs(
        graph.n_nodes,
        edges[:, 0],
        edges[:, 1],
        graph.edge_coupling,
        graph.edge_coupling,
        cost[:, 1] - offsets,
        cost[:, 0] - offsets,
    )


def polytope_inputs(
    graph: PottsGraph,
    field: np.ndarray,
    *,
    integral: bool = False,
    time_limit: float = np.inf,
) -> dict[str, np.ndarray]:
    """HiGHS's script inputs for the local-polytope LP of ``field``, shape ``(n_nodes, n_states)``.

    The unary is ``-field``, the edges as the graph lists them. ``integral``
    makes the node marginals integer, the ILP (#1274), and ``time_limit`` is
    HiGHS's own limit on it, in seconds; the LP reads neither.
    """
    edges = graph.edge_index
    return {
        "unary": np.ascontiguousarray(-field, dtype=np.float64),
        "first": np.ascontiguousarray(edges[:, 0], dtype=np.int64),
        "second": np.ascontiguousarray(edges[:, 1], dtype=np.int64),
        "coupling": np.ascontiguousarray(graph.edge_coupling, dtype=np.float64),
        "integral": np.asarray(integral),
        "time_limit": np.asarray(time_limit, dtype=np.float64),
    }


def integral(marginals: np.ndarray) -> bool:
    """Whether every node marginal is within :data:`INTEGRALITY` of 0 or 1."""
    return bool(np.all(np.minimum(marginals, 1.0 - marginals) <= INTEGRALITY))


def opengm_inputs(
    graph: PottsGraph,
    field: np.ndarray,
    algorithm: str,
    *,
    max_iterations: int,
    tolerance: float,
    start: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """OpenGM's script inputs for ``field``, finite, shape ``(n_nodes, n_states)`` (#1279).

    The unary is ``-field`` and each edge ``-J [a == b]``, so OpenGM's value
    is :func:`sal.sim.potts.energy` with no constant restored; the ends are
    ascending within each pair, as OpenGM requires of a factor's variables,
    and the edges in the graph's order. ``algorithm`` is a key of
    :data:`OPENGM_ALGORITHMS`; ``max_iterations`` and ``tolerance`` set its
    own loop.
    """
    edges = np.sort(graph.edge_index, axis=1)
    inputs = {
        "unary": np.ascontiguousarray(-field, dtype=np.float64),
        "first": np.ascontiguousarray(edges[:, 0], dtype=np.int64),
        "second": np.ascontiguousarray(edges[:, 1], dtype=np.int64),
        "coupling": np.ascontiguousarray(graph.edge_coupling, dtype=np.float64),
        "algorithm": np.asarray(OPENGM_ALGORITHMS[algorithm], dtype=np.int64),
        "max_iterations": np.asarray(max_iterations, dtype=np.int64),
        "tolerance": np.asarray(tolerance, dtype=np.float64),
    }
    if start is not None:
        inputs["start"] = np.ascontiguousarray(start, dtype=np.int64)
    return inputs
