"""HiGHS on the explicit local-polytope LP, the relaxation TRW-S bounds (issue #1063).

:func:`sal.search.trws.trws` and
:func:`sal.search.tightening.dual_bound` both ascend the dual of the
local-polytope relaxation of ``min_x E(x)`` by coordinate moves, which can
stop short of its maximum. HiGHS, through ``scipy.optimize.linprog(method=
"highs")``, solves the primal LP itself, written out with every node and edge
marginal, and shares no code with either. It runs in ``scripts/highs.py``,
in a subprocess, although SciPy is a core dependency, as every framework
here does.

:func:`local_polytope` returns the LP's optimal value, a lower bound on the
minimum energy in :func:`sal.sim.potts.energy`'s sign, and the primal
optimum's node marginals. Where those are integral the LP is tight: the
labelling they select has energy equal to the value, which is then the
minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, log_weight_of, site_field
from sal.validation.runner import run

#: The script this adapter runs.
SCRIPT = "highs"

#: `linprog`'s status for an optimal solution.
OPTIMAL = 0

#: Distance from 0 or 1 within which a node marginal reads as integral:
#: HiGHS's default primal feasibility tolerance, 1e-7.
INTEGRALITY = 1e-7


@dataclass(frozen=True)
class LocalPolytope:
    """HiGHS's optimum of the local-polytope LP, with what its script measured."""

    #: The LP's optimal value: a lower bound on ``min_x E(x)``.
    value: float
    #: The primal optimum's node marginals, ``(n_nodes, n_states)``.
    node_marginals: np.ndarray
    #: `linprog`'s status; :data:`OPTIMAL` is 0.
    status: int
    #: `linprog`'s message.
    message: str
    #: HiGHS's iteration count.
    iterations: int
    #: Wall seconds of ``linprog`` alone.
    seconds: float
    #: Wall seconds assembling the constraint matrix.
    build_seconds: float
    #: Peak resident bytes the assembly and the solve added.
    peak_bytes: int

    @property
    def integral(self) -> bool:
        """Whether every node marginal is within :data:`INTEGRALITY` of 0 or 1."""
        marginals = self.node_marginals
        return bool(np.all(np.minimum(marginals, 1.0 - marginals) <= INTEGRALITY))

    @property
    def labelling(self) -> np.ndarray:
        """Each site's largest marginal: the LP's labelling where it is integral."""
        return np.argmax(self.node_marginals, axis=1).astype(np.int64)


def local_polytope(
    graph: PottsGraph, field: SiteField | np.ndarray, *, timeout: float = 600.0
) -> LocalPolytope:
    """The local-polytope LP of ``min_x E(x)``, solved by HiGHS.

    ``field`` is a log-weight, ``(n_states,)`` or ``(n_nodes, n_states)``, or
    a :class:`~sal.sim.potts.SiteField`, read as :func:`sal.search.trws.trws`
    reads it. ``timeout`` bounds the subprocess, in seconds.
    """
    values = site_field(
        np.asarray(log_weight_of(field), dtype=np.float64), graph.n_nodes
    )
    edges = graph.edge_index
    result = run(
        SCRIPT,
        {
            "unary": np.ascontiguousarray(-values),
            "first": np.ascontiguousarray(edges[:, 0], dtype=np.int64),
            "second": np.ascontiguousarray(edges[:, 1], dtype=np.int64),
            "coupling": np.ascontiguousarray(graph.edge_coupling, dtype=np.float64),
        },
        timeout=timeout,
    )
    outputs = result.outputs
    return LocalPolytope(
        value=float(outputs["value"]),
        node_marginals=outputs["node_marginals"],
        status=int(outputs["status"]),
        message=str(outputs["message"]),
        iterations=int(outputs["iterations"]),
        seconds=result.seconds,
        build_seconds=float(outputs["build_seconds"]),
        peak_bytes=result.peak_bytes,
    )


#: `milp`'s status when the time limit stops it before the gap closes.
TIME_LIMIT = 1


@dataclass(frozen=True)
class Mip:
    """HiGHS's MIP over the local polytope, with every site outside the free region fixed."""

    #: A labelling of every site: the MIP's on the free region, the fixed
    #: labels elsewhere.
    labelling: np.ndarray
    #: The labelling's energy as the MIP counts it, fixed terms included.
    value: float
    #: HiGHS's lower bound on the restricted problem, in the same sign and
    #: with the same constant: :attr:`value` when the optimum is proven.
    dual_bound: float
    #: The sites the MIP labelled.
    free: np.ndarray
    #: `milp`'s status: :data:`OPTIMAL`, or :data:`TIME_LIMIT`.
    status: int
    #: `milp`'s message.
    message: str
    #: Branch-and-bound nodes.
    nodes: int
    #: Wall seconds of ``milp`` alone.
    seconds: float
    #: Peak resident bytes the assembly and the solve added.
    peak_bytes: int

    @property
    def proven(self) -> bool:
        """Whether HiGHS closed the gap: :attr:`value` is the restricted minimum."""
        return self.status == OPTIMAL


def free_region(graph: PottsGraph, marginals: np.ndarray, rings: int) -> np.ndarray:
    """The sites whose node marginals are fractional, grown by ``rings`` lattice neighbours.

    A marginal is fractional more than :data:`INTEGRALITY` from 0 and 1. The
    result is a boolean mask over the sites.
    """
    fractional = np.minimum(marginals, 1.0 - marginals) > INTEGRALITY
    free = np.any(fractional, axis=1)
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    for _ in range(rings):
        grown = free.copy()
        grown[second[free[first]]] = True
        grown[first[free[second]]] = True
        free = grown
    return free


def mip(
    graph: PottsGraph,
    field: SiteField | np.ndarray,
    *,
    fixed: np.ndarray | None = None,
    free: np.ndarray | None = None,
    time_limit: float = 600.0,
    timeout: float | None = None,
) -> Mip:
    """``min_x E(x)`` over the sites in ``free``, every other site held at ``fixed``.

    With ``free`` ``None`` every site is free and the MIP is the whole
    problem, whose proven optimum is the minimum energy. Otherwise the fixed
    sites leave the MIP: an edge from a free site ``i`` to a fixed site at
    label ``b`` adds ``-J`` to ``i``'s energy at ``b``, and the fixed sites'
    own terms are a constant added back to the value and the bound. That
    restricted minimum is an upper bound on the global one, and equals it
    when the fixed labels are those of some optimum. An empty free region
    returns the fixed labelling and its energy without running HiGHS.
    ``time_limit`` is
    HiGHS's, in seconds; ``timeout`` bounds the subprocess and defaults to
    the limit plus ten minutes for the assembly.
    """
    unary = -site_field(
        np.asarray(log_weight_of(field), dtype=np.float64), graph.n_nodes
    )
    edges = graph.edge_index
    coupling = np.asarray(graph.edge_coupling, dtype=np.float64)
    labelling = np.zeros(graph.n_nodes, dtype=np.int64)
    if free is None:
        free = np.ones(graph.n_nodes, dtype=bool)
    else:
        if fixed is None:
            message = "a free region needs the fixed labels of the other sites"
            raise ValueError(message)
        labelling[:] = fixed
    first, second = edges[:, 0], edges[:, 1]
    inside = free[first] & free[second]
    constant = float(unary[~free, labelling[~free]].sum())
    both_fixed = ~free[first] & ~free[second]
    constant -= float(
        coupling[both_fixed][
            labelling[first[both_fixed]] == labelling[second[both_fixed]]
        ].sum()
    )
    restricted = unary.copy()
    for this, other in ((first, second), (second, first)):
        crossing = free[this] & ~free[other]
        np.add.at(
            restricted,
            (this[crossing], labelling[other[crossing]]),
            -coupling[crossing],
        )
    if not free.any():
        # Nothing to solve: the fixed labelling is the restricted problem's
        # one point, and its energy the constant.
        return Mip(
            labelling=labelling,
            value=constant,
            dual_bound=constant,
            free=free,
            status=OPTIMAL,
            message="no free sites",
            nodes=0,
            seconds=0.0,
            peak_bytes=0,
        )
    index = np.full(graph.n_nodes, -1, dtype=np.int64)
    index[free] = np.arange(int(free.sum()))
    result = run(
        SCRIPT,
        {
            "unary": np.ascontiguousarray(restricted[free]),
            "first": np.ascontiguousarray(index[first[inside]]),
            "second": np.ascontiguousarray(index[second[inside]]),
            "coupling": np.ascontiguousarray(coupling[inside]),
            "time_limit": np.asarray(time_limit, dtype=np.float64),
        },
        timeout=time_limit + 600.0 if timeout is None else timeout,
    )
    outputs = result.outputs
    marginals = outputs["node_marginals"]
    if np.all(np.isfinite(marginals)):
        labelling[free] = np.argmax(marginals, axis=1)
    return Mip(
        labelling=labelling,
        value=float(outputs["value"]) + constant,
        dual_bound=float(outputs["dual_bound"]) + constant,
        free=free,
        status=int(outputs["status"]),
        message=str(outputs["message"]),
        nodes=int(outputs["nodes"]),
        seconds=result.seconds,
        peak_bytes=result.peak_bytes,
    )
