"""A Potts ground state as a MIP over the local polytope, whole or on a free region (issue #1069).

Declined, and conserved: it answered the question #1069 posed and was not
adopted, since the whole problem at `spatio_tiling/release` did not leave
HiGHS's root node in 1,800 s. :func:`mip` solves ``min_x E(x)`` in
:func:`sal.sim.potts.energy`'s sign with integer node marginals by
``scipy.optimize.milp`` (HiGHS), in process, on the constraint matrix
`validation.scripts.highs` assembles for the LP. The edge tables stay
continuous: integral node marginals fix every table, so the MIP's optimum is
the minimum energy.

With a free region, every site outside it is held at a given label and leaves
the MIP: an edge from a free site ``i`` to a held site at label ``b`` adds
``-J`` to ``i``'s energy at ``b``, and the held sites' own terms are a
constant added back. That restricted minimum is an upper bound on the global
one, and equals it when the held labels are those of some optimum.
:func:`free_region` grows the LP's fractional sites by lattice rings, the
region #1069 measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, log_weight_of, site_field
from sal.validation.scripts.highs import polytope

#: `milp`'s status for a proven optimum.
OPTIMAL = 0

#: `milp`'s status when the time limit stops it before the gap closes.
TIME_LIMIT = 1

#: Distance from 0 or 1 within which a node marginal reads as integral:
#: HiGHS's default primal feasibility tolerance, as `validation.highs` reads it.
INTEGRALITY = 1e-7


@dataclass(frozen=True)
class Mip:
    """HiGHS's MIP over the local polytope, with every site outside the free region held."""

    #: A labelling of every site: the MIP's on the free region, the held
    #: labels elsewhere.
    labelling: np.ndarray
    #: The labelling's energy as the MIP counts it, held terms included.
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
) -> Mip:
    """``min_x E(x)`` over the sites in ``free``, every other site held at ``fixed``.

    With ``free`` ``None`` every site is free and the MIP is the whole
    problem, whose proven optimum is the minimum energy. An empty free region
    returns the held labelling and its energy without running HiGHS.
    ``time_limit`` is HiGHS's, in seconds.

    Raises
    ------
    ValueError
        If a free region is given without the held labels.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp  # HiGHS

    unary = -site_field(
        np.asarray(log_weight_of(field), dtype=np.float64), graph.n_nodes
    )
    coupling = np.asarray(graph.edge_coupling, dtype=np.float64)
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    labelling = np.zeros(graph.n_nodes, dtype=np.int64)
    if free is None:
        free = np.ones(graph.n_nodes, dtype=bool)
    else:
        if fixed is None:
            message = "a free region needs the held labels of the other sites"
            raise ValueError(message)
        labelling[:] = fixed
    held = ~free
    both_held = held[first] & held[second]
    constant = float(unary[held, labelling[held]].sum()) - float(
        coupling[both_held][
            labelling[first[both_held]] == labelling[second[both_held]]
        ].sum()
    )
    if not free.any():
        return Mip(
            labelling, constant, constant, free, OPTIMAL, "no free sites", 0, 0.0
        )
    restricted = unary.copy()
    for this, other in ((first, second), (second, first)):
        crossing = free[this] & held[other]
        np.add.at(
            restricted,
            (this[crossing], labelling[other[crossing]]),
            -coupling[crossing],
        )
    inside = free[first] & free[second]
    index = np.full(graph.n_nodes, -1, dtype=np.int64)
    index[free] = np.arange(int(free.sum()))
    sub_unary = np.ascontiguousarray(restricted[free])
    cost, matrix, right = polytope(
        sub_unary, index[first[inside]], index[second[inside]], coupling[inside]
    )
    integrality = np.zeros(cost.shape[0], dtype=np.int64)
    integrality[: sub_unary.size] = 1
    started = time.perf_counter()
    result = milp(
        cost,
        integrality=integrality,
        bounds=Bounds(0.0, np.inf),
        constraints=LinearConstraint(matrix, right, right),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    seconds = time.perf_counter() - started
    if result.x is not None:
        marginals = result.x[: sub_unary.size].reshape(sub_unary.shape)
        labelling[free] = np.argmax(marginals, axis=1)
    value = float(result.fun) if result.x is not None else np.nan
    return Mip(
        labelling=labelling,
        value=value + constant,
        dual_bound=float(getattr(result, "mip_dual_bound", np.nan)) + constant,
        free=free,
        status=int(result.status),
        message=str(result.message),
        nodes=int(getattr(result, "mip_node_count", 0)),
        seconds=seconds,
    )
