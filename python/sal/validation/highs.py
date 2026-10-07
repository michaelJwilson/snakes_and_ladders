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

from sal.external.potts_inputs import INTEGRALITY, OPTIMAL, polytope_inputs
from sal.external.potts_inputs import integral as is_integral
from sal.external.runner import run
from sal.sim.graph import PottsGraph
from sal.sim.potts import SiteField, log_weight_of, site_field

#: The script this adapter runs.
SCRIPT = "highs"

__all__ = ["INTEGRALITY", "OPTIMAL", "SCRIPT", "LocalPolytope", "local_polytope"]


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
        return is_integral(self.node_marginals)

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
    reads it. ``timeout`` bounds the subprocess, in seconds. The inputs are
    :func:`~sal.external.potts_inputs.polytope_inputs`, which
    :func:`sal.external.potts.lower_bound` sends too (#1282, step 4); the ILP
    (#1274) is ``lower_bound``'s ``integral``, not this adapter's.
    """
    values = site_field(
        np.asarray(log_weight_of(field), dtype=np.float64), graph.n_nodes
    )
    result = run(
        SCRIPT,
        polytope_inputs(graph, values, time_limit=0.9 * timeout),
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
