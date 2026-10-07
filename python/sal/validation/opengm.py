"""OpenGM's bound trace: what :func:`sal.external.potts.lower_bound` does not carry (issue #1279).

OpenGM (Andres, Beier and Kappes, MIT) is built from source by
``infra/build_opengm.sh`` and runs only in ``scripts/opengm.py``, in a
subprocess. Every answer :mod:`sal.external.potts` returns is made through
it; this adapter keeps the one output its result types do not carry, the
bound after each iteration of OpenGM's TRW-S (``TRWSi``, Savchynskyy's
implementation of Kolmogorov's schedule) or of its subgradient dual
decomposition, which a test reads against
:attr:`sal.search.trws.TrwsResult.trace`.
"""

from __future__ import annotations

import numpy as np

from sal.external.potts_inputs import opengm_inputs
from sal.external.runner import run
from sal.sim.graph import PottsGraph
from sal.sim.potts import site_field

#: The script this adapter runs.
SCRIPT = "opengm"


def bound_trace(
    graph: PottsGraph,
    field: np.ndarray,
    algorithm: str,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """OpenGM's lower bound after each iteration of ``algorithm``, ``"trws"`` or ``"dd"``.

    ``field`` is a finite log-weight, ``(n_states,)`` or
    ``(n_nodes, n_states)``; the bytes are
    :func:`~sal.external.potts_inputs.opengm_inputs`', as
    :func:`sal.external.potts.lower_bound` sends them.
    """
    values = site_field(np.asarray(field, dtype=np.float64), graph.n_nodes)
    inputs = opengm_inputs(
        graph,
        values,
        algorithm,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    trace: np.ndarray = run(SCRIPT, inputs).outputs["trace"]
    return trace
