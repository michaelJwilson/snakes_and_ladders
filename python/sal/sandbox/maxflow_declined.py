"""The three max-flow kernels issue #715 declined, behind the ``sandbox`` feature.

Conserved on this package's declined rule. Issue #715 built four kernels
behind :mod:`sal.search.maxflow.rust`'s seam --- Dinic (the
port issue #220 made), highest-label push-relabel, Boykov--Kolmogorov and a
synchronous parallel push-relabel on rayon --- pinned each to the Python
Dinic arc for arc, and timed them on square lattices with a random per-node
field and as the inner solver of alpha expansion. Boykov--Kolmogorov won both
tables and is the package kernel; the other three lost and live here, so the
comparison that declined them stays re-runnable. `STATUS.md` carries the
tables; the module docstring of ``src/maxflow_declined.rs`` carries each
kernel's number beside its algorithm.

**The case each referees.** Every kernel certifies its cut as the source-
reachable set of the residual graph on termination, the minimal minimum cut
every maximum flow shares, so a declined kernel that still agrees with the
package kernel arc for arc is a second referee of it --- one written to a
different algorithm, which is what conserving it buys. The parallel kernel
referees one more thing: its output is bit-identical at one thread and at
four, because a round's pushes are computed in parallel and applied in
vertex order.

**Built only under the ``sandbox`` Cargo feature.** ``maturin develop
--release`` links none of this; ``--features sandbox`` does. Without it the
extension carries no ``max_flow_declined``, :data:`AVAILABLE` is ``False``,
and every function here raises ``ImportError`` before doing any work --- it
never falls back to the package kernel, because a referee that quietly
becomes the thing it referees asserts nothing. The check is here rather than
at import because ``docs/source/index.rst`` lists every module and
``sphinx-build -W`` imports each one.

Only ``tests/`` and :mod:`sal.qa` import from here.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from sal import oxisal
from sal.search.maxflow import FlowNetwork, MinCut
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy, site_field

#: Whether the extension was built with the ``sandbox`` feature.
AVAILABLE = hasattr(oxisal, "max_flow_declined")

_UNAVAILABLE = (
    "the declined max-flow kernels are not in this build: rebuild the extension "
    "with `maturin develop --release --features sandbox` (DEV.md, the sandbox "
    "feature)"
)


class DeclinedKernel(StrEnum):
    """The kernels issue #715 measured and did not keep."""

    DINIC = "dinic"
    """Level graphs and blocking flows: 234 ms on the 256x256 cut."""

    PUSH_RELABEL = "push-relabel"
    """Goldberg--Tarjan, highest label first, global relabel and gap: 1,680 ms."""

    PARALLEL_PUSH_RELABEL = "parallel-push-relabel"
    """Synchronous push-relabel on rayon: 339 ms, and no faster on four threads."""


def _require() -> None:
    if not AVAILABLE:
        raise ImportError(_UNAVAILABLE)


def min_cut(
    network: FlowNetwork,
    source: int,
    sink: int,
    kernel: DeclinedKernel,
    threads: int | None = None,
) -> MinCut:
    """The minimum cut by a declined kernel; the package's `min_cut` otherwise.

    Parameters
    ----------
    network : FlowNetwork
        Not mutated: the residual graph lives in Rust and is dropped with
        the call, as for the package kernel.
    source, sink : int
        Terminals.
    kernel : DeclinedKernel
        Which one.
    threads : int | None
        The rayon pool the parallel kernel runs on, ``None`` for the global
        pool; the sequential kernels ignore it.

    Raises
    ------
    ImportError
        If the extension was built without the ``sandbox`` feature.
    """
    _require()
    arcs, capacity, reverse = network.as_arrays()
    value, side = oxisal.max_flow_declined(
        network.n_nodes, arcs, capacity, source, sink, reverse, str(kernel), threads
    )
    return MinCut(value=float(value), source_side=np.asarray(side, dtype=bool))


def ising_ground_state(
    graph: PottsGraph,
    field: np.ndarray,
    kernel: DeclinedKernel,
    threads: int | None = None,
) -> tuple[np.ndarray, float]:
    """The exact two-state ground state by a declined kernel.

    The same contract as
    :func:`sal.search.maxflow.rust.ising_ground_state`: the
    configuration, and its energy evaluated in Python on the configuration
    rather than read back from the cut.
    """
    _require()
    values = site_field(np.asarray(field, dtype=float), graph.n_nodes, n_states=2)
    states = oxisal.ising_ground_state_declined(
        graph.n_nodes,
        np.ascontiguousarray(values, dtype=np.float64).reshape(-1),
        graph.edge_index.reshape(-1),
        graph.edge_coupling,
        str(kernel),
        threads,
    )
    configuration = np.asarray(states, dtype=np.int64)
    return configuration, float(energy(graph, values, configuration))
