"""PyMaxflow as an oracle and a timing reference for the minimum cut (issue #973).

PyMaxflow wraps Kolmogorov's own Boykov--Kolmogorov implementation, which
shares no code with :mod:`snakes_and_ladders.search.maxflow` (Dinic, the
Python oracle) or ``src/maxflow.rs`` (the package's Boykov--Kolmogorov). It
is GPL-3.0 and runs only in ``scripts/pymaxflow.py``, in a subprocess.

:func:`min_cut` takes any :class:`~snakes_and_ladders.search.maxflow.FlowNetwork`
with its two terminals, folds each terminal arc into a terminal capacity,
since PyMaxflow holds the terminals implicitly, and returns the flow value
and the source side over the network's own node numbering.
:func:`ising_ground_state` is the two-state ferromagnet the package reduces
to a cut, built with the capacities
:func:`snakes_and_ladders.search.maxflow.ising_ground_state` builds.

**Ties.** On termination Boykov--Kolmogorov leaves a node in neither search
tree when it can neither be reached from the source nor reach the sink in
the residual graph, and PyMaxflow reports such a node on the source side.
The package reads the source side as what the source reaches, so the two
agree node for node exactly when no node is free; the value agrees always.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.search.maxflow import FlowNetwork, GroundState
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import energy, site_field
from snakes_and_ladders.validation.runner import run

#: The script this adapter runs.
SCRIPT = "pymaxflow"


@dataclass(frozen=True)
class Cut:
    """PyMaxflow's answer on one network, with what its script measured."""

    #: The maximum flow, which is the minimum cut's capacity.
    value: float
    #: Per node of the network, terminals included: on the source side.
    source_side: np.ndarray
    #: Wall seconds of ``maxflow()`` alone.
    seconds: float
    #: Wall seconds building the graph in PyMaxflow.
    build_seconds: float
    #: Peak resident bytes the build and the cut added.
    peak_bytes: int


def _cut(
    n_nodes: int,
    tail: np.ndarray,
    head: np.ndarray,
    capacity: np.ndarray,
    reverse: np.ndarray,
    source: np.ndarray,
    sink: np.ndarray,
) -> tuple[float, np.ndarray, float, float, int]:
    """Run the script on non-terminal nodes ``0 .. n_nodes - 1``."""
    result = run(
        SCRIPT,
        {
            "n_nodes": np.asarray(n_nodes, dtype=np.int64),
            "tail": np.ascontiguousarray(tail, dtype=np.int64),
            "head": np.ascontiguousarray(head, dtype=np.int64),
            "capacity": np.ascontiguousarray(capacity, dtype=np.float64),
            "reverse": np.ascontiguousarray(reverse, dtype=np.float64),
            "source": np.ascontiguousarray(source, dtype=np.float64),
            "sink": np.ascontiguousarray(sink, dtype=np.float64),
        },
    )
    return (
        float(result.outputs["value"]),
        result.outputs["sink_side"].astype(bool),
        result.seconds,
        float(result.outputs["build_seconds"]),
        int(result.peak_bytes or 0),
    )


def min_cut(network: FlowNetwork, source: int, sink: int) -> Cut:
    """PyMaxflow's maximum flow on ``network`` between ``source`` and ``sink``.

    The network is read through
    :meth:`~snakes_and_ladders.search.maxflow.FlowNetwork.as_arrays` and not
    mutated. An arc leaving the source, or entering the sink, becomes a
    terminal capacity; an arc into the source or out of the sink carries no
    flow in a maximum flow and is dropped; an arc from source to sink adds its
    capacity to the value directly.
    """
    if source == sink:
        msg = f"source and sink must differ, both are {source}"
        raise ValueError(msg)
    paired = network.as_arrays()
    ends = paired.arcs.reshape(-1, 2)
    tails, heads = ends[:, 0], ends[:, 1]
    forward, backward = paired.capacity, paired.reverse
    inner = np.setdiff1d(np.arange(network.n_nodes), [source, sink])
    position = np.full(network.n_nodes, -1, dtype=np.int64)
    position[inner] = np.arange(inner.size)

    to_source = np.zeros(inner.size)
    to_sink = np.zeros(inner.size)
    direct = 0.0
    # Each edge carries `forward` from tail to head and `backward` back.
    for first, second, capacity in (
        (tails, heads, forward),
        (heads, tails, backward),
    ):
        leaves = (first == source) & (second != sink) & (second != source)
        np.add.at(to_source, position[second[leaves]], capacity[leaves])
        enters = (second == sink) & (first != source) & (first != sink)
        np.add.at(to_sink, position[first[enters]], capacity[enters])
        direct += float(capacity[(first == source) & (second == sink)].sum())
    internal = (position[tails] >= 0) & (position[heads] >= 0)
    value, sink_side, seconds, build_seconds, peak_bytes = _cut(
        int(inner.size),
        position[tails[internal]],
        position[heads[internal]],
        forward[internal],
        backward[internal],
        to_source,
        to_sink,
    )
    side = np.zeros(network.n_nodes, dtype=bool)
    side[inner] = ~sink_side
    side[source] = True
    return Cut(value + direct, side, seconds, build_seconds, peak_bytes)


def ising_ground_state(
    graph: PottsGraph, field_values: np.ndarray
) -> tuple[GroundState, Cut]:
    """The two-state ferromagnetic ground state, cut by PyMaxflow.

    The capacities are :func:`snakes_and_ladders.search.maxflow.ising_ground_state`'s:
    ``source -> i`` costs ``D_i(1)`` and ``i -> sink`` costs ``D_i(0)`` above
    the per-node minimum, and each coupling is a capacity both ways. A node
    on the sink side takes state 1. The cut's :attr:`Cut.source_side` spans
    the lattice's nodes alone.
    """
    values = site_field(
        np.asarray(field_values, dtype=float), graph.n_nodes, n_states=2
    )
    cost = -values
    offsets = cost.min(axis=1)
    edges = graph.edge_index
    value, sink_side, seconds, build_seconds, peak_bytes = _cut(
        graph.n_nodes,
        edges[:, 0],
        edges[:, 1],
        graph.edge_coupling,
        graph.edge_coupling,
        cost[:, 1] - offsets,
        cost[:, 0] - offsets,
    )
    configuration = sink_side.astype(np.int64)
    state = GroundState(configuration, energy(graph, values, configuration))
    return state, Cut(value, ~sink_side, seconds, build_seconds, peak_bytes)
