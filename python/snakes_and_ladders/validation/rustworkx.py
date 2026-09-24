"""rustworkx as an oracle for graph construction and cluster labelling (issue #976).

rustworkx's generators, its multigraph, its connected components and its VF2
isomorphism share no code with :mod:`snakes_and_ladders.sim.graph`,
:mod:`snakes_and_ladders.sim.topology` or the union-find in
:mod:`snakes_and_ladders.sample.potts_mcmc`. It runs only in
``scripts/rustworkx.py``, in a subprocess; this adapter replaces
``PottsGraph.to_rustworkx`` and ``from_rustworkx``, which put it in the
package process (issue #322).

Every answer comes back as arrays: an edge list as its two end columns, a
partition as each node's smallest fellow member, so a comparison is an array
equality and needs no rustworkx object on this side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.validation.runner import run

#: The script this adapter runs.
SCRIPT = "rustworkx"


@dataclass(frozen=True)
class Multigraph:
    """A multigraph as rustworkx reads it back: edges in its order, and degrees."""

    edges: np.ndarray
    weight: np.ndarray
    degree: np.ndarray


@dataclass(frozen=True)
class Components:
    """Each node's smallest fellow member, and what the call cost."""

    labels: np.ndarray
    #: Wall seconds of ``connected_components`` alone.
    seconds: float
    #: Wall seconds building the ``PyGraph``.
    build_seconds: float
    #: Peak resident bytes the build and the call added.
    peak_bytes: int


def _ends(outputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([outputs["first"], outputs["second"]], axis=1)


def grid_edges(shape: tuple[int, int]) -> np.ndarray:
    """``generators.grid_graph(*shape)``'s edges, one row each."""
    inputs = {"mode": np.asarray("grid"), "shape": np.asarray(shape, dtype=np.int64)}
    return _ends(run(SCRIPT, inputs).outputs)


def path_edges(length: int) -> np.ndarray:
    """``generators.path_graph(length)``'s edges, one row each."""
    inputs = {"mode": np.asarray("path"), "length": np.asarray(length, dtype=np.int64)}
    return _ends(run(SCRIPT, inputs).outputs)


def gnp(
    n_nodes: int, probability: float, seeds: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """``undirected_gnp_random_graph``'s edge count per seed, and the first seed's edges."""
    outputs = run(
        SCRIPT,
        {
            "mode": np.asarray("gnp"),
            "n_nodes": np.asarray(n_nodes, dtype=np.int64),
            "probability": np.asarray(probability, dtype=np.float64),
            "seeds": np.asarray(seeds, dtype=np.int64),
        },
    ).outputs
    return outputs["counts"], _ends(outputs)


def multigraph(n_nodes: int, edges: np.ndarray, weight: np.ndarray) -> Multigraph:
    """A ``PyGraph(multigraph=True)`` built from ``edges`` and read back."""
    ends = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    outputs = run(
        SCRIPT,
        {
            "mode": np.asarray("multigraph"),
            "n_nodes": np.asarray(n_nodes, dtype=np.int64),
            "first": np.ascontiguousarray(ends[:, 0]),
            "second": np.ascontiguousarray(ends[:, 1]),
            "weight": np.ascontiguousarray(weight, dtype=np.float64),
        },
    ).outputs
    return Multigraph(_ends(outputs), outputs["weight"], outputs["degree"])


def components(n_nodes: int, edges: np.ndarray) -> Components:
    """``connected_components`` of the graph on ``edges``, as smallest-member labels."""
    ends = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    result = run(
        SCRIPT,
        {
            "mode": np.asarray("components"),
            "n_nodes": np.asarray(n_nodes, dtype=np.int64),
            "first": np.ascontiguousarray(ends[:, 0]),
            "second": np.ascontiguousarray(ends[:, 1]),
        },
    )
    return Components(
        result.outputs["labels"],
        result.seconds,
        float(result.outputs["build_seconds"]),
        int(result.outputs["peak_bytes"]),
    )


def isomorphic(
    graphs: Sequence[tuple[np.ndarray, np.ndarray]],
    pairs: Sequence[tuple[int, int]],
    *,
    match_labels: bool = True,
) -> np.ndarray:
    """``is_isomorphic`` for each pair of ``graphs``, each ``(node_label, edges)``.

    With ``match_labels`` two nodes may be matched only where their integer
    labels are equal.
    """
    labels = [np.asarray(label, dtype=np.int64) for label, _ in graphs]
    ends = [np.asarray(edge, dtype=np.int64).reshape(-1, 2) for _, edge in graphs]
    joined = np.concatenate(ends) if ends else np.zeros((0, 2), dtype=np.int64)
    outputs = run(
        SCRIPT,
        {
            "mode": np.asarray("isomorphic"),
            "node_label": np.concatenate(labels),
            "node_offset": np.cumsum([0, *(label.size for label in labels)]),
            "edge_offset": np.cumsum([0, *(edge.shape[0] for edge in ends)]),
            "first": np.ascontiguousarray(joined[:, 0]),
            "second": np.ascontiguousarray(joined[:, 1]),
            "pairs": np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
            "match_labels": np.asarray(match_labels),
        },
    ).outputs
    return outputs["isomorphic"]
