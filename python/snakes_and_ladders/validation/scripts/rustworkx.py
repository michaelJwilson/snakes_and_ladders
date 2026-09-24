"""rustworkx's graph constructions and queries on graphs the adapter wrote (issue #976).

``mode`` selects one query; every graph arrives as ``n_nodes`` and its edge
ends ``first`` and ``second`` (with ``weight`` where the query reads one):

- ``"grid"``: ``generators.grid_graph(*shape)``; outputs ``first``, ``second``.
- ``"path"``: ``generators.path_graph(length)``; the same outputs.
- ``"gnp"``: ``undirected_gnp_random_graph(n_nodes, probability, seed)`` for
  each of ``seeds``; outputs ``counts``, and ``first``, ``second`` for the
  first seed.
- ``"multigraph"``: a ``PyGraph(multigraph=True)`` from the edges, read back:
  ``first``, ``second``, ``weight`` in edge-index order and ``degree``.
- ``"components"``: ``connected_components`` of the unweighted graph; outputs
  ``labels``, the smallest node of each node's component. The measured
  seconds are the call alone and ``build_seconds`` the graph's construction;
  ``peak_bytes`` is the build and the call together.
- ``"isomorphic"``: ``is_isomorphic`` on node-labelled graphs, ``node_label``
  per node (``-1`` matches only ``-1``) and ``node_offset`` / ``edge_offset``
  splitting the concatenated graphs, for each row of ``pairs``; outputs
  ``isomorphic``, and with ``match_labels`` false the matcher is omitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked, timed


def _graph(rustworkx: Any, n_nodes: int, first: np.ndarray, second: np.ndarray) -> Any:
    graph = rustworkx.PyGraph(multigraph=True)
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from_no_data(
        list(zip(first.tolist(), second.tolist(), strict=True))
    )
    return graph


def main() -> None:
    """Run the requested query and write its answer back."""
    import rustworkx  # the framework, imported only in this interpreter

    given, returned = paths()
    inputs = load(given)
    mode = str(inputs["mode"])
    outputs: dict[str, np.ndarray] = {}
    seconds = 0.0

    def ends(graph: Any) -> None:
        pairs = np.asarray(graph.edge_list(), dtype=np.int64).reshape(-1, 2)
        outputs["first"], outputs["second"] = pairs[:, 0], pairs[:, 1]

    if mode == "grid":
        ends(rustworkx.generators.grid_graph(*inputs["shape"].tolist()))
    elif mode == "path":
        ends(rustworkx.generators.path_graph(int(inputs["length"])))
    elif mode == "gnp":
        n_nodes, probability = int(inputs["n_nodes"]), float(inputs["probability"])
        graphs = [
            rustworkx.undirected_gnp_random_graph(n_nodes, probability, seed=int(s))
            for s in inputs["seeds"]
        ]
        outputs["counts"] = np.array([g.num_edges() for g in graphs], dtype=np.int64)
        ends(graphs[0])
    elif mode == "multigraph":
        graph = rustworkx.PyGraph(multigraph=True)
        graph.add_nodes_from(range(int(inputs["n_nodes"])))
        graph.add_edges_from(
            list(
                zip(
                    inputs["first"].tolist(),
                    inputs["second"].tolist(),
                    inputs["weight"].tolist(),
                    strict=True,
                )
            )
        )
        weighted = graph.weighted_edge_list()
        outputs["first"] = np.array([e[0] for e in weighted], dtype=np.int64)
        outputs["second"] = np.array([e[1] for e in weighted], dtype=np.int64)
        outputs["weight"] = np.array([e[2] for e in weighted], dtype=np.float64)
        outputs["degree"] = np.array(
            [graph.degree(n) for n in graph.node_indices()], dtype=np.int64
        )
    elif mode == "components":
        n_nodes = int(inputs["n_nodes"])

        def build_and_label() -> tuple[Any, float, float]:
            graph, build_seconds = timed(
                lambda: _graph(rustworkx, n_nodes, inputs["first"], inputs["second"])
            )
            found, seconds = timed(lambda: rustworkx.connected_components(graph))
            return found, seconds, build_seconds

        (components, seconds, build_seconds), peak_bytes = peaked(build_and_label)
        labels = np.empty(n_nodes, dtype=np.int64)
        for component in components:
            members = np.fromiter(component, dtype=np.int64)
            labels[members] = members.min()
        outputs["labels"] = labels
        outputs["build_seconds"] = np.asarray(build_seconds)
        outputs["peak_bytes"] = np.asarray(peak_bytes)
    elif mode == "isomorphic":
        node_offset, edge_offset = inputs["node_offset"], inputs["edge_offset"]
        labels, first, second = inputs["node_label"], inputs["first"], inputs["second"]
        graphs = []
        for k in range(node_offset.size - 1):
            graph = rustworkx.PyGraph()
            graph.add_nodes_from(labels[node_offset[k] : node_offset[k + 1]].tolist())
            span = slice(edge_offset[k], edge_offset[k + 1])
            graph.add_edges_from_no_data(
                list(zip(first[span].tolist(), second[span].tolist(), strict=True))
            )
            graphs.append(graph)
        match = bool(inputs["match_labels"])
        outputs["isomorphic"] = np.array(
            [
                rustworkx.is_isomorphic(
                    graphs[a], graphs[b], (lambda x, y: x == y) if match else None
                )
                for a, b in inputs["pairs"].tolist()
            ],
            dtype=np.bool_,
        )
    else:
        message = f"unknown mode {mode!r}"
        raise SystemExit(message)
    dump(returned, outputs, seconds)


if __name__ == "__main__":
    main()
