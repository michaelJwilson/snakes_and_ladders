"""PyMaxflow's Boykov--Kolmogorov cut on a network the adapter wrote (issue #973).

Inputs: ``n_nodes``; ``tail``, ``head``, ``capacity``, ``reverse`` for the
edges between non-terminal nodes; ``source`` and ``sink``, each node's
terminal capacity. Outputs: ``value``, the maximum flow; ``sink_side``, each
node's segment; ``build_seconds``, the graph construction. The measured
seconds are ``maxflow()`` alone; the peak resident memory is the build and
the cut together.
"""

from __future__ import annotations

import numpy as np

from sal.validation.protocol import dump, peaked, received, timed


def main() -> None:
    """Build the graph, cut it under the timer, and write the segments back."""
    import maxflow  # the framework, imported only in this interpreter

    inputs, returned = received()
    n_nodes = int(inputs["n_nodes"])

    def build() -> tuple[object, np.ndarray]:
        graph = maxflow.Graph[float](n_nodes, int(inputs["tail"].size))
        nodes = graph.add_nodes(n_nodes)
        graph.add_edges(
            inputs["tail"], inputs["head"], inputs["capacity"], inputs["reverse"]
        )
        graph.add_grid_tedges(nodes, inputs["source"], inputs["sink"])
        return graph, nodes

    def build_and_cut() -> tuple[object, np.ndarray, float, float, float]:
        (graph, nodes), build_seconds = timed(build)
        value, seconds = timed(graph.maxflow)  # type: ignore[attr-defined]
        return graph, nodes, value, seconds, build_seconds

    (graph, nodes, value, seconds, build_seconds), peak_bytes = peaked(build_and_cut)
    segments = graph.get_grid_segments(nodes)  # type: ignore[attr-defined]
    sink_side = np.array(segments, dtype=np.bool_)
    dump(
        returned,
        {
            "value": np.asarray(value, dtype=np.float64),
            "sink_side": sink_side,
            "build_seconds": np.asarray(build_seconds, dtype=np.float64),
        },
        seconds,
        peak_bytes,
    )


if __name__ == "__main__":
    main()
