"""gco's alpha expansion on a Potts model the adapter wrote (issue #974).

Inputs: ``unary``, the ``(n_nodes, n_states)`` data cost; ``first``,
``second`` and ``weight`` for the edges, ``first < second``; optionally
``start``, the initial labelling (gco's own default is label 0 everywhere).
The smooth cost is Potts, ``[a != b]``. gco's float mode scales each term to
an integer and truncates it; the adapter re-scores the labelling in the
package's own energy. Outputs: ``labels``; ``gco_energy``, gco's own figure;
``build_seconds``. The measured seconds are ``expansion()`` alone; the peak
resident memory is the build and the expansion together. With ``move`` set
to ``swap`` (issue #997) the move is gco's alpha-beta ``swap()`` to
convergence instead, measured the same way.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked, timed


def main() -> None:
    """Build the graph, expand under the timer, and write the labels back."""
    import gco  # the framework, imported only in this interpreter

    given, returned = paths()
    inputs = load(given)
    unary = inputs["unary"]
    n_nodes, n_states = unary.shape

    def build() -> object:
        graph = gco.GCO()
        graph.create_general_graph(n_nodes, n_states, energy_is_float=True)
        graph.set_data_cost(unary)
        graph.set_all_neighbors(inputs["first"], inputs["second"], inputs["weight"])
        graph.set_smooth_cost(1.0 - np.eye(n_states))
        if "start" in inputs:
            for site, label in enumerate(inputs["start"].tolist()):
                graph.init_label_at_site(site, int(label))
        return graph

    def build_and_expand() -> tuple[object, float, float]:
        graph, build_seconds = timed(build)
        move = str(inputs.get("move", np.asarray("expansion")))
        run = graph.swap if move == "swap" else graph.expansion  # type: ignore[attr-defined]
        _, seconds = timed(lambda: run(-1))
        return graph, seconds, build_seconds

    (graph, seconds, build_seconds), peak_bytes = peaked(build_and_expand)
    labels = np.array(graph.get_labels(), dtype=np.int64)  # type: ignore[attr-defined]
    gco_energy = float(graph.compute_energy())  # type: ignore[attr-defined]
    graph.destroy_graph()  # type: ignore[attr-defined]
    dump(
        returned,
        {
            "labels": labels,
            "gco_energy": np.asarray(gco_energy, dtype=np.float64),
            "build_seconds": np.asarray(build_seconds, dtype=np.float64),
        },
        seconds,
        peak_bytes,
    )


if __name__ == "__main__":
    main()
