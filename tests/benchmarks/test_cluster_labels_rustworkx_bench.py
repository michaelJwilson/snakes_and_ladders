"""The Swendsen--Wang pass's union-find beside rustworkx's components, per bond mask (issue #976).

The partition is pinned in `tests/validation/test_rustworkx.py`, the runtime
goals in `test_goals.py`. Each row times the package's union-find labelling
of one bond mask, read in a fresh interpreter by `scripts/package.py`, and
records beside it rustworkx's graph build and `connected_components` on the
same bonds, the medians of `REPEATS` subprocess runs, and both sides' peak
resident memory (#987). The mask is one Swendsen--Wang bond draw at beta = 1
from a random 3-state labelling of an open lattice at `critical_coupling(3)`.

Sizes: 16² is the gate size, 71², 142² and 284² the stress sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.sample.potts_mcmc import _bond_probability
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling
from snakes_and_ladders.validation import rustworkx
from snakes_and_ladders.validation.runner import available, package

pytestmark = pytest.mark.skipif(
    not available("rustworkx"), reason="rustworkx is the validation-rustworkx extra"
)

#: Subprocess runs whose median each figure is.
REPEATS = 3


def bond_mask(side: int) -> tuple[int, np.ndarray]:
    """One bond draw's active edges on the ``side``-square lattice, seed 976."""
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, critical_coupling(3))
    rng = np.random.default_rng(976)
    state = rng.integers(0, 3, graph.n_nodes)
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < _bond_probability(graph, 1.0))
    return graph.n_nodes, graph.edge_index[active]


@pytest.mark.parametrize("side", [16, 71, 142, 284])
def test_union_find_beside_rustworkx_benchmark(
    benchmark: BenchmarkFixture, side: int
) -> None:
    n_nodes, bonds = bond_mask(side)
    theirs = [rustworkx.components(n_nodes, bonds) for _ in range(REPEATS)]
    benchmark.extra_info["rustworkx_s"] = float(
        np.median([one.seconds + one.build_seconds for one in theirs])
    )
    benchmark.extra_info["rustworkx_peak_bytes"] = float(
        np.median([one.peak_bytes for one in theirs])
    )
    inputs = {
        "n_nodes": np.asarray(n_nodes),
        "first": np.ascontiguousarray(bonds[:, 0]),
        "second": np.ascontiguousarray(bonds[:, 1]),
    }
    runs = benchmark.pedantic(  # type: ignore[no-untyped-call]
        lambda: [package("cluster_labels", inputs) for _ in range(REPEATS)],
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["package_s"] = float(np.median([run.seconds for run in runs]))
    benchmark.extra_info["package_peak_bytes"] = float(
        np.median([run.peak_bytes or 0 for run in runs])
    )

    assert runs[0].outputs["roots"].shape == (n_nodes,)
