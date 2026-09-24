"""The package's minimum cut beside PyMaxflow's, on one lattice per size (issue #973).

Correctness is pinned in `tests/validation/test_pymaxflow.py`. Each row times
the package's Rust kernel with its arrays prebuilt --- the graph construction
and the cut, in one call --- and records beside it what PyMaxflow's script
measured on the same capacities: `pymaxflow_build_s` for its graph and
`pymaxflow_cut_s` for `maxflow()`, each the median of `REPEATS` subprocess
runs. The fair pair is the kernel against their sum; the cut alone is the
bound a faster build could reach. Beside them, `pymaxflow_peak_bytes` and
`package_peak_bytes`: the peak resident memory each side's build and cut add,
each read in a fresh interpreter (#987). The Python oracle is timed at the smallest
stress size only, where it is still the baseline a port is judged by.

Sizes: 16² is the gate size, and 71², 142² and 284² the stress sizes a
speedup is read at (root `CLAUDE.md`).
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders import oxisal
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search.ground_state import lattice_rung
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.sim.potts import site_field
from snakes_and_ladders.validation import pymaxflow
from snakes_and_ladders.validation.runner import available, package

pytestmark = pytest.mark.skipif(
    not available("maxflow"), reason="PyMaxflow is the validation-pymaxflow extra"
)

#: Subprocess runs per size whose median PyMaxflow's figures are.
REPEATS = 5

SIDES = [16, 71, 142, 284]


def _pymaxflow_measures(side: int) -> tuple[float, float, float]:
    rung = lattice_rung(side, 2, seed=973)
    cuts = [
        pymaxflow.ising_ground_state(rung.graph, rung.field)[1] for _ in range(REPEATS)
    ]
    return (
        float(np.median([cut.build_seconds for cut in cuts])),
        float(np.median([cut.seconds for cut in cuts])),
        float(np.median([cut.peak_bytes for cut in cuts])),
    )


@pytest.mark.parametrize("side", SIDES)
def test_rust_cut_beside_pymaxflow_benchmark(
    benchmark: BenchmarkFixture, side: int
) -> None:
    rung = lattice_rung(side, 2, seed=973)
    field = np.ascontiguousarray(
        site_field(rung.field, rung.graph.n_nodes, n_states=2), dtype=np.float64
    ).reshape(-1)
    edges = rung.graph.edge_index.reshape(-1)
    coupling = rung.graph.edge_coupling
    build, cut, peak = _pymaxflow_measures(side)
    benchmark.extra_info["pymaxflow_build_s"] = build
    benchmark.extra_info["pymaxflow_cut_s"] = cut
    benchmark.extra_info["pymaxflow_peak_bytes"] = peak
    inputs = {
        "n_nodes": np.asarray(rung.graph.n_nodes),
        "field": field,
        "edges": edges,
        "coupling": coupling,
    }
    benchmark.extra_info["package_peak_bytes"] = float(
        np.median(
            [package("ising_cut", inputs).peak_bytes or 0 for _ in range(REPEATS)]
        )
    )

    states = benchmark(
        oxisal.ising_ground_state,
        rung.graph.n_nodes,
        field,
        edges,
        coupling,
    )

    assert np.asarray(states).shape == (rung.graph.n_nodes,)


def test_python_cut_at_the_smallest_stress_size_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    rung = lattice_rung(71, 2, seed=973)

    state = benchmark(
        ising_ground_state, rung.graph, rung.field, backend=Backend.PYTHON
    )

    assert np.isfinite(state.energy)
