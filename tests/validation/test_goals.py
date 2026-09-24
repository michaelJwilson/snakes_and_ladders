"""The runtimes the package has to meet, each set by an external framework (issue #972).

Each :class:`~tests.validation._goals.Goal` is a framework's runtime on a
declared fixture, measured on the 4-core reference host by the benchmark pair
its row names and written here as a number. The test times the package alone,
so no framework is installed to run it, and it fails until the package's
median meets the figure. It carries `goal` and runs in a non-blocking step of
CI's `validation` job.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.search.ground_state import lattice_rung
from snakes_and_ladders.sim.potts import site_field
from snakes_and_ladders.validation.runner import package

from tests.validation._goals import (
    Goal,
    MemoryGoal,
    assert_fits,
    assert_meets,
    median_seconds,
)

pytestmark = pytest.mark.goal

#: PyMaxflow's graph build and cut on `lattice_rung(side, 2, seed=973)`, the
#: medians of five subprocess runs in `test_maxflow_pymaxflow_bench.py`.
PYMAXFLOW_CUT = {
    side: Goal(
        "pymaxflow",
        f"the Rust cut on lattice_rung({side}, 2, seed=973)",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 2.2, #973",
    )
    for side, seconds in ((142, 15.491e-3), (284, 68.667e-3))
}

#: PyMaxflow's peak added resident memory for the same build and cut, the
#: medians of three subprocess runs (#987).
PYMAXFLOW_CUT_MEMORY = {
    side: MemoryGoal(
        "pymaxflow",
        f"the Rust cut on lattice_rung({side}, 2, seed=973)",
        peak_bytes,
        "2026-09-23, 4-core reference host, #987",
    )
    for side, peak_bytes in ((142, 4_874_240), (284, 20_439_040))
}


def _cut_inputs(side: int) -> dict[str, np.ndarray]:
    """The Rust cut's arrays for `lattice_rung(side, 2, seed=973)`."""
    rung = lattice_rung(side, 2, seed=973)
    field = site_field(rung.field, rung.graph.n_nodes, n_states=2)
    return {
        "n_nodes": np.asarray(rung.graph.n_nodes),
        "field": np.ascontiguousarray(field, dtype=np.float64).reshape(-1),
        "edges": rung.graph.edge_index.reshape(-1),
        "coupling": rung.graph.edge_coupling,
    }


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(PYMAXFLOW_CUT))
def test_the_rust_cut_meets_pymaxflows_runtime(side: int) -> None:
    # The Rust kernel with its arrays prebuilt, which is its graph build and
    # cut in one call, against PyMaxflow's build and cut.
    rung = lattice_rung(side, 2, seed=973)
    field = np.ascontiguousarray(
        site_field(rung.field, rung.graph.n_nodes, n_states=2), dtype=np.float64
    ).reshape(-1)
    edges = rung.graph.edge_index.reshape(-1)
    coupling = rung.graph.edge_coupling
    ours = median_seconds(
        lambda: oxi_snakes_and_ladders.ising_ground_state(
            rung.graph.n_nodes, field, edges, coupling
        )
    )
    assert_meets(ours, PYMAXFLOW_CUT[side])


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(PYMAXFLOW_CUT_MEMORY))
def test_the_rust_cut_fits_pymaxflows_memory(side: int) -> None:
    # Read in a fresh interpreter by `scripts/package.py`, as PyMaxflow's was.
    inputs = _cut_inputs(side)
    peaks = [package("ising_cut", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), PYMAXFLOW_CUT_MEMORY[side])
