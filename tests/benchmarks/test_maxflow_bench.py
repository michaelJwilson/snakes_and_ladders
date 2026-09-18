"""Benchmarks for max flow, and the port decision they justify.

Correctness is pinned in `tests/regression/search/`, per the repo's division
of labor between the two directories.

Root `CLAUDE.md` requires a candidate be benchmarked against the reference
before a Rust port is committed to, and this is that measurement kept rather
than discarded: the Python and Rust paths run the same problem at the same
sizes, so the ratio is visible in one table rather than asserted in prose.

The Rust path is timed twice, as `DEV.md` asks of every port: once through
`snakes_and_ladders.search.maxflow_rust`, which is what a caller pays, and once
on the extension with its arrays already built, which is the kernel plus the
borrow at the boundary. Their difference is the wrapper's own work; issue
#336 measured it and found the energy evaluation, not the boundary.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.search import maxflow_rust
from snakes_and_ladders.search.maxflow import ising_ground_state, site_field
from snakes_and_ladders.sim import fixtures
from snakes_and_ladders.sim.graph import PottsGraph, lattice_graph

# The Python blocking flow recurses to the depth of the level graph; the Rust
# one uses an explicit stack. This raise is itself part of what the port buys.
sys.setrecursionlimit(50_000)


#: `potts_lattice/ci`'s boundary and coupling, read rather than restated
#: (issue #622). Its declared extent is 3 and these cells measure 16, 32 and
#: 64, so the extent stays a parameter and only the lattice's definition is
#: read; 0.6 is the value that was written here.
PARAMS = fixtures.fixture("potts_lattice", "ci").params


def _problem(extent: int) -> tuple[PottsGraph, np.ndarray]:
    graph = lattice_graph((extent, extent), PARAMS.boundary, PARAMS.coupling)
    rng = np.random.default_rng(extent)
    return graph, rng.normal(size=(graph.n_nodes, 2))


@pytest.mark.parametrize("extent", [16, 32, 64])
def test_python_ising_ground_state_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    # The oracle, and the baseline the port is measured against. A random
    # per-node field is the case with content: a uniform one makes the
    # ferromagnetic ground state trivially all-aligned.
    graph, field_values = _problem(extent)

    _, energy = benchmark(ising_ground_state, graph, field_values)

    assert np.isfinite(energy)


@pytest.mark.parametrize("extent", [16, 32, 64])
def test_rust_ising_ground_state_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    graph, field_values = _problem(extent)

    _, energy = benchmark(maxflow_rust.ising_ground_state, graph, field_values)

    assert np.isfinite(energy)


@pytest.mark.parametrize("extent", [16, 32, 64])
def test_rust_kernel_ising_ground_state_benchmark(
    benchmark: BenchmarkFixture, extent: int
) -> None:
    # The kernel with its arrays already contiguous: what remains above the
    # `maxflow_rust` timing is the wrapper's own array construction.
    graph, field_values = _problem(extent)
    field = np.ascontiguousarray(site_field(graph, field_values)).reshape(-1)
    edges = np.asarray(graph.edges, dtype=np.int64).reshape(-1)
    coupling = np.asarray(graph.coupling, dtype=np.float64)

    states = benchmark(
        oxi_snakes_and_ladders.ising_ground_state, graph.n_nodes, field, edges, coupling
    )

    assert states.shape == (graph.n_nodes,)


# --- the kernels behind one seam (issue #715) --------------------------------

#: The sizes the kernels are decided at. 128 and 256 are the stress sizes a
#: speedup is claimed at (root `CLAUDE.md`), over the per-pull-request cap, so
#: they run at the release gate; 16 to 64 keep the ratio measured per pull
#: request.
KERNEL_EXTENTS = [
    16,
    32,
    64,
    pytest.param(128, marks=pytest.mark.release),
    pytest.param(256, marks=pytest.mark.release),
]


@pytest.mark.parametrize("algorithm", list(maxflow_rust.MaxFlowAlgorithm), ids=str)
@pytest.mark.parametrize("extent", KERNEL_EXTENTS)
def test_kernel_ising_ground_state_benchmark(
    benchmark: BenchmarkFixture, extent: int, algorithm: maxflow_rust.MaxFlowAlgorithm
) -> None:
    # Every kernel on the same instance, so one table decides which stays
    # in the package: the lowest wall clock at 128 and 256 wins.
    graph, field_values = _problem(extent)

    _, energy = benchmark(
        maxflow_rust.ising_ground_state, graph, field_values, algorithm
    )

    assert np.isfinite(energy)


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("extent", [64, pytest.param(256, marks=pytest.mark.release)])
def test_parallel_kernel_threads_benchmark(
    benchmark: BenchmarkFixture, extent: int, threads: int
) -> None:
    # The one kernel rayon reaches inside a cut, at one, two and four
    # threads on one pool: what a round's barrier costs against what the
    # parallel pushes buy.
    graph, field_values = _problem(extent)

    _, energy = benchmark(
        maxflow_rust.ising_ground_state,
        graph,
        field_values,
        maxflow_rust.MaxFlowAlgorithm.PARALLEL_PUSH_RELABEL,
        threads,
    )

    assert np.isfinite(energy)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("extent", [64, pytest.param(256, marks=pytest.mark.release)])
def test_batch_ground_states_benchmark(
    benchmark: BenchmarkFixture, extent: int, threads: int
) -> None:
    # The batch entry point: eight independent cuts on the pool, the axis
    # rayon takes by default. Four threads over one is the throughput a
    # sweep of instances or starts pays for.
    graph, _ = _problem(extent)
    fields = np.random.default_rng(extent).normal(size=(8, graph.n_nodes, 2))

    states = benchmark(
        maxflow_rust.ising_ground_states,
        graph,
        fields,
        maxflow_rust.DEFAULT_ALGORITHM,
        threads,
    )

    assert states.shape == (8, graph.n_nodes)
