"""The Rust single-site sweep against the Python oracle, at the sizes that
decide whether the port is kept.

Correctness is pinned in `tests/regression/search/test_potts_mcmc_rust.py`,
distributionally against exhaustive enumeration; this file only measures.

The cells rise in node count because that is the axis the cost is linear in:
issue #232 measured 100 sweeps taking 1.05 s at 1024 nodes against 0.064 s at
64, with the constant set by interpreter overhead rather than arithmetic. A
single cell would report a ratio; three report whether it holds as the problem
grows, which is what the pruning port turned out not to do.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.search.potts_keyed import SwendsenWangMove
from snakes_and_ladders.search.potts_mcmc import (
    _GUARD,
    PottsChain,
    PottsMove,
    parallel_tempering,
    sample_potts,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling, site_field


def _rust_sample_potts(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
) -> PottsChain:
    # The name issue #246 published: the single-site move on the extension.
    return sample_potts(
        graph,
        field,
        PottsMove.SINGLE_SITE,
        rng,
        n_sweeps,
        burn_in,
        thin,
        backend=Backend.RUST,
    )


SWEEPS = 100
FIELD = np.zeros(2)

#: Periodic, so every node has the same degree and the cost is linear in nodes
#: with no boundary term. 32x32 is the size #232 profiled.
EXTENTS = [8, 32]


@pytest.mark.parametrize("extent", EXTENTS, ids=lambda e: f"{e}x{e}")
def test_python_single_site_sweep(benchmark: BenchmarkFixture, extent: int) -> None:
    """The oracle, and the baseline the ratio is against."""
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 0.4)

    chain = benchmark(
        sample_potts,
        graph,
        FIELD,
        PottsMove.SINGLE_SITE,
        np.random.default_rng(1),
        SWEEPS,
        backend=Backend.PYTHON,
    )

    assert chain.states.shape == (SWEEPS, graph.n_nodes)


@pytest.mark.parametrize("extent", EXTENTS, ids=lambda e: f"{e}x{e}")
def test_rust_single_site_sweep(benchmark: BenchmarkFixture, extent: int) -> None:
    """The port, on the same problem.

    Shape and finiteness only. The two backends draw different chains by
    construction -- `f64::exp` and NumPy's differ by a unit in the last place,
    and `searchsorted` is a threshold -- so an equality assertion here would be
    asserting something false. What they must agree on is the distribution,
    which the regression suite checks against enumeration.
    """
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 0.4)

    chain = benchmark(
        _rust_sample_potts,
        graph,
        FIELD,
        np.random.default_rng(1),
        SWEEPS,
    )

    assert chain.states.shape == (SWEEPS, graph.n_nodes)
    assert math.isfinite(float(chain.states.sum()))


LADDER = (2.0, 1.2, 0.7, 0.4)


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
@pytest.mark.parametrize("extent", EXTENTS, ids=lambda e: f"{e}x{e}")
def test_parallel_tempering_benchmark(
    benchmark: BenchmarkFixture, extent: int, backend: Backend
) -> None:
    """Four replicas of 20 sweeps each, on the oracle sweep and on the Rust one.

    The sweep is the whole of a replica's step and the exchange is O(1), so
    this is the #246 ratio applied to tempering (#264): 5.8x at 100 nodes and
    7.8x at 32x32 when it landed, the remainder being the per-sweep energy
    evaluation the exchange needs.
    """
    graph = lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 0.4)

    run = benchmark(
        parallel_tempering,
        graph,
        FIELD,
        LADDER,
        np.random.default_rng(1),
        20,
        backend=backend,
    )

    assert run.states.shape == (20, len(LADDER), graph.n_nodes)


# --- the GIL, released (#604) ------------------------------------------------

THREADS = [1, 2, 4]
THREADED_SWEEPS = 50


def _sweep_task(seed: int) -> np.ndarray:
    """One independent chain, start to finish, inside the kernel."""
    graph = lattice_graph((32, 32), BoundaryCondition.PERIODIC, 0.4)
    rows = site_field(FIELD, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rng = np.random.default_rng(seed)
    state = np.ascontiguousarray(rng.integers(0, 2, size=graph.n_nodes), dtype=np.int64)
    draws = np.ascontiguousarray(
        rng.random(THREADED_SWEEPS * graph.n_nodes), dtype=np.float64
    )
    oxi_snakes_and_ladders.single_site_sweeps(
        state,
        rows,
        offsets,
        neighbours,
        couplings,
        draws,
        THREADED_SWEEPS,
        1.0,
        _GUARD,
        0,
    )
    return state


@pytest.mark.parametrize("n_threads", THREADS, ids=lambda n: f"{n}thread")
def test_rust_sweep_under_python_threads(
    benchmark: BenchmarkFixture, n_threads: int
) -> None:
    """``n_threads`` chains at once, which the kernel held the GIL through.

    The ratio between the cells is the finding and is reported in
    ``STATUS.md``: before issue #604 four threads took 4.03x the wall of one,
    which is serialization exactly. It is **not** asserted here --- a
    wall-clock threshold fails for the machine rather than for the change
    (`DEV.md`, No CI Profiling). What is asserted is `parallel`'s standing
    rule, which is what makes the timing reportable at all: a threaded run is
    bitwise the serial one.
    """
    serial = [_sweep_task(7 + index) for index in range(n_threads)]

    def run() -> list[np.ndarray]:
        out: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_threads

        def body(index: int) -> None:
            out[index] = _sweep_task(7 + index)

        threads = [
            threading.Thread(target=body, args=(index,)) for index in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return out

    threaded = benchmark(run)

    for expected, actual in zip(serial, threaded, strict=True):
        np.testing.assert_array_equal(expected, actual)


#: The instance the stress ranking put the cluster pass at: a 64x64 open
#: lattice at three states and the exact Potts transition, where the bond
#: pass makes a cluster for every 1.7 sites (`docs/experiments/025`). The
#: fixture store's `potts_lattice/stress.yaml` is 12x12 because it declares
#: an *autocorrelation* measurement over 1,500 sweeps; a per-pass ratio is
#: read where the pass does its work, which is here (issue #754).
CLUSTER_EXTENT = 64
CLUSTER_STATES = 3


def _cluster_instance() -> tuple[PottsGraph, np.ndarray, np.ndarray]:
    graph = lattice_graph(
        (CLUSTER_EXTENT, CLUSTER_EXTENT),
        BoundaryCondition.OPEN,
        critical_coupling(CLUSTER_STATES),
    )
    field = np.random.default_rng(1).normal(size=(graph.n_nodes, CLUSTER_STATES))
    state = np.ascontiguousarray(
        np.random.default_rng(4).integers(0, CLUSTER_STATES, size=graph.n_nodes),
        dtype=np.int64,
    )
    return graph, field, state


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
def test_swendsen_wang_propose(benchmark: BenchmarkFixture, backend: Backend) -> None:
    """The ranking's Potts workload: one cluster pass through the keyed move.

    The enclosing call the port is charged against. Shape and finiteness only:
    the two routes draw the same uniforms in a different order, so they return
    chains of one law and not one chain, and the equality belongs in
    `tests/regression/search/test_potts_mcmc_cluster_rust.py`, which pins the
    pass against the oracle on the same draws.
    """
    graph, field, state = _cluster_instance()
    move = SwendsenWangMove(graph, field, backend)

    labels, charge = benchmark(
        move.propose,
        state,
        temperature=1.0,
        site=-1,
        label=-1,
        rng=np.random.default_rng(7),
    )

    assert labels.shape == (graph.n_nodes,)
    assert charge == graph.n_nodes + 2 * len(graph.edges)


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
def test_swendsen_wang_chain(benchmark: BenchmarkFixture, backend: Backend) -> None:
    """The same pass under `sample_potts`, ten sweeps of it."""
    graph, field, _ = _cluster_instance()

    chain = benchmark(
        sample_potts,
        graph,
        field,
        PottsMove.SWENDSEN_WANG,
        np.random.default_rng(7),
        10,
        cluster_backend=backend,
    )

    assert chain.states.shape == (10, graph.n_nodes)
