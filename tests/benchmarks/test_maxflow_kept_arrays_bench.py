"""What keeping the contiguous form is worth (issue #642).

Two pairs. `as_arrays` alone is the ratio the change is about; the
alpha-expansion cell is the one that decides whether it matters, since that is
where `from_arcs` is called 76,928 times on a 32x32 lattice at four labels
(#598) and the Rust backend is the consumer `as_arrays` exists for. The Python
backend never calls it, so it is the control: a ratio there is noise.

Correctness is pinned in tests/regression/search/test_maxflow_kept_arrays.py.
"""

from __future__ import annotations

import math

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.search.alpha_expansion import expand
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.maxflow import FlowNetwork
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

_EDGES = 20000
_NODES = 4096
_SIDE = 16
_LABELS = 4


def _network() -> FlowNetwork:
    rng = np.random.default_rng(0)
    return FlowNetwork.from_arcs(
        _NODES,
        rng.integers(0, _NODES, _EDGES),
        rng.integers(0, _NODES, _EDGES),
        rng.random(_EDGES),
        rng.random(_EDGES),
    )


def test_as_arrays_kept_benchmark(benchmark: BenchmarkFixture) -> None:
    """The form `from_arcs` built, handed back."""
    network = _network()

    result = benchmark(network.as_arrays)

    # Benchmarks assert finiteness only; correctness is pinned in the regression suite.
    assert result[1].size == _EDGES


def test_as_arrays_derived_benchmark(benchmark: BenchmarkFixture) -> None:
    """The same value rebuilt from the lists: the before number."""
    network = _network()
    network._forget_arrays()

    result = benchmark(network.as_arrays)

    assert result[1].size == _EDGES


def _expansion(backend: Backend) -> float:
    graph = lattice_graph((_SIDE, _SIDE), BoundaryCondition.OPEN, 0.5)
    rng = np.random.default_rng(1)
    n_nodes = _SIDE * _SIDE
    field = rng.normal(size=(n_nodes, _LABELS))
    labelling = rng.integers(0, _LABELS, size=n_nodes)
    total = 0.0
    for alpha in range(_LABELS):
        _, value = expand(graph, field, labelling, alpha, backend=backend)
        total += value
    return total


def test_alpha_expansion_rust_benchmark(benchmark: BenchmarkFixture) -> None:
    """The consumer the kept form is for: one sweep of every label."""
    result = benchmark(_expansion, Backend.RUST)

    assert math.isfinite(result)


def test_alpha_expansion_python_benchmark(benchmark: BenchmarkFixture) -> None:
    """The control: this path never calls `as_arrays`."""
    result = benchmark(_expansion, Backend.PYTHON)

    assert math.isfinite(result)
