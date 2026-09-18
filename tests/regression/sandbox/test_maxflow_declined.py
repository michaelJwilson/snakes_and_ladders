"""The declined max-flow kernels referee the package kernel, and each other.

Issue #715. Four kernels were built behind one seam and one was kept; the
three here are pinned exactly as the kept one is --- the value to the last
bits and the side arc for arc against the Python Dinic, the ground state's
energy and configuration against the package kernel, the enumerated minimum
where it fits --- and as the inner solver of alpha expansion, where every
kernel must give the same labelling after every cut and a bitwise-equal
energy. The parallel kernel is additionally pinned bit for bit across thread
counts, which is the property that makes its result a result.

**It skips unless the extension carries the ``sandbox`` Cargo feature.** The
kernels are not in the default build, so a missing ``max_flow_declined``
skips the module rather than failing it; ``infra/release.sh`` rebuilds with
the feature to see these run.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable

import numpy as np
import pytest
from snakes_and_ladders.sandbox import maxflow_declined
from snakes_and_ladders.sandbox.maxflow_declined import DeclinedKernel
from snakes_and_ladders.search import alpha_expansion as expansion_module
from snakes_and_ladders.search import maxflow_rust
from snakes_and_ladders.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.maxflow import (
    FlowNetwork,
    energy,
    ising_ground_state,
    max_flow,
)
from snakes_and_ladders.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    lattice_graph,
    triangular_lattice_graph,
)
from snakes_and_ladders.sim.potts import spatio_only_field

pytestmark = pytest.mark.skipif(
    not maxflow_declined.AVAILABLE,
    reason="the extension was built without the sandbox feature",
)

# The Python blocking flow recurses to the depth of the level graph.
sys.setrecursionlimit(50_000)

KERNELS = list(DeclinedKernel)


def _seeded_network(n_nodes: int, seed: int) -> FlowNetwork:
    """The seeded network with back capacities the package kernel is pinned on."""
    rng = np.random.default_rng(seed)
    tails = rng.integers(0, n_nodes, size=4 * n_nodes)
    heads = rng.integers(0, n_nodes, size=4 * n_nodes)
    forward = rng.uniform(0.1, 3.0, size=4 * n_nodes)
    reverse = rng.uniform(0.0, 1.0, size=4 * n_nodes)
    network = FlowNetwork(n_nodes=n_nodes)
    for tail, head, capacity, back in zip(tails, heads, forward, reverse, strict=True):
        if tail != head:
            network.add_edge(int(tail), int(head), float(capacity), float(back))
    return network


def _spots(extent: int, n_states: int) -> tuple[PottsGraph, np.ndarray]:
    """`spatio_only`'s construction past enumeration: triangular, lognormal sizes."""
    graph = triangular_lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.7)
    rng = np.random.default_rng(extent * 10 + n_states)
    sizes = np.exp(rng.normal(0.0, 0.6, size=graph.n_nodes))
    return graph, spatio_only_field(np.linspace(-0.9, 0.9, n_states), sizes)


@pytest.mark.oracle
@pytest.mark.parametrize("kernel", KERNELS, ids=str)
@pytest.mark.parametrize("n_nodes", [6, 12, 24, 96])
@pytest.mark.parametrize("seed", range(715, 725))
def test_every_declined_kernel_returns_the_python_cut_on_seeded_networks(
    kernel: DeclinedKernel, n_nodes: int, seed: int
) -> None:
    # The value to the last bits and the side arc for arc: the side is the
    # minimal minimum cut every maximum flow shares, so it is compared
    # element for element and not only by its capacity.
    expected = max_flow(_seeded_network(n_nodes, seed), 0, n_nodes - 1)
    realized = maxflow_declined.min_cut(
        _seeded_network(n_nodes, seed), 0, n_nodes - 1, kernel
    )

    assert realized.value == pytest.approx(expected.value, rel=1e-12)
    assert realized.source_side.tolist() == expected.source_side.tolist()


@pytest.mark.oracle
@pytest.mark.parametrize("kernel", KERNELS, ids=str)
@pytest.mark.parametrize("extent", [4, 8, 16])
def test_every_declined_kernel_reproduces_the_package_ground_state(
    kernel: DeclinedKernel, extent: int
) -> None:
    rng = np.random.default_rng(extent)
    graph = lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.6)
    field_values = rng.normal(size=(graph.n_nodes, 2))

    expected_state, expected = maxflow_rust.ising_ground_state(graph, field_values)
    python_state, python_energy = ising_ground_state(graph, field_values)
    realized_state, realized = maxflow_declined.ising_ground_state(
        graph, field_values, kernel
    )

    assert realized == expected == python_energy
    assert np.array_equal(realized_state, expected_state)
    assert np.array_equal(realized_state, python_state)


@pytest.mark.oracle
@pytest.mark.parametrize("kernel", KERNELS, ids=str)
def test_every_declined_kernel_finds_the_enumerated_minimum(
    kernel: DeclinedKernel,
) -> None:
    graph = lattice_graph((4, 3), BoundaryCondition.OPEN, 0.4)
    field_values = np.random.default_rng(715).normal(size=(graph.n_nodes, 2))
    configurations = np.array(
        list(itertools.product(range(2), repeat=graph.n_nodes)), dtype=np.int64
    )
    exact = float(energy(graph, field_values, configurations).min())

    _, realized = maxflow_declined.ising_ground_state(graph, field_values, kernel)

    assert realized == pytest.approx(exact, abs=1e-12)


@pytest.mark.edge_case
@pytest.mark.parametrize("kernel", KERNELS, ids=str)
def test_every_declined_kernel_terminates_when_nothing_reaches_the_sink(
    kernel: DeclinedKernel,
) -> None:
    # The first swap network of `_spots(12, 3)` has 79 arcs out of the source,
    # none into the sink, and a maximum flow of 0: the parallel kernel's
    # global relabel kept every source-side label at `n`, so no node reached
    # `n + 1` and pushed its excess back, and the round never ended. The
    # relabel is now two-sided, as the sequential kernel's, and the value and
    # the side are the package kernel's.
    network = FlowNetwork(n_nodes=4)
    network.add_edge(0, 1, 1.0)
    network.add_edge(1, 2, 0.5)
    network.add_edge(2, 3, 0.0)
    reference = maxflow_rust.min_cut(_copy(network), 0, 3)

    realized = maxflow_declined.min_cut(_copy(network), 0, 3, kernel)

    assert realized.value == 0.0 == reference.value
    assert realized.source_side.tolist() == [True, True, True, False]
    assert np.array_equal(realized.source_side, reference.source_side)


def _copy(network: FlowNetwork) -> FlowNetwork:
    copied = FlowNetwork(n_nodes=network.n_nodes)
    arcs, capacity, _ = network.as_arrays()
    for (tail, head), value in zip(
        arcs.reshape(-1, 2).tolist(), capacity.tolist(), strict=True
    ):
        copied.add_edge(tail, head, value)
    return copied


@pytest.mark.structural
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_the_parallel_kernel_is_bitwise_independent_of_its_thread_count(
    threads: int,
) -> None:
    # A round's pushes are computed in parallel and applied in vertex order,
    # so the floating-point sequence is fixed whatever the pool: the value
    # to the bit and the side element for element.
    n_nodes = 96
    reference = maxflow_declined.min_cut(
        _seeded_network(n_nodes, 715),
        0,
        n_nodes - 1,
        DeclinedKernel.PARALLEL_PUSH_RELABEL,
        threads=1,
    )
    realized = maxflow_declined.min_cut(
        _seeded_network(n_nodes, 715),
        0,
        n_nodes - 1,
        DeclinedKernel.PARALLEL_PUSH_RELABEL,
        threads=threads,
    )

    assert realized.value.hex() == reference.value.hex()
    assert realized.source_side.tolist() == reference.source_side.tolist()


@pytest.fixture
def declined_inner_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[DeclinedKernel], None]:
    """Route alpha expansion's Rust cut through a declined kernel.

    The package seam carries one kernel by design, so the expansion is run
    on a declined one by replacing the cut it calls rather than by a
    parameter nothing in the package needs.
    """

    def install(kernel: DeclinedKernel) -> None:
        def cut(network: FlowNetwork, source: int, sink: int) -> object:
            return maxflow_declined.min_cut(network, source, sink, kernel)

        monkeypatch.setattr(expansion_module, "min_cut", cut)

    return install


@pytest.mark.oracle
@pytest.mark.parametrize("kernel", KERNELS, ids=str)
@pytest.mark.parametrize("n_states", [3, 10])
def test_every_declined_kernel_gives_the_package_expansion(
    declined_inner_solver: Callable[[DeclinedKernel], None],
    kernel: DeclinedKernel,
    n_states: int,
) -> None:
    # The same cut per expansion, so the same labelling after every one, the
    # same cycle count, and the final energy bitwise equal. A labelling may
    # differ only where a cut is degenerate, and none is on this instance.
    graph, field = _spots(12, n_states)
    reference = alpha_expansion(graph, field, n_states, backend=Backend.RUST)
    reference_swap = alpha_beta_swap(graph, field, n_states, backend=Backend.RUST)

    declined_inner_solver(kernel)
    result = alpha_expansion(graph, field, n_states, backend=Backend.RUST)
    swapped = alpha_beta_swap(graph, field, n_states, backend=Backend.RUST)

    assert result.energy.hex() == reference.energy.hex()
    assert np.array_equal(result.labelling, reference.labelling)
    assert result.cycles == reference.cycles
    assert swapped.energy.hex() == reference_swap.energy.hex()
    assert np.array_equal(swapped.labelling, reference_swap.labelling)


@pytest.mark.edge_case
def test_an_unknown_kernel_is_refused_by_name() -> None:
    from snakes_and_ladders import oxi_snakes_and_ladders

    network = FlowNetwork(n_nodes=2)
    network.add_edge(0, 1, 1.0)
    arcs, capacity, reverse = network.as_arrays()
    with pytest.raises(ValueError, match="unknown max-flow algorithm"):
        oxi_snakes_and_ladders.max_flow_declined(
            2, arcs, capacity, 0, 1, reverse, "ford-fulkerson"
        )
