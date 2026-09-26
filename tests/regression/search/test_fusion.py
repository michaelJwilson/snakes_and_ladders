"""The fusion move against enumeration of its binary problem (issue #1070).

`search.alpha_expansion.fuse` chooses per site between two proposals by the
roof dual. It is judged by enumerating all 2^n choices on small lattices and
graphs, with positive and signed couplings, against three claims:

- never worse than either proposal;
- the exact binary optimum wherever it leaves no site open;
- the energy it reports is `sim.potts.energy`'s.

An expansion move is the fusion with a constant labelling, so `expand` is a
second, independent implementation of that case.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal.backend import Backend
from sal.search.alpha_expansion import expand, fuse
from sal.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    erdos_renyi_graph,
    lattice_graph,
)
from sal.sim.potts import energy

#: Fusions drawn per graph, each with its own field and proposals.
DRAWS = 40
#: States per site.
N_STATES = 4


def _signed(graph: PottsGraph, rng: np.random.Generator) -> PottsGraph:
    """The same edges with Gaussian couplings of either sign."""
    return PottsGraph(
        graph.n_nodes,
        graph.edges,
        tuple(float(value) for value in rng.normal(size=len(graph.edges))),
    )


def _graphs() -> dict[str, PottsGraph]:
    rng = np.random.default_rng(1070)
    square = lattice_graph((3, 4), BoundaryCondition.OPEN, 1.0)
    return {
        "square": square,
        "signed": _signed(square, rng),
        "random": erdos_renyi_graph(10, 0.4, 0.8, np.random.default_rng(7)),
    }


GRAPHS = _graphs()


def _binary_optimum(
    graph: PottsGraph, field: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    """The lowest energy over every per-site choice between the two proposals."""
    return min(
        energy(graph, field, np.where(np.array(choice, dtype=bool), second, first))
        for choice in itertools.product((False, True), repeat=graph.n_nodes)
    )


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON], ids=str)
@pytest.mark.parametrize("name", list(GRAPHS))
def test_the_fusion_is_never_worse_and_exact_where_nothing_is_open(
    name: str, backend: Backend
) -> None:
    graph = GRAPHS[name]
    rng = np.random.default_rng([1070, len(name)])
    exact = 0
    for _ in range(DRAWS):
        field = rng.normal(size=(graph.n_nodes, N_STATES))
        first = rng.integers(0, N_STATES, graph.n_nodes)
        second = rng.integers(0, N_STATES, graph.n_nodes)

        fused = fuse(graph, field, first, second, backend=backend)
        optimum = _binary_optimum(graph, field, first, second)

        assert fused.energy == pytest.approx(energy(graph, field, fused.labelling))
        assert (
            fused.energy
            <= min(energy(graph, field, first), energy(graph, field, second)) + 1e-9
        )
        assert fused.energy >= optimum - 1e-9
        if fused.unlabelled == 0:
            assert fused.energy == pytest.approx(optimum, abs=1e-9)
            exact += 1
    # The open case is the exception: most draws are solved outright.
    assert exact >= DRAWS // 2


@pytest.mark.oracle
@pytest.mark.parametrize("alpha", range(N_STATES))
def test_fusing_with_a_constant_is_the_expansion_move(alpha: int) -> None:
    graph = lattice_graph((6, 5), BoundaryCondition.OPEN, 0.7)
    rng = np.random.default_rng([1070, alpha])
    field = rng.normal(size=(graph.n_nodes, N_STATES))
    labelling = rng.integers(0, N_STATES, graph.n_nodes)

    fused = fuse(graph, field, labelling, np.full(graph.n_nodes, alpha))
    moved = expand(graph, field, labelling, alpha)

    # Every pair of an expansion is submodular, so nothing is left open.
    assert fused.unlabelled == 0
    assert fused.energy == pytest.approx(moved.energy, abs=1e-9)


@pytest.mark.oracle
@pytest.mark.backend
def test_both_cuts_fuse_to_one_labelling() -> None:
    graph = lattice_graph((12, 12), BoundaryCondition.OPEN, 0.7)
    rng = np.random.default_rng(3)
    field = rng.normal(size=(graph.n_nodes, N_STATES))
    first = rng.integers(0, N_STATES, graph.n_nodes)
    second = rng.integers(0, N_STATES, graph.n_nodes)

    rust = fuse(graph, field, first, second, backend=Backend.RUST)
    python = fuse(graph, field, first, second, backend=Backend.PYTHON)

    assert np.array_equal(rust.labelling, python.labelling)
    assert rust.unlabelled == python.unlabelled


@pytest.mark.smoke
def test_identical_proposals_fuse_to_themselves() -> None:
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 1.0)
    field = np.random.default_rng(0).normal(size=(graph.n_nodes, 3))
    labelling = np.random.default_rng(1).integers(0, 3, graph.n_nodes)

    fused = fuse(graph, field, labelling, labelling.copy())

    assert np.array_equal(fused.labelling, labelling)
    assert fused.unlabelled == 0
