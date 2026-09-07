"""The two exact Potts evaluators, each pinned against something independent.

`strip_log_partition` is the oracle belief propagation's loopy regime is
measured against, so it cannot itself rest on belief propagation. It is pinned
twice, in opposite directions: against exhaustive enumeration at sizes where
that is affordable, and by reduction --- a strip of width 1 is a chain, and
must reproduce `phylo.opt.potts.log_partition`, a transfer matrix written
before this module existed and sharing no code with it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from phylo.likelihood.potts import (
    ExactPotts,
    enumerate_potts,
    log_weights,
    strip_log_partition,
)
from phylo.opt.potts import log_partition
from phylo.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

# `likelihood/CLAUDE.md`'s float64 bound. Both sides here are float64, so this
# is the applicable one; nothing in this module runs on a device.
RELATIVE_TOLERANCE = 1e-11

FIELD = np.array([0.3, -0.7, 0.15])


def _relative(realized: float, reference: float) -> float:
    return abs(realized - reference) / abs(reference)


@pytest.mark.oracle
@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (2, 4), (5, 2)])
def test_the_transfer_matrix_reproduces_exhaustive_enumeration(
    shape: tuple[int, int],
) -> None:
    # (4, 2) and (2, 4) are the same lattice transposed. Both are checked
    # because the recursion runs along one axis and sums within the other, so
    # a swapped convention would agree with enumeration on squares only.
    coupling = 0.6
    graph = lattice_graph(shape, BoundaryCondition.OPEN, coupling)
    exact = enumerate_potts(graph, FIELD)
    realized = strip_log_partition(shape, BoundaryCondition.OPEN, coupling, FIELD)

    assert _relative(realized, exact.log_partition) < RELATIVE_TOLERANCE


@pytest.mark.oracle
@pytest.mark.parametrize("length", [2, 3, 5, 8])
def test_a_strip_of_width_one_reduces_to_the_chain_transfer_matrix(
    length: int,
) -> None:
    # A second exact reference, not a second sampler: `phylo.opt.potts`
    # transfers a single site at a time in torch, this transfers a column of
    # `M` sites in numpy, and at `M = 1` they must agree to machine precision.
    coupling = 0.6
    realized = strip_log_partition((length, 1), BoundaryCondition.OPEN, coupling, FIELD)
    reference = float(
        log_partition(
            torch.tensor(coupling, dtype=torch.float64),
            torch.tensor(FIELD, dtype=torch.float64),
            length,
        )
    )

    assert _relative(realized, reference) < RELATIVE_TOLERANCE


@pytest.mark.edge_case
def test_a_periodic_lattice_is_refused_rather_than_answered_as_open() -> None:
    # The forward recursion computes an open strip. Returning that number for
    # a periodic lattice would be wrong by a whole ring of bonds and silent.
    with pytest.raises(ValueError, match="trace over the transfer operator"):
        strip_log_partition((3, 3), BoundaryCondition.PERIODIC, 0.6, FIELD)


@pytest.mark.edge_case
def test_a_three_dimensional_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="takes a 2-D shape"):
        strip_log_partition((2, 2, 2), BoundaryCondition.OPEN, 0.6, FIELD)  # type: ignore[arg-type]


@pytest.mark.edge_case
def test_enumeration_refuses_a_size_it_cannot_do() -> None:
    # Refused rather than attempted: 3**20 configurations is an out-of-memory
    # kill inside a test, which reads as broken infrastructure rather than as
    # a caller asking for a size this cannot reach.
    graph = lattice_graph((5, 4), BoundaryCondition.OPEN, 0.6)
    with pytest.raises(
        ValueError, match=r"refusing to enumerate .*3\*\*20 spin configurations"
    ):
        enumerate_potts(graph, FIELD)


@pytest.mark.edge_case
def test_a_configuration_of_the_wrong_width_is_refused() -> None:
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.6)
    with pytest.raises(ValueError, match="columns for a graph of 4 nodes"):
        log_weights(graph, FIELD, np.zeros((3, 5), dtype=np.int64))


@pytest.mark.oracle
def test_log_weights_matches_the_hamiltonian_evaluated_by_hand() -> None:
    # Two sites, one bond. Agreeing costs `h[a] + h[a] + J`; disagreeing costs
    # `h[a] + h[b]`. Written out rather than derived from the same loop the
    # implementation uses.
    graph = PottsGraph(n_nodes=2, edges=((0, 1),), coupling=(0.6,))
    configurations = np.array([[0, 0], [0, 1], [2, 2], [1, 2]], dtype=np.int64)

    realized = log_weights(graph, FIELD, configurations)

    expected = np.array(
        [
            FIELD[0] + FIELD[0] + 0.6,
            FIELD[0] + FIELD[1],
            FIELD[2] + FIELD[2] + 0.6,
            FIELD[1] + FIELD[2],
        ]
    )
    np.testing.assert_allclose(realized, expected, rtol=RELATIVE_TOLERANCE)


def _enumerated() -> tuple[PottsGraph, ExactPotts]:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.6)
    return graph, enumerate_potts(graph, FIELD)


@pytest.mark.mathematical
def test_enumerated_marginals_are_distributions() -> None:
    _, exact = _enumerated()

    np.testing.assert_allclose(exact.single_site.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(exact.pairwise.sum(axis=(1, 2)), 1.0, atol=1e-12)


@pytest.mark.mathematical
def test_the_pairwise_marginal_reduces_to_the_single_site_one() -> None:
    # The consistency a joint distribution owes its own marginals. It holds
    # for enumeration by construction and *not* for belief propagation on a
    # loop, which is why the same invariant is asserted there only on a tree.
    graph, exact = _enumerated()

    for position, (first, second) in enumerate(graph.edges):
        np.testing.assert_allclose(
            exact.pairwise[position].sum(axis=1), exact.single_site[first], atol=1e-12
        )
        np.testing.assert_allclose(
            exact.pairwise[position].sum(axis=0), exact.single_site[second], atol=1e-12
        )


@pytest.mark.edge_case
@pytest.mark.oracle
def test_a_zero_coupling_lattice_factorizes_into_independent_sites() -> None:
    # With no bonds the model is `n_nodes` independent draws from softmax(h),
    # so `log Z` is `n_nodes` copies of one normalizer -- an analytic result,
    # not another computation.
    shape = (3, 3)
    graph = lattice_graph(shape, BoundaryCondition.OPEN, 0.0)
    exact = enumerate_potts(graph, FIELD)

    single = float(np.log(np.exp(FIELD).sum()))
    assert _relative(exact.log_partition, graph.n_nodes * single) < RELATIVE_TOLERANCE
    independent = np.tile(np.exp(FIELD) / np.exp(FIELD).sum(), (graph.n_nodes, 1))
    np.testing.assert_allclose(exact.single_site, independent, atol=1e-12)
