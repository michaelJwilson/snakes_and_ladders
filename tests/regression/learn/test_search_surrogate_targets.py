"""The surrogate's three exact targets and its augmentation, against enumeration (issue #729).

Each was unrefereed at #729's measurement and is pinned to an answer computed
without it: :func:`strip_log_partition_target` against
:func:`~sal.likelihood.potts.enumerate_potts` on the 3x3
lattice; :func:`ground_state_target` against the enumerated minimum;
:func:`ground_state_offset` as a bound on it, per instance; and
:func:`shuffle_children` against the maximized log-likelihood it may not move.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest
from sal.backend import Backend
from sal.learn.ranking import (
    ground_state_offset,
    ground_state_target,
    lattice_instances,
    shuffle_children,
    strip_log_partition_target,
)
from sal.likelihood.potts import enumerate_potts, log_weights
from sal.search.infer import score_topology
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.newick import to_newick, validate_unrooted_newick
from sal.sim.potts import SpatioOnlyParams
from sal.sim.simulator import simulate_tree
from sal.sim.topology import leaf_bipartitions, random_topology

from tests._fixtures import SMALL_SITES, load_fixture

#: The declared 3x3 lattice at three classes: 19,683 configurations, so the
#: enumeration is the referee for every lattice target below.
CI: SpatioOnlyParams = fixture("spatio_only", "ci").params

#: The float64 agreement two exact routes to one number are held to. Realized
#: 1.3e-16 for the strip transfer matrix against enumeration and 1.8e-15
#: absolute for alpha expansion against the enumerated minimum.
EXACT_RTOL = 1e-12


def _configurations(n_nodes: int, n_states: int) -> np.ndarray:
    return np.array(
        list(itertools.product(range(n_states), repeat=n_nodes)), dtype=np.int64
    )


def _enumerated_minimum(graph: object, field: np.ndarray, n_states: int) -> float:
    """``min_s E(s)`` over every configuration, as the negated maximum log weight."""
    states = _configurations(int(field.shape[0]), n_states)
    return -float(log_weights(graph, field, states).max())  # type: ignore[arg-type]


@pytest.mark.oracle
def test_the_strip_target_is_the_enumerated_log_partition() -> None:
    # Transfer matrix against 19,683 configurations, sharing no recursion:
    # 13.5439055020 either way, 1.3e-16 relative apart.
    exact = enumerate_potts(CI.graph, CI.field).log_partition

    assert strip_log_partition_target()(CI.graph, CI.field) == pytest.approx(
        exact, rel=EXACT_RTOL
    )


@pytest.mark.oracle
def test_the_strip_target_refuses_the_lattices_it_is_not_exact_on() -> None:
    # The preconditions are part of the claim: the transfer matrix is exact on
    # a 2-D lattice at one uniform coupling and is wrong elsewhere, so a
    # silent answer on a 1-D chain or a mixed coupling would be an unrefereed
    # target rather than a refused one.
    target = strip_log_partition_target()
    chain = lattice_graph((4,), BoundaryCondition.OPEN, 0.5)
    field = np.zeros((chain.n_nodes, 2))

    with pytest.raises(ValueError, match="needs a 2-D lattice"):
        target(chain, field)

    square = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.5)
    mixed = replace(
        square, coupling=tuple(float(1 + i) for i in range(len(square.edges)))
    )
    with pytest.raises(ValueError, match="one uniform coupling"):
        target(mixed, np.zeros((square.n_nodes, 2)))


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
def test_the_ground_state_target_is_the_enumerated_minimum_energy(
    backend: Backend,
) -> None:
    # Expansion is an upper bound, attained here: -7.9364200936 from both cut
    # backends and enumerated, 1.8e-15 apart.
    exact = _enumerated_minimum(CI.graph, CI.field, CI.n_classes)

    reached = ground_state_target(CI.n_classes, backend=backend)(CI.graph, CI.field)

    assert reached == pytest.approx(exact, rel=EXACT_RTOL)


@pytest.mark.oracle
def test_the_ground_state_offset_bounds_the_enumerated_minimum_of_every_instance() -> (
    None
):
    # The sum of each term's optimum, judged against each instance's enumerated
    # minimum: gaps 2.8818, 3.3203, 3.3153, 2.5391, so it holds and is not tight.
    graphs, fields, _ = lattice_instances(CI, 2, 2)

    offsets = ground_state_offset(graphs, fields)

    assert offsets.shape == (len(graphs),)
    gaps = []
    for position, (graph, field) in enumerate(zip(graphs, fields, strict=True)):
        minimum = _enumerated_minimum(graph, field, CI.n_classes)
        bound = float(offsets[position])
        # The stack is per instance and in order: the bound of one instance
        # alone must be the entry the stack carries for it.
        assert bound == float(ground_state_offset([graph], [field])[0])
        assert bound <= minimum
        gaps.append(minimum - bound)
    assert np.allclose(gaps, [2.8818, 3.3203, 3.3153, 2.5391], atol=5e-4), gaps


@pytest.mark.oracle
def test_shuffling_a_node_s_children_moves_no_likelihood() -> None:
    # Refereed by the refitted maximized log-likelihood: over 8 shuffles of the
    # 4-taxon fixture, splits identical, 5 distinct valid spellings, worst
    # relative movement 1.3e-16.
    params = load_fixture(SMALL_SITES)
    alignment = dict(
        simulate_tree(params, np.random.default_rng(params.seed), n_sites=200).alignment
    )
    rng = np.random.default_rng(17)
    topology = random_topology(sorted(alignment), np.random.default_rng(2))
    expected = score_topology(topology, alignment, params.n_states)

    moved = 0.0
    spellings = set()
    for _ in range(8):
        shuffled = shuffle_children(topology, rng)
        spellings.add(to_newick(shuffled))
        assert validate_unrooted_newick(to_newick(shuffled))
        assert leaf_bipartitions(shuffled) == leaf_bipartitions(topology)
        score = score_topology(shuffled, alignment, params.n_states)
        moved = max(moved, abs(score - expected) / abs(expected))
    assert moved < EXACT_RTOL, moved
    # A shuffle that never reordered anything would pass every line above.
    assert len(spellings) > 1, spellings
