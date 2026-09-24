"""One Wolff move and a field whose sign is declared (issue #921).

`SiteField` carries the log-weight every sampler and solver reads, built from
a log-weight or from an energy, which it negates. Referees: the negation is
bitwise; a solver handed the same model as a bare array, as a log-weight and
as the negated energy returns one labelling; and the coupled model's own
Wolff move, deleted by this issue and kept here as the referee, is
`wolff_sweep` on `SiteField.from_energy`, bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sample.accept import accept
from snakes_and_ladders.sample.potts_mcmc import (
    PottsMove,
    adjacency_lists,
    anneal_potts,
    wolff_sweep,
)
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.icm import iterated_conditional_modes
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import SiteField, log_weight_of, site_field

from tests._rows import every_value


def _deleted_wolff_update(
    labels: np.ndarray,
    graph: PottsGraph,
    field: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> None:
    """`search.spatio_sequential._wolff_update` as it stood before #921: the referee."""
    n_nodes = labels.shape[0]
    offsets, neighbour_index, edge_couplings = graph.compressed_adjacency()
    bounds = offsets.tolist()
    neighbours, couplings = neighbour_index.tolist(), edge_couplings.tolist()
    root = int(rng.integers(n_nodes))
    colour = int(labels[root])
    members = [root]
    inside = np.zeros(n_nodes, dtype=bool)
    inside[root] = True
    frontier = [root]
    while frontier:
        node = frontier.pop()
        for position in range(bounds[node], bounds[node + 1]):
            neighbour, coupling = neighbours[position], couplings[position]
            if inside[neighbour] or labels[neighbour] != colour:
                continue
            if rng.random() < 1.0 - np.exp(-beta * coupling):
                inside[neighbour] = True
                members.append(neighbour)
                frontier.append(neighbour)
    proposed = int(rng.integers(field.shape[1]))
    if proposed == colour:
        return
    cluster = np.array(members, dtype=np.int64)
    difference = beta * float((field[cluster, proposed] - field[cluster, colour]).sum())
    if accept(difference, rng):
        labels[cluster] = proposed


def _model() -> tuple[PottsGraph, np.ndarray]:
    graph = lattice_graph((8, 8), BoundaryCondition.OPEN, 0.9)
    energy = np.random.default_rng(921).normal(0.0, 1.0, (graph.n_nodes, 3))
    return graph, energy


@pytest.mark.critical
@pytest.mark.oracle
def test_the_folded_move_is_the_deleted_one_bitwise() -> None:
    # 500 steps of each from one state and one stream: the same labels after
    # every step.
    def check(beta: float) -> None:
        graph, energy = _model()
        offsets, neighbours, couplings = graph.compressed_adjacency()
        lists = adjacency_lists(offsets, neighbours, couplings)
        declared = SiteField.from_energy(energy)
        start = np.random.default_rng(1).integers(0, 3, graph.n_nodes)
        old, new = start.copy(), start.copy()
        old_rng, new_rng = np.random.default_rng(2), np.random.default_rng(2)
        for _ in range(500):
            _deleted_wolff_update(old, graph, -energy, beta, old_rng)
            wolff_sweep(
                new,
                declared,
                offsets,
                neighbours,
                couplings,
                new_rng,
                beta=beta,
                lists=lists,
            )
            assert np.array_equal(old, new)

    every_value([0.3, 1.0, 3.0], check)


@pytest.mark.smoke
def test_the_declared_field_is_one_model_whichever_way_it_arrives() -> None:
    graph, energy = _model()
    assert np.array_equal(SiteField.from_energy(energy).log_weight, -energy)
    shared = np.array([0.2, -0.1, 0.4])
    assert np.array_equal(
        SiteField.widened(shared, graph.n_nodes).log_weight,
        site_field(shared, graph.n_nodes),
    )
    bare = -energy
    forms = (bare, SiteField.from_log_weight(bare), SiteField.from_energy(energy))
    assert all(np.array_equal(log_weight_of(f), bare) for f in forms)
    expansions = [alpha_expansion(graph, f, 3).labelling for f in forms]
    modes = [
        iterated_conditional_modes(graph, f, 3, np.random.default_rng(0)).labelling
        for f in forms
    ]
    schedule = ExponentialTempSchedule(start=2.0, end=0.5, n_steps=20)
    annealed = [
        anneal_potts(
            graph, f, schedule, np.random.default_rng(0), move=PottsMove.WOLFF
        ).labelling
        for f in forms
    ]
    for runs in (expansions, modes, annealed):
        assert all(np.array_equal(runs[0], other) for other in runs[1:])
    with pytest.raises(ValueError, match="n_nodes, n_states"):
        SiteField(shared)
