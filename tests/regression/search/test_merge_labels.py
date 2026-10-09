"""The closed-form label merge: `merge_labels` and the `merge-labels` stage (issue #1142).

Referees: every entry of the predicted ``(q, q)`` change against the energy
recomputed by :func:`sal.sim.potts.energy` after the merge; each merge the
run makes against the energy recomputed after it, never positive; and the
result against an exhaustive check that no pair of its labels lowers the
energy. The couplings are unequal per edge, so a formula that read one
coupling for all, or halved the boundary, would fail.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.search.ground_state import ground_state
from sal.search.icm import _merge_rounds, merge_deltas, merge_labels
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import energy

N_STATES = 6
LATTICE = lattice_graph((8, 8), BoundaryCondition.OPEN, 1.0)
#: The lattice's edges, each with its own coupling, from a declared seed.
GRAPH = PottsGraph(
    LATTICE.n_nodes,
    LATTICE.edges,
    tuple(np.random.default_rng(1142).uniform(0.2, 1.4, len(LATTICE.edges))),
)
FIELD = np.random.default_rng(1143).normal(0.0, 1.0, (GRAPH.n_nodes, N_STATES))
START = np.random.default_rng(1144).integers(0, N_STATES, GRAPH.n_nodes)
#: Agreement between a predicted change and the recomputed one.
TOLERANCE = 1e-12
BUDGET = Budget(Cost.SITE_VISITS, 400 * GRAPH.n_nodes)
VISITS_PER_SWEEP = GRAPH.n_nodes + 2 * len(GRAPH.edges)


def _merged(labelling: np.ndarray, u: int, v: int) -> np.ndarray:
    out = labelling.copy()
    out[out == u] = v
    return out


def _tables(labelling: np.ndarray) -> np.ndarray:
    """The start's predicted changes, from the tables `merge_labels` builds."""
    q = N_STATES
    summed = np.zeros((q, q))
    np.add.at(summed, labelling, FIELD)
    boundary = np.zeros((q, q))
    first, second = GRAPH.edge_index[:, 0], GRAPH.edge_index[:, 1]
    np.add.at(boundary, (labelling[first], labelling[second]), GRAPH.edge_coupling)
    alive = np.bincount(labelling, minlength=q) > 0
    return merge_deltas(summed, boundary, alive)


@pytest.mark.oracle
def test_every_predicted_change_is_the_recomputed_energy_change() -> None:
    # All 30 ordered pairs of live labels at the start, to 1e-12.
    before = energy(GRAPH, FIELD, START)
    delta = _tables(START)

    for u in range(N_STATES):
        for v in range(N_STATES):
            if u == v:
                continue
            after = energy(GRAPH, FIELD, _merged(START, u, v))
            assert abs(delta[u, v] - (after - before)) < TOLERANCE, (u, v)


@pytest.mark.oracle
def test_each_merge_lowers_the_energy_by_its_prediction() -> None:
    # Each merge the run makes, its tables updated in place rather than
    # rebuilt: the predicted change against the energy recomputed after it.
    labelling, rounds = START.copy(), 0
    for u, v, change, after in _merge_rounds(GRAPH, FIELD, START):
        assert np.array_equal(after, _merged(labelling, u, v))
        recomputed = energy(GRAPH, FIELD, after) - energy(GRAPH, FIELD, labelling)
        assert change < 0.0
        assert abs(change - recomputed) < TOLERANCE, (u, v)
        labelling, rounds = after, rounds + 1

    assert rounds >= 2


@pytest.mark.oracle
def test_the_merge_ends_where_no_pair_lowers_the_energy() -> None:
    merged = merge_labels(GRAPH, FIELD, START)
    labels = np.unique(merged.labelling)
    final = energy(GRAPH, FIELD, merged.labelling)

    assert merged.termination.converged
    assert 1 <= merged.termination.iterations <= N_STATES - 1
    assert merged.termination.iterations == N_STATES - labels.size
    assert merged.energy == final
    assert final < energy(GRAPH, FIELD, START)
    for u in labels:
        for v in labels:
            if u != v:
                lowered = energy(GRAPH, FIELD, _merged(merged.labelling, u, v))
                assert lowered >= final - TOLERANCE, (u, v)


@pytest.mark.oracle
def test_a_shared_field_is_its_rows_repeated() -> None:
    shared = FIELD[0]
    one = merge_labels(GRAPH, shared, START)
    two = merge_labels(GRAPH, np.tile(shared, (GRAPH.n_nodes, 1)), START)

    assert np.array_equal(one.labelling, two.labelling)
    assert one.energy == two.energy


@pytest.mark.oracle
def test_the_stage_is_the_solver_then_the_merge_run_by_hand() -> None:
    # The anneal holds one sweep back for the merge, which lowers its energy.
    # The seed is one whose anneal the merge lowers, so the equality is not
    # vacuous: 3 and 4 of seeds 0..15 on the Python and Rust loops. Seed 5,
    # -114.415 -> -115.648 on the Python loop, ends at -118.304 on the Rust
    # loop, the default since #1370, where no merge lowers it; seed 2 on the
    # Rust loop is -117.032 -> -118.443.
    chained = ground_state(
        GRAPH,
        FIELD,
        "anneal(reserve_sweeps=1)>merge-labels",
        BUDGET,
        np.random.default_rng(2),
    )
    rng = np.random.default_rng(2)
    held = Budget(BUDGET.unit, BUDGET.size - VISITS_PER_SWEEP)
    annealed = ground_state(GRAPH, FIELD, "anneal", held, rng)
    merged = merge_labels(GRAPH, FIELD, annealed.labelling)

    assert np.array_equal(chained.labelling, merged.labelling)
    assert chained.energy == merged.energy
    assert chained.spent == annealed.spent + VISITS_PER_SWEEP <= BUDGET.size
    assert merged.energy < annealed.energy


@pytest.mark.smoke
def test_a_merge_without_a_labelling_or_its_sweep_is_refused() -> None:
    with pytest.raises(ValueError, match="no first stage"):
        ground_state(
            GRAPH, FIELD, "merge-labels>descent", BUDGET, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="reserve_sweeps=1"):
        ground_state(
            GRAPH, FIELD, "anneal>merge-labels", BUDGET, np.random.default_rng(0)
        )
