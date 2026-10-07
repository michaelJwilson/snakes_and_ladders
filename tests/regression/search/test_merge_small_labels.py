"""A labelling floored after any solver: `merge_small_labels` and `merge(min_sites=k)` (issue #1081).

Referees: the merge is ICM's floor from the handed labelling, bitwise; no
class it returns holds fewer than ``min_sites`` sites; the chain stage is
the expansion then the merge run by hand, bitwise. And a stage's own floor
is the one it runs: the chain-level default no longer overrides it. The
deterministic floor, ``FloorPolicy.SMALLEST_FIRST_BEST_FIELD`` (#1324): the
numba kernel equals its Python oracle bitwise on random labellings with
and without forbidden labels, and a hand-checked case fixes its order and
ties.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import ground_state
from sal.backend import Backend
from sal.search.icm import (
    FloorPolicy,
    _floor_smallest_first,
    iterated_conditional_modes,
    merge_small_labels,
)
from sal.search.icm.numba import floor_smallest_first
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energy, forbid

GRAPH = lattice_graph((8, 8), BoundaryCondition.OPEN, 0.3)
FIELD = np.random.default_rng(1081).normal(0.0, 1.0, (GRAPH.n_nodes, 5))
MIN_SITES = 10
BUDGET = Budget(Cost.SITE_VISITS, 400 * GRAPH.n_nodes)
#: What one ICM sweep is charged: a write per site, a read per edge end.
VISITS_PER_SWEEP = GRAPH.n_nodes + 2 * len(GRAPH.edges)


def _expanded() -> np.ndarray:
    return alpha_expansion(GRAPH, FIELD).labelling


@pytest.mark.oracle
def test_the_merge_is_icms_floor_from_the_labelling_and_leaves_no_small_class() -> None:
    start = _expanded()
    assert int(np.bincount(start, minlength=5)[np.bincount(start) > 0].min()) < (
        MIN_SITES
    )

    merged = merge_small_labels(
        GRAPH, FIELD, start, np.random.default_rng(3), min_sites=MIN_SITES
    )
    floored = iterated_conditional_modes(
        GRAPH, FIELD, np.random.default_rng(3), start=start, min_sites=MIN_SITES
    )

    assert np.array_equal(merged.labelling, floored.labelling)
    counts = np.bincount(merged.labelling, minlength=5)
    assert bool(np.all((counts == 0) | (counts >= MIN_SITES)))
    assert merged.energy == energy(GRAPH, FIELD, merged.labelling)


@pytest.mark.oracle
def test_the_merge_stage_is_the_expansion_then_the_merge_run_by_hand() -> None:
    chained = ground_state(
        GRAPH,
        FIELD,
        f"alpha-expansion>merge(min_sites={MIN_SITES})",
        BUDGET,
        np.random.default_rng(5),
    )
    rng = np.random.default_rng(5)
    expansion = ground_state(GRAPH, FIELD, "alpha-expansion", BUDGET, rng)
    remaining = BUDGET.size - expansion.spent
    merged = merge_small_labels(
        GRAPH,
        FIELD,
        expansion.labelling,
        rng,
        min_sites=MIN_SITES,
        max_iterations=remaining // VISITS_PER_SWEEP,
    )

    assert np.array_equal(chained.labelling, merged.labelling)
    assert chained.spent == expansion.spent + merged.sweeps * VISITS_PER_SWEEP


@pytest.mark.smoke
def test_a_stages_own_floor_is_the_floor_it_runs() -> None:
    staged = ground_state(
        GRAPH, FIELD, f"icm(min_sites={MIN_SITES})", BUDGET, np.random.default_rng(2)
    )
    given = ground_state(
        GRAPH, FIELD, "icm", BUDGET, np.random.default_rng(2), min_sites=MIN_SITES
    )

    assert np.array_equal(staged.labelling, given.labelling)


@pytest.mark.smoke
def test_a_merge_without_a_labelling_or_a_floor_is_refused() -> None:
    with pytest.raises(ValueError, match="no first stage"):
        ground_state(
            GRAPH, FIELD, "merge(min_sites=3)", BUDGET, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="needs min_sites"):
        ground_state(
            GRAPH, FIELD, "alpha-expansion>merge", BUDGET, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="at least one site"):
        merge_small_labels(
            GRAPH, FIELD, _expanded(), np.random.default_rng(0), min_sites=0
        )


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(6))
def test_the_smallest_first_kernel_is_its_oracle_bitwise(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_nodes, n_states = 200, 3 + seed
    # Integer fields make ties common, so the lowest-index rule is exercised.
    values = rng.integers(-3, 4, (n_nodes, n_states)).astype(np.float64)
    if seed % 2:
        allowed = rng.random(values.shape) > 0.3
        allowed[np.arange(n_nodes), rng.integers(0, n_states, n_nodes)] = True
        values = forbid(values, allowed)
    # Skewed label sizes, so several states sit below the floor at once.
    weights = rng.dirichlet(np.full(n_states, 0.5))
    labels = rng.choice(n_states, n_nodes, p=weights).astype(np.int64)
    if seed % 2:
        labels = np.array([rng.choice(np.flatnonzero(row)) for row in allowed])
    min_sites = 15 + 5 * seed
    by_kernel, by_oracle = labels.copy(), labels.copy()

    moved = floor_smallest_first(by_kernel, values, min_sites)
    assert moved == _floor_smallest_first(by_oracle, values, min_sites)
    np.testing.assert_array_equal(by_kernel, by_oracle)
    assert np.isfinite(values[np.arange(n_nodes), by_kernel]).all()


@pytest.mark.oracle
def test_the_smallest_first_floor_on_a_hand_checked_case() -> None:
    # Counts 4, 1, 2 at min_sites 3: state 1 (one site) goes first, to the
    # better of states 0 and 2 by field; then state 2, now 2 or 3 sites.
    values = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # state 1's site: best at 2, so 2 reaches 3
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 0, 0, 1, 2, 2], dtype=np.int64)
    assert floor_smallest_first(labels, values, 3) == 1
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 2, 2, 2])
    # A tie between 0 and 2 goes to 0, the lower index.
    tied = np.array([0, 0, 0, 0, 1, 2, 2, 2], dtype=np.int64)
    flat = np.zeros((8, 3))
    assert floor_smallest_first(tied, flat, 3) == 1
    assert tied[4] == 0


@pytest.mark.oracle
def test_the_smallest_first_merge_leaves_no_small_class_on_both_backends() -> None:
    start = _expanded()
    runs = [
        merge_small_labels(
            GRAPH,
            FIELD,
            start,
            np.random.default_rng(5),
            min_sites=MIN_SITES,
            backend=backend,
            policy=FloorPolicy.SMALLEST_FIRST_BEST_FIELD,
        )
        for backend in (Backend.NUMBA, Backend.PYTHON)
    ]
    np.testing.assert_array_equal(runs[0].labelling, runs[1].labelling)
    counts = np.bincount(runs[0].labelling)
    assert counts[counts > 0].min() >= MIN_SITES
    default = merge_small_labels(
        GRAPH, FIELD, start, np.random.default_rng(5), min_sites=MIN_SITES
    )
    uniform = merge_small_labels(
        GRAPH,
        FIELD,
        start,
        np.random.default_rng(5),
        min_sites=MIN_SITES,
        policy=FloorPolicy.UNIFORM,
    )
    np.testing.assert_array_equal(default.labelling, uniform.labelling)


@pytest.mark.oracle
def test_the_merge_stage_carries_the_policy() -> None:
    policy = FloorPolicy.SMALLEST_FIRST_BEST_FIELD
    chained = ground_state(
        GRAPH,
        FIELD,
        f"alpha-expansion>merge(min_sites={MIN_SITES},policy={policy})",
        BUDGET,
        np.random.default_rng(5),
    )
    rng = np.random.default_rng(5)
    expansion = ground_state(GRAPH, FIELD, "alpha-expansion", BUDGET, rng)
    merged = merge_small_labels(
        GRAPH,
        FIELD,
        expansion.labelling,
        rng,
        min_sites=MIN_SITES,
        max_iterations=(BUDGET.size - expansion.spent) // VISITS_PER_SWEEP,
        policy=policy,
    )
    np.testing.assert_array_equal(chained.labelling, merged.labelling)
    assert chained.energy == merged.energy
    with pytest.raises(ValueError, match="only 'merge' floors by a policy"):
        ground_state(
            GRAPH, FIELD, "icm(policy=uniform)", BUDGET, np.random.default_rng(2)
        )
