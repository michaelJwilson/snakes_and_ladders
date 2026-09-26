"""A labelling floored after any solver: `merge_small_labels` and `merge(min_sites=k)` (issue #1081).

Referees: the merge is ICM's floor from the handed labelling, bitwise; no
class it returns holds fewer than ``min_sites`` sites; the chain stage is
the expansion then the merge run by hand, bitwise. And a stage's own floor
is the one it runs: the chain-level default no longer overrides it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import ground_state
from sal.search.icm import iterated_conditional_modes, merge_small_labels
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energy

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
