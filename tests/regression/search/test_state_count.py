"""The state count is the field's, never a second argument that can disagree (issue #1091).

Each Potts solver reads ``n_states`` from the field's state axis; a caller
may still state it, and a stated count that differs is refused. Inferred and
stated runs are bitwise one run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sal.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from sal.search.bifurcation import simulated_bifurcation
from sal.search.icm import iterated_conditional_modes
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import states_of

GRAPH = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.7)
FIELD = np.random.default_rng(1091).normal(size=(9, 4))


@pytest.mark.analytic
def test_every_solver_reads_the_count_from_the_field_bitwise() -> None:
    pairs: list[tuple[Any, Any]] = [
        (
            iterated_conditional_modes(GRAPH, FIELD, np.random.default_rng(0)),
            iterated_conditional_modes(
                GRAPH, FIELD, np.random.default_rng(0), n_states=4
            ),
        ),
        (alpha_expansion(GRAPH, FIELD), alpha_expansion(GRAPH, FIELD, n_states=4)),
        (alpha_beta_swap(GRAPH, FIELD), alpha_beta_swap(GRAPH, FIELD, n_states=4)),
    ]
    inferred_sb = simulated_bifurcation(
        GRAPH, FIELD, np.random.default_rng(0), steps=20
    )
    stated_sb = simulated_bifurcation(
        GRAPH, FIELD, np.random.default_rng(0), steps=20, n_states=4
    )
    for inferred, stated in pairs:
        assert np.array_equal(inferred.labelling, stated.labelling)
        assert inferred.energy == stated.energy
    assert np.array_equal(inferred_sb.labelling, stated_sb.labelling)
    assert inferred_sb.energy == stated_sb.energy


@pytest.mark.smoke
def test_a_stated_count_the_field_contradicts_is_refused() -> None:
    assert states_of(FIELD, 9) == 4
    assert states_of(FIELD[0], 9) == 4
    for call in (
        lambda: iterated_conditional_modes(
            GRAPH, FIELD, np.random.default_rng(0), n_states=3
        ),
        lambda: alpha_expansion(GRAPH, FIELD, n_states=3),
        lambda: alpha_beta_swap(GRAPH, FIELD, n_states=5),
        lambda: simulated_bifurcation(
            GRAPH, FIELD, np.random.default_rng(0), n_states=3
        ),
    ):
        with pytest.raises(ValueError, match="must have . columns, got 4"):
            call()
