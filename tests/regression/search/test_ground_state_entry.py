"""Every ground-state method on a ``(graph, field)`` problem, with no fixture rung (issue #933, R10).

`search.ground_state.ground_state` runs a `METHODS` entry on any Potts
problem. Referees: on a fixture rung's own graph and field every entry
returns what the rung-keyed method returns, bitwise under one seed; and at
two states the cut-based entries reach the exact ground state the graph cut
gives.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.search.ground_state import (
    METHODS,
    ground_state,
    lattice_rung,
)
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.sim.potts import energy


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("method", sorted(METHODS))
def test_the_entry_is_the_rung_keyed_method(method: str) -> None:
    rung = lattice_rung(8, 3, seed=933)
    budget = Budget(Cost.SITE_VISITS, 20 * rung.visits_per_sweep)
    keyed = METHODS[method](rung, budget, np.random.default_rng(1))
    entry = ground_state(
        rung.graph, rung.field, method, budget, np.random.default_rng(1)
    )
    np.testing.assert_array_equal(entry.labelling, keyed.labelling)
    assert entry.energy == keyed.energy
    assert entry.spent == keyed.spent


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("method", ["alpha-expansion", "alpha-beta-swap"])
def test_two_states_reach_the_graph_cut(method: str) -> None:
    rung = lattice_rung(10, 2, seed=5)
    exact = ising_ground_state(rung.graph, rung.field)
    budget = Budget(Cost.SITE_VISITS, 50 * rung.visits_per_sweep)
    run = ground_state(rung.graph, rung.field, method, budget, np.random.default_rng(0))
    assert abs(run.energy - exact.energy) <= 1e-9
    assert exact.energy == energy(rung.graph, rung.field, exact.configuration)


@pytest.mark.smoke
def test_an_unknown_method_a_wrong_field_or_unit_is_refused() -> None:
    rung = lattice_rung(4, 2, seed=1)
    budget = Budget(Cost.SITE_VISITS, 100)
    with pytest.raises(ValueError, match="no ground-state method"):
        ground_state(
            rung.graph, rung.field, "gradient", budget, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="one row per node"):
        ground_state(
            rung.graph, rung.field[:3], "icm", budget, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="site visits"):
        ground_state(
            rung.graph,
            rung.field,
            "icm",
            Budget(Cost.SWEEPS, 5),
            np.random.default_rng(0),
        )
