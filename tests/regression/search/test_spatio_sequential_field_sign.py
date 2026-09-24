"""The coupled model hands the cluster move its field negated (issue #919).

``H`` is an energy; the cluster moves read a log-weight ``h``, so `label_step`
and the annealed start use `SiteField.from_energy` (#921). At a temperature
where every equal-label bond forms, a Wolff step cannot lower agreement and
accepts no rising field, so the textbook's label objective never increases.
Declared as ``+H`` the field term rises; the objective can still fall via the
coupling, so the control reads the field term.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.likelihood.spatio_sequential import external_field
from snakes_and_ladders.sample.potts_mcmc import wolff_sweep
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.potts import SiteField, energy
from snakes_and_ladders.sim.spatio_sequential import simulate_spatio_sequential

#: Steps per run on the stress fixture (100 sites, 2 classes, 4 positions): the
#: CI fixture has 4 sites, where a random start is already a cold fixed point.
STEPS = 300


def _traces(kept: bool) -> tuple[list[float], list[float]]:
    """The ground-state objective and its field term after each cold Wolff step."""
    params = fixture("spatio_sequential", "stress").params
    data = simulate_spatio_sequential(params, np.random.default_rng(919))
    rng = np.random.default_rng(1)
    labels = rng.integers(0, params.n_classes, params.graph.n_nodes)
    field = external_field(params, data.observations, labels)
    graph = params.scaled_graph()
    nodes = np.arange(params.graph.n_nodes)
    # beta * J >= 60 on every edge: every bond between equal labels forms.
    _, _, couplings = params.graph.compressed_adjacency()
    beta = 60.0 / float(couplings.min())
    objective = [energy(graph, -field, labels)]
    field_term = [float(field[nodes, labels].sum())]
    # Kept: H declared as the energy it is. Dropped: H read as a log-weight.
    declared = (
        SiteField.from_energy(field) if kept else SiteField.from_log_weight(field)
    )
    offsets, neighbours, adjacency = params.graph.compressed_adjacency()
    for _ in range(STEPS):
        wolff_sweep(labels, declared, offsets, neighbours, adjacency, rng, beta=beta)
        objective.append(energy(graph, -field, labels))
        field_term.append(float(field[nodes, labels].sum()))
    return objective, field_term


@pytest.mark.analytic
def test_a_cold_wolff_step_never_raises_the_label_objective() -> None:
    # A whole monochrome region recoloured cannot lose an agreement, and the
    # accept step admits no rise in the field term, so neither can rise.
    objective, field_term = _traces(kept=True)
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(objective))
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(field_term))
    assert objective[-1] < objective[0]


@pytest.mark.smoke
def test_the_field_with_its_sign_dropped_raises_its_term() -> None:
    # The control: handed +H, the accept step admits only rises in the field
    # term, so the pin above has the power to see a dropped negation.
    _, field_term = _traces(kept=False)
    assert any(b > a + 1e-9 for a, b in itertools.pairwise(field_term))
