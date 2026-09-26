"""`sense="max"` is the minimum of the negated problem, in every Potts solver (issue #1081).

Referees: each solver's ``max`` on ``(J, h)`` returns the labelling its
``min`` returns on ``(-J, -h)``, bitwise, with the energy in the original
sign; and on a 3x3 lattice TRW-S's bound under ``max`` sits above the
enumerated maximum, with a gap that is never negative.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal.search.alpha_expansion import (
    Sense,
    alpha_beta_swap,
    alpha_expansion,
    oriented,
)
from sal.search.icm import iterated_conditional_modes
from sal.search.trws import trws
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energies, energy

# Repulsive, so the negated coupling is attractive and the graph cuts take it.
GRAPH = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.5)
FIELD = np.random.default_rng(1081).normal(0.0, 1.0, (GRAPH.n_nodes, 3))


def _solvers() -> dict[str, object]:
    return {
        "icm": lambda g, f, s: iterated_conditional_modes(
            g, f, n_states=3, rng=np.random.default_rng(0), sense=s
        ),
        "alpha-expansion": lambda g, f, s: alpha_expansion(g, f, n_states=3, sense=s),
        "alpha-beta-swap": lambda g, f, s: alpha_beta_swap(g, f, n_states=3, sense=s),
        "trws": lambda g, f, s: trws(g, f, sense=s),
    }


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(_solvers()))
def test_max_is_min_on_the_negated_problem_bitwise(name: str) -> None:
    solve = _solvers()[name]
    negated_graph, negated_field = oriented(GRAPH, FIELD, Sense.MAX)

    highest = solve(GRAPH, FIELD, Sense.MAX)  # type: ignore[operator]
    lowest = solve(negated_graph, negated_field, Sense.MIN)  # type: ignore[operator]

    assert np.array_equal(highest.labelling, lowest.labelling)
    assert highest.energy == -lowest.energy
    assert highest.energy == pytest.approx(
        energy(GRAPH, FIELD, highest.labelling), rel=1e-12
    )
    assert highest.sense is Sense.MAX


@pytest.mark.oracle
def test_the_bound_under_max_sits_above_the_enumerated_maximum() -> None:
    every = np.array(list(itertools.product(range(3), repeat=GRAPH.n_nodes)))
    maximum = float(energies(GRAPH, FIELD, every).max())

    bounded = trws(GRAPH, FIELD, sense=Sense.MAX)

    assert bounded.bound >= maximum - 1e-9
    assert bounded.energy <= maximum + 1e-9
    assert bounded.gap >= -1e-9
