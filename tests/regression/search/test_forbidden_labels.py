"""A label the field forbids, ``-inf``, is never returned (issue #1081).

Referees: enumeration over every allowed labelling of a 3x3 lattice with 3
states. Every ground-state method, TRW-S and the dual bound return an
allowed labelling with a finite energy; the exact ones reach the enumerated
constrained minimum; the dual's bound stays below it. And ``forbid``'s mask
is the ``-inf`` field, bitwise, so the two spellings are one problem.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.search.bifurcation import simulated_bifurcation
from sal.search.ground_state import METHODS, ground_state
from sal.search.tightening import dual_bound
from sal.search.trws import trws
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energies, forbid

GRAPH = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8)
FINITE = np.random.default_rng(1081).normal(0.0, 1.0, (GRAPH.n_nodes, 3))
ALLOWED = np.ones_like(FINITE, dtype=bool)
ALLOWED[:, 0] = False  # the first label nowhere
ALLOWED[::2, 2] = False  # the third on every other site
FIELD = forbid(FINITE, ALLOWED)
BUDGET = Budget(Cost.SITE_VISITS, 400 * GRAPH.n_nodes * 5)


def _constrained_minimum() -> float:
    every = np.array(list(itertools.product(range(3), repeat=GRAPH.n_nodes)))
    keep = ALLOWED[np.arange(GRAPH.n_nodes), every].all(axis=1)
    return float(energies(GRAPH, FINITE, every[keep]).min())


def _allowed(labelling: np.ndarray) -> bool:
    return bool(ALLOWED[np.arange(GRAPH.n_nodes), labelling].all())


@pytest.mark.oracle
@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_returns_an_allowed_labelling(method: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = ground_state(GRAPH, FIELD, method, BUDGET, np.random.default_rng(3))

    assert _allowed(run.labelling)
    assert np.isfinite(run.energy)
    assert run.energy >= _constrained_minimum() - 1e-9


@pytest.mark.oracle
def test_the_bounded_solvers_bound_the_constrained_minimum() -> None:
    minimum = _constrained_minimum()
    for bounded in (trws(GRAPH, FIELD), dual_bound(GRAPH, FIELD)):
        assert _allowed(bounded.labelling)
        assert np.isfinite(bounded.energy)
        assert np.isfinite(bounded.bound)
        assert bounded.bound <= minimum + 1e-9 <= bounded.energy + 2e-9
    relaxed = simulated_bifurcation(
        GRAPH, FIELD, n_states=3, rng=np.random.default_rng(0)
    )
    assert _allowed(relaxed.labelling)


@pytest.mark.smoke
def test_a_mask_is_the_negative_infinite_field_and_a_site_allowing_nothing_is_refused() -> (
    None
):
    written = FINITE.copy()
    written[~ALLOWED] = -np.inf
    assert np.array_equal(forbid(FINITE, ALLOWED), written)
    nothing = ALLOWED.copy()
    nothing[4] = False
    with pytest.raises(ValueError, match="allows no label"):
        forbid(FINITE, nothing)
