"""Forbidden-label instances the gco and PyMaxflow pins share (issue #1274).

A forbidden pair is ``-inf`` in the package's field (:func:`sal.sim.potts.forbid`,
issue #1139). Neither framework carries ``-inf``, so each receives
:func:`sal.external.potts_inputs.stand_in`, the rule ``sal.external`` applies
itself (#1282); it and :func:`~sal.external.potts_inputs.allowed_by` are
imported from there, not copied.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest
from sal.backend import Backend
from sal.external.potts_inputs import allowed_by, stand_in
from sal.search.alpha_expansion import alpha_expansion
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, energies, energy, forbid

from tests.regression.search.test_forbidden_labels import (
    ALLOWED,
    FINITE,
    GRAPH,
    SMALL,
    _small_problem,
)

#: Agreement of two energies of one labelling, or of an expansion and the
#: enumerated minimum it reaches: sums over the same terms in other orders.
ENERGY_AGREEMENT = 1e-12

#: ``test_gco.py``'s declared agreement of two expansions' energies at 71².
EXPANSION_AGREEMENT = 1e-2

Row = tuple[str, PottsGraph, np.ndarray, np.ndarray]


def constrained_minimum(
    graph: PottsGraph, finite: np.ndarray, allowed: np.ndarray
) -> float:
    """The minimum energy over every allowed labelling, enumerated."""
    n_states = finite.shape[1]
    every = np.array(list(itertools.product(range(n_states), repeat=graph.n_nodes)))
    keep = allowed[np.arange(graph.n_nodes), every].all(axis=1)
    return float(energies(graph, finite, every[keep]).min())


def enumerable() -> list[Row]:
    """`test_forbidden_labels.py`'s 3x3 instance and its 2x4 ones at 2 and 3 states."""
    rows: list[Row] = [("3x3/q3", GRAPH, FINITE, ALLOWED)]
    for n_states, seed in itertools.product((2, 3), range(6)):
        finite, allowed = _small_problem(seed, n_states)
        rows.append((f"2x4/q{n_states}/{seed}", SMALL, finite, allowed))
    return rows


def lattices() -> list[Row]:
    """Open lattices at 16² and 71², 3 and 10 states, a third of the pairs forbidden.

    Each at its critical coupling under a standard normal field, seed 1139;
    each site keeps one allowed label at least.
    """
    rows: list[Row] = []
    for side, n_states in itertools.product((16, 71), (3, 10)):
        graph = lattice_graph(
            (side, side), BoundaryCondition.OPEN, critical_coupling(n_states)
        )
        rng = np.random.default_rng(1139)
        finite = rng.normal(size=(graph.n_nodes, n_states))
        allowed = rng.random((graph.n_nodes, n_states)) < 2 / 3
        allowed[
            np.arange(graph.n_nodes), rng.integers(n_states, size=graph.n_nodes)
        ] = True
        rows.append((f"{side}^2/q{n_states}", graph, finite, allowed))
    return rows


def check_expansion(
    name: str,
    graph: PottsGraph,
    finite: np.ndarray,
    allowed: np.ndarray,
    theirs: Callable[[PottsGraph, np.ndarray, int], np.ndarray],
    *,
    exact: bool,
) -> None:
    """Row ``name``: the package's expansion on ``-inf`` beside a framework's on :func:`~sal.external.potts_inputs.stand_in`.

    Neither holds a forbidden label; the framework's labelling is a fixed
    point of the package's move, energy bitwise; the two energies agree
    within :data:`EXPANSION_AGREEMENT`; with ``exact``, both are the
    enumerated constrained minimum within :data:`ENERGY_AGREEMENT`.
    """
    n_states = finite.shape[1]
    field = forbid(finite, allowed)
    ours = alpha_expansion(graph, field, backend=Backend.RUST)
    labelling = theirs(graph, stand_in(graph, finite, allowed), n_states)
    attained = energy(graph, field, labelling)
    from_theirs = alpha_expansion(graph, field, start=labelling, backend=Backend.RUST)

    assert allowed_by(allowed, ours.labelling), name
    assert allowed_by(allowed, labelling), name
    assert from_theirs.moves == 0, name
    assert np.array_equal(from_theirs.labelling, labelling), name
    assert from_theirs.energy == attained, name
    assert attained == pytest.approx(ours.energy, rel=EXPANSION_AGREEMENT), name
    if exact:
        minimum = constrained_minimum(graph, finite, allowed)
        scale = ENERGY_AGREEMENT * max(1.0, abs(minimum))
        assert abs(ours.energy - minimum) <= scale, (name, ours.energy, minimum)
        assert abs(attained - minimum) <= scale, (name, attained, minimum)
