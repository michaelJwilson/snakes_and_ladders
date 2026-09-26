"""Every Potts entry point reads one field whichever form it arrives in (issue #1091).

`DEV.md`'s vocabulary types the field `SiteField | np.ndarray`, read through
`log_weight_of`. Referee: each entry point typed so here, handed the bare
log-weight and the `SiteField` built from its negated energy, returns the same
run on the same stream, bitwise, wall-clock fields aside.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import pytest
from sal.cost import Cost
from sal.likelihood.surrogate import ground_state_energy_bounds
from sal.opt.budget import Budget
from sal.sample.annealed import (
    annealed_importance_sampling,
    population_annealing,
    simulated_tempering,
)
from sal.sample.potts_keyed import cluster_moves
from sal.sample.potts_mcmc import PottsMove
from sal.sample.potts_mcmc.chains import (
    adapt_ladder_potts,
    parallel_tempering,
    sample_potts_pair,
)
from sal.sample.schedule import InverseTemperatures
from sal.sample.tempered import adapt_ladder_round_trips, tempered_potts_pair
from sal.search.ground_state import ground_state
from sal.search.maxflow import cut_energy, ising_ground_state
from sal.search.tightening import dual_bound
from sal.sim.factor_graph import from_potts
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import SiteField, simulate_potts

#: Result fields that time a run rather than report it.
WALL_CLOCK = frozenset({"seconds", "wall_seconds", "elapsed"})

GRAPH = lattice_graph((4, 4), BoundaryCondition.OPEN, 0.7)
ENERGY = np.random.default_rng(1091).normal(0.0, 1.0, (GRAPH.n_nodes, 3))
BETAS = InverseTemperatures((0.0, 1.0, 2.0))
SWEEPS = 6


def _rng() -> np.random.Generator:
    return np.random.default_rng(7)


#: One call per entry point, on a field given in either form.
CALLS: dict[str, Callable[[PottsGraph, Any], Any]] = {
    "annealed_importance_sampling": lambda g, f: annealed_importance_sampling(
        g, f, BETAS, _rng(), 4
    ),
    "population_annealing": lambda g, f: population_annealing(g, f, BETAS, _rng(), 4),
    "simulated_tempering": lambda g, f: simulated_tempering(
        g, f, BETAS, np.zeros(3), _rng(), SWEEPS
    ),
    "tempered_potts_pair": lambda g, f: tempered_potts_pair(
        g, f, (1.0, 2.0), _rng(), SWEEPS, houdayer=False
    ),
    "adapt_ladder_round_trips": lambda g, f: adapt_ladder_round_trips(
        g, f, (1.0, 1.5, 2.0), _rng(), SWEEPS, 0.02, 2
    ),
    "parallel_tempering": lambda g, f: parallel_tempering(
        g, f, (1.0, 2.0), _rng(), SWEEPS
    ),
    "adapt_ladder_potts": lambda g, f: adapt_ladder_potts(
        g, f, (2.0, 0.4), _rng(), SWEEPS, (0.1, 0.9), 2, 4
    ),
    "sample_potts_pair": lambda g, f: sample_potts_pair(
        g, f, PottsMove.SINGLE_SITE, _rng(), SWEEPS, houdayer=False
    ),
    "cluster_moves": lambda g, f: {
        kind: move._field  # the field each move was built on
        for kind, move in cluster_moves(g, f).items()
    },
    "simulate_potts": lambda g, f: simulate_potts(g, f, _rng(), 3, burn_in=5),
    "ground_state_energy_bounds": lambda g, f: ground_state_energy_bounds(g, f, 1.0),
    "ground_state": lambda g, f: ground_state(
        g, f, "alpha-expansion", Budget(Cost.SITE_VISITS, 50 * g.n_nodes), _rng()
    ),
    "dual_bound": lambda g, f: dual_bound(g, f, iterations=20),
    "from_potts": lambda g, f: [
        factor.log_table for factor in from_potts(g, f).factors
    ],
}

#: The cuts take two states: the first two columns of the same energy.
BINARY: dict[str, Callable[[PottsGraph, Any], Any]] = {
    "ising_ground_state": lambda g, f: ising_ground_state(g, f),
    "cut_energy": lambda g, f: cut_energy(g, f, 1.5),
}


def _same(first: Any, second: Any) -> bool:
    """Equal field by field, arrays bitwise, wall-clock fields skipped."""
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        return type(first) is type(second) and all(
            _same(getattr(first, name.name), getattr(second, name.name))
            for name in dataclasses.fields(first)
            if name.name not in WALL_CLOCK
        )
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second, equal_nan=True)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same(first[key], second[key]) for key in first
        )
    if isinstance(first, list | tuple):
        return len(first) == len(second) and all(
            _same(a, b) for a, b in zip(first, second, strict=True)
        )
    if isinstance(first, float):
        return bool(first == second or (np.isnan(first) and np.isnan(second)))
    if isinstance(first, int | str | bool | Enum) or first is None:
        return bool(first == second)
    if hasattr(first, "numpy"):
        return np.array_equal(first.numpy(), second.numpy())
    return type(first) is type(second)


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_site_field_and_its_log_weight_run_alike(name: str) -> None:
    call = CALLS[name]

    bare = call(GRAPH, -ENERGY)
    declared = call(GRAPH, SiteField.from_energy(ENERGY))

    assert _same(bare, declared)


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(BINARY))
def test_a_two_state_site_field_and_its_log_weight_cut_alike(name: str) -> None:
    call, energy = BINARY[name], ENERGY[:, :2]

    assert _same(call(GRAPH, -energy), call(GRAPH, SiteField.from_energy(energy)))
