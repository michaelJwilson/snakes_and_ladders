"""What a decision costs in each of #706's four arms, and what a move costs.

Correctness is pinned in `tests/regression/learn/test_cluster_arms.py`, per the
division of labor between the two directories.

Two quantities, benchmarked together because the ticket's feasibility argument
is a *ratio* between them. `features` is the policy's own overhead --- what it
pays to look at its candidates --- and `step` is the move it then spends. An
arm is affordable when the first is small against the second, and #706's answer
to "score the mixed set in full?" came from reading exactly this pair: a
full 13,830-candidate pass is 0.86 of a sweep's work, a 128-candidate subset
1/120th of it.

The arms are benchmarked at the same lattice for that reason and not to rank
them. A Swendsen-Wang decision is cheap because its action names nothing, and
a Wolff step is cheap because it touches one cluster --- so "cheapest arm" and
"cheapest arm per site visit bought" are different claims and only the pair of
numbers separates them.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from sal.learn.potts_nd import (
    PottsAction,
    PottsNDEnvironment,
)
from sal.sample.potts_keyed import cluster_moves
from sal.sample.potts_mcmc import MoveKind
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import critical_coupling, spatio_only_field

#: The lattice the ticket's costs were measured at, halved per side so a
#: benchmark run stays inside the suite's budget: 144 sites rather than 576.
SIDE = 12

#: The arms, spelled as `learn.potts_nd` takes them.
ARMS = {
    "single-site": (MoveKind.FLIP, MoveKind.SWEEP),
    "wolff": (MoveKind.WOLFF,),
    "swendsen-wang": (MoveKind.SWENDSEN_WANG,),
    "mixed": tuple(MoveKind),
}


def _arm(kinds: tuple[MoveKind, ...], n_states: int = 3) -> PottsNDEnvironment:
    """One arm at the critical coupling, with `spatio_only`'s lognormal sizes."""
    coupling = critical_coupling(n_states)
    graph = lattice_graph((SIDE, SIDE), BoundaryCondition.OPEN, coupling)
    sizes = np.random.default_rng(706).lognormal(0.0, 0.6, size=SIDE * SIDE)
    field = spatio_only_field(np.linspace(-1.0, 1.0, n_states), sizes)
    return PottsNDEnvironment(
        edges=list(graph.edges),
        n_nodes=SIDE * SIDE,
        coupling=coupling,
        field=field,
        kinds=kinds,
        moves=cluster_moves(graph, field),
        generator=np.random.default_rng(706),
    )


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_scoring_one_decisions_candidates(
    benchmark: BenchmarkFixture, arm: str
) -> None:
    """One `features` pass over what the arm offers at a state: the overhead."""
    environment = _arm(ARMS[arm])
    state = environment.reset(np.random.default_rng(0))
    offered = environment.actions(state)
    benchmark(environment.features, state, offered)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_listing_one_decisions_candidates(
    benchmark: BenchmarkFixture, arm: str
) -> None:
    """The keyed subset draw, which `features` is charged after.

    Separate from the pass above because the two scale differently: the subset
    is drawn once per state whatever the ladder's length, while the scoring is
    linear in the candidates the ladder multiplies it into.
    """
    environment = _arm(ARMS[arm])
    state = environment.reset(np.random.default_rng(0))
    benchmark(environment.actions, state)


@pytest.mark.parametrize(
    ("kind", "rung"),
    [
        (MoveKind.FLIP, 3),
        (MoveKind.SWEEP, 3),
        (MoveKind.WOLFF, 0),
        (MoveKind.WOLFF, 3),
        (MoveKind.SWENDSEN_WANG, 0),
        (MoveKind.SWENDSEN_WANG, 3),
    ],
)
def test_one_move(benchmark: BenchmarkFixture, kind: MoveKind, rung: int) -> None:
    """One `step`, the thing a decision buys.

    The cluster moves are read at two rungs because their cost is the cluster
    they grow and the bond probability is what sets its size: rung ``0`` is
    the ``T = 0`` limit, where every like-coloured bond is active and the
    cluster is the whole monochromatic component, and rung ``3`` is a
    temperature where it is not. A single rung would report a move's cost as
    if it had one.
    """
    environment = _arm(tuple(MoveKind))
    state = environment.reset(np.random.default_rng(0))
    parametric = kind in (MoveKind.FLIP, MoveKind.WOLFF)
    action = PottsAction(kind, 7 if parametric else -1, 1 if parametric else -1, rung)
    benchmark(environment.step, state, action)
