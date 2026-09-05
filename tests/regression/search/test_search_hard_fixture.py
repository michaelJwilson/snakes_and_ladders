"""A fixture hill climbing does not already solve, and why it does not.

`STATUS.md` records that no claim about a learned tree policy is possible in
either direction, because on the 6-taxon fixture greedy reaches the enumerated
optimum from every start. Issue #177 is that gap. What is asserted here is a
property of the *fixture*, not of any agent: that a single-move neighbourhood
gets trapped on it, that the trap is a genuine local optimum rather than a
truncated episode, and that the generating topology is nonetheless the
best-scoring one -- without which a search failing would say nothing.

The surface is the fixed-branch-length one `phylo.search.rl` scores
(`RewardModel.KNOWN`), because that is the surface an agent optimizes and the
one issue #178 trains against. All 945 unrooted topologies on 7 leaves are
enumerated, so "the best topology" is an enumerated fact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phylo.learn.rollout import greedy_rollout
from phylo.search.infer import MoveSet
from phylo.search.rl import RewardModel, TopologyEnvironment
from phylo.search.topology import Topology, enumerate_topologies
from phylo.sim.params import SimulationParams, load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import edges

FIXTURE = Path("tests/regression/fixtures/simulation_params_hard.yaml")

# 50 starts at seed + 1000, 30 steps each. Both numbers are part of the
# measurement: a different start set or a shorter horizon is a different
# success rate, so they are named here rather than passed at each call site.
STARTS = 50
HORIZON = 30
START_SEED_OFFSET = 1000

# Realized on the committed fixture. NNI reaches the enumerated maximum from
# 24 of 50 starts; SPR from 50 of 50. The assertions below are bounds around
# these rather than equalities: pinning 0.48 exactly would fail on a numpy
# version that reorders a tie, and the claim the fixture has to support is
# "greedy fails often enough to leave room", not "greedy fails 52% of the
# time".
_NNI_SUCCESS = 0.48
_SPR_SUCCESS = 1.00
# Median log-likelihood shortfall of an NNI run that does not reach the
# maximum. Reported so a fixture that became hard by becoming flat -- every
# topology within a rounding error of every other -- would be visible.
_NNI_MEDIAN_GAP = 38.2


@pytest.fixture(scope="module")
def params() -> SimulationParams:
    return load_simulation_params(FIXTURE)


@pytest.fixture(scope="module")
def alignment(params: SimulationParams) -> dict[str, np.ndarray]:
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    return dict(dataset.alignment)


def _environment(
    params: SimulationParams, alignment: dict[str, np.ndarray], moves: MoveSet
) -> TopologyEnvironment:
    """The reward surface an agent sees, at the tree's own mean branch length."""
    return TopologyEnvironment(
        alignment,
        params.k,
        np.asarray(params.pi),
        branch_length=float(
            np.mean([child.branch_length for _, child in edges(params.tau)])
        ),
        reward=RewardModel.KNOWN,
        moves=moves,
    )


def _enumerated_maximum(
    environment: TopologyEnvironment, alignment: dict[str, np.ndarray]
) -> float:
    return max(
        environment.score(topology)
        for topology in enumerate_topologies(sorted(alignment))
    )


def _endpoints(environment: TopologyEnvironment, seed: int) -> list[Topology]:
    """Where greedy stops, from each of `STARTS` seeded starting topologies."""
    rng = np.random.default_rng(seed + START_SEED_OFFSET)
    return [
        greedy_rollout(environment, environment.reset(rng), HORIZON).states[-1]
        for _ in range(STARTS)
    ]


def test_the_generating_topology_is_the_enumerated_maximum(
    params: SimulationParams, alignment: dict[str, np.ndarray]
) -> None:
    # Without this the fixture is unusable: a search that fails to find the
    # best topology would be finding the right answer, and a search that
    # succeeded would be finding the wrong one.
    environment = _environment(params, alignment, MoveSet.NNI)
    assert environment.score(params.tau) == _enumerated_maximum(environment, alignment)


def test_the_fixture_enumerates_every_unrooted_topology_on_seven_leaves(
    alignment: dict[str, np.ndarray],
) -> None:
    # (2n - 5)!! at n = 7. The oracle every other assertion here rests on is
    # only an oracle if it is exhaustive.
    assert len(list(enumerate_topologies(sorted(alignment)))) == 945


def test_nni_hill_climbing_fails_from_a_substantial_fraction_of_starts(
    params: SimulationParams, alignment: dict[str, np.ndarray]
) -> None:
    environment = _environment(params, alignment, MoveSet.NNI)
    best = _enumerated_maximum(environment, alignment)
    reached = [
        environment.score(state) for state in _endpoints(environment, params.seed)
    ]
    success = float(np.mean([value == best for value in reached]))

    # Both bounds matter. A fixture greedy always solves cannot separate a
    # policy from it; a fixture greedy never solves cannot either, because
    # then the comparison is against a baseline that is not trying.
    assert 0.2 <= success <= 0.8, f"realized NNI success {success}"
    assert abs(success - _NNI_SUCCESS) < 0.2

    shortfalls = [best - value for value in reached if value != best]
    assert np.median(shortfalls) > 1.0, (
        "a fixture that is hard only because every topology scores the same "
        "is flat, not rugged"
    )
    assert abs(float(np.median(shortfalls)) - _NNI_MEDIAN_GAP) < 10.0


def test_every_nni_failure_stops_at_a_genuine_local_optimum(
    params: SimulationParams, alignment: dict[str, np.ndarray]
) -> None:
    # The difference between a hard fixture and too short an episode. If a
    # run stopped with an improving move still available, the failure would
    # measure HORIZON rather than the landscape.
    environment = _environment(params, alignment, MoveSet.NNI)
    best = _enumerated_maximum(environment, alignment)
    endpoints = _endpoints(environment, params.seed)
    failures = [state for state in endpoints if environment.score(state) != best]

    assert failures, "expected some NNI runs to fall short on this fixture"
    assert all(environment.is_terminal(state) for state in failures)


def test_spr_reaches_the_optimum_where_nni_does_not(
    params: SimulationParams, alignment: dict[str, np.ndarray]
) -> None:
    # The 6-taxon fixture cannot show this: both move sets reach the maximum
    # there, so `search_trajectory`'s caption records that it "does not yet
    # separate the two move sets". This one does, which is a second reason to
    # keep it -- the larger SPR neighbourhood escapes the traps NNI sits in.
    environment = _environment(params, alignment, MoveSet.SPR)
    best = _enumerated_maximum(environment, alignment)
    reached = [
        environment.score(state) for state in _endpoints(environment, params.seed)
    ]
    success = float(np.mean([value == best for value in reached]))

    assert success == _SPR_SUCCESS
