"""A fixture hill climbing does not already solve, and why it does not.

On the 6-taxon fixture greedy reached the optimum from every start, so no
learned-policy claim was possible (issue #177). Asserted of the fixture: a
single-move neighbourhood is trapped, the trap is a genuine local optimum, and
the generating topology scores best. The surface is `RewardModel.KNOWN`, the
one issue #178 trains on; all 945 unrooted topologies on 7 leaves are enumerated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sal.fixtures import load_params
from sal.learn.rollout import greedy_rollout
from sal.learn.tree import RewardModel, TreeEnvironment
from sal.sim.params import SimulationParams
from sal.sim.simulator import simulate_tree
from sal.sim.topology import MoveSet, Topology, enumerate_topologies
from sal.sim.tree import edges

FIXTURE = Path("tests/regression/fixtures/tree_search/release.yaml")

# 50 starts at seed + 1000, 30 steps each. Both numbers are part of the
# measurement: a different start set or a shorter horizon is a different
# success rate, so they are named here rather than passed at each call site.
STARTS = 50
HORIZON = 30
START_SEED_OFFSET = 1000

# Realized: NNI reaches the maximum from 24 of 50 starts, SPR from 50 of 50.
# Asserted as bounds; a tie reordered by numpy would move 0.48.
_NNI_SUCCESS = 0.48
_SPR_SUCCESS = 1.00
# Median log-likelihood shortfall of an NNI run that does not reach the
# maximum. Reported so a fixture that became hard by becoming flat -- every
# topology within a rounding error of every other -- would be visible.
_NNI_MEDIAN_GAP = 38.2


@pytest.fixture(scope="module")
def params() -> SimulationParams:
    return load_params(FIXTURE, SimulationParams)


@pytest.fixture(scope="module")
def alignment(params: SimulationParams) -> dict[str, np.ndarray]:
    dataset = simulate_tree(params, np.random.default_rng(params.seed))
    return dict(dataset.alignment)


def _environment(
    params: SimulationParams, alignment: dict[str, np.ndarray], moves: MoveSet
) -> TreeEnvironment:
    """The reward surface an agent sees, at the tree's own mean branch length."""
    return TreeEnvironment(
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
    environment: TreeEnvironment, alignment: dict[str, np.ndarray]
) -> float:
    return max(
        environment.score(topology)
        for topology in enumerate_topologies(sorted(alignment))
    )


def _endpoints(environment: TreeEnvironment, seed: int) -> list[Topology]:
    """Where greedy stops, from each of `STARTS` seeded starting topologies."""
    rng = np.random.default_rng(seed + START_SEED_OFFSET)
    return [
        greedy_rollout(
            environment, start=environment.reset(rng), max_steps=HORIZON
        ).states[-1]
        for _ in range(STARTS)
    ]


@pytest.fixture(scope="module")
def nni(params: SimulationParams, alignment: dict[str, np.ndarray]) -> TreeEnvironment:
    """The NNI reward surface three tests below score against, built once."""
    return _environment(params, alignment, MoveSet.NNI)


@pytest.fixture(scope="module")
def nni_maximum(nni: TreeEnvironment, alignment: dict[str, np.ndarray]) -> float:
    """The enumerated maximum over all 945 topologies, scored once."""
    return _enumerated_maximum(nni, alignment)


@pytest.fixture(scope="module")
def nni_endpoints(nni: TreeEnvironment, params: SimulationParams) -> list[Topology]:
    """Where the `STARTS` seeded NNI runs stop, rolled out once."""
    return _endpoints(nni, params.seed)


@pytest.mark.oracle
def test_the_generating_topology_is_the_enumerated_maximum(
    params: SimulationParams, nni: TreeEnvironment, nni_maximum: float
) -> None:
    # Without this the fixture is unusable: a search that fails to find the
    # best topology would be finding the right answer, and a search that
    # succeeded would be finding the wrong one.
    assert nni.score(params.tau) == nni_maximum


@pytest.mark.oracle
def test_the_fixture_enumerates_every_unrooted_topology_on_seven_leaves(
    alignment: dict[str, np.ndarray],
) -> None:
    # Refereed by the closed form `(2n - 5)!!`, which is 945 at n = 7. The
    # oracle every other assertion here rests on is only an oracle if it is
    # exhaustive. Realized 945.
    assert len(list(enumerate_topologies(sorted(alignment)))) == 945


@pytest.mark.end2end
def test_nni_hill_climbing_fails_from_a_substantial_fraction_of_starts(
    nni: TreeEnvironment, nni_maximum: float, nni_endpoints: list[Topology]
) -> None:
    best = nni_maximum
    reached = [nni.score(state) for state in nni_endpoints]
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


@pytest.mark.analytic
def test_every_nni_failure_stops_at_a_genuine_local_optimum(
    nni: TreeEnvironment, nni_maximum: float, nni_endpoints: list[Topology]
) -> None:
    # The difference between a hard fixture and too short an episode. If a
    # run stopped with an improving move still available, the failure would
    # measure HORIZON rather than the environment.
    failures = [state for state in nni_endpoints if nni.score(state) != nni_maximum]

    assert failures, "expected some NNI runs to fall short on this fixture"
    assert all(nni.is_terminal(state) for state in failures)


@pytest.mark.oracle
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
