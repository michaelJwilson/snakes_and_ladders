"""What a learned policy does on a environment hill climbing does not solve.

Milestone 2.1 asks whether a learned proposal policy beats hill climbing on
#177's fixture; measured, it does not. Pinned: the policy learns (untrained is
far below greedy), and the environment cannot support better: an episode ends
when no move improves, so the agent chooses which local optimum to enter, not
how to leave one. Together they place the null result in the environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sal.fixtures import load_params
from sal.learn.policy import LinearPolicy
from sal.learn.reinforce import reinforce
from sal.learn.rollout import greedy_rollout, rollout
from sal.learn.tree import RewardModel, TreeEnvironment
from sal.qa.rl_tree_policy import (
    BATCH,
    HORIZON,
    ITERATIONS,
    ROLLOUTS_PER_START,
    STARTS,
)
from sal.sim.params import SimulationParams
from sal.sim.simulator import simulate_tree
from sal.sim.topology import MoveSet, Topology, enumerate_topologies
from sal.sim.tree import edges

FIXTURE = Path("tests/regression/fixtures/tree_search/release.yaml")

# Realized over 16 seeds at 640 episodes: 0.485 against greedy's 0.480 (+0.005,
# sd 0.014, 8 of 16 ahead, sign test p = 1.0); untrained 0.018.
_GREEDY = 0.48
_UNTRAINED = 0.018
_TRAINED_LOWER = 0.44


@pytest.fixture(scope="module")
def params() -> SimulationParams:
    return load_params(FIXTURE, SimulationParams)


@pytest.fixture(scope="module")
def taxa(params: SimulationParams) -> list[str]:
    return sorted(simulate_tree(params, np.random.default_rng(params.seed)).alignment)


@pytest.fixture(scope="module")
def environment(params: SimulationParams) -> TreeEnvironment:
    dataset = simulate_tree(params, np.random.default_rng(params.seed))
    return TreeEnvironment(
        dict(dataset.alignment),
        params.k,
        np.asarray(params.pi),
        branch_length=float(
            np.mean([child.branch_length for _, child in edges(params.tau)])
        ),
        reward=RewardModel.KNOWN,
        moves=MoveSet.NNI,
    )


@pytest.fixture(scope="module")
def starts(environment: TreeEnvironment, params: SimulationParams) -> list[Topology]:
    rng = np.random.default_rng(params.seed + 1000)
    return [environment.reset(rng) for _ in range(STARTS)]


@pytest.fixture(scope="module")
def maximum(environment: TreeEnvironment, taxa: list[str]) -> float:
    """The enumerated maximum over all 945 unrooted topologies."""
    return max(environment.score(topology) for topology in enumerate_topologies(taxa))


def _rate(
    environment: TreeEnvironment, endpoints: list[Topology], best: float
) -> float:
    return float(
        np.mean([abs(environment.score(state) - best) < 1e-9 for state in endpoints])
    )


@pytest.mark.smoke
def test_every_episode_ends_where_no_move_improves(
    environment: TreeEnvironment, starts: list[Topology]
) -> None:
    # The claim the null result rests on. If an episode could end with an
    # improving move still on the table, the comparison would measure HORIZON
    # rather than the environment, and a longer budget might change the answer.
    policy = LinearPolicy(environment.n_features())
    reinforce(
        environment,
        policy,
        np.random.default_rng(0),
        iterations=ITERATIONS,
        batch=BATCH,
        max_steps=HORIZON,
    )
    probe = np.random.default_rng(1000)

    greedy_ends = [
        greedy_rollout(environment, start=start, max_steps=HORIZON).states[-1]
        for start in starts
    ]
    learned_ends = [
        rollout(environment, policy, probe, max_steps=HORIZON, start=start).states[-1]
        for start in starts
    ]

    assert all(environment.is_terminal(state) for state in greedy_ends)
    assert all(environment.is_terminal(state) for state in learned_ends)


@pytest.mark.smoke
@pytest.mark.release
def test_an_untrained_policy_is_far_worse_than_greedy(
    environment: TreeEnvironment, starts: list[Topology], maximum: float
) -> None:
    # The control. Without it "the policy ties greedy" is consistent with the
    # environment being so easy that anything ties greedy.
    best = maximum
    untrained = LinearPolicy(environment.n_features())
    rng = np.random.default_rng(99)
    rate = _rate(
        environment,
        [
            rollout(environment, untrained, rng, max_steps=HORIZON, start=start).states[
                -1
            ]
            for start in starts
            for _ in range(ROLLOUTS_PER_START)
        ],
        best,
    )

    assert rate < 0.1, f"untrained policy reached the maximum on {rate}"
    assert abs(rate - _UNTRAINED) < 0.02


@pytest.mark.smoke
def test_a_trained_policy_is_no_worse_than_hill_climbing(
    environment: TreeEnvironment, starts: list[Topology], maximum: float
) -> None:
    # Weaker than the measurement (+0.005, sd 0.014, p = 1.0): a "wins"
    # threshold would assert noise. Checked: out of the untrained regime.
    best = maximum
    greedy = _rate(
        environment,
        [
            greedy_rollout(environment, start=start, max_steps=HORIZON).states[-1]
            for start in starts
        ],
        best,
    )
    assert greedy == pytest.approx(_GREEDY)

    policy = LinearPolicy(environment.n_features())
    reinforce(
        environment,
        policy,
        np.random.default_rng(0),
        iterations=ITERATIONS,
        batch=BATCH,
        max_steps=HORIZON,
    )
    probe = np.random.default_rng(1000)
    learned = _rate(
        environment,
        [
            rollout(environment, policy, probe, max_steps=HORIZON, start=start).states[
                -1
            ]
            for start in starts
            for _ in range(ROLLOUTS_PER_START)
        ],
        best,
    )

    assert learned > _UNTRAINED * 10, "the policy did not train"
    assert learned >= _TRAINED_LOWER
