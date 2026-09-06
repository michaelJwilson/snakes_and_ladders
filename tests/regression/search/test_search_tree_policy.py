"""What a learned policy does on a landscape hill climbing does not solve.

Milestone 2.1's phylogenetic half asks whether a learned proposal policy beats
hill climbing. Issue #177 supplied a fixture on which hill climbing fails, so
the question is finally askable. The answer measured here is that it does not,
and the tests below pin the two facts that make that answer meaningful rather
than merely disappointing.

The first is that the policy learns: an untrained policy, uniform over the same
moves, is far worse than greedy, so "ties greedy" is most of the distance from
chance rather than a failure to train.

The second is that this environment cannot support a better answer. An episode
ends when no move improves, so every run terminates at a state greedy would
also have stopped at, and the agent chooses which local optimum to enter rather
than how to leave one. Both halves are asserted, because together they say the
null result is a property of the environment and not of the training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout
from snakes_and_ladders.qa.rl_tree_policy import (
    BATCH,
    HORIZON,
    ITERATIONS,
    ROLLOUTS_PER_START,
    STARTS,
)
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import RewardModel, TopologyEnvironment
from snakes_and_ladders.search.topology import Topology, enumerate_topologies
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

FIXTURE = Path("tests/regression/fixtures/simulation_params_hard.yaml")

# Realized over 16 training seeds at 640 episodes: the policy reaches the
# enumerated maximum on 0.485 of episodes against greedy's 0.480, a difference
# of +0.005 with a standard deviation of 0.014, 8 of 16 seeds ahead, and an
# exact two-sided sign test at p = 1.0. An untrained policy reaches it on
# 0.018.
_GREEDY = 0.48
_UNTRAINED = 0.018
_TRAINED_LOWER = 0.44


@pytest.fixture(scope="module")
def params() -> SimulationParams:
    return load_simulation_params(FIXTURE)


@pytest.fixture(scope="module")
def taxa(params: SimulationParams) -> list[str]:
    return sorted(
        simulate_alignment(
            tau=params.tau,
            k=params.k,
            pi=params.pi,
            rng=np.random.default_rng(params.seed),
            n_sites=params.n_sites,
        ).alignment
    )


@pytest.fixture(scope="module")
def environment(params: SimulationParams) -> TopologyEnvironment:
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    return TopologyEnvironment(
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
def starts(
    environment: TopologyEnvironment, params: SimulationParams
) -> list[Topology]:
    rng = np.random.default_rng(params.seed + 1000)
    return [environment.reset(rng) for _ in range(STARTS)]


@pytest.fixture(scope="module")
def maximum(environment: TopologyEnvironment, taxa: list[str]) -> float:
    """The enumerated maximum over all 945 unrooted topologies."""
    return max(environment.score(topology) for topology in enumerate_topologies(taxa))


def _rate(
    environment: TopologyEnvironment, endpoints: list[Topology], best: float
) -> float:
    return float(
        np.mean([abs(environment.score(state) - best) < 1e-9 for state in endpoints])
    )


@pytest.mark.mathematical
def test_every_episode_ends_where_no_move_improves(
    environment: TopologyEnvironment, starts: list[Topology]
) -> None:
    # The claim the null result rests on. If an episode could end with an
    # improving move still on the table, the comparison would measure HORIZON
    # rather than the landscape, and a longer budget might change the answer.
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
        greedy_rollout(environment, start, HORIZON).states[-1] for start in starts
    ]
    learned_ends = [
        rollout(environment, policy, probe, HORIZON, start=start).states[-1]
        for start in starts
    ]

    assert all(environment.is_terminal(state) for state in greedy_ends)
    assert all(environment.is_terminal(state) for state in learned_ends)


@pytest.mark.mathematical
def test_an_untrained_policy_is_far_worse_than_greedy(
    environment: TopologyEnvironment, starts: list[Topology], maximum: float
) -> None:
    # The control. Without it "the policy ties greedy" is consistent with the
    # environment being so easy that anything ties greedy.
    best = maximum
    untrained = LinearPolicy(environment.n_features())
    rng = np.random.default_rng(99)
    rate = _rate(
        environment,
        [
            rollout(environment, untrained, rng, HORIZON, start=start).states[-1]
            for start in starts
            for _ in range(ROLLOUTS_PER_START)
        ],
        best,
    )

    assert rate < 0.1, f"untrained policy reached the maximum on {rate}"
    assert abs(rate - _UNTRAINED) < 0.02


@pytest.mark.mathematical
def test_a_trained_policy_is_no_worse_than_hill_climbing(
    environment: TopologyEnvironment, starts: list[Topology], maximum: float
) -> None:
    # Deliberately weaker than the measurement, following
    # `test_the_learned_policy_is_at_least_as_good_as_hill_climbing`: over 16
    # seeds the difference is +0.005 with a standard deviation of 0.014 and an
    # exact sign test at p = 1.0, so a threshold asserting the policy *wins*
    # would be asserting noise. What is checked is that it climbs out of the
    # untrained regime and lands on the baseline.
    best = maximum
    greedy = _rate(
        environment,
        [greedy_rollout(environment, start, HORIZON).states[-1] for start in starts],
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
            rollout(environment, policy, probe, HORIZON, start=start).states[-1]
            for start in starts
            for _ in range(ROLLOUTS_PER_START)
        ],
        best,
    )

    assert learned > _UNTRAINED * 10, "the policy did not train"
    assert learned >= _TRAINED_LOWER
