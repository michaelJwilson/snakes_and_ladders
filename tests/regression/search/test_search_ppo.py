"""PPO on the tree environment, against REINFORCE and hill climbing at a matched budget (issue #313).

The hard fixture of #177 is where greedy fails often enough to leave room:
NNI hill climbing reaches the enumerated maximum from 0.48 of the starts,
REINFORCE at 640 episodes lands on that rate (#178). PPO trains at the same
budget from the same starts, and the comparison is reported; a scheduled
epsilon-greedy behaviour policy is the off-policy variant #194 asked for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.critic import Critic, n_state_features
from snakes_and_ladders.learn.policy import EpsilonGreedyPolicy, LinearPolicy
from snakes_and_ladders.learn.ppo import ppo
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout
from snakes_and_ladders.opt.schedule import Linear
from snakes_and_ladders.qa.rl_tree_policy import BATCH, HORIZON, ITERATIONS, STARTS
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import RewardModel, TopologyEnvironment
from snakes_and_ladders.search.topology import Topology, enumerate_topologies
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

FIXTURE = Path("tests/regression/fixtures/simulation_params_hard.yaml")
ROLLOUTS_PER_START = 4


@pytest.fixture(scope="module")
def environment() -> TopologyEnvironment:
    params = load_simulation_params(FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    lengths = [
        child.branch_length for _, child in edges(params.tau) if child.branch_length
    ]
    return TopologyEnvironment(
        dict(dataset.alignment),
        params.k,
        params.pi,
        branch_length=float(np.mean(lengths)),
        reward=RewardModel.KNOWN,
        moves=MoveSet.NNI,
    )


@pytest.fixture(scope="module")
def maximum(environment: TopologyEnvironment) -> float:
    params = load_simulation_params(FIXTURE)
    leaves = sorted(node.name for _, node in edges(params.tau) if node.is_leaf)
    return max(environment.score(t) for t in enumerate_topologies(leaves))


def _rate(
    environment: TopologyEnvironment, finals: list[Topology], best: float
) -> float:
    return float(np.mean([abs(environment.score(t) - best) < 1e-9 for t in finals]))


@pytest.mark.release
@pytest.mark.oracle
def test_ppo_on_the_hard_fixture_is_no_worse_than_reinforce_at_the_same_budget(
    environment: TopologyEnvironment, maximum: float
) -> None:
    # Same 640 episodes, same starts, same horizon as test_search_tree_policy.
    # Measured (recorded in STATUS.md): the rates at which greedy, REINFORCE
    # and PPO reach the enumerated maximum from the 50 seeded starts.
    rng = np.random.default_rng(load_simulation_params(FIXTURE).seed + 1000)
    starts = [environment.reset(rng) for _ in range(STARTS)]
    greedy = _rate(
        environment,
        [greedy_rollout(environment, s, HORIZON).states[-1] for s in starts],
        maximum,
    )
    outcomes = {}
    for label in ("reinforce", "ppo", "ppo-epsilon"):
        policy = LinearPolicy(environment.n_features())
        if label == "reinforce":
            reinforce(
                environment,
                policy,
                np.random.default_rng(0),
                iterations=ITERATIONS,
                batch=BATCH,
                max_steps=HORIZON,
            )
        else:
            critic = Critic(
                n_state_features(environment),
                hidden=None,
                generator=torch.Generator().manual_seed(0),
            )
            behaviour = (
                EpsilonGreedyPolicy(policy, 0.2) if label == "ppo-epsilon" else None
            )
            schedule = Linear(0.3, 0.02, ITERATIONS) if label == "ppo-epsilon" else None
            ppo(
                environment,
                policy,
                critic,
                np.random.default_rng(0),
                iterations=ITERATIONS,
                batch=BATCH,
                max_steps=HORIZON,
                behaviour=behaviour,
                epsilon_schedule=schedule,
            )
        probe = np.random.default_rng(1000)
        outcomes[label] = _rate(
            environment,
            [
                rollout(environment, policy, probe, HORIZON, start=s).states[-1]
                for s in starts
                for _ in range(ROLLOUTS_PER_START)
            ],
            maximum,
        )
    assert greedy == pytest.approx(0.48)
    assert outcomes["ppo"] >= outcomes["reinforce"] - 0.05, outcomes
    assert all(rate > 0.18 for rate in outcomes.values()), outcomes


@pytest.mark.structural
def test_epsilon_greedy_log_probabilities_are_a_distribution_over_the_neighbourhood(
    environment: TopologyEnvironment,
) -> None:
    policy = LinearPolicy(environment.n_features())
    policy.set_weights(torch.tensor([5.0], dtype=torch.float64))
    wrapper = EpsilonGreedyPolicy(policy, 0.25)
    state = environment.reset(np.random.default_rng(3))
    features = environment.features(state, environment.actions(state))
    log_probabilities = wrapper.log_probabilities(features)
    assert float(torch.exp(log_probabilities).sum()) == pytest.approx(1.0, abs=1e-12)
    greedy = policy.greedy(features)
    n_actions = features.shape[0]
    assert float(torch.exp(log_probabilities[greedy])) == pytest.approx(
        0.75 + 0.25 / n_actions
    )
    wrapper.epsilon = 0.0
    assert float(
        torch.exp(wrapper.log_probabilities(features)[greedy])
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="epsilon must lie"):
        wrapper.epsilon = 1.5
