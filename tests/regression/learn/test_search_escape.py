"""Leaving a local optimum, and the baseline that does it better.

#193 found no policy could beat hill climbing on the #177 fixture: every
episode ended at the first local optimum. Issue #194 lets `rollout` run past
one and `EpsilonGreedyPolicy` take a worsening move. Pinned: the mechanism
works, and random-restart hill climbing solves the fixture outright at the
same budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sal.fixtures import load_params
from sal.learn.policy import EpsilonGreedyPolicy, LinearPolicy
from sal.learn.rollout import greedy_rollout, rollout
from sal.learn.tree import RewardModel, TreeEnvironment
from sal.sim.params import SimulationParams
from sal.sim.simulator import simulate_tree
from sal.sim.topology import MoveSet, Topology, enumerate_topologies
from sal.sim.tree import edges

FIXTURE = Path("tests/regression/fixtures/tree_search/release.yaml")

BUDGET = 60
TRAP_RUNS = 8
PROBE_STARTS = 20

# Realized over 180 runs from each of the 9 non-global local optima, and over
# 50 starts x 8 rollouts at a budget of 60 decisions:
#
#   epsilon   escape from a trap   matched-budget success
#   0.00      0.111                0.560
#   0.05      0.450                0.693
#   0.10      0.694                0.753
#   0.20      0.794                0.833
#   0.40      0.883                0.908
#   restarts  --                   1.000
#
# Asserted as bounds around these at fewer runs.
_HIGH_EPSILON = 0.4
_LOW_EPSILON = 0.0


@pytest.fixture(scope="module")
def params() -> SimulationParams:
    return load_params(FIXTURE, SimulationParams)


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
def taxa(params: SimulationParams) -> list[str]:
    return sorted(simulate_tree(params, np.random.default_rng(params.seed)).alignment)


@pytest.fixture(scope="module")
def maximum(environment: TreeEnvironment, taxa: list[str]) -> float:
    return max(
        environment.log_weight(topology) for topology in enumerate_topologies(taxa)
    )


@pytest.fixture(scope="module")
def traps(
    environment: TreeEnvironment, taxa: list[str], maximum: float
) -> list[Topology]:
    """The local optima that are not the global one."""
    return [
        topology
        for topology in enumerate_topologies(taxa)
        if environment.is_terminal(topology)
        and abs(environment.log_weight(topology) - maximum) >= 1e-9
    ]


def _hill_climbing_policy(environment: TreeEnvironment) -> LinearPolicy:
    """A policy whose greedy action is the best-rewarded one: hill climbing.

    A positive weight on the one feature; see the untrained-policy test below.
    """
    policy = LinearPolicy(environment.n_features())
    policy.set_weights(torch.tensor([1.0], dtype=torch.float64))
    return policy


def _best_seen(environment: TreeEnvironment, states: tuple[Topology, ...]) -> float:
    """A wandering searcher keeps its best state, not its last."""
    return max(environment.log_weight(state) for state in states)


@pytest.mark.smoke
def test_wrapping_an_untrained_policy_is_not_hill_climbing(
    environment: TreeEnvironment, traps: list[Topology]
) -> None:
    # An untrained `LinearPolicy` scores every action alike, so the wrapper
    # explores around "always take action 0", which is not a searcher.
    untrained = LinearPolicy(environment.n_features())
    climbing = _hill_climbing_policy(environment)
    features = [
        environment.features(state, environment.actions(state)) for state in traps
    ]

    # Always the first action, whatever the rewards are.
    assert [untrained.greedy(row) for row in features] == [0] * len(traps)
    # Sometimes the first action, because sometimes the first action is best;
    # the claim is that it is chosen on the reward rather than on position.
    assert any(climbing.greedy(row) != 0 for row in features)


@pytest.mark.oracle
def test_epsilon_zero_reproduces_hill_climbing_exactly(
    environment: TreeEnvironment, traps: list[Topology]
) -> None:
    # The control. Without it, a rising escape rate is not attributable to
    # epsilon, because the thing epsilon is added to might not be the
    # baseline at all -- which is precisely how the first run of this
    # experiment went wrong.
    agent = EpsilonGreedyPolicy(_hill_climbing_policy(environment), 0.0)
    for start in traps[:3]:
        under_policy = rollout(
            environment, agent, np.random.default_rng(0), max_steps=BUDGET, start=start
        )
        under_greedy = greedy_rollout(environment, start=start, max_steps=BUDGET)
        assert under_policy.states == under_greedy.states


@pytest.mark.oracle
def test_an_episode_can_leave_a_local_optimum(
    environment: TreeEnvironment, traps: list[Topology], maximum: float
) -> None:
    # By default every run ends where it started, so the floor is exactly 0.
    # Refereed by the exhaustive `maximum` over every unrooted topology.
    # Realized over 9 traps x 8 runs: 0.111 at epsilon 0, 0.875 at 0.4.
    policy = _hill_climbing_policy(environment)
    rates = {}
    for epsilon in (_LOW_EPSILON, _HIGH_EPSILON):
        agent = EpsilonGreedyPolicy(policy, epsilon)
        rng = np.random.default_rng(7)
        escaped = [
            abs(
                _best_seen(
                    environment,
                    rollout(
                        environment,
                        agent,
                        rng,
                        max_steps=BUDGET,
                        start=trap,
                        stop_at_local_optimum=False,
                    ).states,
                )
                - maximum
            )
            < 1e-9
            for trap in traps
            for _ in range(TRAP_RUNS)
        ]
        rates[epsilon] = float(np.mean(escaped))

    assert rates[_LOW_EPSILON] > 0.0, "continuing past a local optimum escapes some"
    assert rates[_HIGH_EPSILON] > 0.5, f"realized {rates[_HIGH_EPSILON]}"
    assert rates[_HIGH_EPSILON] > rates[_LOW_EPSILON], "exploration must pay"


@pytest.mark.smoke
def test_stopping_at_a_local_optimum_never_escapes(
    environment: TreeEnvironment, traps: list[Topology], maximum: float
) -> None:
    # The floor: under the default rule an episode started at a local optimum
    # has terminated; the score line restates the `traps` filter.
    agent = EpsilonGreedyPolicy(_hill_climbing_policy(environment), _HIGH_EPSILON)
    rng = np.random.default_rng(7)
    for trap in traps:
        episode = rollout(environment, agent, rng, max_steps=BUDGET, start=trap)
        assert episode.states == (trap,)
        assert abs(environment.log_weight(trap) - maximum) >= 1e-9


@pytest.mark.oracle
def test_random_restart_hill_climbing_solves_this_fixture(
    environment: TreeEnvironment, params: SimulationParams, maximum: float
) -> None:
    # A single greedy run reaches the maximum from 48% of starts (#193);
    # restarts at the same budget from all of them, so Stage 2's baseline is
    # 1.00. Realized 20 of 20 within 1e-9 of the exhaustive maximum.
    start_rng = np.random.default_rng(params.seed + 1000)
    restart_rng = np.random.default_rng(11)
    solved = []
    for _ in range(PROBE_STARTS):
        state, spent, seen = environment.reset(start_rng), 0, -np.inf
        while spent < BUDGET:
            episode = greedy_rollout(environment, start=state, max_steps=BUDGET - spent)
            spent += max(len(episode.actions), 1)
            seen = max(seen, _best_seen(environment, episode.states))
            state = environment.reset(restart_rng)
        solved.append(abs(seen - maximum) < 1e-9)

    assert float(np.mean(solved)) == 1.0
