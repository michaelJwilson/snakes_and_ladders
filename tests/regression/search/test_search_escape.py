"""Leaving a local optimum, and the baseline that does it better.

#193 measured that no policy over this action set can beat hill climbing on
the #177 fixture, because every episode ends at the first state no move
improves. Issue #194 removes that: `rollout` can run past a local optimum,
and `EpsilonGreedyPolicy` can take the worsening move that leaves one.

What is pinned here is that the mechanism works *and* that it is not the best
available answer. Random-restart hill climbing solves this fixture outright at
the same budget, so a reader is not left to infer from a rising escape rate
that epsilon-greedy is what tree search should use. The rising rate and the
baseline that beats it are the same result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.policy import EpsilonGreedyPolicy, LinearPolicy
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import RewardModel, TopologyEnvironment
from snakes_and_ladders.search.topology import Topology, enumerate_topologies
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

FIXTURE = Path("tests/regression/fixtures/simulation_params_hard.yaml")

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
# The assertions below are bounds around these, at a smaller run count so the
# test stays inside the per-pull-request suite.
_HIGH_EPSILON = 0.4
_LOW_EPSILON = 0.0


@pytest.fixture(scope="module")
def params() -> SimulationParams:
    return load_simulation_params(FIXTURE)


@pytest.fixture(scope="module")
def environment(params: SimulationParams) -> TopologyEnvironment:
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
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
def taxa(params: SimulationParams) -> list[str]:
    return sorted(
        simulate_alignment(
            tau=params.tau,
            k=params.k,
            pi=params.pi,
            seed=params.seed,
            n_sites=params.n_sites,
        ).alignment
    )


@pytest.fixture(scope="module")
def maximum(environment: TopologyEnvironment, taxa: list[str]) -> float:
    return max(environment.score(topology) for topology in enumerate_topologies(taxa))


@pytest.fixture(scope="module")
def traps(
    environment: TopologyEnvironment, taxa: list[str], maximum: float
) -> list[Topology]:
    """The local optima that are not the global one."""
    return [
        topology
        for topology in enumerate_topologies(taxa)
        if environment.is_terminal(topology)
        and abs(environment.score(topology) - maximum) >= 1e-9
    ]


def _hill_climbing_policy(environment: TopologyEnvironment) -> LinearPolicy:
    """A policy whose greedy action is the best-rewarded one.

    The environment's single feature is the improvement a move buys, so any
    positive weight makes ``greedy`` pick the highest-improvement action --
    hill climbing exactly. A zero-weight policy would not: see
    ``test_wrapping_an_untrained_policy_is_not_hill_climbing``.
    """
    policy = LinearPolicy(environment.n_features())
    policy.set_weights(torch.tensor([1.0], dtype=torch.float64))
    return policy


def _best_seen(environment: TopologyEnvironment, states: tuple[Topology, ...]) -> float:
    """A wandering searcher keeps its best state, not its last."""
    return max(environment.score(state) for state in states)


def test_wrapping_an_untrained_policy_is_not_hill_climbing(
    environment: TopologyEnvironment, traps: list[Topology]
) -> None:
    # A trap for the next caller, pinned rather than left to be rediscovered.
    # `EpsilonGreedyPolicy` takes the *wrapped policy's* greedy action, and an
    # untrained `LinearPolicy` scores every action alike, so `greedy` returns
    # the first one. The wrapper then explores around "always take action 0",
    # which looks like a searcher and is not one.
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


def test_epsilon_zero_reproduces_hill_climbing_exactly(
    environment: TopologyEnvironment, traps: list[Topology]
) -> None:
    # The control. Without it, a rising escape rate is not attributable to
    # epsilon, because the thing epsilon is added to might not be the
    # baseline at all -- which is precisely how the first run of this
    # experiment went wrong.
    agent = EpsilonGreedyPolicy(_hill_climbing_policy(environment), 0.0)
    for start in traps[:3]:
        under_policy = rollout(
            environment, agent, np.random.default_rng(0), BUDGET, start=start
        )
        under_greedy = greedy_rollout(environment, start, BUDGET)
        assert under_policy.states == under_greedy.states


def test_an_episode_can_leave_a_local_optimum(
    environment: TopologyEnvironment, traps: list[Topology], maximum: float
) -> None:
    # The claim the ticket exists for, against a floor that cannot drift:
    # with `stop_at_local_optimum` left at its default every one of these
    # runs ends where it started, so the rate is exactly 0 by construction.
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
                        BUDGET,
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


def test_stopping_at_a_local_optimum_never_escapes(
    environment: TopologyEnvironment, traps: list[Topology], maximum: float
) -> None:
    # The floor the previous test is measured against, asserted rather than
    # asserted-about: under the default rule an episode started at a local
    # optimum has already terminated, so no exploration rate can help.
    agent = EpsilonGreedyPolicy(_hill_climbing_policy(environment), _HIGH_EPSILON)
    rng = np.random.default_rng(7)
    for trap in traps:
        episode = rollout(environment, agent, rng, BUDGET, start=trap)
        assert episode.states == (trap,)
        assert abs(environment.score(trap) - maximum) >= 1e-9


def test_random_restart_hill_climbing_solves_this_fixture(
    environment: TopologyEnvironment, params: SimulationParams, maximum: float
) -> None:
    # The result that matters more than the mechanism. #193 compared a policy
    # against a *single* greedy run, which reaches the maximum from 48% of
    # starts. Restarting greedy until the same budget is spent reaches it from
    # all of them, so the baseline Stage 2 has to beat on this fixture is 1.00
    # and not 0.48 -- and no epsilon measured here comes close.
    start_rng = np.random.default_rng(params.seed + 1000)
    restart_rng = np.random.default_rng(11)
    solved = []
    for _ in range(PROBE_STARTS):
        state, spent, seen = environment.reset(start_rng), 0, -np.inf
        while spent < BUDGET:
            episode = greedy_rollout(environment, state, BUDGET - spent)
            spent += max(len(episode.actions), 1)
            seen = max(seen, _best_seen(environment, episode.states))
            state = environment.reset(restart_rng)
        solved.append(abs(seen - maximum) < 1e-9)

    assert float(np.mean(solved)) == 1.0
