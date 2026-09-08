"""The Gymnasium adapter reproduces the in-house rollout, and passes Farama's own checker (issue #322).

Two referees. ``gymnasium.utils.env_checker.check_env`` is the API's own
statement of what an environment must do -- seeded resets that agree,
observations inside their space, a deterministic step -- and it is run
rather than paraphrased. The round trip is ours: one seed, one policy, the
same episode through :func:`rollout` and through ``reset``/``step``, with
the same states, rewards and candidate count, so the adapter adds a
vocabulary and not a search.

Everything here skips without the ``frameworks`` extra; the core suite does
not install ``gymnasium``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.rollout import rollout
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import RewardModel, TopologyEnvironment
from snakes_and_ladders.search.topology import Topology, leaf_bipartitions
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FIXTURES_DIR

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402
from snakes_and_ladders.search.gym import GymnasiumEnvironment  # noqa: E402

FIELD = np.array([0.4, -0.1, -0.3])
FIXTURE = FIXTURES_DIR / "tree_search/ci.yaml"
BRANCH_LENGTH = 0.1629


def _potts(chain_length: int = 4) -> tuple[PottsLandscape, int]:
    landscape = PottsLandscape(coupling=0.75, field=FIELD, chain_length=chain_length)
    return landscape, chain_length * (landscape.n_states - 1)


def _tree() -> tuple[TopologyEnvironment, int]:
    params = load_simulation_params(FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    environment = TopologyEnvironment(
        dict(dataset.alignment),
        params.k,
        params.pi,
        BRANCH_LENGTH,
        reward=RewardModel.KNOWN,
        moves=MoveSet.NNI,
    )
    # An unrooted tree on n taxa has 2 (n - 3) NNI neighbours before
    # deduplication, so no state can exceed this.
    return environment, 2 * (len(dataset.alignment) - 3)


def _round_trip[S, A](
    environment: Environment[S, A],
    n_max: int,
    policy: LinearPolicy,
    seed: int,
    max_steps: int,
) -> tuple[list[S], list[float], int, bool, bool]:
    """One episode through the Gymnasium interface, actions drawn from the adapter's generator."""
    adapter = GymnasiumEnvironment(environment, n_max=n_max, max_steps=max_steps)
    observation, info = adapter.reset(seed=seed)
    states, rewards = [adapter.state], []
    terminated, truncated = bool(info["terminal"]), False
    while not terminated and not truncated:
        features = torch.from_numpy(observation[info["action_mask"]])
        index = policy.sample(features, adapter.np_random)
        observation, reward, terminated, truncated, info = adapter.step(index)
        assert info["action_valid"]
        states.append(adapter.state)
        rewards.append(reward)
    return states, rewards, int(info["evaluations"]), terminated, truncated


def _candidates_scored[S, A](environment: Environment[S, A], states: list[S]) -> int:
    return sum(len(environment.actions(state)) for state in states)


# --- Farama's checker ----------------------------------------------------


@pytest.mark.structural
def test_the_potts_adapter_passes_gymnasium_s_environment_checker() -> None:
    landscape, n_max = _potts()
    with warnings.catch_warnings():
        # The checker warns that the environment has no `spec` and so no
        # registered render modes; neither is a claim this adapter makes.
        warnings.simplefilter("ignore", UserWarning)
        check_env(GymnasiumEnvironment(landscape, n_max=n_max, max_steps=8))


@pytest.mark.structural
def test_the_tree_adapter_passes_gymnasium_s_environment_checker() -> None:
    environment, n_max = _tree()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check_env(GymnasiumEnvironment(environment, n_max=n_max, max_steps=8))


# --- the round trip -------------------------------------------------------


@pytest.mark.structural
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_an_episode_on_the_potts_chain_is_the_same_through_both_interfaces(
    seed: int,
) -> None:
    landscape, n_max = _potts()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, 0.9], dtype=torch.float64))
    episode = rollout(landscape, policy, np.random.default_rng(seed), max_steps=6)
    states, rewards, evaluations, terminated, truncated = _round_trip(
        landscape, n_max, policy, seed, max_steps=6
    )
    assert states == list(episode.states)
    assert_allclose(rewards, episode.rewards, atol=1e-12)
    assert terminated == episode.terminated
    assert truncated == (not episode.terminated)
    assert evaluations == _candidates_scored(landscape, list(episode.states))


@pytest.mark.structural
@pytest.mark.parametrize("seed", [0, 3])
def test_an_episode_on_the_tree_is_the_same_through_both_interfaces(
    seed: int,
) -> None:
    environment, n_max = _tree()
    policy = LinearPolicy(1)
    policy.set_weights(torch.tensor([2.0], dtype=torch.float64))
    episode = rollout(environment, policy, np.random.default_rng(seed), max_steps=4)
    states, rewards, evaluations, terminated, _ = _round_trip(
        environment, n_max, policy, seed, max_steps=4
    )
    assert [leaf_bipartitions(s) for s in states] == [
        leaf_bipartitions(s) for s in episode.states
    ]
    assert_allclose(rewards, episode.rewards, atol=1e-12)
    assert terminated == episode.terminated
    assert evaluations == _candidates_scored(environment, list(episode.states))


@pytest.mark.structural
def test_the_observation_is_the_neighbourhood_s_features_padded_with_zeros() -> None:
    landscape, n_max = _potts()
    adapter = GymnasiumEnvironment(landscape, n_max=n_max + 3, max_steps=5)
    observation, info = adapter.reset(seed=2)
    available = landscape.actions(adapter.state)
    assert info["action_mask"].sum() == len(available) == n_max
    assert_allclose(
        observation[: len(available)],
        landscape.features(adapter.state, available).numpy(),
        atol=0.0,
    )
    assert not observation[len(available) :].any()
    assert observation.shape == adapter.observation_space.shape
    assert observation in adapter.observation_space


# --- termination, truncation, refusal ------------------------------------


@pytest.mark.structural
def test_the_episode_truncates_at_the_decision_budget() -> None:
    # Weights that make downhill moves likely, so the episode does not
    # terminate before the budget bites -- the rollout test's construction.
    landscape, n_max = _potts(chain_length=8)
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([-1.0, -1.0], dtype=torch.float64))
    episode = rollout(landscape, policy, np.random.default_rng(1), max_steps=3)
    states, rewards, _, terminated, truncated = _round_trip(
        landscape, n_max, policy, 1, max_steps=3
    )
    assert not episode.terminated
    assert truncated
    assert not terminated
    assert len(rewards) == 3
    assert states == list(episode.states)


@pytest.mark.structural
def test_the_episode_terminates_at_a_local_optimum() -> None:
    landscape, n_max = _potts()
    policy = LinearPolicy(2)
    policy.set_weights(landscape.greedy_weights() * 50.0)
    start = (2, 1, 1, 0)
    adapter = GymnasiumEnvironment(landscape, n_max=n_max, max_steps=50)
    observation, info = adapter.reset(seed=2, options={"start": start})
    assert adapter.state == start
    steps = 0
    terminated = truncated = False
    while not terminated and not truncated:
        features = torch.from_numpy(observation[info["action_mask"]])
        index = policy.sample(features, adapter.np_random)
        observation, _, terminated, truncated, info = adapter.step(index)
        steps += 1
    assert terminated
    assert not truncated
    assert landscape.is_terminal(adapter.state)
    assert steps < 50
    episode = rollout(
        landscape, policy, np.random.default_rng(2), max_steps=50, start=start
    )
    assert adapter.state == episode.states[-1]
    assert steps == len(episode.actions)


@pytest.mark.edge_case
def test_a_start_at_a_local_optimum_is_terminal_before_any_decision() -> None:
    landscape, n_max = _potts()
    optimum = (0, 0, 0, 0)
    assert landscape.is_terminal(optimum)
    adapter = GymnasiumEnvironment(landscape, n_max=n_max, max_steps=5)
    _, info = adapter.reset(seed=0, options={"start": optimum})
    # What `rollout` reads to take no action; Gymnasium has no slot for it.
    assert info["terminal"]
    # Stepping anyway is a caller's choice: every move off a local maximum
    # loses, and the episode is then wherever that move landed.
    _, reward, terminated, _, _ = adapter.step(0)
    assert reward < 0.0
    assert terminated == landscape.is_terminal(adapter.state)


@pytest.mark.edge_case
def test_an_unknown_reset_option_is_refused() -> None:
    landscape, n_max = _potts()
    adapter = GymnasiumEnvironment(landscape, n_max=n_max, max_steps=5)
    with pytest.raises(ValueError, match="reset options are"):
        adapter.reset(seed=0, options={"seed": 3})


@pytest.mark.edge_case
def test_a_masked_action_is_refused_without_moving() -> None:
    landscape, n_max = _potts()
    adapter = GymnasiumEnvironment(landscape, n_max=n_max + 2, max_steps=4)
    _, info = adapter.reset(seed=1)
    before = adapter.state
    assert not info["action_mask"][n_max]
    _, reward, _, _, info = adapter.step(n_max)
    assert adapter.state == before
    assert reward == 0.0
    assert not info["action_valid"]
    # It spent a decision: an agent ignoring the mask still meets the budget.
    _, _, _, truncated, _ = adapter.step(n_max)
    _, _, _, truncated, _ = adapter.step(n_max)
    _, _, _, truncated, _ = adapter.step(n_max)
    assert truncated


@pytest.mark.edge_case
def test_a_neighbourhood_wider_than_n_max_is_refused() -> None:
    landscape, n_max = _potts()
    adapter = GymnasiumEnvironment(landscape, n_max=n_max - 1, max_steps=4)
    with pytest.raises(ValueError, match="exceeds n_max"):
        adapter.reset(seed=0)


@pytest.mark.edge_case
def test_a_non_positive_width_or_budget_is_refused() -> None:
    landscape, n_max = _potts()
    with pytest.raises(ValueError, match="n_max must be >= 1"):
        GymnasiumEnvironment(landscape, n_max=0, max_steps=4)
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        GymnasiumEnvironment(landscape, n_max=n_max, max_steps=0)


@pytest.mark.edge_case
def test_the_state_is_unavailable_before_the_first_reset() -> None:
    landscape, n_max = _potts()
    with pytest.raises(RuntimeError, match="call reset"):
        _ = GymnasiumEnvironment(landscape, n_max=n_max, max_steps=4).state


@pytest.mark.structural
def test_the_tree_state_is_a_topology() -> None:
    environment, n_max = _tree()
    adapter = GymnasiumEnvironment(environment, n_max=n_max, max_steps=2)
    adapter.reset(seed=0)
    assert isinstance(adapter.state, Topology)
    assert adapter.environment is environment
