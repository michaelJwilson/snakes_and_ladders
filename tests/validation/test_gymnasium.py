"""The Gymnasium adapter reproduces the in-house rollout and passes Farama's checker, in a subprocess (issues #322, #977).

Gymnasium runs in `validation.gymnasium`, and the adapter that was
`learn.gym.GymnasiumEnvironment` lives in its script, so this module replaces
`learn/test_search_gym.py`. Referees:

- ``gymnasium.utils.env_checker.check_env``, the API's own statement of what
  an environment must do, passes on the Potts chain and the tree search;
- one seed, one policy, the same episode through :func:`rollout` and through
  ``reset``/``step``: the same states, rewards, end flags and candidate
  count, on the Potts chain at three seeds and the tree at two;
- the observation is the neighbourhood's features padded with zeros;
- the episode truncates at the decision budget and terminates at a local
  optimum, from a given start, where the rollout does.

Dropped with the move, since the adapter left the package API and its
argument checks are no claim about the science: the refusals of an unknown
reset option, a masked action, a neighbourhood wider than ``n_max``, a
non-positive width or budget, and a state read before the first reset; and
the step taken after a terminal start, which asserted the environment's
reward sign and not the adapter.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.rollout import rollout
from snakes_and_ladders.validation import gymnasium

from tests._fixtures import FIXTURES_DIR
from tests._frameworks import requires
from tests.regression.learn.conftest import COUPLING, FIELD

pytestmark = [
    pytest.mark.validation,
    requires("gymnasium"),
]

TREE = gymnasium.Spec(
    kind="tree",
    fixture=str(FIXTURES_DIR / "tree_search/ci.yaml"),
    branch_length=0.1629,
)


def _potts(chain_length: int = 4) -> gymnasium.Spec:
    return gymnasium.Spec(
        kind="potts",
        coupling=COUPLING,
        field=tuple(FIELD.tolist()),
        chain_length=chain_length,
    )


def _rolled[S, A](
    environment: Environment[S, A],
    weights: list[float],
    seed: int,
    max_steps: int,
    start: S | None = None,
) -> tuple[list[str], tuple[float, ...], int, bool]:
    """The rollout's encoded states, rewards, candidates scored, and end flag."""
    policy = LinearPolicy(len(weights))
    policy.set_weights(torch.tensor(weights, dtype=torch.float64))
    episode = rollout(
        environment, policy, np.random.default_rng(seed), max_steps, start=start
    )
    scored = sum(len(environment.actions(state)) for state in episode.states)
    states = [gymnasium.encode(state) for state in episode.states]
    return states, episode.rewards, scored, episode.terminated


@pytest.mark.smoke
@pytest.mark.parametrize("spec", [_potts(), TREE], ids=["potts", "tree"])
def test_the_adapter_passes_gymnasium_s_environment_checker(
    spec: gymnasium.Spec,
) -> None:
    assert gymnasium.check(spec, max_steps=8)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("spec", "weights", "seeds", "max_steps"),
    [(_potts(), [0.3, 0.9], [0, 1, 7], 6), (TREE, [2.0], [0, 3], 4)],
    ids=["potts", "tree"],
)
def test_an_episode_is_the_same_through_both_interfaces(
    spec: gymnasium.Spec, weights: list[float], seeds: list[int], max_steps: int
) -> None:
    environment, _ = gymnasium.build(spec)
    theirs = gymnasium.episodes(spec, weights, seeds, max_steps=max_steps)
    for seed, episode in zip(seeds, theirs, strict=True):
        states, rewards, scored, terminated = _rolled(
            environment, weights, seed, max_steps
        )
        assert episode.states == states
        assert_allclose(episode.rewards, rewards, atol=1e-12)
        assert episode.terminated == terminated
        assert episode.truncated == (not terminated)
        assert episode.evaluations == scored


@pytest.mark.smoke
def test_the_observation_is_the_neighbourhood_s_features_padded_with_zeros() -> None:
    spec = _potts()
    environment, n_max = gymnasium.build(spec)
    (episode,) = gymnasium.episodes(spec, [0.3, 0.9], [2], max_steps=5, extra_width=3)
    start = tuple(int(s) for s in episode.states[0].split(","))
    available_actions = environment.actions(start)
    assert episode.mask.sum() == len(available_actions) == n_max
    assert episode.observation.shape == (n_max + 3, environment.n_features())
    assert_allclose(
        episode.observation[:n_max],
        environment.features(start, available_actions),
        atol=0.0,
    )
    assert not episode.observation[n_max:].any()


@pytest.mark.smoke
def test_the_episode_truncates_at_the_decision_budget() -> None:
    # Weights that make downhill moves likely, so the budget bites first.
    spec = _potts(chain_length=8)
    environment, _ = gymnasium.build(spec)
    weights = [-1.0, -1.0]
    (episode,) = gymnasium.episodes(spec, weights, [1], max_steps=3)
    states, _, _, terminated = _rolled(environment, weights, 1, 3)
    assert not terminated
    assert episode.truncated
    assert not episode.terminated
    assert episode.rewards.size == 3
    assert episode.states == states


@pytest.mark.smoke
def test_the_episode_terminates_at_a_local_optimum() -> None:
    spec = _potts()
    environment, _ = gymnasium.build(spec)
    weights = (environment.greedy_weights() * 50.0).tolist()
    start = (2, 1, 1, 0)
    (episode,) = gymnasium.episodes(spec, weights, [2], max_steps=50, start=start)
    states, _, _, terminated = _rolled(environment, weights, 2, 50, start=start)
    assert episode.states[0] == gymnasium.encode(start)
    assert episode.terminated
    assert not episode.truncated
    assert terminated
    assert episode.states == states
    assert episode.rewards.size < 50
