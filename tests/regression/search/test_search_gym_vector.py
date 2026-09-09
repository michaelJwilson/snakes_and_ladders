"""The batched rollout is the sequential one reordered, and its bookkeeping is Gymnasium's (issue #392).

One claim carries this module. ``rollout_batch`` with a single copy equals
repeated :func:`rollout` under the same generator, episode for episode and
therefore draw for draw; at ``n`` copies every episode it returns is one the
generator that owns that copy would have produced on its own. Batching
changes the order episodes arrive in and nothing about the episodes.

The wrappers are pinned separately, because they are what replaced the
counters: ``TimeLimit`` truncates on the decision the adapter's own counter
truncates on, and ``RecordEpisodeStatistics`` reports the return and length
the assembled :class:`Episode` carries.

Skips without the ``frameworks`` extra; the core suite does not install
``gymnasium``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.environment import Episode
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.rollout import rollout
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import RewardModel, TopologyEnvironment
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import FIXTURES_DIR

gymnasium = pytest.importorskip("gymnasium")
from snakes_and_ladders.search.gym import (  # noqa: E402
    rollout_batch,
    vector_environment,
)

FIELD = np.array([0.4, -0.1, -0.3])
FIXTURE = FIXTURES_DIR / "tree_search/ci.yaml"
BRANCH_LENGTH = 0.1629
HORIZON = 6
WEIGHTS = torch.tensor([0.3, -0.6], dtype=torch.float64)


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
    return environment, 2 * (len(dataset.alignment) - 3)


def _policy(n_features: int = 2) -> LinearPolicy:
    policy = LinearPolicy(n_features)
    policy.set_weights(WEIGHTS[:n_features])
    return policy


def _sequential[S, A](
    environment: object, policy: LinearPolicy, seed: int, episodes: int
) -> list[Episode[S, A]]:
    rng = np.random.default_rng(seed)
    return [
        rollout(environment, policy, rng, HORIZON)  # type: ignore[arg-type]
        for _ in range(episodes)
    ]


# --- the pin: one copy is the sequential rollout ------------------------


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_one_copy_reproduces_the_sequential_potts_rollout_draw_for_draw(
    seed: int,
) -> None:
    landscape, n_max = _potts()
    policy = _policy()
    episodes = 12
    batched = rollout_batch(
        landscape,
        policy,
        [np.random.default_rng(seed)],
        n_max=n_max,
        max_steps=HORIZON,
        episodes=episodes,
    )
    assert batched == _sequential(landscape, policy, seed, episodes)
    # The batch is not a set of empty episodes agreeing trivially.
    assert sum(len(episode.actions) for episode in batched) > 0


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 3])
def test_one_copy_reproduces_the_sequential_tree_rollout_draw_for_draw(
    seed: int,
) -> None:
    environment, n_max = _tree()
    policy = _policy(environment.n_features())
    episodes = 4
    batched = rollout_batch(
        environment,
        policy,
        [np.random.default_rng(seed)],
        n_max=n_max,
        max_steps=HORIZON,
        episodes=episodes,
    )
    assert batched == _sequential(environment, policy, seed, episodes)


@pytest.mark.oracle
@pytest.mark.parametrize("n", [2, 4, 8])
def test_every_episode_of_a_batch_is_the_one_its_own_generator_would_have_rolled(
    n: int,
) -> None:
    # A batch is the interleaving of n independent sequences: each copy reads
    # one generator and nothing else, so the episodes it contributes are that
    # generator's own, in that generator's order. How *many* each contributes
    # is not fixed --- copies whose episodes end sooner are reset sooner and
    # so run more of them --- which is why the claim is stated per sequence
    # and not per copy count.
    landscape, n_max = _potts()
    policy = _policy()
    wanted = 3 * n
    batched = rollout_batch(
        landscape,
        policy,
        list(np.random.default_rng(11).spawn(n)),
        n_max=n_max,
        max_steps=HORIZON,
        episodes=wanted,
    )
    assert len(batched) == wanted
    # One reference stream per copy, long enough that no copy can outrun it.
    remaining = [
        [rollout(landscape, policy, rng, HORIZON) for _ in range(wanted)]
        for rng in np.random.default_rng(11).spawn(n)
    ]
    for episode in batched:
        matched = next(
            (sequence for sequence in remaining if sequence and sequence[0] == episode),
            None,
        )
        assert matched is not None, "an episode no copy's generator would produce"
        matched.pop(0)
    # Every stream was drawn from, or the batch collapsed onto one copy and
    # the interleaving claim is untested.
    assert all(len(sequence) < wanted for sequence in remaining)


@pytest.mark.oracle
def test_a_terminal_start_is_the_empty_episode_the_sequential_rollout_records() -> None:
    # A start already at a local optimum ends before any decision. Gymnasium
    # autoresets on a `done` returned by `step` and there is none, so the
    # batched path has to restart the copy itself; the seed below reaches
    # such a start inside the batch.
    landscape, n_max = _potts(chain_length=3)
    policy = _policy()
    episodes = 24
    batched = rollout_batch(
        landscape,
        policy,
        [np.random.default_rng(5)],
        n_max=n_max,
        max_steps=HORIZON,
        episodes=episodes,
    )
    assert batched == _sequential(landscape, policy, 5, episodes)
    empty = [episode for episode in batched if not episode.actions]
    assert empty, "the fixture no longer draws a terminal start; the path is untested"
    assert all(episode.terminated for episode in empty)


# --- the wrappers that replaced the counters ----------------------------


@pytest.mark.structural
def test_timelimit_truncates_on_the_decision_the_adapter_s_own_counter_does() -> None:
    # Both are the same budget, so the wrapper is the batched path's
    # authority without being a second rule. Chain length 8 so the episode
    # reaches the budget rather than a local optimum.
    landscape, n_max = _potts(chain_length=8)
    policy = _policy()
    budget = 3
    vector = vector_environment(landscape, n_max=n_max, max_steps=budget, n=1)
    adapter = vector.env.envs[0].unwrapped  # type: ignore[attr-defined]
    adapter.np_random = np.random.default_rng(1)
    observations, _ = vector.reset()
    truncated = np.zeros(1, dtype=np.bool_)
    steps = 0
    while not truncated[0]:
        rows = torch.from_numpy(observations[0][: len(adapter.available)])
        index = policy.sample(rows, adapter.np_random)
        observations, _, terminated, truncated, _ = vector.step(np.array([index]))
        steps += 1
        assert not terminated[0]
    assert steps == budget
    assert adapter._steps == budget


@pytest.mark.structural
def test_the_recorded_return_and_length_are_the_assembled_episode_s() -> None:
    # `RecordEpisodeStatistics` is what `rollout_batch` reads its episode
    # boundaries from; this pins the numbers it reports against the episode
    # the sequential rollout produces from the same generator.
    landscape, n_max = _potts()
    policy = _policy()
    vector = vector_environment(landscape, n_max=n_max, max_steps=HORIZON, n=1)
    adapter = vector.env.envs[0].unwrapped  # type: ignore[attr-defined]
    adapter.np_random = np.random.default_rng(2)
    expected = rollout(landscape, policy, np.random.default_rng(2), HORIZON)
    observations, _ = vector.reset()
    infos: dict[str, object] = {}
    done = np.zeros(1, dtype=np.bool_)
    while not done[0]:
        rows = torch.from_numpy(observations[0][: len(adapter.available)])
        index = policy.sample(rows, adapter.np_random)
        observations, _, terminated, truncated, infos = vector.step(np.array([index]))
        done = terminated | truncated
    recorded = cast(
        "dict[str, np.ndarray[Any, np.dtype[np.float64]]]", infos["episode"]
    )
    assert int(recorded["l"][0]) == len(expected.actions)
    assert float(recorded["r"][0]) == pytest.approx(expected.total_reward)


# --- refusals ------------------------------------------------------------


@pytest.mark.edge_case
def test_an_empty_generator_list_or_a_non_positive_count_is_refused() -> None:
    landscape, n_max = _potts()
    policy = _policy()
    with pytest.raises(ValueError, match="at least one generator"):
        rollout_batch(landscape, policy, [], n_max=n_max, max_steps=HORIZON, episodes=1)
    with pytest.raises(ValueError, match="episodes must be >= 1"):
        rollout_batch(
            landscape,
            policy,
            [np.random.default_rng(0)],
            n_max=n_max,
            max_steps=HORIZON,
            episodes=0,
        )
    with pytest.raises(ValueError, match="n must be >= 1"):
        vector_environment(landscape, n_max=n_max, max_steps=HORIZON, n=0)
