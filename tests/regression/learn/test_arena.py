"""The table's harness: a seam that re-spells nothing, and a metric with a cost.

Issue #705. Four kinds of claim:

* **the registry re-spells, it does not re-implement.** Each learner reached
  through :data:`~snakes_and_ladders.learn.arena.LEARNERS` returns **bitwise**
  what calling its function directly returns under the same seed --- the same
  states, the same actions, the same rewards. That is the whole claim the table
  rests on: a harness that re-derived anything would make its rows
  incomparable with every number already published against these functions;
* **the metric reproduces the table it generalizes.** The Potts chain's five
  rows come back at 80.2 / 88.9 / 88.9 / 96.3 / 97.5% of its 81 starts (#313).
  A harness that cannot reproduce that is not measuring it;
* **greedy is a row and not a policy.** `greedy_rollout` stops at a local
  maximum and a rolled-out policy walks past one, so the two are different
  baselines and the suite pins that they differ rather than assuming the
  distinction;
* **the cost is reported in a stated unit.** `evaluations` is the mean scored
  actions per evaluation episode, counted from the trajectory, and greedy's is
  17.4 on the chain rather than the 48 the published comparison quotes ---
  which is ``max_steps x |actions|``, the budget greedy does not spend.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.actor_critic import actor_critic
from snakes_and_ladders.learn.arena import (
    CRITIC_HIDDEN,
    LEARNERS,
    MLP_HIDDEN,
    MLP_LEARNING_RATE,
    Learner,
    Outcome,
    Row,
    Scoring,
    Streams,
    TrainingBudget,
    evaluations,
    reached_optimum,
    row,
    table,
)
from snakes_and_ladders.learn.critic import Critic, n_state_features
from snakes_and_ladders.learn.environment import Episode
from snakes_and_ladders.learn.policy import (
    LinearPolicy,
    MLPPolicy,
    Policy,
    TrainablePolicy,
)
from snakes_and_ladders.learn.potts import (
    PottsEnvironment,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.ppo import ppo
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout

#: The chain issue #313's five rows were measured on.
FIELD = np.array([0.4, -0.1, -0.3])

#: What #313 gave each learned row.
PUBLISHED = TrainingBudget(iterations=60, batch=32, max_steps=6)

#: Enough training to prove a bitwise identity and not enough to be a result.
#: Identity does not depend on the budget, so the fast tier buys it at 1/20th
#: of the published one.
SHORT = TrainingBudget(iterations=3, batch=32, max_steps=6)

#: The five rows as issue #313 measured them, out of 81 starts.
CHAIN_ROWS = {
    "greedy": 65,
    "reinforce": 72,
    "actor-critic": 72,
    "ppo": 78,
    "mlp-ppo": 79,
}


def _environment() -> PottsEnvironment:
    return PottsEnvironment(coupling=0.75, field=FIELD, chain_length=4)


def _starts(environment: PottsEnvironment) -> list[tuple[int, ...]]:
    return list(
        enumerate_configurations(environment.n_states, environment.chain_length)
    )


def _streams(seed: int = 0) -> Streams:
    """The four streams the published rows were measured on.

    Written out here rather than defaulted in the package: the shape is a
    property of how #313 was run --- four independent generators, the two
    `torch` ones seeded alike --- and a harness that chose it silently would be
    choosing what the reproduction means.
    """
    return Streams(
        training=np.random.default_rng(seed),
        evaluation=np.random.default_rng(seed + 1),
        policy=torch.Generator().manual_seed(seed),
        critic=torch.Generator().manual_seed(seed),
    )


def _critic(environment: PottsEnvironment, seed: int) -> Critic:
    return Critic(
        n_state_features(environment),
        hidden=CRITIC_HIDDEN,
        generator=torch.Generator().manual_seed(seed),
    )


def _direct(
    name: str, environment: PottsEnvironment, budget: TrainingBudget, seed: int
) -> Policy:
    """The policy a caller writing the training call by hand would end up with.

    Spelled out per learner rather than dispatched, because that is the point:
    these are the four call shapes issue #705 found in four test files, and the
    registry has to reproduce each exactly.
    """
    rng = np.random.default_rng(seed)
    if name == "reinforce":
        policy = LinearPolicy(environment.n_features())
        reinforce(
            environment, policy, rng, budget.iterations, budget.batch, budget.max_steps
        )
        return policy
    if name == "actor-critic":
        policy = LinearPolicy(environment.n_features())
        actor_critic(
            environment,
            policy,
            _critic(environment, seed),
            rng,
            iterations=budget.iterations,
            batch=budget.batch,
            max_steps=budget.max_steps,
        )
        return policy
    trainable: TrainablePolicy = (
        LinearPolicy(environment.n_features())
        if name == "ppo"
        else MLPPolicy(
            environment.n_features(),
            hidden=MLP_HIDDEN,
            generator=torch.Generator().manual_seed(seed),
        )
    )
    ppo(
        environment,
        trainable,
        _critic(environment, seed),
        rng,
        iterations=budget.iterations,
        batch=budget.batch,
        max_steps=budget.max_steps,
        learning_rate=0.05 if name == "ppo" else MLP_LEARNING_RATE,
    )
    return trainable


# --- the seam re-spells nothing ------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["reinforce", "actor-critic", "ppo", "mlp-ppo"])
def test_a_registered_learner_is_bitwise_the_direct_call(name: str) -> None:
    """Same seed, same episodes: states, actions and rewards to the bit.

    Not a tolerance. The registry's job is to be a list of these four calls,
    so any difference at all is a second implementation --- and a second
    implementation is what makes a table's rows incomparable with the numbers
    published against the functions it wraps.
    """
    environment = _environment()
    starts = _starts(environment)
    through = LEARNERS[name].episodes(environment, starts, SHORT, _streams())
    policy = _direct(name, environment, SHORT, 0)
    rng = np.random.default_rng(1)
    direct = [
        rollout(environment, policy, rng, SHORT.max_steps, start=start)
        for start in starts
    ]

    assert [episode.states for episode in through] == [
        episode.states for episode in direct
    ]
    assert [episode.actions for episode in through] == [
        episode.actions for episode in direct
    ]
    assert [episode.rewards for episode in through] == [
        episode.rewards for episode in direct
    ]


@pytest.mark.smoke
def test_the_greedy_row_is_the_greedy_rollout_and_not_a_policy() -> None:
    """The row runs `greedy_rollout`, which stops where a policy walks on.

    Pinned rather than argued. `greedy_rollout` ends at a local maximum and a
    rolled-out policy takes the least-bad move and keeps going, so a "greedy
    policy" is a different baseline from the one 80.2% was measured on --- and
    on this fixture they differ on most starts, which is what makes the
    distinction load-bearing rather than pedantic.
    """
    environment = _environment()
    starts = _starts(environment)
    through = LEARNERS["greedy"].episodes(environment, starts, PUBLISHED, _streams())
    direct = [
        greedy_rollout(environment, start, PUBLISHED.max_steps) for start in starts
    ]

    assert not LEARNERS["greedy"].trains
    assert [episode.states for episode in through] == [
        episode.states for episode in direct
    ]
    trained = _direct("ppo", environment, SHORT, 0)
    rng = np.random.default_rng(1)
    walked = [
        rollout(environment, trained, rng, PUBLISHED.max_steps, start=start)
        for start in starts
    ]
    assert sum(episode.terminated for episode in direct) > sum(
        episode.terminated for episode in walked
    )


# --- the metric, and what it costs ---------------------------------------


@pytest.mark.analytic
def test_the_cost_is_the_scored_actions_and_greedy_spends_fewer_than_its_budget() -> (
    None
):
    """`evaluations` counts what an episode scored, not what it was allowed.

    Greedy's mean on the chain is **17.4** scored actions --- eight actions at
    each of 2.2 steps --- where the published comparison quotes **48**, which
    is ``max_steps x |actions|``: the budget, which greedy does not spend
    because it stops at a local maximum. The two readings are both defensible
    and they are not the same number, so the unit is stated here and the
    figure with it.
    """
    environment = _environment()
    starts = _starts(environment)
    episodes = [
        greedy_rollout(environment, start, PUBLISHED.max_steps) for start in starts
    ]
    per_state = len(environment.actions(starts[0]))
    counted = [evaluations(environment, episode) for episode in episodes]

    assert per_state == 8
    assert counted == [per_state * len(episode.actions) for episode in episodes]
    assert abs(float(np.mean(counted)) - 17.4) < 0.05
    assert float(np.mean(counted)) < per_state * PUBLISHED.max_steps


@pytest.mark.analytic
def test_the_two_scorings_differ_exactly_where_an_episode_walks_out() -> None:
    """`VISITED` is the module's best-state rule, `FINAL` the published one.

    Constructed rather than trained, because the case that separates them is
    the one a fixture may not contain: an episode that touches the optimum and
    then leaves it. On the chain the five rows agree under both, which is
    measured in the row test below and is a property of that fixture, not of
    the metric.
    """
    episode: Episode[int, int] = Episode(
        states=(0, 1, 2), actions=(0, 1), rewards=(1.0, -1.0), terminated=False
    )
    value = float

    assert reached_optimum(episode, value, 1.0, scoring=Scoring.VISITED)
    assert not reached_optimum(episode, value, 1.0, scoring=Scoring.FINAL)
    assert reached_optimum(episode, value, 2.0, scoring=Scoring.FINAL)


@pytest.mark.analytic
def test_a_row_reads_beat_match_or_lose_against_its_baseline() -> None:
    """Three outcomes, compared as fractions so unequal start counts are safe.

    A match is the outcome the ticket exists for, and it stays a match when it
    cost less: the cost sits beside the outcome rather than inside it, so a
    reader sees "the same answer at a fraction of the price" as a sentence
    rather than as a rank the comparison invented.
    """
    baseline = Row("greedy", 65, 81, 17.4, 0, Scoring.VISITED)
    better = Row("ppo", 78, 81, 23.9, 11_520, Scoring.VISITED)
    same = Row("reinforce", 65, 81, 9.0, 11_520, Scoring.VISITED)
    coarser = Row("other", 130, 162, 9.0, 11_520, Scoring.VISITED)

    assert better.against(baseline) is Outcome.BEAT
    assert baseline.against(better) is Outcome.LOSE
    assert same.against(baseline) is Outcome.MATCH
    assert coarser.against(baseline) is Outcome.MATCH
    assert same.evaluations < baseline.evaluations


@pytest.mark.smoke
def test_a_budget_is_three_positive_numbers_and_reports_their_product() -> None:
    """The decision count a row states, and the refusal that keeps it meaningful."""
    assert PUBLISHED.decisions == 60 * 32 * 6
    for bad in ((0, 32, 6), (60, 0, 6), (60, 32, 0)):
        with pytest.raises(ValueError, match="must be >= 1"):
            TrainingBudget(*bad)


@pytest.mark.smoke
def test_the_table_names_its_rows_and_refuses_one_it_does_not_carry() -> None:
    """A partial table says which rows it is; an unknown name is not silent."""
    environment = _environment()
    starts = _starts(environment)
    best = optimum(environment)[1]
    partial = table(
        environment,
        starts,
        environment.energy,
        best,
        SHORT,
        _streams,
        names=["greedy"],
    )

    assert [entry.name for entry in partial] == ["greedy"]
    assert list(LEARNERS) == ["greedy", "reinforce", "actor-critic", "ppo", "mlp-ppo"]
    with pytest.raises(KeyError, match="unregistered learners"):
        table(
            environment,
            starts,
            environment.energy,
            best,
            SHORT,
            _streams,
            names=["sarsa"],
        )


@pytest.mark.smoke
def test_an_untrained_row_spends_no_training_decisions() -> None:
    """A row states what it spent, and greedy spent none."""
    environment = _environment()
    starts = _starts(environment)
    best = optimum(environment)[1]
    greedy = row(
        LEARNERS["greedy"],
        environment,
        starts,
        environment.energy,
        best,
        PUBLISHED,
        _streams(),
    )

    assert greedy.training_decisions == 0
    assert greedy.reached == CHAIN_ROWS["greedy"]
    assert isinstance(LEARNERS["greedy"], Learner)


# --- the rows the harness has to reproduce -------------------------------


@pytest.mark.release
@pytest.mark.end2end
def test_the_chains_five_rows_come_back_at_the_published_fractions() -> None:
    """80.2 / 88.9 / 88.9 / 96.3 / 97.5% of 81 starts, from one call (#313).

    The reproduction the harness exists to pass. Each row was measured in its
    own test file against its own hand-written training call; here they come
    from :func:`~snakes_and_ladders.learn.arena.table` and agree to the start.

    Measured with it: the two scorings give the *same* fraction for all five
    rows on this fixture, so nothing here depends on which was chosen --- a
    policy that touches the chain's optimum inside six steps does not leave it
    again. The chain's evaluations are 17.4 (greedy), 23.5, 23.5, 23.9 and
    21.9.
    """
    environment = _environment()
    starts = _starts(environment)
    best = optimum(environment)[1]

    for scoring in (Scoring.VISITED, Scoring.FINAL):
        rows = table(
            environment,
            starts,
            environment.energy,
            best,
            PUBLISHED,
            _streams,
            scoring=scoring,
        )
        assert {entry.name: entry.reached for entry in rows} == CHAIN_ROWS
        assert all(entry.starts == 81 for entry in rows)

    rows = table(environment, starts, environment.energy, best, PUBLISHED, _streams)
    baseline = rows[0]
    assert [entry.against(baseline) for entry in rows[1:]] == [Outcome.BEAT] * 4
