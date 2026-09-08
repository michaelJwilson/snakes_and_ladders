"""A feature set beyond the improvement a move buys, and what it changes (issue #328).

With one feature the policy is an inverse temperature and no optimizer can
learn what the class cannot express, which is why the hard fixture of issue
#177 measured every policy at hill climbing's rate. ``FeatureSet.FULL`` adds
six columns a move already computes without a fit. Three things are pinned
before any training: every column is refereed by an independent computation
in ``test_search_rl.py`` and ``test_search_support.py``; every column varies
across a neighbourhood, and a planted constant column is refused, because
``learn/CLAUDE.md`` says a constant is unidentifiable; and the greedy searcher
is still inside the policy class, since standardizing a column is a positive
affine map and cannot reorder it.

Then the measurement: issue #178's comparison on the hard fixture, single
feature against the full set, at 640 episodes and 50 starts, with an exact
sign test over training seeds. The per-pull-request test runs one seed at a
sixth of the budget and pins its rates; the 16-seed run at the full budget is
release-gated, and ``docs/experiments/005-tree-policy-features.md`` records
what it found.

Both of those are release-gated, so what remains per pull request is the
sibling at the end of this module (issue #401): the same training, one seed,
on the six-taxon fixture, against the untrained rate and the hill-climbing
rate read from that fixture's baseline record rather than measured again. It
pins that the training runs and lands where it is expected to; it cannot pin
the full set *ahead* of the single feature, because at six taxa both reach
the enumerated maximum from every start and there is no order left to see.
That claim has no fast form and stays at the release gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from snakes_and_ladders.fixtures import Scale
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout
from snakes_and_ladders.search.infer import MoveSet
from snakes_and_ladders.search.rl import (
    FEATURE_NAMES,
    FeatureSet,
    RewardModel,
    TopologyEnvironment,
)
from snakes_and_ladders.search.statistics import sign_test_p_value
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    normalized_robinson_foulds,
)
from snakes_and_ladders.sim.fixtures import baseline, fixture
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import edges

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "tree_search/release.yaml"

# Issue #178's budget, unchanged so the comparison is the one that ticket
# specified: 50 starts at seed + 1000 and 30 decisions per episode, as
# `test_search_hard_fixture.py` measures greedy; 640 training episodes; and
# 16 rollouts per start for the stochastic policy, averaged rather than
# best-of, so the per-episode rate is budget-matched against greedy's one
# deterministic run.
HORIZON = 30
STARTS = 50
START_SEED_OFFSET = 1000
ITERATIONS = 40
BATCH = 16
ROLLOUTS_PER_START = 16
TRAINING_SEEDS = tuple(range(16))

# The per-pull-request tier: one seed, 320 episodes, 4 rollouts per start.
# Half the budget, because at a quarter of it the full set has not trained
# (0.375 at 160 episodes against 0.595 at 320) and a pin there would measure
# the budget. Its numbers are its own; the release tier's are the experiment's.
CI_SEEDS = (0,)
CI_ITERATIONS = 20
CI_ROLLOUTS_PER_START = 4

# Realized at the per-pull-request budget on the committed fixture, seed 0.
# Greedy is deterministic and the hard fixture's 0.48; an untrained policy is
# uniform over the moves.
_GREEDY = 0.48
_UNTRAINED = 0.02
_CI_RATES = {FeatureSet.IMPROVEMENT: 0.435, FeatureSet.FULL: 0.595}

# Realized at the full budget over the 16 seeds: the mean rate per feature
# set (single 0.465-0.535, full 0.695-0.849) and the exact sign test of the
# full set against the single feature, ahead in 16 of 16 seeds; the single
# feature against greedy is p = 0.79. The per-seed table is
# `docs/experiments/005-tree-policy-features.md`.
_RELEASE_MEAN = {FeatureSet.IMPROVEMENT: 0.487, FeatureSet.FULL: 0.796}
_RELEASE_P_FULL_VS_SINGLE = 3.05e-5


def load_params() -> SimulationParams:
    return load_simulation_params(FIXTURE)


def environment(
    params: SimulationParams, feature_set: FeatureSet
) -> tuple[TopologyEnvironment, list[str]]:
    """The reward surface an agent sees, and the taxa it is over."""
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)
    built = TopologyEnvironment(
        alignment,
        params.k,
        np.asarray(params.pi),
        branch_length=float(
            np.mean([child.branch_length for _, child in edges(params.tau)])
        ),
        reward=RewardModel.KNOWN,
        moves=MoveSet.NNI,
        features=feature_set,
    )
    return built, sorted(alignment)


def starting_topologies(
    built: TopologyEnvironment, params: SimulationParams
) -> list[Topology]:
    rng = np.random.default_rng(params.seed + START_SEED_OFFSET)
    return [built.reset(rng) for _ in range(STARTS)]


def enumerated_maximum(built: TopologyEnvironment, taxa: list[str]) -> float:
    return max(built.score(topology) for topology in enumerate_topologies(taxa))


def _reached(
    built: TopologyEnvironment, endpoints: list[Topology], best: float
) -> float:
    return float(
        np.mean([abs(built.score(state) - best) < 1e-9 for state in endpoints])
    )


def greedy_rate(
    built: TopologyEnvironment, starts: list[Topology], best: float
) -> float:
    return _reached(
        built,
        [greedy_rollout(built, start, HORIZON).states[-1] for start in starts],
        best,
    )


def policy_rate(
    built: TopologyEnvironment,
    policy: LinearPolicy,
    starts: list[Topology],
    best: float,
    rng: np.random.Generator,
    rollouts_per_start: int,
) -> float:
    return _reached(
        built,
        [
            rollout(built, policy, rng, HORIZON, start=start).states[-1]
            for start in starts
            for _ in range(rollouts_per_start)
        ],
        best,
    )


def trained_rate(
    feature_set: FeatureSet, seed: int, iterations: int, rollouts_per_start: int
) -> float:
    """Train one policy from ``seed`` and measure it from the shared starts.

    Self-contained on purpose: the release run spreads seeds over processes,
    and a process must be able to build everything it needs from the
    fixture alone. The evaluation generator is seeded from the training
    seed, so a seed is one number.
    """
    params = load_params()
    built, taxa = environment(params, feature_set)
    policy = LinearPolicy(built.n_features())
    reinforce(
        built,
        policy,
        np.random.default_rng(seed),
        iterations=iterations,
        batch=BATCH,
        max_steps=HORIZON,
    )
    return policy_rate(
        built,
        policy,
        starting_topologies(built, params),
        enumerated_maximum(built, taxa),
        np.random.default_rng(10_000 + seed),
        rollouts_per_start,
    )


@dataclass(frozen=True)
class Comparison:
    """Issue #178's comparison, per training seed, with its sign tests."""

    greedy: float
    single: tuple[float, ...]
    full: tuple[float, ...]

    @property
    def p_full_vs_single(self) -> float:
        return sign_test_p_value(np.array(self.full) - np.array(self.single))

    @property
    def p_full_vs_greedy(self) -> float:
        return sign_test_p_value(np.array(self.full) - self.greedy)

    @property
    def p_single_vs_greedy(self) -> float:
        return sign_test_p_value(np.array(self.single) - self.greedy)


def assemble(
    seeds: tuple[int, ...],
    rates: dict[tuple[FeatureSet, int], float],
) -> Comparison:
    """The comparison from per-(feature set, seed) rates, however they were run."""
    params = load_params()
    built, taxa = environment(params, FeatureSet.IMPROVEMENT)
    return Comparison(
        greedy=greedy_rate(
            built, starting_topologies(built, params), enumerated_maximum(built, taxa)
        ),
        single=tuple(rates[FeatureSet.IMPROVEMENT, seed] for seed in seeds),
        full=tuple(rates[FeatureSet.FULL, seed] for seed in seeds),
    )


def compare(
    seeds: tuple[int, ...], iterations: int, rollouts_per_start: int
) -> Comparison:
    """Run the comparison serially; the release script runs the same in parallel."""
    rates = {
        (feature_set, seed): trained_rate(
            feature_set, seed, iterations, rollouts_per_start
        )
        for feature_set in (FeatureSet.IMPROVEMENT, FeatureSet.FULL)
        for seed in seeds
    }
    return assemble(seeds, rates)


@pytest.fixture(scope="module")
def params() -> SimulationParams:
    return load_params()


@pytest.fixture(scope="module")
def full(params: SimulationParams) -> tuple[TopologyEnvironment, list[str]]:
    return environment(params, FeatureSet.FULL)


@pytest.fixture(scope="module")
def starts(
    full: tuple[TopologyEnvironment, list[str]], params: SimulationParams
) -> list[Topology]:
    return starting_topologies(full[0], params)


# --- the columns are identifiable ---------------------------------------


def _constant_columns(built: TopologyEnvironment, states: list[Topology]) -> set[int]:
    """Columns that take one value across every action, in every one of ``states``."""
    constant: set[int] | None = None
    for state in states:
        actions = built.actions(state)
        rows = built.features(state, actions)
        here = {
            column
            for column in range(rows.shape[1])
            if float(rows[:, column].max() - rows[:, column].min()) == 0.0
        }
        constant = here if constant is None else constant & here
    return constant or set()


@pytest.mark.structural
def test_every_full_set_column_varies_within_some_neighbourhood(
    full: tuple[TopologyEnvironment, list[str]], starts: list[Topology]
) -> None:
    # The invariance rule of `learn/CLAUDE.md`, applied column by column: a
    # feature the softmax cancels in every state has no identifiable weight.
    # Over the 50 starts every column varies somewhere, and the improvement,
    # parsimony and support columns vary in every one.
    built, _ = full
    assert built.n_features() == 7 == len(FEATURE_NAMES[FeatureSet.FULL])
    assert _constant_columns(built, starts) == set()
    always_varying = {0, 1, 2, 3}
    for state in starts:
        rows = built.features(state, built.actions(state))
        for column in always_varying:
            assert float(rows[:, column].std()) > 0.0, (state, column)


@pytest.mark.edge_case
def test_a_planted_constant_column_is_refused(
    full: tuple[TopologyEnvironment, list[str]], starts: list[Topology]
) -> None:
    # The normalized Robinson-Foulds distance from the state is 1 / (n - 3)
    # for every NNI neighbour, so as a column it is exactly the thing the
    # rule excludes -- and the check above must say so, or it checks nothing.
    built, _ = full

    class Planted:
        n_features = staticmethod(lambda: 8)
        actions = staticmethod(built.actions)

        @staticmethod
        def features(state: Topology, actions: list[Topology]) -> torch.Tensor:
            distances = torch.tensor(
                [[normalized_robinson_foulds(state, action)] for action in actions],
                dtype=torch.float64,
            )
            assert torch.all(distances == distances[0])
            return torch.cat([built.features(state, actions), distances], dim=1)

    assert _constant_columns(Planted, starts) == {7}  # type: ignore[arg-type]


@pytest.mark.mathematical
def test_shifting_a_column_by_a_constant_leaves_the_policy_unchanged(
    full: tuple[TopologyEnvironment, list[str]], starts: list[Topology]
) -> None:
    # The invariance itself, for every new column: adding a constant to one
    # column of the neighbourhood's features shifts every score alike.
    built, _ = full
    policy = LinearPolicy(built.n_features())
    policy.set_weights(
        torch.linspace(-1.0, 1.0, built.n_features(), dtype=torch.float64)
    )
    for state in starts[:10]:
        rows = built.features(state, built.actions(state))
        reference = policy.log_probabilities(rows).detach()
        for column in range(rows.shape[1]):
            shifted = rows.clone()
            shifted[:, column] += 3.5
            assert torch.allclose(
                policy.log_probabilities(shifted).detach(), reference, atol=1e-12
            )


@pytest.mark.oracle
def test_the_greedy_weights_reproduce_the_greedy_searcher(
    full: tuple[TopologyEnvironment, list[str]], starts: list[Topology]
) -> None:
    # `learn/CLAUDE.md`: the greedy searcher must be inside the policy class.
    # Weight on the improvement column only, and the policy's argmax is the
    # move greedy takes at every state of every greedy trajectory -- the
    # standardization cannot reorder a column. Asserted on the trajectories
    # themselves rather than on a rollout, since a sampled rollout at a
    # large weight would only be greedy with high probability.
    built, _ = full
    policy = LinearPolicy(built.n_features())
    weights = torch.zeros(built.n_features(), dtype=torch.float64)
    weights[0] = 1.0
    policy.set_weights(weights)
    decisions = 0
    for start in starts:
        episode = greedy_rollout(built, start, HORIZON)
        for state, action in zip(episode.states, episode.actions, strict=False):
            actions = built.actions(state)
            chosen = actions[policy.greedy(built.features(state, actions))]
            assert leaf_bipartitions(chosen) == leaf_bipartitions(action)
            decisions += 1
    assert decisions > STARTS


# --- the measurement -----------------------------------------------------


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_full_set_is_ahead_of_the_single_feature_at_the_ci_budget(
    full: tuple[TopologyEnvironment, list[str]], starts: list[Topology]
) -> None:
    # One seed at half the budget. The control is the uniform policy; both
    # trained policies must leave it far behind, land within 0.1 of the rates
    # realized here, and keep their order, so a regression in either feature
    # set's training is visible. The 16-seed claim is the release run's. The
    # one seed measured 33.1 s on the reference host, over the 10 s cap, so
    # this runs at the release gate too (issue #372).
    built, taxa = full
    best = enumerated_maximum(built, taxa)
    assert greedy_rate(built, starts, best) == pytest.approx(_GREEDY)
    untrained = policy_rate(
        built,
        LinearPolicy(built.n_features()),
        starts,
        best,
        np.random.default_rng(99),
        CI_ROLLOUTS_PER_START,
    )
    assert abs(untrained - _UNTRAINED) < 0.03

    rates = {}
    for feature_set in (FeatureSet.IMPROVEMENT, FeatureSet.FULL):
        rate = trained_rate(
            feature_set, CI_SEEDS[0], CI_ITERATIONS, CI_ROLLOUTS_PER_START
        )
        assert rate > 10 * _UNTRAINED, f"{feature_set} did not train: {rate}"
        assert abs(rate - _CI_RATES[feature_set]) < 0.1, (feature_set, rate)
        rates[feature_set] = rate
    assert rates[FeatureSet.FULL] > rates[FeatureSet.IMPROVEMENT]
    assert rates[FeatureSet.FULL] > _GREEDY


@pytest.mark.release
@pytest.mark.simulated_truth
def test_the_full_set_against_the_single_feature_over_sixteen_seeds() -> None:
    # Issue #178's comparison at its budget, single feature against the full
    # set. The bounds are around the realized means at 0.05 and the sign
    # test's direction, not its exact p-value, so a numpy that reorders a
    # tie does not fail a true result.
    comparison = compare(TRAINING_SEEDS, ITERATIONS, ROLLOUTS_PER_START)
    assert comparison.greedy == pytest.approx(_GREEDY)
    for feature_set, rates in (
        (FeatureSet.IMPROVEMENT, comparison.single),
        (FeatureSet.FULL, comparison.full),
    ):
        assert abs(float(np.mean(rates)) - _RELEASE_MEAN[feature_set]) < 0.05
    assert (comparison.p_full_vs_single < 0.05) == (_RELEASE_P_FULL_VS_SINGLE < 0.05)


# --- the per-pull-request sibling (issue #401) ---------------------------

#: The six-taxon fixture, whose 105 topologies enumerate in 0.1 s against the
#: hard fixture's 945 in 1.4 s, and whose episodes are shorter because the NNI
#: neighbourhood is smaller. Small enough for the training above to run twice
#: inside the per-pull-request cap, and still refereed by enumeration.
SIBLING = "tree_search"
SIBLING_TIER = Scale.STRESS

# An eighth of the release budget: 80 episodes, and two rollouts per start
# rather than sixteen. Both feature sets reach every start's maximum at this
# budget, so a regression in either would have to survive a smaller one.
SIBLING_ITERATIONS = 10
SIBLING_BATCH = 8
SIBLING_ROLLOUTS_PER_START = 2
SIBLING_SEED = 0
#: Realized at that budget on the six-taxon fixture, both feature sets.
_SIBLING_RATE = 1.0


@pytest.mark.simulated_truth
def test_both_feature_sets_train_away_from_the_recorded_untrained_rate() -> None:
    # The fast sibling of the two release-tier measurements above. The
    # control and the baseline are the fixture's committed record -- the
    # untrained policy reaches the enumerated maximum on 0.17 of episodes and
    # hill climbing on 1.00 -- so the 6 s of uniform rollouts and the
    # enumeration behind them are not paid again; `infra/baselines.py`
    # recomputes both at the release gate, and a change to the environment
    # makes this read raise rather than serve a stale number.
    record = baseline(SIBLING, SIBLING_TIER)
    untrained, greedy = record.value("untrained_rate"), record.value("greedy_rate")
    assert untrained < 0.25 < greedy, (untrained, greedy)

    params = fixture(SIBLING, SIBLING_TIER).params
    rates = {}
    for feature_set in (FeatureSet.IMPROVEMENT, FeatureSet.FULL):
        built, _ = environment(params, feature_set)
        policy = LinearPolicy(built.n_features())
        reinforce(
            built,
            policy,
            np.random.default_rng(SIBLING_SEED),
            iterations=SIBLING_ITERATIONS,
            batch=SIBLING_BATCH,
            max_steps=HORIZON,
        )
        rng = np.random.default_rng(params.seed + START_SEED_OFFSET)
        starts = [built.reset(rng) for _ in range(STARTS)]
        rates[feature_set] = policy_rate(
            built,
            policy,
            starts,
            record.value("enumerated_maximum"),
            np.random.default_rng(10_000 + SIBLING_SEED),
            SIBLING_ROLLOUTS_PER_START,
        )

    for feature_set, rate in rates.items():
        assert rate > untrained + 0.5, (feature_set, rate)
        assert rate == pytest.approx(_SIBLING_RATE), (feature_set, rate)
    assert rates[FeatureSet.FULL] >= greedy
