"""A feature set beyond the improvement a move buys, and what it changes (issue #328).

One feature makes the policy an inverse temperature (#177). ``FeatureSet.FULL``
adds six columns a move computes without a fit, each refereed in
``test_search_rl.py`` and ``test_search_support.py``; every column varies over
a neighbourhood and a planted constant is refused (``learn/CLAUDE.md``); greedy
stays inside the class, as standardization cannot reorder a column. Issue
#178's comparison (640 episodes, 50 starts, exact sign test) is release-gated
(``docs/experiments/005-tree-policy-features.md``). The per-PR sibling (#401)
trains one seed on six taxa against the baseline record; there both sets reach
the maximum from every start, so "full ahead" stays at release.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from sal.fixtures import Scale, load_params
from sal.learn.policy import LinearPolicy
from sal.learn.reinforce import reinforce
from sal.learn.rollout import greedy_rollout, rollout
from sal.learn.tree import (
    FEATURE_NAMES,
    FeatureSet,
    RewardModel,
    TreeEnvironment,
)
from sal.sample.statistics import sign_test_p_value
from sal.sim.fixtures import baseline, fixture
from sal.sim.params import SimulationParams
from sal.sim.simulator import simulate_tree
from sal.sim.topology import (
    MoveSet,
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    normalized_robinson_foulds,
)
from sal.sim.tree import edges

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "tree_search/release.yaml"

# Issue #178's budget: 50 starts at seed + 1000, 30 decisions per episode, 640
# training episodes, 16 averaged rollouts per start, budget-matched to greedy.
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

# Realized over 16 seeds: single 0.465-0.535, full 0.695-0.849; full ahead in
# 16 of 16; single against greedy p = 0.79 (`docs/experiments/005-...`).
_RELEASE_MEAN = {FeatureSet.IMPROVEMENT: 0.487, FeatureSet.FULL: 0.796}
_RELEASE_P_FULL_VS_SINGLE = 3.05e-5


def _params() -> SimulationParams:
    return load_params(FIXTURE, SimulationParams)


def environment(
    params: SimulationParams, feature_set: FeatureSet
) -> tuple[TreeEnvironment, list[str]]:
    """The reward surface an agent sees, and the taxa it is over."""
    dataset = simulate_tree(params, np.random.default_rng(params.seed))
    alignment = dict(dataset.alignment)
    built = TreeEnvironment(
        alignment,
        params.n_states,
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
    built: TreeEnvironment, params: SimulationParams
) -> list[Topology]:
    rng = np.random.default_rng(params.seed + START_SEED_OFFSET)
    return [built.reset(rng) for _ in range(STARTS)]


def enumerated_maximum(built: TreeEnvironment, taxa: list[str]) -> float:
    return max(built.log_weight(topology) for topology in enumerate_topologies(taxa))


def _reached(built: TreeEnvironment, endpoints: list[Topology], best: float) -> float:
    return float(
        np.mean([abs(built.log_weight(state) - best) < 1e-9 for state in endpoints])
    )


def greedy_rate(built: TreeEnvironment, starts: list[Topology], best: float) -> float:
    return _reached(
        built,
        [
            greedy_rollout(built, start=start, max_steps=HORIZON).states[-1]
            for start in starts
        ],
        best,
    )


def policy_rate(
    built: TreeEnvironment,
    policy: LinearPolicy,
    starts: list[Topology],
    best: float,
    rng: np.random.Generator,
    rollouts_per_start: int,
) -> float:
    return _reached(
        built,
        [
            rollout(built, policy, rng, max_steps=HORIZON, start=start).states[-1]
            for start in starts
            for _ in range(rollouts_per_start)
        ],
        best,
    )


def trained_rate(
    feature_set: FeatureSet, seed: int, iterations: int, rollouts_per_start: int
) -> float:
    """Train one policy from ``seed`` and measure it from the shared starts.

    Self-contained for worker processes; the evaluation seed derives from ``seed``.
    """
    params = _params()
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
    params = _params()
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
    return _params()


@pytest.fixture(scope="module")
def full(params: SimulationParams) -> tuple[TreeEnvironment, list[str]]:
    return environment(params, FeatureSet.FULL)


@pytest.fixture(scope="module")
def starts(
    full: tuple[TreeEnvironment, list[str]], params: SimulationParams
) -> list[Topology]:
    return starting_topologies(full[0], params)


# --- the columns are identifiable ---------------------------------------


def _constant_columns(built: TreeEnvironment, states: list[Topology]) -> set[int]:
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


@pytest.mark.smoke
def test_every_full_set_column_varies_within_some_neighbourhood(
    full: tuple[TreeEnvironment, list[str]], starts: list[Topology]
) -> None:
    # The invariance rule of `learn/CLAUDE.md`, column by column: a feature the
    # softmax cancels in every state has no identifiable weight. Over the 50
    # starts every column varies somewhere, and the improvement, parsimony and
    # support columns vary in every one.
    built, _ = full
    assert built.n_features() == 7 == len(FEATURE_NAMES[FeatureSet.FULL])
    assert _constant_columns(built, starts) == set()
    always_varying = {0, 1, 2, 3}
    for state in starts:
        rows = built.features(state, built.actions(state))
        for column in always_varying:
            assert float(rows[:, column].std()) > 0.0, (state, column)


@pytest.mark.smoke
def test_a_planted_constant_column_is_refused(
    full: tuple[TreeEnvironment, list[str]], starts: list[Topology]
) -> None:
    # The normalized Robinson-Foulds distance from the state is 1 / (n - 3)
    # for every NNI neighbour, so as a column it is exactly the thing the
    # rule excludes -- and the check above must say so, or it checks nothing.
    built, _ = full

    class Planted:
        n_features = staticmethod(lambda: 8)
        actions = staticmethod(built.actions)

        @staticmethod
        def features(state: Topology, actions: list[Topology]) -> NDArray[np.float64]:
            distances = np.array(
                [[normalized_robinson_foulds(state, action)] for action in actions],
                dtype=np.float64,
            )
            assert np.all(distances == distances[0])
            return np.concatenate([built.features(state, actions), distances], axis=1)

    assert _constant_columns(Planted, starts) == {7}  # type: ignore[arg-type]


@pytest.mark.analytic
def test_shifting_a_column_by_a_constant_leaves_the_policy_unchanged(
    full: tuple[TreeEnvironment, list[str]], starts: list[Topology]
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
            shifted = rows.copy()
            shifted[:, column] += 3.5
            assert torch.allclose(
                policy.log_probabilities(shifted).detach(), reference, atol=1e-12
            )


@pytest.mark.oracle
def test_the_greedy_weights_reproduce_the_greedy_searcher(
    full: tuple[TreeEnvironment, list[str]], starts: list[Topology]
) -> None:
    # Greedy inside the class (`learn/CLAUDE.md`): improvement-only weights take
    # greedy's move at every state of its trajectories, checked on trajectories.
    built, _ = full
    policy = LinearPolicy(built.n_features())
    weights = torch.zeros(built.n_features(), dtype=torch.float64)
    weights[0] = 1.0
    policy.set_weights(weights)
    decisions = 0
    for start in starts:
        episode = greedy_rollout(built, start=start, max_steps=HORIZON)
        for state, action in zip(episode.states, episode.actions, strict=False):
            actions = built.actions(state)
            chosen = actions[policy.greedy(built.features(state, actions))]
            assert leaf_bipartitions(chosen) == leaf_bipartitions(action)
            decisions += 1
    assert decisions > STARTS


# --- the measurement -----------------------------------------------------


@pytest.mark.end2end
@pytest.mark.release
def test_the_full_set_is_ahead_of_the_single_feature_at_the_ci_budget(
    full: tuple[TreeEnvironment, list[str]], starts: list[Topology]
) -> None:
    # One seed, half the budget: within 0.1 of these rates, order kept. 33.1 s
    # on the reference host, over the 10 s cap, so release too (issue #372).
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
@pytest.mark.end2end
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


@pytest.mark.end2end
def test_both_feature_sets_train_away_from_the_recorded_untrained_rate() -> None:
    # The control and baseline are the committed record (untrained 0.17, hill
    # climbing 1.00), recomputed at release; a changed environment raises.
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
