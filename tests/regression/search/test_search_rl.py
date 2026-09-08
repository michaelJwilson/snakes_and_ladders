"""The phylogenetic environment: its rewards, its caching, and its boundaries.

Two things are worth asserting here that neither `snakes_and_ladders.learn` nor
`snakes_and_ladders.search.infer` can assert for itself. That the environment's reward is
a difference of the log-likelihood it names -- both reward models, each
against the scorer it claims to call -- and that the cheap model is cheap for
the stated reason rather than by accident: it memoizes on a key that
recognizes the same topology however it is spelled, and it evaluates a closed
form rather than an optimization.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.rollout import greedy_rollout, rollout
from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.likelihood.pruning_torch import branch_order
from snakes_and_ladders.likelihood.pruning_torch import (
    log_likelihood as log_likelihood_torch,
)
from snakes_and_ladders.search.infer import Model, MoveSet, score_topology
from snakes_and_ladders.search.rl import (
    FEATURE_NAMES,
    FeatureSet,
    RewardModel,
    TopologyEnvironment,
    exchanged_subtrees,
    standardize,
    with_uniform_branch_lengths,
)
from snakes_and_ladders.search.support import internal_splits, split_pattern_support
from snakes_and_ladders.search.topology import (
    Topology,
    branch_splits,
    enumerate_topologies,
    leaf_bipartitions,
    nni_neighbours,
)
from snakes_and_ladders.sim.gtr import gtr_rate_matrix
from snakes_and_ladders.sim.jc import jc_rate_matrix
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR
from tests._scale import stress_only

FIXTURE = FIXTURES_DIR / "simulation_params_5taxa.yaml"
_BRANCH_LENGTH = 0.1629


def _params() -> SimulationParams:
    return load_simulation_params(FIXTURE)


def _alignment(params: SimulationParams) -> dict[str, np.ndarray]:
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    return dict(dataset.alignment)


def _environment(
    reward: RewardModel = RewardModel.KNOWN,
    moves: MoveSet = MoveSet.NNI,
    features: FeatureSet = FeatureSet.IMPROVEMENT,
) -> tuple[TopologyEnvironment, SimulationParams, dict[str, np.ndarray]]:
    params = _params()
    alignment = _alignment(params)
    environment = TopologyEnvironment(
        alignment,
        params.k,
        params.pi,
        _BRANCH_LENGTH,
        reward=reward,
        moves=moves,
        features=features,
    )
    return environment, params, alignment


# A general-Q truth for the known reward: unequal exchangeabilities and a
# non-uniform stationary distribution, so nothing about it reduces to
# Jukes-Cantor by accident.
_GTR_PI = np.array([0.1, 0.2, 0.3, 0.4])
_GTR_EXCHANGEABILITIES = np.array([1.5, 0.5, 1.0, 0.8, 2.0, 1.0])


# 200 sites rather than the fixture's 1200: the fitted-GTR pin below fits
# every parameter of the model per topology, and its claim is an inequality
# that holds at any site count.
_GTR_SITES = 200


def _gtr_alignment(
    params: SimulationParams,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rate_matrix = gtr_rate_matrix(_GTR_EXCHANGEABILITIES, _GTR_PI)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=_GTR_PI,
        rng=np.random.default_rng(params.seed),
        n_sites=_GTR_SITES,
        rate_matrix=rate_matrix,
    )
    return dict(dataset.alignment), rate_matrix


def _mirrored(topology: Topology) -> Topology:
    """The same topology with every child list reversed.

    A different spelling of the same tree, which is what the memoization key
    has to see through.
    """
    return Node(
        name=topology.name,
        branch_length=topology.branch_length,
        children=tuple(_mirrored(child) for child in reversed(topology.children)),
    )


@pytest.mark.structural
def test_the_environment_satisfies_the_protocol() -> None:
    environment, _, _ = _environment()
    assert isinstance(environment, Environment)


# --- uniform branch lengths ----------------------------------------------


@pytest.mark.structural
def test_uniform_branch_lengths_relabel_every_edge_but_the_root() -> None:
    params = _params()
    labelled = with_uniform_branch_lengths(params.tau, 0.25)
    lengths = [node.branch_length for node in preorder(labelled)]
    assert lengths[0] is None
    assert lengths[1:] == [0.25] * (len(lengths) - 1)
    # Shape preserved: same leaves, same splits.
    assert leaf_bipartitions(labelled) == leaf_bipartitions(params.tau)


@pytest.mark.edge_case
@pytest.mark.parametrize("branch_length", [0.0, -0.1])
def test_a_non_positive_branch_length_is_rejected(branch_length: float) -> None:
    # A zero-length branch identifies two nodes, and a likelihood evaluated
    # there no longer distinguishes the topology it was given.
    with pytest.raises(ValueError, match="branch_length must be > 0"):
        with_uniform_branch_lengths(_params().tau, branch_length)


# --- the rewards are the log-likelihoods they claim to be ----------------


@pytest.mark.oracle
def test_the_known_score_is_the_likelihood_at_a_fixed_branch_length() -> None:
    # Against a direct call to the pruning recursion, built independently
    # here: the environment is a wrapper, and this is the claim that it wraps
    # what it says it does.
    environment, params, alignment = _environment(RewardModel.KNOWN)
    for topology in enumerate_topologies(sorted(alignment)):
        expected = log_likelihood(
            with_uniform_branch_lengths(topology, _BRANCH_LENGTH),
            params.k,
            params.pi,
            alignment,
        )
        assert_allclose(environment.score(topology), expected, rtol=1e-12)


@pytest.mark.oracle
def test_the_fitted_score_is_the_maximized_likelihood() -> None:
    environment, params, alignment = _environment(RewardModel.FITTED)
    topology = next(iter(enumerate_topologies(sorted(alignment))))
    assert_allclose(
        environment.score(topology),
        score_topology(topology, alignment, params.k),
        rtol=1e-9,
    )


@pytest.mark.mathematical
def test_the_fitted_score_is_never_below_the_known_one() -> None:
    # The known surface fixes the branch lengths the fitted one optimizes, so
    # it can only do worse. This is the sense in which the cheap reward is a
    # different surface rather than a noisy estimate of the same one, and it
    # holds topology by topology.
    _, params, alignment = _environment()
    known, fitted = (
        TopologyEnvironment(
            alignment, params.k, params.pi, _BRANCH_LENGTH, reward=reward
        )
        for reward in (RewardModel.KNOWN, RewardModel.FITTED)
    )
    for topology in enumerate_topologies(sorted(alignment)):
        assert fitted.score(topology) >= known.score(topology) - 1e-9


@pytest.mark.mathematical
def test_a_reward_is_the_improvement_it_reports() -> None:
    environment, _, alignment = _environment()
    state = next(iter(enumerate_topologies(sorted(alignment))))
    for action in environment.actions(state):
        successor, reward = environment.step(state, action)
        assert leaf_bipartitions(successor) == leaf_bipartitions(action)
        assert_allclose(
            reward, environment.score(action) - environment.score(state), atol=1e-12
        )


@pytest.mark.mathematical
def test_an_episode_return_telescopes_to_its_total_improvement() -> None:
    environment, _, alignment = _environment()
    start = next(iter(enumerate_topologies(sorted(alignment))))
    episode = greedy_rollout(environment, start, max_steps=10)
    assert_allclose(
        episode.total_reward,
        environment.score(episode.states[-1]) - environment.score(start),
        atol=1e-9,
    )


# --- what the policy sees --------------------------------------------------


@pytest.mark.structural
def test_the_only_feature_is_the_reward_the_move_would_buy() -> None:
    # One feature, so the policy is a Boltzmann distribution over moves whose
    # single weight is an inverse temperature. This is the claim that makes
    # the greedy searcher its zero-temperature limit, so it is pinned rather
    # than left to the docstring.
    environment, _, alignment = _environment()
    state = next(iter(enumerate_topologies(sorted(alignment))))
    actions = environment.actions(state)
    features = environment.features(state, actions)

    assert environment.n_features() == 1
    assert features.shape == (len(actions), 1)
    assert_allclose(
        features[:, 0].numpy(),
        [environment.step(state, action)[1] for action in actions],
        atol=1e-12,
    )


@pytest.mark.mathematical
def test_a_policy_rollout_telescopes_like_the_greedy_one() -> None:
    # The environment exists to be driven by a policy, not only by the
    # baseline; this is the path `snakes_and_ladders.learn.rollout.rollout` takes through
    # it. At a large positive weight the policy is effectively greedy, so the
    # two agree -- the same zero-temperature limit checked on the Potts
    # landscape, now on trees.
    environment, _, alignment = _environment()
    start = next(iter(enumerate_topologies(sorted(alignment))))
    policy = LinearPolicy(1)
    policy.set_weights(torch.tensor([200.0], dtype=torch.float64))

    episode = rollout(environment, policy, np.random.default_rng(0), 20, start=start)

    assert_allclose(
        episode.total_reward,
        environment.score(episode.states[-1]) - environment.score(start),
        atol=1e-9,
    )
    assert episode.actions == greedy_rollout(environment, start, 20).actions


# --- caching --------------------------------------------------------------


@pytest.mark.structural
def test_a_topology_is_scored_once_however_it_is_spelled() -> None:
    # `leaf_bipartitions` is rooting- and child-order-independent, so a tree
    # reached by two different move sequences costs one evaluation. Without
    # this the cheap reward is not cheap: `is_terminal` alone scores the whole
    # neighbourhood, and an SPR neighbourhood overlaps its predecessor
    # heavily.
    environment, _, alignment = _environment()
    topology = next(iter(enumerate_topologies(sorted(alignment))))
    first = environment.score(topology)
    assert environment.evaluations == 1
    assert environment.score(topology) == first
    assert environment.score(_mirrored(topology)) == first
    assert environment.evaluations == 1


@pytest.mark.structural
def test_the_neighbourhood_excludes_the_state_and_repeats() -> None:
    environment, _, alignment = _environment()
    state = next(iter(enumerate_topologies(sorted(alignment))))
    actions = environment.actions(state)
    keys = [leaf_bipartitions(action) for action in actions]
    assert leaf_bipartitions(state) not in keys
    assert len(set(keys)) == len(keys)
    # Every neighbour the move set produces is still represented.
    assert set(keys) == {leaf_bipartitions(n) for n in nni_neighbours(state)} - {
        leaf_bipartitions(state)
    }


# --- terminal states and the baseline ------------------------------------


@pytest.mark.structural
def test_a_terminal_state_is_one_no_move_improves() -> None:
    environment, _, alignment = _environment()
    for topology in enumerate_topologies(sorted(alignment)):
        improvable = any(
            environment.score(action) > environment.score(topology)
            for action in environment.actions(topology)
        )
        assert environment.is_terminal(topology) is not improvable


@pytest.mark.oracle
def test_greedy_search_reaches_the_enumerated_optimum() -> None:
    # The exhaustive oracle, on the surface the agent actually sees. Measured:
    # hill climbing reaches it from all 15 starts here and recovers the
    # generating topology, which is the same saturation issue #128 found on
    # the 6-taxon fitted surface -- so this fixture validates the environment,
    # not any policy's advantage over the baseline.
    environment, params, alignment = _environment()
    topologies = list(enumerate_topologies(sorted(alignment)))
    best = max(environment.score(topology) for topology in topologies)
    for start in topologies:
        episode = greedy_rollout(environment, start, max_steps=20)
        assert environment.score(episode.states[-1]) == pytest.approx(best)
        assert episode.terminated


@pytest.mark.simulated_truth
def test_the_known_optimum_is_the_generating_topology_here() -> None:
    environment, params, alignment = _environment()
    scored = {
        leaf_bipartitions(t): environment.score(t)
        for t in enumerate_topologies(sorted(alignment))
    }
    best = max(scored, key=lambda key: scored[key])
    assert best == leaf_bipartitions(params.tau)


# --- boundaries -----------------------------------------------------------


@pytest.mark.structural
def test_reset_is_reproducible_from_its_seed() -> None:
    environment, _, _ = _environment()
    first = environment.reset(np.random.default_rng(3))
    second = environment.reset(np.random.default_rng(3))
    assert leaf_bipartitions(first) == leaf_bipartitions(second)


@pytest.mark.edge_case
def test_the_known_reward_under_a_general_model_needs_its_rate_matrix() -> None:
    # A general model at known parameters is a Q and a pi, not a branch
    # length alone. Refusing is better than silently scoring GTR as if it
    # were Jukes-Cantor, which is what the environment did before #328.
    params = _params()
    alignment = _alignment(params)
    with pytest.raises(ValueError, match="under gtr needs a rate_matrix"):
        TopologyEnvironment(
            alignment,
            params.k,
            params.pi,
            _BRANCH_LENGTH,
            model=Model.GTR,
            reward=RewardModel.KNOWN,
        )
    with pytest.raises(ValueError, match="must have shape"):
        TopologyEnvironment(
            alignment,
            params.k,
            params.pi,
            _BRANCH_LENGTH,
            model=Model.GTR,
            reward=RewardModel.KNOWN,
            rate_matrix=np.eye(3),
        )
    with pytest.raises(ValueError, match="fixes its rate matrix"):
        TopologyEnvironment(
            alignment,
            params.k,
            params.pi,
            _BRANCH_LENGTH,
            model=Model.JC,
            rate_matrix=jc_rate_matrix(params.k),
        )


# --- the general-Q known reward -----------------------------------------


@pytest.mark.oracle
def test_the_known_gtr_score_is_the_pruning_recursion_at_the_fixed_length() -> None:
    # The environment wraps the differentiable recursion with the model's Q
    # at one length on every branch; here the recursion is called directly
    # at the same lengths, for every topology on the leaf set.
    params = _params()
    alignment, rate_matrix = _gtr_alignment(params)
    environment = TopologyEnvironment(
        alignment,
        params.k,
        _GTR_PI,
        _BRANCH_LENGTH,
        model=Model.GTR,
        reward=RewardModel.KNOWN,
        rate_matrix=rate_matrix,
    )
    for topology in enumerate_topologies(sorted(alignment)):
        lengths = torch.full(
            (len(branch_order(topology)),), _BRANCH_LENGTH, dtype=torch.float64
        )
        expected = float(
            log_likelihood_torch(
                topology,
                params.k,
                _GTR_PI,
                alignment,
                lengths,
                rate_matrix=torch.as_tensor(rate_matrix),
            )
        )
        assert_allclose(environment.score(topology), expected, rtol=1e-12)


@pytest.mark.oracle
def test_the_general_q_path_reduces_to_jukes_cantor_at_its_rate_matrix() -> None:
    # The reduction `search/CLAUDE.md` asks of every general construction:
    # with Jukes-Cantor's own Q and uniform pi, the matrix-exponential path
    # must reproduce the closed-form path, topology by topology. The two
    # share no transition-probability code.
    environment, params, alignment = _environment(RewardModel.KNOWN)
    general = TopologyEnvironment(
        alignment,
        params.k,
        params.pi,
        _BRANCH_LENGTH,
        model=Model.GTR,
        reward=RewardModel.KNOWN,
        rate_matrix=jc_rate_matrix(params.k),
    )
    for topology in enumerate_topologies(sorted(alignment)):
        assert_allclose(general.score(topology), environment.score(topology), rtol=1e-9)


@pytest.mark.mathematical
@stress_only(
    "a GTR fit optimizes lengths, exchangeabilities and pi per topology, "
    "and the same inequality is pinned under Jukes-Cantor in the CI tier"
)
def test_the_fitted_gtr_score_is_never_below_the_known_one() -> None:
    # The fitted surface optimizes what the known one fixes -- lengths, Q
    # and pi alike -- so it cannot do worse, at the generating Q as at any.
    params = _params()
    alignment, rate_matrix = _gtr_alignment(params)
    known = TopologyEnvironment(
        alignment,
        params.k,
        _GTR_PI,
        _BRANCH_LENGTH,
        model=Model.GTR,
        reward=RewardModel.KNOWN,
        rate_matrix=rate_matrix,
    )
    fitted = TopologyEnvironment(
        alignment,
        params.k,
        _GTR_PI,
        _BRANCH_LENGTH,
        model=Model.GTR,
        reward=RewardModel.FITTED,
    )
    for topology in list(enumerate_topologies(sorted(alignment)))[:2]:
        assert fitted.score(topology) >= known.score(topology) - 1e-6


# --- the full feature set --------------------------------------------------


@pytest.mark.structural
def test_the_full_set_is_seven_standardized_columns() -> None:
    environment, _, alignment = _environment(features=FeatureSet.FULL)
    assert environment.feature_set is FeatureSet.FULL
    assert environment.n_features() == len(FEATURE_NAMES[FeatureSet.FULL]) == 7
    for state in enumerate_topologies(sorted(alignment)):
        actions = environment.actions(state)
        rows = environment.features(state, actions)
        assert rows.shape == (len(actions), 7)
        assert_allclose(rows.mean(dim=0).numpy(), np.zeros(7), atol=1e-12)
        spread = rows.std(dim=0, unbiased=False).numpy()
        assert np.all((np.abs(spread - 1.0) < 1e-12) | (spread == 0.0))
        assert_allclose(
            rows.numpy(),
            standardize(environment.raw_features(state, actions)).numpy(),
            atol=1e-12,
        )


@pytest.mark.oracle
def test_the_parsimony_column_is_the_change_in_fitch_score() -> None:
    environment, params, alignment = _environment(features=FeatureSet.FULL)
    for state in enumerate_topologies(sorted(alignment)):
        actions = environment.actions(state)
        raw = environment.raw_features(state, actions)
        expected = [
            fitch_score(action, alignment, params.k)
            - fitch_score(state, alignment, params.k)
            for action in actions
        ]
        assert_allclose(raw[:, 1].numpy(), expected, atol=0.0)
        assert_allclose(
            raw[:, 0].numpy(),
            [environment.step(state, action)[1] for action in actions],
            atol=1e-12,
        )


@pytest.mark.oracle
def test_the_support_columns_are_the_pattern_support_of_the_split_broken_and_made() -> (
    None
):
    # An NNI move breaks exactly one internal split and makes exactly one;
    # the columns are their pattern supports, taken here from the split sets
    # directly rather than from the environment's cache.
    environment, params, alignment = _environment(features=FeatureSet.FULL)
    for state in enumerate_topologies(sorted(alignment)):
        actions = environment.actions(state)
        raw = environment.raw_features(state, actions)
        for row, action in enumerate(actions):
            (broken,) = internal_splits(state) - internal_splits(action)
            (made,) = internal_splits(action) - internal_splits(state)
            assert raw[row, 2] == split_pattern_support(broken, alignment, params.k)
            assert raw[row, 3] == split_pattern_support(made, alignment, params.k)
            assert 0.0 < raw[row, 2] <= 1.0
            assert 0.0 < raw[row, 3] <= 1.0


@pytest.mark.oracle
def test_the_subtree_columns_are_the_sizes_of_the_exchanged_subtrees() -> None:
    # Against `branch_splits`: each exchanged set is the leaf set below a
    # branch of both topologies, the two are disjoint, and swapping them in
    # the broken split yields the made split -- which is what "the subtrees
    # an NNI move exchanges" means. The four blocks around the edge
    # partition the leaves.
    environment, _, alignment = _environment(features=FeatureSet.FULL)
    leaves = frozenset(alignment)
    for state in enumerate_topologies(sorted(alignment)):
        actions = environment.actions(state)
        raw = environment.raw_features(state, actions)
        for row, action in enumerate(actions):
            detached, attached = exchanged_subtrees(
                leaf_bipartitions(state), leaf_bipartitions(action)
            )
            (broken,) = internal_splits(state) - internal_splits(action)
            (made,) = internal_splits(action) - internal_splits(state)
            for subtree in (detached, attached):
                for topology in (state, action):
                    sides = set(branch_splits(topology))
                    sides |= {leaves - split for split in sides}
                    assert subtree in sides, (subtree, topology)
            assert detached
            assert attached
            assert not detached & attached
            assert (broken - detached) | attached == made
            shared, rest = broken & made, leaves - (broken | made)
            assert len(detached) + len(attached) + len(shared) + len(rest) == len(
                leaves
            )
            assert float(raw[row, 4]) == len(detached)
            assert float(raw[row, 5]) == len(attached)
            assert raw[row, 6] == abs(len(detached) - len(attached))


@pytest.mark.edge_case
def test_standardizing_one_row_or_a_constant_column_gives_zeros() -> None:
    # One action, or a column every action shares: no spread to divide by,
    # and zero is what a constant is worth to a softmax.
    one = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    assert_allclose(standardize(one).numpy(), np.zeros((1, 2)))
    rows = torch.tensor([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=torch.float64)
    expected = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]) * np.array(
        [np.sqrt(3.0 / 2.0), 1.0]
    )
    assert_allclose(standardize(rows).numpy(), expected, atol=1e-12)
    same = frozenset({frozenset({"A", "B"})})
    assert exchanged_subtrees(same, same) == (frozenset(), frozenset())


@pytest.mark.edge_case
def test_too_few_taxa_is_rejected() -> None:
    params = _params()
    alignment = _alignment(params)
    trimmed = {name: alignment[name] for name in sorted(alignment)[:3]}
    with pytest.raises(ValueError, match="need at least 4 taxa"):
        TopologyEnvironment(trimmed, params.k, params.pi, _BRANCH_LENGTH)


@pytest.mark.oracle
def test_the_spr_neighbourhood_is_larger_than_the_nni_one() -> None:
    # The environment is parameterized by move set, and the two must actually
    # differ or that parameter is decoration.
    nni, _, alignment = _environment(moves=MoveSet.NNI)
    spr, _, _ = _environment(moves=MoveSet.SPR)
    state = next(iter(enumerate_topologies(sorted(alignment))))
    assert len(spr.actions(state)) > len(nni.actions(state))
