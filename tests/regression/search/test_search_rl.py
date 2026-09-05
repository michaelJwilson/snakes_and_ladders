"""The phylogenetic environment: its rewards, its caching, and its boundaries.

Two things are worth asserting here that neither `phylo.learn` nor
`phylo.search.infer` can assert for itself. That the environment's reward is
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
from phylo.learn.environment import Environment
from phylo.learn.policy import LinearPolicy
from phylo.learn.rollout import greedy_rollout, rollout
from phylo.likelihood.pruning import log_likelihood
from phylo.search.infer import Model, MoveSet, score_topology
from phylo.search.rl import (
    RewardModel,
    TopologyEnvironment,
    with_uniform_branch_lengths,
)
from phylo.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
    nni_neighbours,
)
from phylo.sim.params import SimulationParams, load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params_5taxa.yaml"
_BRANCH_LENGTH = 0.1629


def _params() -> SimulationParams:
    return load_simulation_params(FIXTURE)


def _alignment(params: SimulationParams) -> dict[str, np.ndarray]:
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )
    return dict(dataset.alignment)


def _environment(
    reward: RewardModel = RewardModel.KNOWN, moves: MoveSet = MoveSet.NNI
) -> tuple[TopologyEnvironment, SimulationParams, dict[str, np.ndarray]]:
    params = _params()
    alignment = _alignment(params)
    environment = TopologyEnvironment(
        alignment, params.k, params.pi, _BRANCH_LENGTH, reward=reward, moves=moves
    )
    return environment, params, alignment


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
    # baseline; this is the path `phylo.learn.rollout.rollout` takes through
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
def test_the_known_reward_refuses_a_model_it_cannot_score() -> None:
    # The closed-form scorer is the Jukes-Cantor pruning path, which is where
    # the cheapness comes from. Refusing is better than silently scoring a
    # general model as if it were Jukes-Cantor.
    params = _params()
    with pytest.raises(ValueError, match="reward is implemented for jc only"):
        TopologyEnvironment(
            _alignment(params),
            params.k,
            params.pi,
            _BRANCH_LENGTH,
            model=Model.GTR,
            reward=RewardModel.KNOWN,
        )


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
