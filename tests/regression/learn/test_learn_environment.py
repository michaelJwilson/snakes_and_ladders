"""The RL interface: protocol conformance, episode arithmetic, and the gauge.

What is asserted here is what no instance can assert for itself -- that the
return telescopes to the improvement it claims to be, that a rollout obeys
its budget and reports truncation, and that a score shared by every action is
unidentifiable. The last is the same failure ``log_simplex`` exists to
prevent in ``phylo.opt``, one module over, and it is the reason no
environment here supplies a bias feature.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import phylo.learn
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.learn.environment import Environment, Episode
from phylo.learn.policy import LinearPolicy
from phylo.learn.potts import PottsLandscape
from phylo.learn.rollout import greedy_rollout, rollout

# Same rule, same wording, same reason as `tests/regression/test_opt_objective.py`.
FORBIDDEN_PREFIXES = ("phylo.sim", "phylo.likelihood", "phylo.search")

FIELD = np.array([0.4, -0.1, -0.3])


def _landscape(chain_length: int = 4) -> PottsLandscape:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=chain_length)


def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


@pytest.mark.critical
@pytest.mark.structural
def test_learn_imports_nothing_from_the_application_modules() -> None:
    # The structural claim this package exists to make. An agent that has
    # seen a tree is an agent shaped by trees, and neither ruff nor mypy
    # would notice a single convenience import.
    package = Path(phylo.learn.__file__).parent
    offenders: dict[str, set[str]] = {}
    for source in sorted(package.glob("*.py")):
        bad = {
            name
            for name in _imported_modules(source)
            if name.startswith(FORBIDDEN_PREFIXES)
        }
        if bad:
            offenders[source.name] = bad
    assert offenders == {}


@pytest.mark.structural
def test_the_reference_environment_satisfies_the_protocol() -> None:
    assert isinstance(_landscape(), Environment)


# --- the return telescopes -----------------------------------------------


@pytest.mark.mathematical
def test_total_reward_is_the_improvement_between_first_and_last_state() -> None:
    # eq. (16): the return of an episode is exactly the total improvement it
    # achieved, independent of the path. This is what licenses gamma = 1, so
    # it is checked against the objective rather than assumed from the algebra.
    environment = _landscape()
    policy = LinearPolicy(2)
    episode = rollout(environment, policy, np.random.default_rng(0), max_steps=5)
    improvement = environment.energy(episode.states[-1]) - environment.energy(
        episode.states[0]
    )
    assert_allclose(episode.total_reward, improvement, atol=1e-12)


@pytest.mark.mathematical
def test_returns_to_go_are_undiscounted_suffix_sums() -> None:
    episode: Episode[int, str] = Episode(
        states=(0, 1, 2, 3),
        actions=("a", "b", "c"),
        rewards=(1.0, -2.0, 4.0),
        terminated=True,
    )
    assert episode.returns_to_go() == (3.0, 2.0, 4.0)
    assert episode.total_reward == 3.0


@pytest.mark.edge_case
def test_an_empty_episode_has_zero_return() -> None:
    episode: Episode[int, str] = Episode(
        states=(0,), actions=(), rewards=(), terminated=True
    )
    assert episode.total_reward == 0.0
    assert episode.returns_to_go() == ()


# --- rollout semantics ---------------------------------------------------


@pytest.mark.structural
def test_a_rollout_respects_its_budget_and_reports_truncation() -> None:
    environment = _landscape(chain_length=8)
    policy = LinearPolicy(2)
    # Weights that make downhill moves likely, so the episode does not
    # terminate at a local maximum before the budget bites.
    policy.set_weights(torch.tensor([-1.0, -1.0], dtype=torch.float64))
    episode = rollout(environment, policy, np.random.default_rng(1), max_steps=3)
    assert len(episode.actions) == 3
    assert not episode.terminated


@pytest.mark.structural
def test_a_rollout_stops_on_reaching_a_local_maximum() -> None:
    environment = _landscape()
    policy = LinearPolicy(2)
    policy.set_weights(environment.greedy_weights() * 50.0)
    episode = rollout(
        environment, policy, np.random.default_rng(2), max_steps=50, start=(2, 1, 1, 0)
    )
    assert episode.terminated
    assert environment.is_terminal(episode.states[-1])
    assert len(episode.actions) < 50


@pytest.mark.edge_case
@pytest.mark.structural
def test_a_rollout_started_at_a_local_maximum_takes_no_action() -> None:
    environment = _landscape()
    optimum_state = (0, 0, 0, 0)
    assert environment.is_terminal(optimum_state)
    episode = rollout(
        environment, LinearPolicy(2), np.random.default_rng(3), 10, start=optimum_state
    )
    assert episode.actions == ()
    assert episode.terminated


@pytest.mark.structural
def test_a_rollout_is_reproducible_from_its_seed() -> None:
    environment = _landscape()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, 0.9], dtype=torch.float64))
    first = rollout(environment, policy, np.random.default_rng(11), max_steps=6)
    second = rollout(environment, policy, np.random.default_rng(11), max_steps=6)
    assert first.states == second.states
    assert first.actions == second.actions


@pytest.mark.oracle
def test_greedy_takes_the_best_rewarded_action_at_every_step() -> None:
    environment = _landscape()
    episode = greedy_rollout(environment, (2, 1, 1, 0), max_steps=20)
    for state, taken, reward in zip(
        episode.states, episode.actions, episode.rewards, strict=False
    ):
        best = max(
            environment.step(state, action)[1] for action in environment.actions(state)
        )
        assert_allclose(reward, best, atol=1e-12)
        assert reward > 0.0
        assert taken in environment.actions(state)


@pytest.mark.edge_case
def test_a_negative_budget_is_rejected_by_a_policy_rollout() -> None:
    with pytest.raises(ValueError, match="max_steps must be >= 0"):
        rollout(_landscape(), LinearPolicy(2), np.random.default_rng(0), -1)


@pytest.mark.edge_case
def test_a_negative_budget_is_rejected_by_the_greedy_rollout() -> None:
    with pytest.raises(ValueError, match="max_steps must be >= 0"):
        greedy_rollout(_landscape(), (0, 1, 0, 1), -1)


# --- the gauge -----------------------------------------------------------


@pytest.mark.edge_case
def test_a_score_shared_by_every_action_is_unidentifiable() -> None:
    # Adding the same feature row to every action shifts every score by the
    # same amount, and the softmax is invariant to that. So a feature that
    # does not vary across a state's actions carries no information and its
    # weight has no value -- which is why there is no bias term. Exactly the
    # softmax gauge of `phylo.opt.constrain`, restated for a policy.
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.7, -1.3], dtype=torch.float64))
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    shifted = features + torch.tensor([3.0, -2.0], dtype=torch.float64)
    assert_allclose(
        policy.log_probabilities(features).detach().numpy(),
        policy.log_probabilities(shifted).detach().numpy(),
        atol=1e-14,
    )


@pytest.mark.mathematical
def test_scaling_the_weights_drives_the_policy_to_its_argmax() -> None:
    # The zero-temperature limit. It is why a greedy searcher is a member of
    # this policy class rather than a different kind of thing, which is what
    # makes the two comparable at matched budget at all.
    policy = LinearPolicy(2)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]], dtype=torch.float64)
    direction = torch.tensor([1.0, 0.25], dtype=torch.float64)
    policy.set_weights(direction * 200.0)
    probabilities = torch.exp(policy.log_probabilities(features)).detach()
    assert int(torch.argmax(probabilities)) == 0
    assert float(probabilities[0]) > 1.0 - 1e-9


@pytest.mark.mathematical
def test_log_probabilities_are_normalized() -> None:
    policy = LinearPolicy(3)
    policy.set_weights(torch.tensor([0.2, -0.5, 1.1], dtype=torch.float64))
    features = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    assert_allclose(
        float(torch.exp(policy.log_probabilities(features)).sum().detach()),
        1.0,
        rtol=1e-14,
    )


@pytest.mark.edge_case
def test_a_policy_rejects_features_of_the_wrong_width() -> None:
    policy = LinearPolicy(2)
    with pytest.raises(ValueError, match=r"expected features of shape"):
        policy.log_probabilities(torch.zeros((3, 5), dtype=torch.float64))


@pytest.mark.edge_case
def test_a_policy_needs_at_least_one_feature() -> None:
    with pytest.raises(ValueError, match="n_features must be >= 1"):
        LinearPolicy(0)


@pytest.mark.edge_case
def test_set_weights_rejects_the_wrong_shape() -> None:
    policy = LinearPolicy(2)
    with pytest.raises(ValueError, match="expected weights of shape"):
        policy.set_weights(torch.zeros(3, dtype=torch.float64))
