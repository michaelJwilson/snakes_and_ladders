"""The RL interface: protocol conformance, episode arithmetic, and the gauge.

What is asserted here is what no instance can assert for itself -- that the
return telescopes to the improvement it claims to be, that a rollout obeys
its budget and reports truncation, and that a score shared by every action is
unidentifiable. The last is the same failure ``log_simplex`` exists to
prevent in ``sal.opt``, one module over, and it is the reason no
environment here supplies a bias feature.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import sal.learn
import torch
from numpy.testing import assert_allclose
from sal.learn.environment import Environment, Episode, features_tensor
from sal.learn.policy import LinearPolicy
from sal.learn.potts import PottsEnvironment
from sal.learn.rollout import greedy_rollout, rollout

from tests.regression.learn.conftest import potts_environment

# Same rule, same wording, same reason as `tests/regression/test_opt_objective.py`.
FORBIDDEN_PREFIXES = (
    "sal.sim",
    "sal.likelihood",
    "sal.search",
    # `sample/`'s application half: `sample.schedule` shapes no agent, while
    # `potts_mcmc` and `potts_keyed` are the lattice's move sets (#777).
    "sal.sample.potts",
)

#: The modules issue #779 exempted, listed so a fourth is a reader's decision.
#: `tree.py` and `ranking.py` are `learn/` interfaces over one application's
#: objects; `potts_nd.py` imports `sample.potts_mcmc.MoveKind` and
#: `sample.accept` (#857), neither an application.
APPLICATION_INSTANCES = ("potts.py", "potts_nd.py", "ranking.py", "tree.py")


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
@pytest.mark.smoke
def test_learn_imports_nothing_from_the_application_modules() -> None:
    # The structural claim this package exists to make. An agent that has
    # seen a tree is an agent shaped by trees, and neither ruff nor mypy
    # would notice a single convenience import.
    package = Path(sal.learn.__file__).parent
    offenders: dict[str, set[str]] = {}
    for source in sorted(package.glob("*.py")):
        if source.name in APPLICATION_INSTANCES:
            continue
        bad = {
            name
            for name in _imported_modules(source)
            if name.startswith(FORBIDDEN_PREFIXES)
        }
        if bad:
            offenders[source.name] = bad
    assert offenders == {}


@pytest.mark.critical
@pytest.mark.smoke
def test_every_named_application_instance_exists_and_imports_one() -> None:
    # The exemption is two-sided: a name that no longer names a module would
    # exempt nothing and read as though it did, and a module listed here that
    # imports no application module does not need the exemption.
    package = Path(sal.learn.__file__).parent
    for name in APPLICATION_INSTANCES:
        source = package / name
        assert source.exists(), name
        imported = _imported_modules(source)
        assert {n for n in imported if n.startswith(FORBIDDEN_PREFIXES)} != set(), name


@pytest.mark.smoke
def test_the_reference_environment_satisfies_the_protocol() -> None:
    # Annotated so the module names its problem by import (`tests/_problems.py`,
    # #863); the claim: the concrete class satisfies the protocol.
    environment: PottsEnvironment = potts_environment()

    assert isinstance(environment, Environment)


# --- the return telescopes -----------------------------------------------


@pytest.mark.analytic
def test_total_reward_is_the_improvement_between_first_and_last_state() -> None:
    # eq:return of docs/tex/textbook.tex: the return of an episode is exactly the
    # total improvement it
    # achieved, independent of the path. This is what licenses gamma = 1, so
    # it is checked against the objective rather than assumed from the algebra.
    environment = potts_environment()
    policy = LinearPolicy(2)
    episode = rollout(environment, policy, np.random.default_rng(0), max_steps=5)
    improvement = environment.energy(episode.states[-1]) - environment.energy(
        episode.states[0]
    )
    assert_allclose(episode.total_reward, improvement, atol=1e-12)


@pytest.mark.analytic
def test_returns_to_go_are_undiscounted_suffix_sums() -> None:
    episode: Episode[int, str] = Episode(
        states=(0, 1, 2, 3),
        actions=("a", "b", "c"),
        rewards=(1.0, -2.0, 4.0),
        terminated=True,
    )
    assert episode.returns_to_go() == (3.0, 2.0, 4.0)
    assert episode.total_reward == 3.0


@pytest.mark.smoke
def test_an_empty_episode_has_zero_return() -> None:
    episode: Episode[int, str] = Episode(
        states=(0,), actions=(), rewards=(), terminated=True
    )
    assert episode.total_reward == 0.0
    assert episode.returns_to_go() == ()


# --- rollout semantics ---------------------------------------------------


@pytest.mark.smoke
def test_a_rollout_respects_its_budget_and_reports_truncation() -> None:
    environment = potts_environment(chain_length=8)
    policy = LinearPolicy(2)
    # Weights that make downhill moves likely, so the episode does not
    # terminate at a local maximum before the budget bites.
    policy.set_weights(torch.tensor([-1.0, -1.0], dtype=torch.float64))
    episode = rollout(environment, policy, np.random.default_rng(1), max_steps=3)
    assert len(episode.actions) == 3
    assert not episode.terminated


@pytest.mark.smoke
def test_a_rollout_stops_on_reaching_a_local_maximum() -> None:
    environment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(environment.greedy_weights() * 50.0)
    episode = rollout(
        environment, policy, np.random.default_rng(2), max_steps=50, start=(2, 1, 1, 0)
    )
    assert episode.terminated
    assert environment.is_terminal(episode.states[-1])
    assert len(episode.actions) < 50


@pytest.mark.smoke
def test_a_rollout_started_at_a_local_maximum_takes_no_action() -> None:
    environment = potts_environment()
    optimum_state = (0, 0, 0, 0)
    assert environment.is_terminal(optimum_state)
    episode = rollout(
        environment,
        LinearPolicy(2),
        np.random.default_rng(3),
        max_steps=10,
        start=optimum_state,
    )
    assert episode.actions == ()
    assert episode.terminated


@pytest.mark.smoke
def test_a_rollout_is_reproducible_from_its_seed() -> None:
    environment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, 0.9], dtype=torch.float64))
    first = rollout(environment, policy, np.random.default_rng(11), max_steps=6)
    second = rollout(environment, policy, np.random.default_rng(11), max_steps=6)
    assert first.states == second.states
    assert first.actions == second.actions


@pytest.mark.oracle
def test_greedy_takes_the_best_rewarded_action_at_every_step() -> None:
    environment = potts_environment()
    episode = greedy_rollout(environment, start=(2, 1, 1, 0), max_steps=20)
    for state, taken, reward in zip(
        episode.states, episode.actions, episode.rewards, strict=False
    ):
        best = max(
            environment.step(state, action)[1] for action in environment.actions(state)
        )
        assert_allclose(reward, best, atol=1e-12)
        assert reward > 0.0
        assert taken in environment.actions(state)


@pytest.mark.smoke
def test_a_negative_budget_is_rejected_by_a_policy_rollout() -> None:
    with pytest.raises(ValueError, match="max_steps must be >= 0"):
        rollout(
            potts_environment(), LinearPolicy(2), np.random.default_rng(0), max_steps=-1
        )


@pytest.mark.smoke
def test_a_negative_budget_is_rejected_by_the_greedy_rollout() -> None:
    with pytest.raises(ValueError, match="max_steps must be >= 0"):
        greedy_rollout(potts_environment(), start=(0, 1, 0, 1), max_steps=-1)


# --- the gauge -----------------------------------------------------------


@pytest.mark.smoke
def test_a_score_shared_by_every_action_is_unidentifiable() -> None:
    # A feature constant across a state's actions shifts every score alike and
    # the softmax cancels it: the gauge of `opt.constrain`, hence no bias term.
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.7, -1.3], dtype=torch.float64))
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    shifted = features + torch.tensor([3.0, -2.0], dtype=torch.float64)
    assert_allclose(
        policy.log_probabilities(features).detach().numpy(),
        policy.log_probabilities(shifted).detach().numpy(),
        atol=1e-14,
    )


@pytest.mark.analytic
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


@pytest.mark.analytic
def test_log_probabilities_are_normalized() -> None:
    policy = LinearPolicy(3)
    policy.set_weights(torch.tensor([0.2, -0.5, 1.1], dtype=torch.float64))
    features = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    assert_allclose(
        float(torch.exp(policy.log_probabilities(features)).sum().detach()),
        1.0,
        rtol=1e-14,
    )


@pytest.mark.smoke
def test_a_policy_rejects_features_of_the_wrong_width() -> None:
    policy = LinearPolicy(2)
    with pytest.raises(ValueError, match=r"expected features of shape"):
        policy.log_probabilities(torch.zeros((3, 5), dtype=torch.float64))


@pytest.mark.smoke
def test_a_policy_needs_at_least_one_feature() -> None:
    with pytest.raises(ValueError, match="n_features must be >= 1"):
        LinearPolicy(0)


@pytest.mark.smoke
def test_set_weights_rejects_the_wrong_shape() -> None:
    policy = LinearPolicy(2)
    with pytest.raises(ValueError, match="expected weights of shape"):
        policy.set_weights(torch.zeros(3, dtype=torch.float64))


@pytest.mark.smoke
@pytest.mark.patch
def test_every_rollout_loop_reads_terminated_from_the_state_it_ended_in() -> None:
    # `terminated` is read from the last state under both stopping rules, at
    # a truncating budget, none, and for greedy (issue #862).
    environment = potts_environment()
    rng = np.random.default_rng(0)
    policy = LinearPolicy(2)
    episodes = [
        rollout(environment, policy, rng, max_steps=budget, stop_at_local_optimum=stop)
        for budget in (0, 1, 8)
        for stop in (True, False)
    ]
    episodes += [
        greedy_rollout(environment, start=environment.reset(rng), max_steps=budget)
        for budget in (0, 8)
    ]

    for episode in episodes:
        assert episode.terminated == environment.is_terminal(episode.states[-1])
        assert len(episode.states) == len(episode.actions) + 1
        assert len(episode.rewards) == len(episode.actions)


# --- The array boundary (issue #1011) --------------------------------------


@pytest.mark.smoke
@pytest.mark.patch
def test_a_policy_scores_an_array_as_it_scored_the_tensor() -> None:
    # `features` is an array and the policy converts it once, where it meets
    # the weights. The tensor route it replaced is the referee, bitwise: the
    # log-probabilities, their gradient, and the greedy index, at every
    # state of the reference chain.
    environment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(np.array([0.9, -0.4]))
    for state in environment_states(environment):
        rows = environment.features(state, environment.actions(state))
        assert isinstance(rows, np.ndarray)
        assert rows.dtype == np.float64
        from_array = policy.log_probabilities(rows)
        from_tensor = policy.log_probabilities(torch.as_tensor(rows))
        assert from_array.detach().numpy().tobytes() == (
            from_tensor.detach().numpy().tobytes()
        )
        (by_array,) = torch.autograd.grad(from_array[0], policy.weights)
        (by_tensor,) = torch.autograd.grad(from_tensor[0], policy.weights)
        assert by_array.numpy().tobytes() == by_tensor.numpy().tobytes()
        assert policy.greedy(rows) == policy.greedy(torch.as_tensor(rows))


def environment_states(environment: PottsEnvironment) -> list[tuple[int, ...]]:
    """Twenty seeded states of ``environment``."""
    rng = np.random.default_rng(1011)
    return [environment.reset(rng) for _ in range(20)]


@pytest.mark.smoke
@pytest.mark.patch
def test_the_deprecated_tensor_accessors_return_the_arrays_bitwise() -> None:
    # Release 0.3.0 returned tensors from `features` and `greedy_weights`;
    # the accessors keep that for one release and warn that they go.
    environment = potts_environment()
    state = (0, 1, 2, 0)
    actions = environment.actions(state)
    with pytest.warns(DeprecationWarning, match="features_tensor is deprecated"):
        tensor = features_tensor(environment, state, actions)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float64
    assert tensor.numpy().tobytes() == environment.features(state, actions).tobytes()
    with pytest.warns(DeprecationWarning, match="greedy_weights_tensor is deprecated"):
        weights = environment.greedy_weights_tensor()
    assert weights.numpy().tobytes() == environment.greedy_weights().tobytes()


@pytest.mark.smoke
@pytest.mark.hmm
@pytest.mark.parametrize(
    "module", ["canonical", "environment", "hmm", "potts", "potts_nd"]
)
def test_an_environment_module_imports_no_torch(module: str) -> None:
    # Root `CLAUDE.md`: no autodiff package where no derivative is taken. An
    # environment's features are constants to every loss, so importing one
    # loads no torch; a fresh interpreter, since this one has it loaded.
    code = f"import sys, sal.learn.{module}; print('torch' in sys.modules)"
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "False"
