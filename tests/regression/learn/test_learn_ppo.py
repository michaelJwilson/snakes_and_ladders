"""PPO against the identities that define it, and against REINFORCE at a matched budget (issue #313).

The generalized advantage at ``lambda = 1`` is the return less the value
and at ``lambda = 0`` the one-step temporal difference; the clipped
objective at the collecting policy, with no clipping, has the actor-critic's
gradient exactly; and the enumerated expected return -- exact on the Potts
chain -- rises over training and ends above REINFORCE's at the same number
of episodes.

The last of those is release-gated at its 1,920-episode budget, so the
sibling at the end of this module carries it per pull request (issue #401):
a quarter of the budget on the *declared* chain, with the untrained policy's
exact expected return read from the ``potts_chain/ci`` baseline record
instead of computed a second time. The exact return is a sum over all 81
configurations and costs 1.9 s whoever asks for it, so measuring the policy
before and after training is most of a test that only needs the after.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.actor_critic import advantage_surrogate_loss
from snakes_and_ladders.learn.critic import Critic, n_state_features
from snakes_and_ladders.learn.exact import exact_expected_return
from snakes_and_ladders.learn.policy import LinearPolicy, MLPPolicy
from snakes_and_ladders.learn.potts import (
    PottsLandscape,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.ppo import (
    _log_probabilities,
    episode_advantages,
    generalized_advantages,
    ppo,
    ppo_loss,
)
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.rollout import rollout
from snakes_and_ladders.sim.fixtures import baseline, fixture

FIELD = np.array([0.4, -0.1, -0.3])


def _landscape() -> PottsLandscape:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=4)


def _states(landscape: PottsLandscape) -> list[tuple[int, ...]]:
    return list(enumerate_configurations(landscape.n_states, landscape.chain_length))


def _reached(landscape: PottsLandscape, policy: LinearPolicy | MLPPolicy) -> float:
    best = optimum(landscape)[1]
    rng = np.random.default_rng(1)
    return float(
        np.mean(
            [
                abs(
                    landscape.energy(
                        rollout(landscape, policy, rng, 6, start=s).states[-1]
                    )
                    - best
                )
                < 1e-9
                for s in _states(landscape)
            ]
        )
    )


def _mean_return(landscape: PottsLandscape, policy: LinearPolicy | MLPPolicy) -> float:
    return float(
        np.mean(
            [
                float(exact_expected_return(landscape, policy, s, 3).detach())
                for s in _states(landscape)
            ]
        )
    )


@pytest.mark.mathematical
def test_generalized_advantages_reduce_to_their_two_limits() -> None:
    rewards, values = [1.0, -0.5, 2.0], [1.0, 2.0, 0.5, 0.0]
    monte_carlo = generalized_advantages(rewards, values, lam=1.0, terminated=True)
    assert monte_carlo == pytest.approx(
        [sum(rewards[t:]) - values[t] for t in range(3)]
    )
    one_step = generalized_advantages(rewards, values, lam=0.0, terminated=False)
    assert one_step == pytest.approx(
        [rewards[t] + values[t + 1] - values[t] for t in range(3)]
    )
    # A truncated episode keeps the final state's value; a terminated one drops it.
    truncated = generalized_advantages(
        rewards, [1.0, 2.0, 0.5, 4.0], lam=1.0, terminated=False
    )
    assert truncated[-1] == pytest.approx(rewards[-1] + 4.0 - 0.5)
    with pytest.raises(ValueError, match="need 4 values"):
        generalized_advantages(rewards, values[:3], lam=1.0, terminated=True)
    with pytest.raises(ValueError, match="lam must be"):
        generalized_advantages(rewards, values, lam=1.5, terminated=True)


@pytest.mark.mathematical
def test_unclipped_ppo_at_the_collecting_policy_has_the_actor_critic_gradient() -> None:
    # Every ratio is one, so min(rho A, clip(rho) A) = A and the gradient of
    # the clipped objective is the advantage-weighted score function.
    landscape = _landscape()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    rng = np.random.default_rng(0)
    episodes = [rollout(landscape, policy, rng, 3) for _ in range(40)]
    advantages = episode_advantages(landscape, episodes, critic, lam=1.0)
    old = [t.detach() for t in _log_probabilities(landscape, policy, episodes)]
    loss, fraction = ppo_loss(
        _log_probabilities(landscape, policy, episodes), old, advantages, clip=math.inf
    )
    (ppo_gradient,) = torch.autograd.grad(loss, policy.weights)
    (ac_gradient,) = torch.autograd.grad(
        advantage_surrogate_loss(landscape, policy, episodes, advantages),
        policy.weights,
    )
    torch.testing.assert_close(ppo_gradient, ac_gradient)
    assert fraction == 0.0
    # Moving the policy makes ratios differ from one and the clip bind.
    policy.set_weights(torch.tensor([2.0, -3.0], dtype=torch.float64))
    _, moved = ppo_loss(
        _log_probabilities(landscape, policy, episodes), old, advantages, clip=0.2
    )
    assert moved > 0.0
    with pytest.raises(ValueError, match="clip must be positive"):
        ppo_loss(old, old, advantages, clip=0.0)


@pytest.mark.oracle
@pytest.mark.release
def test_ppo_raises_the_enumerated_expected_return_and_beats_reinforce_at_a_matched_budget() -> (
    None
):
    # 60 iterations of 32 episodes, as #135 trained REINFORCE. Measured:
    # REINFORCE reaches the enumerated optimum from 88.9% of the 81 starts
    # with mean exact return 2.21; PPO from 96.3% with 2.28; greedy from
    # 80.2%. At a quarter of the budget (480 episodes) REINFORCE reaches it
    # from 32.1% and PPO from 87.7%. Asserted at the margins below.
    landscape = _landscape()
    ppo_policy = LinearPolicy(2)
    before = _mean_return(landscape, ppo_policy)
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    training = ppo(
        landscape,
        ppo_policy,
        critic,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=6,
    )
    assert training.episodes == 1920
    assert len(training.clipped_fraction) == 60
    after = _mean_return(landscape, ppo_policy)
    assert after > before
    reinforce_policy = LinearPolicy(2)
    reinforce(
        landscape,
        reinforce_policy,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=6,
    )
    assert _reached(landscape, ppo_policy) >= _reached(landscape, reinforce_policy)
    assert _reached(landscape, ppo_policy) >= 72 / 81
    short_ppo, short_reinforce = LinearPolicy(2), LinearPolicy(2)
    ppo(
        landscape,
        short_ppo,
        Critic(
            n_state_features(landscape),
            hidden=None,
            generator=torch.Generator().manual_seed(0),
        ),
        np.random.default_rng(0),
        iterations=15,
        batch=32,
        max_steps=6,
    )
    reinforce(
        landscape,
        short_reinforce,
        np.random.default_rng(0),
        iterations=15,
        batch=32,
        max_steps=6,
    )
    assert _reached(landscape, short_ppo) > _reached(landscape, short_reinforce) + 0.3


@pytest.mark.release
@pytest.mark.simulated_truth
def test_an_mlp_policy_trained_by_ppo_reaches_the_optimum() -> None:
    # Measured 97.5% of the 81 starts with mean exact return 2.55, against
    # the linear policy's 96.3% and 2.28: the deeper scorer can represent the
    # worsening move a chain needs, which the two linear features cannot.
    landscape = _landscape()
    policy = MLPPolicy(2, hidden=8, generator=torch.Generator().manual_seed(0))
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    ppo(
        landscape,
        policy,
        critic,
        np.random.default_rng(0),
        iterations=60,
        batch=32,
        max_steps=6,
        learning_rate=0.01,
    )
    assert _reached(landscape, policy) >= 72 / 81
    assert _mean_return(landscape, policy) > 2.3


@pytest.mark.edge_case
def test_ppo_refuses_a_non_positive_budget() -> None:
    landscape = _landscape()
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    with pytest.raises(ValueError, match="must be >= 1"):
        ppo(
            landscape,
            LinearPolicy(2),
            critic,
            np.random.default_rng(0),
            iterations=0,
            batch=1,
            max_steps=1,
        )


# --- the per-pull-request sibling (issue #401) ---------------------------

#: The chain the registry declares, which is the one `opt` fits. The RL
#: environment is that model at the length exhaustive enumeration reaches;
#: the length is read from the record rather than repeated here, so the
#: environment measured and the environment trained on cannot drift apart.
DECLARED = ("potts_chain", "ci")

# A quarter of the release budget: 480 episodes. Measured at it, PPO reaches
# the enumerated optimum from 0.877 of the 81 starts with mean exact return
# 2.155, against REINFORCE's 0.321 and 1.204 -- the separation the release
# test asserts at the full budget, at the budget that fits the tier.
SIBLING_ITERATIONS = 15
SIBLING_BATCH = 32


def _declared_landscape() -> PottsLandscape:
    """The declared chain as a single-flip search at the enumerable length."""
    record = baseline(*DECLARED)
    params = fixture(*DECLARED).params
    return PottsLandscape(
        coupling=params.coupling,
        field=params.field,
        chain_length=int(
            record.measurement("enumerated_optimum").budget["chain_length"]
        ),
    )


@pytest.mark.oracle
def test_ppo_raises_the_recorded_expected_return_and_stays_ahead_of_reinforce() -> None:
    # The fast sibling of the 1,920-episode comparison above. The untrained
    # policy's exact expected return and its optimum-reaching rate are the
    # fixture's committed baseline, so the 1.9 s enumeration behind the
    # "before" is not paid here; `infra/baselines.py` recomputes it at the
    # release gate. What is measured is the training: the return must rise
    # above the recorded one and PPO must still be ahead of REINFORCE at a
    # matched budget, which is the claim, not the exact number.
    record = baseline(*DECLARED)
    landscape = _declared_landscape()
    before = record.value("untrained_expected_return")
    assert record.value("untrained_reached") < 0.2

    policy = LinearPolicy(2)
    critic = Critic(
        n_state_features(landscape),
        hidden=None,
        generator=torch.Generator().manual_seed(0),
    )
    training = ppo(
        landscape,
        policy,
        critic,
        np.random.default_rng(0),
        iterations=SIBLING_ITERATIONS,
        batch=SIBLING_BATCH,
        max_steps=6,
    )
    assert training.episodes == SIBLING_ITERATIONS * SIBLING_BATCH

    baselined = LinearPolicy(2)
    reinforce(
        landscape,
        baselined,
        np.random.default_rng(0),
        iterations=SIBLING_ITERATIONS,
        batch=SIBLING_BATCH,
        max_steps=6,
    )
    assert _mean_return(landscape, policy) > before + 1.0
    assert _reached(landscape, policy) > _reached(landscape, baselined) + 0.3
    assert _reached(landscape, policy) > record.value("untrained_reached") + 0.5
