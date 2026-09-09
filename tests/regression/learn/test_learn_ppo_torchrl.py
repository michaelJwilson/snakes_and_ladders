"""`learn.ppo`'s advantage estimate and clipped loss against the TorchRL fronts of them (issues #322, #391).

TorchRL's ``GAE`` and ``ClipPPOLoss`` are a second implementation of the two
equations ``learn.ppo`` states -- the generalized advantage and the clipped
surrogate -- written by people who never saw ours: the validation use issue
#315 scoped. They were then measured as replacements and declined, so they
live in ``snakes_and_ladders.sandbox`` as fronts carrying our signatures
(``sandbox/CLAUDE.md``), and this module pins each front against the
implementation it did not replace. Same rollout in, same advantages and the
same objective out, to ``1e-10``.

Pinning the sandbox module rather than TorchRL assembled here is what keeps
the decline re-checkable: the front is the code
`tests/benchmarks/test_learn_ppo_bench.py` times, so a TorchRL release that
changes an answer fails here and one that changes a ratio shows up there,
instead of both being claims in a merged pull request.

Two conventions differ and are mapped inside the fronts rather than hidden.
TorchRL reads a truncated episode from ``done`` without ``terminated``, which
is the flag ``generalized_advantages`` takes; and its objective is the mean
over decisions where ours is the mean over episodes of the sum over
decisions, so the two agree up to the factor ``n_decisions / n_episodes``.

Skips without the ``frameworks`` extra, and until ``learn.ppo`` lands
(issue #313): a pin against a module that is not there pins nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.rollout import rollout

pytest.importorskip("torchrl")
pytest.importorskip("tensordict")
ppo = pytest.importorskip(
    "snakes_and_ladders.learn.ppo",
    reason="learn.ppo lands with issue #313; nothing to pin until it does",
)

from snakes_and_ladders.sandbox import torchrl_advantage, torchrl_clip  # noqa: E402

FIELD = np.array([0.4, -0.1, -0.3])

# One fixed rollout, in the shape `generalized_advantages` takes: rewards per
# decision and one value per state, the last being the final state's.
ROLLOUTS = (
    ([1.0, -0.5, 2.0], [1.0, 2.0, 0.5, 4.0]),
    ([0.3], [0.2, -0.7]),
    ([-1.0, 0.25, 0.5, 0.125, -2.0], [0.0, 1.0, -1.0, 0.5, 0.25, 3.0]),
)

type Neighbourhoods = list[list[tuple[torch.Tensor, int]]]


def _landscape() -> PottsLandscape:
    return PottsLandscape(coupling=0.75, field=FIELD, chain_length=4)


@pytest.mark.oracle
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize(("rewards", "values"), ROLLOUTS)
def test_the_advantages_are_torchrl_s_gae_at_gamma_one(
    rewards: list[float], values: list[float], lam: float, terminated: bool
) -> None:
    ours = ppo.generalized_advantages(rewards, values, lam=lam, terminated=terminated)

    theirs = torchrl_advantage.generalized_advantages(
        rewards, values, lam=lam, terminated=terminated
    )

    torch.testing.assert_close(
        torch.tensor(theirs, dtype=torch.float64),
        torch.tensor(ours, dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.oracle
def test_a_batch_of_advantages_agrees_episode_for_episode() -> None:
    # The front the benchmark times is the batch one, so it is the batch one
    # that is pinned: agreement per episode does not say the loop over
    # episodes pairs the episodes up the same way.
    rewards = [list(r) for r, _ in ROLLOUTS]
    values = [list(v) for _, v in ROLLOUTS]
    flags = [True, False, True]
    ours = [
        ppo.generalized_advantages(r, v, lam=0.95, terminated=t)
        for r, v, t in zip(rewards, values, flags, strict=True)
    ]

    theirs = torchrl_advantage.episode_advantages(rewards, values, flags, lam=0.95)

    torch.testing.assert_close(
        torch.tensor([a for episode in theirs for a in episode], dtype=torch.float64),
        torch.tensor([a for episode in ours for a in episode], dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.edge_case
def test_the_advantage_front_refuses_the_episode_of_no_decisions_ours_answers() -> None:
    # A start state already at a local optimum. `generalized_advantages`
    # returns an empty list; TorchRL has no trajectory of length zero, so the
    # front refuses rather than returning an answer TorchRL did not give.
    assert ppo.generalized_advantages([], [0.5], lam=0.95, terminated=True) == []

    with pytest.raises(ValueError, match="no decisions"):
        torchrl_advantage.generalized_advantages([], [0.5], lam=0.95, terminated=True)


def _neighbourhoods(
    landscape: PottsLandscape, policy: LinearPolicy, seed: int, episodes: int
) -> Neighbourhoods:
    """Each episode's decisions, as neighbourhood features and the index taken."""
    rng = np.random.default_rng(seed)
    rolled = [rollout(landscape, policy, rng, 3) for _ in range(episodes)]
    out: Neighbourhoods = []
    for episode in rolled:
        steps = []
        for state, action in zip(episode.states, episode.actions, strict=False):
            available = landscape.actions(state)
            steps.append(
                (landscape.features(state, available), available.index(action))
            )
        out.append(steps)
    return out


def _log_probabilities(
    policy: LinearPolicy, neighbourhoods: Neighbourhoods
) -> list[torch.Tensor]:
    return [
        torch.stack([policy.log_probabilities(rows)[index] for rows, index in steps])
        for steps in neighbourhoods
    ]


@pytest.mark.oracle
@pytest.mark.parametrize("clip", [0.1, 0.2, 0.5])
def test_the_clipped_objective_and_its_gradient_are_torchrl_s(clip: float) -> None:
    landscape = _landscape()
    collector = LinearPolicy(2)
    collector.set_weights(torch.tensor([0.1, 0.2], dtype=torch.float64))
    neighbourhoods = _neighbourhoods(landscape, collector, seed=0, episodes=12)
    old = [t.detach() for t in _log_probabilities(collector, neighbourhoods)]
    rng = np.random.default_rng(1)
    advantages = [list(rng.normal(size=len(steps))) for steps in neighbourhoods]

    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    current = _log_probabilities(policy, neighbourhoods)
    ours, _ = ppo.ppo_loss(current, old, advantages, clip=clip)
    (our_gradient,) = torch.autograd.grad(ours, policy.weights)

    theirs = torchrl_clip.clipped_objective(
        policy, neighbourhoods, old, advantages, clip=clip
    )
    (their_gradient,) = torch.autograd.grad(theirs, policy.weights)

    torch.testing.assert_close(theirs, ours, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(their_gradient, our_gradient, rtol=0.0, atol=1e-10)
    # Clipping is active somewhere, or the pin says nothing about the clip.
    ratio = torch.exp(torch.cat(current) - torch.cat(old))
    assert bool(((ratio < 1 - clip) | (ratio > 1 + clip)).any())


@pytest.mark.oracle
def test_the_clipped_objective_agrees_on_ragged_neighbourhoods() -> None:
    # Every neighbourhood on the chain has the same width, so the pin above
    # says nothing about the padding the front needs for a tree. A padded row
    # is a move that does not exist and must carry neither probability nor
    # gradient.
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.3, -0.6], dtype=torch.float64))
    generator = torch.Generator().manual_seed(0)
    neighbourhoods: Neighbourhoods = [
        [
            (torch.rand(width, 2, generator=generator, dtype=torch.float64), width - 1)
            for width in widths
        ]
        for widths in ([2, 5, 3], [4, 2])
    ]
    collector = LinearPolicy(2)
    collector.set_weights(torch.tensor([0.1, 0.2], dtype=torch.float64))
    old = [t.detach() for t in _log_probabilities(collector, neighbourhoods)]
    advantages = [[1.0, -2.0, 0.5], [0.25, -0.75]]
    ours, _ = ppo.ppo_loss(
        _log_probabilities(policy, neighbourhoods), old, advantages, clip=0.2
    )
    (our_gradient,) = torch.autograd.grad(ours, policy.weights)

    theirs = torchrl_clip.clipped_objective(
        policy, neighbourhoods, old, advantages, clip=0.2
    )
    (their_gradient,) = torch.autograd.grad(theirs, policy.weights)

    torch.testing.assert_close(theirs, ours, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(their_gradient, our_gradient, rtol=0.0, atol=1e-10)


@pytest.mark.edge_case
def test_the_loss_front_refuses_the_empty_batch_ours_answers() -> None:
    # A batch whose episodes all started at a local optimum. `ppo_loss`
    # returns zero; `ClipPPOLoss` has no such batch, so the front refuses.
    empty = torch.zeros(0, dtype=torch.float64)
    loss, fraction = ppo.ppo_loss([empty], [empty], [[]], clip=0.2)
    assert float(loss) == 0.0
    assert fraction == 0.0

    with pytest.raises(ValueError, match="no decisions"):
        torchrl_clip.clipped_objective(LinearPolicy(2), [[]], [empty], [[]], clip=0.2)
