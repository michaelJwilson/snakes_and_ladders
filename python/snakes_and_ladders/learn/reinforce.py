"""REINFORCE with a baseline: the score-function estimator ``docs/tex`` states.

The estimator is eq. (17) of the technical document, undiscounted::

    grad J = E[ sum_t grad log pi(a_t | s_t) * (G_t - b(s_t)) ]

with ``G_t`` the return-to-go. ``gamma = 1`` because the reward telescopes,
so the return of an episode is exactly the improvement it achieved and a
discount would prefer improvement found early over the same improvement
found late.

**The baseline is the previous iterations' mean return, not this batch's.**
Subtracting a constant leaves the estimator unbiased because it multiplies a
term of zero expectation, and that argument needs the constant to be
independent of the sample it is subtracted from. A within-batch mean is not:
it is correlated with the returns it is centring, which buys variance at the
price of an ``O(1/N)`` bias. Carrying the running mean forward costs nothing
and keeps the unbiasedness claim exact -- which matters here because
:mod:`snakes_and_ladders.learn.exact` checks it against an enumerated gradient rather than
taking it on trust.

**What the baseline is for.** ``docs/tex`` is explicit that variance, not
bias, is the reason: ``G_t`` is a sum of objective differences whose scale
depends on the problem, so an uncentred gradient varies in magnitude by
orders of magnitude between instances. The measured variance reduction is
reported by the regression suite rather than asserted in the abstract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.learn.environment import Environment, Episode
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.rollout import rollout

# Adam rather than plain SGD, for the reason the baseline exists: the
# gradient's scale is set by the objective's units, which differ between
# instances. Its per-parameter scaling makes one learning rate usable across
# them, where SGD would need retuning per problem and hide that fact.
_DEFAULT_LEARNING_RATE = 0.05


@dataclass(frozen=True)
class Training:
    """The outcome of a training run.

    Parameters
    ----------
    weights : np.ndarray
        The learned policy parameters, shape ``(n_features,)``.
    mean_returns : tuple[float, ...]
        Mean sampled return per iteration. A diagnostic, *not* a result: it
        is a Monte Carlo estimate under a changing policy, so a rising curve
        is consistent with a broken estimator. The claim worth making is
        against the enumerated ``J`` in :mod:`snakes_and_ladders.learn.exact`.
    episodes : int
        Total episodes sampled, which is what the budget counts.
    """

    weights: np.ndarray
    mean_returns: tuple[float, ...]
    episodes: int


def surrogate_loss[S, A](
    environment: Environment[S, A],
    policy: LinearPolicy,
    episodes: Sequence[Episode[S, A]],
    baseline: float,
) -> torch.Tensor:
    """A scalar whose gradient is the REINFORCE estimator, negated to minimize.

    The log-probabilities are recomputed here rather than cached during the
    rollout: the rollout samples under ``no_grad`` because sampling is not
    the thing being differentiated, and a cached graph would tie the
    estimator to the policy that *collected* the data instead of the one
    being updated. That distinction is what separates REINFORCE from the
    importance-weighted objective PPO uses.

    Parameters
    ----------
    environment : Environment[S, A]
        The environment the episodes were collected in.
    policy : LinearPolicy
        The policy being updated.
    episodes : Sequence[Episode[S, A]]
        Sampled trajectories.
    baseline : float
        Constant subtracted from every return-to-go. Must not depend on
        ``episodes``, or the estimator is no longer unbiased.

    Returns
    -------
    torch.Tensor
        Scalar; its gradient is minus the estimated ``grad J``, averaged over
        episodes.

    Raises
    ------
    ValueError
        If ``episodes`` is empty.
    """
    if not episodes:
        msg = "need at least one episode to form an estimate"
        raise ValueError(msg)

    total = torch.zeros((), dtype=policy.weights.dtype)
    for episode in episodes:
        advantages = [value - baseline for value in episode.returns_to_go()]
        for step, action in enumerate(episode.actions):
            state = episode.states[step]
            available = environment.actions(state)
            log_probabilities = policy.log_probabilities(
                environment.features(state, available)
            )
            total = (
                total + log_probabilities[available.index(action)] * advantages[step]
            )
    return -total / len(episodes)


def reinforce[S, A](
    environment: Environment[S, A],
    policy: LinearPolicy,
    rng: np.random.Generator,
    iterations: int,
    batch: int,
    max_steps: int,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
    use_baseline: bool = True,
) -> Training:
    """Train ``policy`` in place by REINFORCE.

    Parameters
    ----------
    environment : Environment[S, A]
        The problem to learn in.
    policy : LinearPolicy
        Updated in place; its weights carry the result.
    rng : np.random.Generator
        The only source of randomness, covering starting states and every
        action sampled, so a run is reproducible from its seed.
    iterations : int
        Gradient updates to perform.
    batch : int
        Episodes sampled per update.
    max_steps : int
        Decision budget per episode.
    learning_rate : float
        Adam step size.
    use_baseline : bool
        Whether to subtract the running mean return. Exposed so the variance
        reduction it buys can be measured rather than assumed.

    Returns
    -------
    Training
        The learned weights and the per-iteration diagnostics.

    Raises
    ------
    ValueError
        If ``iterations`` or ``batch`` is not positive.
    """
    if iterations < 1:
        msg = f"iterations must be >= 1, got {iterations}"
        raise ValueError(msg)
    if batch < 1:
        msg = f"batch must be >= 1, got {batch}"
        raise ValueError(msg)

    optimizer = torch.optim.Adam([policy.weights], lr=learning_rate)
    mean_returns: list[float] = []
    # Running mean of every return seen in *earlier* iterations. Independent
    # of the current batch, which is what keeps the estimator unbiased.
    seen_total, seen_count = 0.0, 0

    for _ in range(iterations):
        episodes = [rollout(environment, policy, rng, max_steps) for _ in range(batch)]
        returns = [episode.total_reward for episode in episodes]
        baseline = seen_total / seen_count if use_baseline and seen_count else 0.0

        optimizer.zero_grad()
        surrogate_loss(environment, policy, episodes, baseline).backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        mean_returns.append(float(np.mean(returns)))
        seen_total += float(np.sum(returns))
        seen_count += len(returns)

    return Training(
        weights=policy.weights.detach().numpy().copy(),
        mean_returns=tuple(mean_returns),
        episodes=iterations * batch,
    )
