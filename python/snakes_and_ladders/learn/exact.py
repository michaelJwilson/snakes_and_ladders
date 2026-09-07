"""Exact expected return by trajectory enumeration: the oracle for the agents.

Root ``CLAUDE.md`` requires expected values to be pinned against analytic
results, brute-force computations or a secondary implementation, and forbids
tests that assert only that something ran. For a policy-gradient method the
brute-force computation is available and cheap, provided the environment is
small: with a finite action set and a finite horizon the set of trajectories
is finite, so

    J(w) = sum over trajectories of P(trajectory | w) * G(trajectory)

is a *closed form*, and its gradient follows by autodiff through that sum.
That gives two independent checks on a sampled estimator, neither of which is
"the return went up":

* the enumerated gradient against central finite differences of the
  enumerated ``J`` -- autodiff against numerical differentiation;
* the sampled REINFORCE estimator against the enumerated gradient -- an
  unbiasedness claim, checked to a stated Monte Carlo tolerance.

The cost is ``|A| ** horizon``, so this is an oracle for the deliberately
small reference instance and for nothing else. That is the same bargain
``snakes_and_ladders.search.topology.enumerate_topologies`` strikes below eight taxa.
"""

from __future__ import annotations

import torch

from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.learn.policy import LinearPolicy


def exact_expected_return[S, A](
    environment: Environment[S, A],
    policy: LinearPolicy,
    start: S,
    horizon: int,
) -> torch.Tensor:
    """``J(w)``: expected undiscounted return from ``start``, enumerated exactly.

    Parameters
    ----------
    environment : Environment[S, A]
        The problem. Its action set and horizon must be small enough that
        ``|A| ** horizon`` trajectories are affordable.
    policy : LinearPolicy
        The policy whose expected return is wanted.
    start : S
        Starting state.
    horizon : int
        Maximum decisions, matching the ``max_steps`` a rollout would use.

    Returns
    -------
    torch.Tensor
        Scalar, differentiable with respect to ``policy.weights``.

    Raises
    ------
    ValueError
        If ``horizon`` is negative.
    """
    if horizon < 0:
        msg = f"horizon must be >= 0, got {horizon}"
        raise ValueError(msg)

    def value(state: S, remaining: int) -> torch.Tensor:
        if remaining == 0 or environment.is_terminal(state):
            return torch.zeros((), dtype=policy.weights.dtype)
        available = environment.actions(state)
        probabilities = torch.exp(
            policy.log_probabilities(environment.features(state, available))
        )
        total = torch.zeros((), dtype=policy.weights.dtype)
        for index, action in enumerate(available):
            successor, reward = environment.step(state, action)
            total = total + probabilities[index] * (
                reward + value(successor, remaining - 1)
            )
        return total

    return value(start, horizon)


def exact_policy_gradient[S, A](
    environment: Environment[S, A],
    policy: LinearPolicy,
    start: S,
    horizon: int,
) -> torch.Tensor:
    """``grad J(w)``, by autodiff through the enumerated expectation.

    The quantity every sampled estimator in this package is claiming to be an
    unbiased estimate *of*, so it is what they are compared against.

    Returns
    -------
    torch.Tensor
        Shape ``(policy.n_features,)``.
    """
    objective = exact_expected_return(environment, policy, start, horizon)
    (gradient,) = torch.autograd.grad(objective, policy.weights)
    return gradient


def finite_difference_gradient[S, A](
    environment: Environment[S, A],
    policy: LinearPolicy,
    start: S,
    horizon: int,
    step: float = 1e-5,
) -> torch.Tensor:
    """Central differences of the enumerated ``J``, entry by entry.

    An independent route to the same gradient: it touches neither autograd
    nor the score-function identity, so agreement between the two rules out
    an error in either.

    Parameters
    ----------
    step : float
        Half-width of the central difference. ``1e-5`` on ``float64`` puts
        truncation and rounding error at a comparable, small size.

    Returns
    -------
    torch.Tensor
        Shape ``(policy.n_features,)``.
    """
    baseline = policy.weights.detach().clone()
    gradient = torch.zeros_like(baseline)
    for index in range(baseline.shape[0]):
        shifted = baseline.clone()
        shifted[index] = baseline[index] + step
        policy.set_weights(shifted)
        upper = float(
            exact_expected_return(environment, policy, start, horizon).detach()
        )
        shifted[index] = baseline[index] - step
        policy.set_weights(shifted)
        lower = float(
            exact_expected_return(environment, policy, start, horizon).detach()
        )
        gradient[index] = (upper - lower) / (2.0 * step)
    policy.set_weights(baseline)
    return gradient
