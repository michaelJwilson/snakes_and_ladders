"""JAX twins of the HMM objectives' negative log-likelihood, for issue #1000's decision.

Each twin reads the objective's own ``theta`` layout and constraint maps --- a
simplex row is ``log_softmax([0, free])``, a positive parameter ``exp``, a
probability the logistic --- and scores the observations by the log-space
forward recursion under ``lax.scan``, so ``jit(value_and_grad)`` of it is the
gradient PyTorch's autograd takes through ``__call__``. PyTorch is the
oracle. A twin leaves the sandbox for the package only where issue #1000's
rule holds: at most half PyTorch's runtime at no more than 1.25x its memory,
or the converse.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from snakes_and_ladders.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    GaussianHmmObjective,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
)

#: The objectives a twin is written for.
type Twinned = (
    HmmObjective
    | GaussianHmmObjective
    | PoissonHmmObjective
    | BinomialHmmObjective
    | NegativeBinomialHmmObjective
    | BetaBinomialHmmObjective
)


def value_and_grad(
    objective: Twinned,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """``theta -> (U(theta), dU/dtheta)`` under ``jit``, the objective's negative log-likelihood.

    Every objective of :data:`Twinned` is covered as the package states it:
    a negative binomial's covariate is an exposure scaling each rate, a
    beta-binomial's a trial count per observation, and the other families
    refuse one when they are built, so no twin meets it.

    Raises
    ------
    TypeError
        If the objective has no twin.
    """
    import jax  # an optional backend, imported where it is used

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    jnp = jax.numpy
    observations = jnp.asarray(objective.observations.numpy())
    m = objective._n_states
    initial_slice = objective._initial_slice
    transition_slice = objective._transition_slice
    log_density = _emission(objective, observations, jax)

    def simplex(free: Any) -> Any:
        pinned = jnp.concatenate([jnp.zeros(free.shape[:-1] + (1,)), free], axis=-1)
        return jax.nn.log_softmax(pinned, axis=-1)

    def negative_log_likelihood(theta: Any) -> Any:
        log_initial = simplex(theta[initial_slice])
        log_transition = simplex(theta[transition_slice].reshape(m, m - 1))
        emit = log_density(theta)
        alpha = log_initial + emit[:, 0]

        def step(alpha: Any, column: Any) -> tuple[Any, None]:
            reach = jax.nn.logsumexp(alpha[:, :, None] + log_transition, axis=1)
            return reach + column, None

        alpha, _ = jax.lax.scan(step, alpha, jnp.moveaxis(emit[:, 1:], 1, 0))
        return -jnp.sum(jax.nn.logsumexp(alpha, axis=1))

    compiled = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def call(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = compiled(jnp.asarray(theta))
        return float(value), np.asarray(gradient)

    return call


def _emission(objective: Twinned, observations: Any, jax: Any) -> Callable[[Any], Any]:
    """``theta -> (n_sequences, length, m)`` log-densities, as the objective's family scores them."""
    jnp = jax.numpy
    gammaln = jax.scipy.special.gammaln
    m = objective._n_states
    block = objective._emission_slice
    y = observations[..., None]
    # Shaped (..., 1) by the objective, so it broadcasts along the states.
    given = objective.covariate
    covariate = None if given is None else jnp.asarray(given.numpy())
    if isinstance(objective, GaussianHmmObjective):
        means, scales = objective._mean_slice(), objective._log_scale_slice()

        def gaussian(theta: Any) -> Any:
            mean, log_scale = theta[means], theta[scales]
            z = (y - mean) / jnp.exp(log_scale)
            return -0.5 * jnp.log(2.0 * jnp.pi) - log_scale - 0.5 * z * z

        return gaussian
    if isinstance(objective, PoissonHmmObjective):

        def poisson(theta: Any) -> Any:
            rate = jnp.exp(theta[block])
            return y * jnp.log(rate) - rate - gammaln(y + 1.0)

        return poisson
    if isinstance(objective, NegativeBinomialHmmObjective):
        dispersions = objective._log_dispersion_slice()
        means = objective._log_mean_slice()

        def negative_binomial(theta: Any) -> Any:
            r, mu = jnp.exp(theta[dispersions]), jnp.exp(theta[means])
            if covariate is not None:
                mu = covariate * mu
            total = r + mu
            return (
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1.0)
                + r * jnp.log(r / total)
                + y * jnp.log(mu / total)
            )

        return negative_binomial
    if isinstance(objective, BetaBinomialHmmObjective):
        alphas = objective._log_alpha_slice()
        betas = objective._log_beta_slice()
        declared = jnp.asarray(objective._trials.numpy())
        n = declared if covariate is None else covariate

        def beta_binomial(theta: Any) -> Any:
            a, b = jnp.exp(theta[alphas]), jnp.exp(theta[betas])
            return (
                gammaln(n + 1.0)
                - gammaln(y + 1.0)
                - gammaln(n - y + 1.0)
                + gammaln(y + a)
                + gammaln(n - y + b)
                - gammaln(n + a + b)
                + gammaln(a + b)
                - gammaln(a)
                - gammaln(b)
            )

        return beta_binomial
    if isinstance(objective, BinomialHmmObjective):
        trials = jnp.asarray(objective._trials.numpy())

        def binomial(theta: Any) -> Any:
            p = jax.nn.sigmoid(theta[block])
            return (
                gammaln(trials + 1.0)
                - gammaln(y + 1.0)
                - gammaln(trials - y + 1.0)
                + y * jnp.log(p)
                + (trials - y) * jnp.log1p(-p)
            )

        return binomial
    if type(objective) is HmmObjective:
        k = objective._n_symbols
        symbols = observations.astype(int)

        def categorical(theta: Any) -> Any:
            free = theta[block].reshape(m, k - 1)
            pinned = jnp.concatenate([jnp.zeros((m, 1)), free], axis=1)
            log_emission = jax.nn.log_softmax(pinned, axis=1)
            return jnp.moveaxis(log_emission[:, symbols], 0, -1)

        return categorical
    msg = f"no JAX twin for {type(objective).__name__}"
    raise TypeError(msg)
