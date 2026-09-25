"""The HMM objectives' negative log-likelihood and gradient under JAX, their default route (issue #1000).

Each twin reads the objective's own ``theta`` layout and constraint maps --- a
simplex row is ``log_softmax([0, free])``, a positive parameter ``exp``, a
probability the logistic --- and scores the observations by the scaled
forward recursion, whose reverse pass is the backward recursion (see
:func:`_forward`), so ``jit(value_and_grad)`` of it is the gradient PyTorch's
autograd takes through ``__call__``. PyTorch autograd, asked for as
:data:`~sal.backend.Backend.TORCH`, is the oracle.

What is compiled depends on the objective's structure alone --- family,
state and symbol counts, ``theta`` layout, covariate and table --- and is
cached on it (:func:`_compiled`), so a second objective of the same structure
and shape reuses the program; the observations are its arguments, not
constants folded into it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln

from sal.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    GaussianHmmObjective,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
    _HmmObjective,
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

#: Above this argument the log rising factorial is taken through ``betaln``.
RISING_FROM = 1e3

#: A table is kept where the distinct cells are at most this share of the positions.
TABLE_SHARE = 0.25


@dataclass(frozen=True)
class _Structure:
    """What the compiled program depends on, and nothing it is given at a call."""

    family: str
    n_states: int
    n_symbols: int
    #: ``(start, stop)`` of the initial, transition, and each emission block.
    blocks: tuple[tuple[int, int], ...]
    covariate: bool
    tabled: bool


def value_and_grad(
    objective: _HmmObjective,
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
    structure, host = _prepared(objective)
    compiled = _compiled(structure)
    # Placed once: every call reads the same device buffers.
    data = {key: jax.device_put(value) for key, value in host.items()}

    def call(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = compiled(np.asarray(theta, dtype=np.float64), data)
        return float(value), np.asarray(gradient)

    return call


def _span(block: slice) -> tuple[int, int]:
    return int(block.start), int(block.stop)


def _prepared(objective: _HmmObjective) -> tuple[_Structure, dict[str, np.ndarray]]:
    """The objective's structure, and its data in NumPy with the terms free of ``theta`` read once."""
    observations = objective.observations.numpy()
    y = observations.astype(np.float64)[..., None]
    data: dict[str, np.ndarray] = {"y": y}
    given = objective.covariate
    covariate = given is not None
    if given is not None:
        # Shaped (..., 1) by the objective, so it broadcasts along the states.
        data["covariate"] = given.numpy()
    emission: tuple[tuple[int, int], ...]
    n_symbols = 0
    if isinstance(objective, GaussianHmmObjective):
        family = "gaussian"
        emission = (_span(objective._mean_slice()), _span(objective._log_scale_slice()))
    elif isinstance(objective, PoissonHmmObjective):
        family = "poisson"
        emission = (_span(objective._emission_slice),)
        data["constant"] = -gammaln(y + 1.0)
    elif isinstance(objective, NegativeBinomialHmmObjective):
        family = "negative_binomial"
        emission = (
            _span(objective._log_dispersion_slice()),
            _span(objective._log_mean_slice()),
        )
        data["constant"] = -gammaln(y + 1.0)
    elif isinstance(objective, BetaBinomialHmmObjective):
        family = "beta_binomial"
        emission = (
            _span(objective._log_alpha_slice()),
            _span(objective._log_beta_slice()),
        )
        n = data["covariate"] if covariate else objective._trials.numpy()
        data["trials"] = np.asarray(n, dtype=np.float64)
        data["constant"] = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
    elif isinstance(objective, BinomialHmmObjective):
        family = "binomial"
        emission = (_span(objective._emission_slice),)
        trials = objective._trials.numpy()
        data["trials"] = trials
        data["constant"] = (
            gammaln(trials + 1.0) - gammaln(y + 1.0) - gammaln(trials - y + 1.0)
        )
    elif type(objective) is HmmObjective:
        family = "categorical"
        emission = (_span(objective._emission_slice),)
        n_symbols = objective._n_symbols
        data = {"symbols": observations.astype(np.int64)}
    else:
        msg = f"no JAX twin for {type(objective).__name__}"
        raise TypeError(msg)
    blocks = (
        _span(objective._initial_slice),
        _span(objective._transition_slice),
        *emission,
    )
    tabled = family not in ("gaussian", "categorical") and _table(data)
    structure = _Structure(
        family, objective._n_states, n_symbols, blocks, covariate, tabled
    )
    return structure, data


def _table(data: dict[str, np.ndarray]) -> bool:
    """Reduce ``data`` in place to one row per distinct cell, with the ``index`` back to the positions.

    A count family's density at a position depends on the position only
    through its count and covariate, so over counts that repeat --- the
    common case --- the ``gammaln`` terms and their ``digamma`` gradients are
    taken once per distinct cell and the gather's transpose, a scatter-add,
    sums the posteriors into them. Each cell's value is the per-position one,
    elementwise, so nothing moves, and the padding rows gather no posterior.
    Where the cells are more than :data:`TABLE_SHARE` of the positions the
    gather does not repay itself, ``data`` is left whole and ``False``
    returned.
    """
    shape = data["y"].shape
    positions = shape[:-1]
    # Per position is what spans the counts' axes; the cell is the per-position
    # values without an axis of states, and what has one --- a constant term
    # under per-state trials --- is a function of the cell and is gathered.
    # A per-state array such as a binomial's declared trials stays whole.
    per_position = [
        key
        for key, value in data.items()
        if value.ndim == len(shape) and value.shape[:-1] == positions
    ]
    keys = [key for key in per_position if data[key].shape[-1] == 1]
    cells = np.stack([data[key].reshape(-1) for key in keys], axis=1)
    _, first, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    if len(first) > TABLE_SHARE * len(cells):
        return False
    # Padded to a power of two with copies of the first cell that no position
    # indexes, so a table of another length reuses the compiled program.
    rows = np.zeros(1 << (len(first) - 1).bit_length(), dtype=np.int64)
    rows[: len(first)] = first
    for key in per_position:
        data[key] = data[key].reshape(-1, data[key].shape[-1])[rows]
    data["index"] = inverse.reshape(positions)
    return True


@functools.cache
def _negative_log_likelihood(
    structure: _Structure,
) -> Callable[[Any, dict[str, Any]], Any]:
    """``(theta, data) -> U(theta)``, traceable, once per structure (issue #1008).

    What :func:`_compiled` differentiates and a compiled chain
    (:mod:`sal.sample.hmc_jax`) traces inside its own loop.
    """
    import jax

    jnp = jax.numpy
    m = structure.n_states
    (initial, transition, *emission) = (slice(*block) for block in structure.blocks)
    log_density = _density(structure, emission, jax)
    forward = _forward(jax)

    def simplex(free: Any) -> Any:
        pinned = jnp.concatenate([jnp.zeros(free.shape[:-1] + (1,)), free], axis=-1)
        return jax.nn.log_softmax(pinned, axis=-1)

    def negative_log_likelihood(theta: Any, data: dict[str, Any]) -> Any:
        log_initial = simplex(theta[initial])
        log_transition = simplex(theta[transition].reshape(m, m - 1))
        emit = log_density(theta, data)
        if structure.tabled:
            emit = emit[data["index"]]
        return -forward(log_initial, log_transition, emit)

    return negative_log_likelihood


@functools.cache
def _compiled(structure: _Structure) -> Any:
    """``jit(value_and_grad)`` of the negative log-likelihood, once per structure."""
    import jax

    return jax.jit(jax.value_and_grad(_negative_log_likelihood(structure)))


def jax_energy(
    objective: _HmmObjective,
) -> tuple[Callable[[Any, Any], Any], dict[str, Any]]:
    """The objective's negative log-likelihood as a traceable ``(theta, data)`` function, and its data on the device (issue #1008)."""
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    structure, host = _prepared(objective)
    return _negative_log_likelihood(structure), {
        key: jax.device_put(value) for key, value in host.items()
    }


def _density(
    structure: _Structure, emission: list[slice], jax: Any
) -> Callable[[Any, dict[str, Any]], Any]:
    """``(theta, data) -> (..., m)`` log-densities, as the objective's family scores them."""
    jnp = jax.numpy
    m, k = structure.n_states, structure.n_symbols
    family = structure.family

    rising = functools.partial(_rising, jax=jax)
    if family == "gaussian":
        means, scales = emission

        def gaussian(theta: Any, data: dict[str, Any]) -> Any:
            mean, log_scale = theta[means], theta[scales]
            z = (data["y"] - mean) / jnp.exp(log_scale)
            return -0.5 * jnp.log(2.0 * jnp.pi) - log_scale - 0.5 * z * z

        return gaussian
    if family == "poisson":
        (block,) = emission

        def poisson(theta: Any, data: dict[str, Any]) -> Any:
            rate = jnp.exp(theta[block])
            return data["y"] * jnp.log(rate) - rate + data["constant"]

        return poisson
    if family == "negative_binomial":
        dispersions, means = emission

        def negative_binomial(theta: Any, data: dict[str, Any]) -> Any:
            y = data["y"]
            r, mu = jnp.exp(theta[dispersions]), jnp.exp(theta[means])
            if structure.covariate:
                mu = data["covariate"] * mu
            # `-r log1p(mu / r)` is `r log(r / (r + mu))` without the
            # cancellation at large `r`.
            return (
                rising(y, r)
                + data["constant"]
                - r * jnp.log1p(mu / r)
                + y * jnp.log(mu / (r + mu))
            )

        return negative_binomial
    if family == "beta_binomial":
        alphas, betas = emission

        def beta_binomial(theta: Any, data: dict[str, Any]) -> Any:
            y, n = data["y"], data["trials"]
            a, b = jnp.exp(theta[alphas]), jnp.exp(theta[betas])
            return data["constant"] + rising(y, a) + rising(n - y, b) - rising(n, a + b)

        return beta_binomial
    if family == "binomial":
        (block,) = emission

        def binomial(theta: Any, data: dict[str, Any]) -> Any:
            y, trials = data["y"], data["trials"]
            p = jax.nn.sigmoid(theta[block])
            return data["constant"] + y * jnp.log(p) + (trials - y) * jnp.log1p(-p)

        return binomial
    (block,) = emission

    def categorical(theta: Any, data: dict[str, Any]) -> Any:
        free = theta[block].reshape(m, k - 1)
        pinned = jnp.concatenate([jnp.zeros((m, 1)), free], axis=1)
        log_emission = jax.nn.log_softmax(pinned, axis=1)
        return jnp.moveaxis(log_emission[:, data["symbols"]], 0, -1)

    return categorical


def _rising(count: Any, x: Any, jax: Any) -> Any:
    """``gammaln(count + x) - gammaln(x)``, the log rising factorial, for ``count >= 0``.

    The difference cancels: at ``x = 5.8e7`` it is off by 1e-7 and at
    ``4e11`` by 1e-3, against 40-digit ``mpmath``, which stalls a fit
    whose dispersion runs to a flat likelihood (issue #1000). Above
    :data:`RISING_FROM` it is ``gammaln(count) - betaln(count, x)``, whose
    error there is under 5e-13; below, ``betaln`` is the worse of the two
    (8e-10 at ``x = 95``) and the difference is kept.
    """
    jnp, gammaln = jax.numpy, jax.scipy.special.gammaln
    betaln = jax.scipy.special.betaln
    positive = jnp.where(count > 0, count, 1.0)
    large = jnp.where(count > 0, gammaln(positive) - betaln(positive, x), 0.0)
    small = gammaln(count + x) - gammaln(x)
    return jnp.where(x > RISING_FROM, large, small)


def _forward(jax: Any) -> Any:
    """``(log_initial, log_transition, emit) -> ln P(x)`` summed over sequences, with its own reverse pass.

    The forward pass runs in probability space scaled per position (Rabiner
    1989, section V.A), each emission column shifted by its maximum so no
    ``exp`` underflows: ``alpha_t = (alpha_{t-1} T) * b_t / c_t`` and
    ``ln P(x) = sum_t ln c_t + sum_t shift_t``. The reverse pass is the
    backward recursion rather than autodiff through the scan: the gradient in
    ``emit`` is the posterior ``gamma``, in ``log_transition`` the expected
    pair counts, in ``log_initial`` the first posterior (the Fisher
    identity), so what is kept is ``alpha`` and ``c``, not every step's
    ``(n, m, m)`` intermediate.
    """
    jnp = jax.numpy

    def run(log_initial: Any, log_transition: Any, emit: Any) -> tuple[Any, Any]:
        transition = jnp.exp(log_transition)
        shift = jnp.max(emit, axis=-1, keepdims=True)
        # (length, n_sequences, m): the scan walks the leading axis.
        b = jnp.moveaxis(jnp.exp(emit - shift), 1, 0)
        first = jnp.exp(log_initial) * b[0]
        c0 = first.sum(-1)
        first = first / c0[:, None]

        def step(alpha: Any, column: Any) -> tuple[Any, tuple[Any, Any]]:
            alpha = (alpha @ transition) * column
            c = alpha.sum(-1)
            alpha = alpha / c[:, None]
            return alpha, (alpha, c)

        _, (alphas, cs) = jax.lax.scan(step, first, b[1:])
        alphas = jnp.concatenate([first[None], alphas])
        cs = jnp.concatenate([c0[None], cs])
        return jnp.sum(jnp.log(cs)) + jnp.sum(shift), (transition, b, alphas, cs)

    @jax.custom_vjp  # type: ignore[untyped-decorator]
    def forward(log_initial: Any, log_transition: Any, emit: Any) -> Any:
        return run(log_initial, log_transition, emit)[0]

    def backward(residual: Any, g: Any) -> tuple[Any, Any, Any]:
        transition, b, alphas, cs = residual

        def step(carry: Any, xs: Any) -> tuple[Any, Any]:
            beta, pairs = carry
            previous, column, c = xs
            onward = column * beta / c[:, None]
            pairs = pairs + previous.T @ onward
            beta = onward @ transition.T
            return (beta, pairs), previous * beta

        last = jnp.ones_like(alphas[-1])
        (_, pairs), posterior = jax.lax.scan(
            step,
            (last, jnp.zeros_like(transition)),
            (alphas[:-1], b[1:], cs[1:]),
            reverse=True,
        )
        gamma = jnp.concatenate([posterior, alphas[-1][None]])
        return (
            g * gamma[0].sum(0),
            g * pairs * transition,
            g * jnp.moveaxis(gamma, 0, 1),
        )

    forward.defvjp(run, backward)
    return forward
