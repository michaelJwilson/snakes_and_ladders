"""An HMM objective's negative log-likelihood and gradient under JAX (issues #1000, #1189, #1206).

Each twin is keyed on :class:`~sal.opt.hmm.EmissionHmmObjective`'s start
family and reads the objective's own ``theta`` layout (:attr:`blocks`) and
constraint maps --- a
simplex row is ``log_softmax([0, free])``, a positive parameter ``exp``, a
probability the logistic --- and scores the observations by the scaled
forward recursion, whose reverse pass is the backward recursion (see
:func:`_forward`), so ``jit(value_and_grad)`` of it is the gradient PyTorch's
autograd takes through ``__call__``, the oracle. A twin exists for a
categorical, Gaussian, Poisson, binomial, beta-binomial (either reading) or
negative binomial family of scalar observations, and for a count pair ---
joint or independent, either reading of its success channel --- of two
channels (:func:`twinned`).

Segments of unequal length are padded to the longest and masked
(:func:`_layout`): a padded step of the forward recursion is the identity
and carries no posterior, so one program serves every segment. A count
family's rising factorials, the density's cost, are taken once per distinct
count (:func:`_count_tables`).

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

from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
    RateConcentrationBetaBinomialEmission,
    RateConcentrationCountPairEmission,
)
from sal.opt.hmm import EmissionHmmObjective
from sal.ragged import Ragged

#: The families a twin is written for, by exact type: a subclass may score
#: otherwise.
FAMILIES: dict[type, str] = {
    CategoricalEmission: "categorical",
    GaussianEmission: "gaussian",
    PoissonEmission: "poisson",
    BinomialEmission: "binomial",
    BetaBinomialEmission: "beta_binomial",
    RateConcentrationBetaBinomialEmission: "beta_binomial",
    NegativeBinomialEmission: "negative_binomial",
    CountPairEmission: "count_pair",
    RateConcentrationCountPairEmission: "count_pair",
}

#: ``sim.count_pairs.IndependentCountPair``, keyed by its qualified name: `opt/`
#: imports nothing from `sim/` (``test_directory_imports.py``), and the
#: family is the independent count pair whatever imports it.
INDEPENDENT_PAIR = "sal.sim.count_pairs.IndependentCountPair"

#: The success channel's two readings, by exact type.
_READINGS: dict[type, tuple[str, str]] = {
    BetaBinomialEmission: ("alpha", "beta"),
    RateConcentrationBetaBinomialEmission: ("rate", "concentration"),
}

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
    #: A beta-binomial channel read by ``(rate, concentration)``, not ``(a, b)``.
    rate_concentration: bool = False
    #: A count pair's success trials are its observed total.
    joint: bool = False
    #: Segments of unequal length, scored end to end, then gathered into a
    #: padded layout and masked.
    masked: bool = False
    #: A zero exposure marks a total unobserved somewhere: it scores log 1.
    unseen: bool = False
    #: The rising-factorial arguments taken once per distinct count
    #: (:func:`_count_tables`).
    tables: tuple[str, ...] = ()


def _kind(family: EmissionFamily) -> str | None:
    """The twin ``family`` takes, or ``None``."""
    kind = FAMILIES.get(type(family))
    if kind is not None:
        return kind
    named = f"{type(family).__module__}.{type(family).__qualname__}"
    if (
        named == INDEPENDENT_PAIR
        and type(getattr(family, "total", None)) is NegativeBinomialEmission
        and type(getattr(family, "successes", None)) in _READINGS
    ):
        return "count_pair"
    return None


def twinned(objective: EmissionHmmObjective) -> bool:
    """Whether ``objective`` has a twin: a family :func:`_kind` names, its observations' shape, and JAX installed.

    A scalar family takes ``(total,)`` observations and a count pair
    ``(total, 2)``; the segments may differ in length. A joint count pair
    with a covariate has none: the family refuses one when it scores.
    """
    import importlib.util

    kind = _kind(objective.start)
    observations = objective.observations
    if kind == "count_pair":
        shaped = observations.dim() == 2 and observations.shape[1] == 2
        if getattr(objective.start, "joint", False) and objective.covariate is not None:
            return False
    else:
        shaped = observations.dim() == 1
    return kind is not None and shaped and importlib.util.find_spec("jax") is not None


def value_and_grad(
    objective: EmissionHmmObjective,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """``theta -> (U(theta), dU/dtheta)`` under ``jit``, the objective's negative log-likelihood.

    Every family of :data:`FAMILIES` is covered as the package states it:
    a negative binomial's covariate is an exposure scaling each rate, a
    beta-binomial's a trial count per observation, an independent count
    pair's one of each by channel, and the other families refuse one when
    they score, so no twin meets it.

    Raises
    ------
    TypeError
        If the objective has no twin (:func:`twinned`).
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


def _layout(lengths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray | None]:
    """``(rows, mask)``: each ``(segment, step)``'s row of the observations, and which steps are real.

    :meth:`~sal.ragged.Ragged.padded` of the row numbers. Segments of one
    length are read back as ``(n_sequences, length)`` and ``mask`` is
    ``None``. Otherwise every segment is padded to the longest with its own
    first row, so a padded step gathers a density the family supports,
    finite; ``mask`` is ``False`` there.
    """
    block, mask = Ragged(np.arange(sum(lengths)), lengths).padded()
    if bool(mask.all()):
        return block, None
    return np.where(mask, block, block[:, :1]), mask


def _names(family: EmissionFamily, kind: str) -> tuple[tuple[str, ...], bool]:
    """The emission blocks' names in the order the density reads them, and whether a beta-binomial channel is read by rate and concentration."""
    if kind == "beta_binomial":
        reading = _READINGS[type(family)]
        return reading, reading[0] == "rate"
    if kind == "count_pair":
        if isinstance(family, CountPairEmission):
            success: tuple[str, ...] = (
                ("rate", "concentration")
                if type(family) is RateConcentrationCountPairEmission
                else ("alpha", "beta")
            )
            return ("dispersion", "mean", *success), success[0] == "rate"
        reading = _READINGS[type(family.successes)]  # type: ignore[attr-defined]
        return (
            "total.dispersion",
            "total.mean",
            *(f"successes.{name}" for name in reading),
        ), reading[0] == "rate"
    return {
        "gaussian": ("mean", "scale"),
        "poisson": ("mean",),
        "negative_binomial": ("dispersion", "mean"),
        "binomial": ("probability",),
        "categorical": ("log_emission",),
    }[kind], False


def _binomial_constant(successes: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """``log C(n, y)``, and ``-inf`` where ``y > n``, outside the support."""
    remaining = np.maximum(trials - successes, 0.0)
    constant = (
        gammaln(trials + 1.0) - gammaln(successes + 1.0) - gammaln(remaining + 1.0)
    )
    return np.where(successes > trials, -np.inf, constant)


def _prepared(
    objective: EmissionHmmObjective,
) -> tuple[_Structure, dict[str, np.ndarray]]:
    """The objective's structure, and its data in NumPy with the terms free of ``theta`` read once.

    Segments of one length are read back as ``(n_sequences, length)`` rows
    (:func:`_layout`), a scalar observation and each channel of a pair as
    ``(n_sequences, length, 1)``, which broadcasts along the states, and the
    covariate likewise, by channel for a pair. Segments of unequal length
    stay end to end, ``(total, 1)``, and ``index`` gathers their densities
    into the padded ``(n_sequences, longest)`` layout, so a padded step costs
    a gather and no density (issue #1206). A zero exposure scores a
    total at log 1 and a zero trial count the successes, as the families
    score them (issue #933).
    """
    family = objective.start
    if not twinned(objective):
        msg = f"no JAX twin for an HMM of {type(family).__name__}"
        raise TypeError(msg)
    kind = _kind(family)
    assert kind is not None
    rows, mask = _layout(objective.lengths)
    # Unequal segments are scored where they are, end to end, and gathered
    # into the padded layout after: no density is taken at a padded step.
    layout: Any = rows if mask is None else slice(None)
    observations = objective.observations.numpy()[layout]
    given = objective.covariate
    covariate = given is not None
    cov = None if given is None else given.numpy()[layout]
    data: dict[str, np.ndarray] = {}
    unseen = False
    joint = False
    n_symbols = 0

    def exposed(exposure: np.ndarray) -> np.ndarray:
        # A zero exposure marks the count unobserved: scored at a unit
        # exposure, then weighted out by `seen`.
        nonlocal unseen
        seen = exposure != 0.0
        if not bool(seen.all()):
            unseen = True
            data["seen"] = seen.astype(np.float64)
        return np.where(seen, exposure, 1.0)

    def weighted(constant: np.ndarray) -> np.ndarray:
        return data["seen"] * constant if unseen else constant

    if kind == "count_pair":
        y = observations[..., 0:1].astype(np.float64)
        s = observations[..., 1:2].astype(np.float64)
        joint = bool(getattr(family, "joint", False))
        data["y"] = y
        if cov is not None:
            data["exposure"] = exposed(cov[..., 0:1].astype(np.float64))
            n = cov[..., 1:2].astype(np.float64)
        elif joint:
            n = y
        else:
            # The independent form's fixed count, on the pair or on its
            # success channel.
            holder: Any = (
                family if isinstance(family, CountPairEmission) else family.successes  # type: ignore[attr-defined]
            )
            n = np.asarray(holder.trials.numpy(), dtype=np.float64)
        # A zero trial count marks the successes unobserved; scored at zero
        # successes, every term is zero.
        if cov is not None:
            s = np.where(n == 0.0, 0.0, s)
        data["rising.total"] = y
        _beta_binomial_terms(data, s, n)
        data["constant"] = weighted(-gammaln(y + 1.0)) + _binomial_constant(s, n)
    elif kind == "categorical":
        n_symbols = family.n_symbols  # type: ignore[attr-defined]
        data["symbols"] = observations.astype(np.int64)
    else:
        y = observations.astype(np.float64)[..., None]
        if kind == "beta_binomial":
            n = (
                cov.astype(np.float64)
                if cov is not None
                else np.asarray(family.trials.numpy(), dtype=np.float64)  # type: ignore[attr-defined]
            )
            if cov is not None:
                y = np.where(n == 0.0, 0.0, y)
            _beta_binomial_terms(data, y, n)
            data["constant"] = _binomial_constant(y, n)
        elif kind == "binomial":
            trials = family.trials.numpy()  # type: ignore[attr-defined]
            data["trials"] = trials
            data["constant"] = _binomial_constant(y, trials)
        elif kind in ("poisson", "negative_binomial"):
            if cov is not None:
                data["covariate"] = exposed(cov.astype(np.float64))
            data["constant"] = weighted(-gammaln(y + 1.0))
            if kind == "negative_binomial":
                data["rising.total"] = y
        data["y"] = y
    names, rate_concentration = _names(family, kind)
    blocks = objective.blocks
    spans = (
        _span(blocks["log_initial"]),
        _span(blocks["log_transition"]),
        *(_span(blocks[name]) for name in names),
    )
    tabled = kind not in ("gaussian", "categorical") and _table(data)
    tables = () if kind == "categorical" else _count_tables(data)
    if mask is not None:
        data["index"] = data["index"][rows] if tabled else rows
        data["mask"] = mask.astype(np.float64)
    structure = _Structure(
        kind,
        objective.n_states,
        n_symbols,
        spans,
        covariate,
        tabled,
        rate_concentration=rate_concentration,
        joint=joint,
        masked=mask is not None,
        unseen=unseen,
        tables=tables,
    )
    return structure, data


def _beta_binomial_terms(
    data: dict[str, np.ndarray], successes: np.ndarray, trials: np.ndarray
) -> None:
    """The beta-binomial's three rising-factorial arguments: ``y``, ``n - y`` and ``n``.

    ``n - y`` is held at zero or above, so a pair outside the support, whose
    constant is ``-inf``, has a finite term and no ``nan`` reaches a
    gradient. A trial count per state gives ``n - y`` an axis of states.
    """
    data["rising.successes"] = successes
    data["rising.remaining"] = np.maximum(trials - successes, 0.0)
    data["rising.trials"] = trials


def _count_tables(data: dict[str, np.ndarray]) -> tuple[str, ...]:
    """Reduce each rising-factorial argument in place to its distinct rows, with an ``index`` back; the names reduced.

    ``gammaln(y + x) - gammaln(x)`` and its ``digamma`` gradient are the
    density's cost, and a count takes few values: on the 7,644 rows of
    issue #1206 the totals take a few hundred. Each argument is taken once per
    distinct value and per state, and gathered back to the positions; the
    gather's transpose, a scatter-add, sums the cotangents into the table.
    Each value is the per-position one, elementwise. An argument whose
    distinct rows are more than :data:`TABLE_SHARE` of its rows is kept
    whole. The table is padded to a power of two with copies of its first
    row, which nothing indexes, so a table of another length reuses the
    compiled program.
    """
    positions = data["y"].shape[:-1]
    reduced: list[str] = []
    for key in sorted(key for key in data if key.startswith("rising.")):
        value = data[key]
        if value.ndim != len(positions) + 1 or value.shape[:-1] != positions:
            continue
        flat = value.reshape(-1, value.shape[-1])
        _, first, inverse = np.unique(
            flat, axis=0, return_index=True, return_inverse=True
        )
        if len(first) > TABLE_SHARE * len(flat):
            continue
        rows = np.zeros(1 << (len(first) - 1).bit_length(), dtype=np.int64)
        rows[: len(first)] = first
        data[key] = flat[rows]
        data[f"{key}.index"] = inverse.reshape(positions)
        reduced.append(key.removeprefix("rising."))
    return tuple(reduced)


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
    (:mod:`sal.sample.hmc.jax`) traces inside its own loop.
    """
    import jax

    jnp = jax.numpy
    m = structure.n_states
    (initial, transition, *emission) = (slice(*block) for block in structure.blocks)
    log_density = _density(structure, emission, jax)
    forward = _forward(jax, masked=structure.masked)

    def simplex(free: Any) -> Any:
        pinned = jnp.concatenate([jnp.zeros(free.shape[:-1] + (1,)), free], axis=-1)
        return jax.nn.log_softmax(pinned, axis=-1)

    def negative_log_likelihood(theta: Any, data: dict[str, Any]) -> Any:
        log_initial = simplex(theta[initial])
        log_transition = simplex(theta[transition].reshape(m, m - 1))
        emit = log_density(theta, data)
        if structure.tabled or structure.masked:
            emit = emit[data["index"]]
        if structure.masked:
            return -forward(log_initial, log_transition, emit, data["mask"])
        return -forward(log_initial, log_transition, emit)

    return negative_log_likelihood


@functools.cache
def _compiled(structure: _Structure) -> Any:
    """``jit(value_and_grad)`` of the negative log-likelihood, once per structure."""
    import jax

    return jax.jit(jax.value_and_grad(_negative_log_likelihood(structure)))


def jax_energy(
    objective: EmissionHmmObjective,
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

    def success(theta: Any, first: slice, second: slice) -> tuple[Any, Any]:
        # ``(a, b)``: both positive, or ``(tau p, tau (1 - p))`` as
        # `emissions.counts._rate_concentration` forms them.
        if not structure.rate_concentration:
            return jnp.exp(theta[first]), jnp.exp(theta[second])
        p, tau = jax.nn.sigmoid(theta[first]), jnp.exp(theta[second])
        return tau * p, tau * (1.0 - p)

    def seen(data: dict[str, Any], scores: Any) -> Any:
        return data["seen"] * scores if structure.unseen else scores

    def at(data: dict[str, Any], name: str, x: Any) -> Any:
        # The rising factorial of the argument `name`, per position: taken
        # over its table and gathered where it has one.
        value = rising(data[f"rising.{name}"], x)
        if name in structure.tables:
            return value[data[f"rising.{name}.index"]]
        return value

    def beta_binomial_terms(data: dict[str, Any], a: Any, b: Any) -> Any:
        # The beta-binomial's log-density less ``log C(n, y)``.
        return (
            at(data, "successes", a)
            + at(data, "remaining", b)
            - at(data, "trials", a + b)
        )

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
            r, mu = jnp.exp(theta[dispersions]), jnp.exp(theta[means])
            if structure.covariate:
                mu = data["covariate"] * mu
            return data["constant"] + seen(
                data, _negative_binomial(data["y"], r, mu, jnp, at(data, "total", r))
            )

        return negative_binomial
    if family == "beta_binomial":
        first, second = emission

        def beta_binomial(theta: Any, data: dict[str, Any]) -> Any:
            return data["constant"] + beta_binomial_terms(
                data, *success(theta, first, second)
            )

        return beta_binomial
    if family == "count_pair":
        dispersions, means, first, second = emission

        def count_pair(theta: Any, data: dict[str, Any]) -> Any:
            r, mu = jnp.exp(theta[dispersions]), jnp.exp(theta[means])
            if structure.covariate:
                mu = data["exposure"] * mu
            total = _negative_binomial(data["y"], r, mu, jnp, at(data, "total", r))
            return (
                data["constant"]
                + seen(data, total)
                + beta_binomial_terms(data, *success(theta, first, second))
            )

        return count_pair
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


def _negative_binomial(y: Any, r: Any, mu: Any, jnp: Any, rising: Any) -> Any:
    """The negative binomial's log-density less ``-log y!``, given ``rising``, the rising factorial of ``(y, r)``.

    ``-r log1p(mu / r)`` is ``r log(r / (r + mu))`` without the cancellation
    at large ``r``.
    """
    return rising - r * jnp.log1p(mu / r) + y * jnp.log(mu / (r + mu))


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

    def small(count: Any, x: Any) -> Any:
        return gammaln(count + x) - gammaln(x)

    def either(count: Any, x: Any) -> Any:
        positive = jnp.where(count > 0, count, 1.0)
        large = jnp.where(count > 0, gammaln(positive) - betaln(positive, x), 0.0)
        return jnp.where(x > RISING_FROM, large, small(count, x))

    # ``x`` is a parameter per state, so whether any reaches the threshold is
    # one predicate: below it, the common case, ``betaln`` is not taken at
    # every position (issue #1206), and the value is the one ``either``
    # selects, bitwise.
    shape = jnp.broadcast_shapes(jnp.shape(count), jnp.shape(x))
    count = jnp.asarray(count, dtype=jnp.result_type(count, x))
    return jax.lax.cond(
        jnp.any(x > RISING_FROM),
        lambda c, v: jnp.broadcast_to(either(c, v), shape),
        lambda c, v: jnp.broadcast_to(small(c, v), shape),
        count,
        x,
    )


def _forward(jax: Any, *, masked: bool = False) -> Any:
    """``(log_initial, log_transition, emit[, mask]) -> ln P(x)`` summed over sequences, with its own reverse pass.

    The forward pass runs in probability space scaled per position (Rabiner
    1989, section V.A), each emission column shifted by its maximum so no
    ``exp`` underflows: ``alpha_t = (alpha_{t-1} T) * b_t / c_t`` and
    ``ln P(x) = sum_t ln c_t + sum_t shift_t``. The reverse pass is the
    backward recursion rather than autodiff through the scan: the gradient in
    ``emit`` is the posterior ``gamma``, in ``log_transition`` the expected
    pair counts, in ``log_initial`` the first posterior (the Fisher
    identity), so what is kept is ``alpha`` and ``c``, not every step's
    ``(n, m, m)`` intermediate.

    ``masked`` takes a ``(n_sequences, length)`` mask, ``1`` on a real step
    and ``0`` on padding past a segment's end (:func:`_layout`). A padded
    step is the identity in both passes --- ``alpha`` carried, ``c = 1``,
    ``beta`` carried, no pair counted --- and its emission column is read as
    zero, so its shift is zero and its posterior, the gradient in ``emit``,
    is zero: the value and gradient are the unpadded segments'. Padding sits
    at the end of a segment, so the backward pass starts each from
    ``beta = 1`` at its own last step.
    """
    jnp = jax.numpy

    def run(
        log_initial: Any, log_transition: Any, emit: Any, mask: Any
    ) -> tuple[Any, Any]:
        transition = jnp.exp(log_transition)
        # `None` is an empty pytree: unmasked, the scans carry no mask.
        steps = None
        if masked:
            keep = mask[..., None] > 0.0
            emit = jnp.where(keep, emit, 0.0)
            steps = jnp.moveaxis(keep, 1, 0)
        shift = jnp.max(emit, axis=-1, keepdims=True)
        # (length, n_sequences, m): the scan walks the leading axis.
        b = jnp.moveaxis(jnp.exp(emit - shift), 1, 0)
        first = jnp.exp(log_initial) * b[0]
        c0 = first.sum(-1)
        first = first / c0[:, None]

        def step(alpha: Any, xs: Any) -> tuple[Any, tuple[Any, Any]]:
            column, real = xs
            onward = (alpha @ transition) * column
            c = onward.sum(-1)
            onward = onward / c[:, None]
            if masked:
                onward = jnp.where(real, onward, alpha)
                c = jnp.where(real[:, 0], c, 1.0)
            return onward, (onward, c)

        _, (alphas, cs) = jax.lax.scan(
            step, first, (b[1:], None if steps is None else steps[1:])
        )
        alphas = jnp.concatenate([first[None], alphas])
        cs = jnp.concatenate([c0[None], cs])
        return jnp.sum(jnp.log(cs)) + jnp.sum(shift), (
            transition,
            b,
            alphas,
            cs,
            steps,
        )

    def backward(residual: Any, g: Any) -> tuple[Any, ...]:
        transition, b, alphas, cs, steps = residual

        def step(carry: Any, xs: Any) -> tuple[Any, Any]:
            beta, pairs = carry
            previous, column, c, real = xs
            onward = column * beta / c[:, None]
            if masked:
                pairs = pairs + previous.T @ jnp.where(real, onward, 0.0)
                beta = jnp.where(real, onward @ transition.T, beta)
            else:
                pairs = pairs + previous.T @ onward
                beta = onward @ transition.T
            return (beta, pairs), previous * beta

        last = jnp.ones_like(alphas[-1])
        (_, pairs), posterior = jax.lax.scan(
            step,
            (last, jnp.zeros_like(transition)),
            (alphas[:-1], b[1:], cs[1:], None if steps is None else steps[1:]),
            reverse=True,
        )
        gamma = jnp.concatenate([posterior, alphas[-1][None]])
        if masked:
            gamma = jnp.where(steps, gamma, 0.0)
        return (
            g * gamma[0].sum(0),
            g * pairs * transition,
            g * jnp.moveaxis(gamma, 0, 1),
        )

    if masked:

        @jax.custom_vjp  # type: ignore[untyped-decorator]
        def forward(log_initial: Any, log_transition: Any, emit: Any, mask: Any) -> Any:
            return run(log_initial, log_transition, emit, mask)[0]

        def reverse(residual: Any, g: Any) -> tuple[Any, ...]:
            *rest, mask = residual
            return (*backward(tuple(rest), g), jnp.zeros_like(mask))

        def ahead(*arguments: Any) -> tuple[Any, Any]:
            value, residual = run(*arguments)
            return value, (*residual, arguments[3])

        forward.defvjp(ahead, reverse)
        return forward

    @jax.custom_vjp  # type: ignore[untyped-decorator]
    def unmasked(log_initial: Any, log_transition: Any, emit: Any) -> Any:
        return run(log_initial, log_transition, emit, None)[0]

    def plain(log_initial: Any, log_transition: Any, emit: Any) -> tuple[Any, Any]:
        return run(log_initial, log_transition, emit, None)

    unmasked.defvjp(plain, backward)
    return unmasked
