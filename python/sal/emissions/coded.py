"""Coded count observations and their log-emission (issue #1340).

A count is an integer, so the integer part of a count family's log-density
is a function of the count alone. :func:`encode` codes the observations once
per distinct ``(label, count)``, in Rust (``src/coded_emission.rs``): an
``int32`` inverse, ``-1`` where the covariate is zero (unobserved), a weight
per code, the distinct counts, and the covariate per observation, which is
never rounded. :func:`log_emission` builds the tables at the distinct counts
only --- not at every count up to the largest (#719: 7,109 rows in place of
154,549) --- completes the covariate term per observation in
:mod:`sal.emissions.nb`'s and :mod:`sal.emissions.bb`'s order (#1334,
#1336), and gathers. The kernel is ``src/dense_emission.rs``'s, read through
the inverse. A :class:`Dense` is scored by coding it first: one implementation,
so a :class:`Dense` score is its :class:`Coded` score bitwise. Each channel
reads its own table, one row per distinct value of that channel, so a count
shared by several labels, or a total shared by several pair codes, is one
row.

**Which families.** :class:`~sal.emissions.NegativeBinomialEmission`,
:class:`~sal.emissions.BetaBinomialEmission` (so
:class:`~sal.emissions.RateConcentrationBetaBinomialEmission`) and the
independent :class:`~sal.emissions.CountPairEmission`.

**Labels.** ``label`` is one integer array, one value per observation,
which the caller builds (for example ``x + Z y``); any number of distinct
values is coded. A tuple or a 2-D label is refused.

**Shift.** ``shift`` is a per-label log-rate offset on the negative binomial,
``lambda = c mu exp(shift[label])``, indexed by the label's value. It enters
per observation as the exposure ``c exp(shift[label])``, so no table varies
by label and none carries a label axis; ``shift=None`` is the unshifted
route, bitwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from sal import oxisal
from sal.emissions.bb import family_log_pmf, log_factorial, trial_tables
from sal.emissions.counts import BetaBinomialEmission, NegativeBinomialEmission
from sal.emissions.dense import IndependentPair, checked_counts
from sal.emissions.nb import count_log_factor
from sal.emissions.rising import digamma_rising, scaled_rising_table

__all__ = [
    "Coded",
    "Dense",
    "encode",
    "log_emission",
    "log_emission_partials",
    "log_emission_sum",
]

Family = NegativeBinomialEmission | BetaBinomialEmission | IndependentPair


@dataclass(frozen=True)
class Dense:
    """Every observation: ``counts`` ``(n,)``, or ``(n, 2)`` for a pair, and its covariate.

    ``covariate`` is ``(n,)`` for one channel, the exposure or the trial
    count, or ``(n, 2)`` for a pair; ``None`` for none. A zero marks the
    channel unobserved. ``label`` is ``(n,)`` integers, as :func:`encode`
    takes it.
    """

    counts: NDArray[np.float64]
    covariate: NDArray[np.float64] | None = None
    label: NDArray[np.int64] | None = None


@dataclass(frozen=True)
class Coded:
    """Observations coded per distinct ``(label, count)``, from :func:`encode`.

    Parameters
    ----------
    label : np.ndarray
        ``(U,)`` ``int64``, each code's label; 0 without one.
    counts : np.ndarray
        ``(U,)`` ``uint32`` distinct counts, ascending, or ``(U, 2)`` for a pair.
    inverse : np.ndarray
        ``(n,)`` ``int32``, the code of each observation, ``-1`` unobserved.
    weight : np.ndarray
        ``(U,)`` ``int64``, the observations carrying each code.
    covariate : np.ndarray | None
        ``(n,)`` or ``(n, 2)`` ``float64`` per observation, as :class:`Dense`.
    """

    label: NDArray[np.int64]
    counts: NDArray[np.uint32]
    inverse: NDArray[np.int32]
    weight: NDArray[np.int64]
    covariate: NDArray[np.float64] | None


def _counts(counts: NDArray[np.float64] | NDArray[np.uint32]) -> NDArray[np.uint32]:
    """``counts`` as ``(n,)`` or ``(n, 2)`` ``uint32``, refused otherwise."""
    values = np.asarray(counts)
    if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 2):
        msg = f"counts are (n,) or a pair's (n, 2), got {values.shape}"
        raise ValueError(msg)
    return checked_counts(values, "count").reshape(values.shape)


def _covariate(
    covariate: object | None, counts: NDArray[np.uint32]
) -> NDArray[np.float64] | None:
    """``covariate`` as ``float64`` shaped as ``counts``, refused otherwise."""
    if covariate is None:
        return None
    values = np.ascontiguousarray(covariate, dtype=np.float64)
    if values.shape != counts.shape:
        msg = f"a covariate is shaped as its counts, {counts.shape}, got {values.shape}"
        raise ValueError(msg)
    return values


def _labels(label: object | None, n: int) -> NDArray[np.int64] | None:
    """``label`` as one ``(n,)`` ``int64`` array, refused otherwise."""
    if label is None:
        return None
    if isinstance(label, tuple | list) and any(np.ndim(one) > 0 for one in label):
        msg = (
            "label is one integer array per observation; fold several into "
            "one code, for example x + Z * y"
        )
        raise ValueError(msg)
    values = np.asarray(label)
    if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
        msg = f"label is one ({n},) integer array, got {values.dtype} {values.shape}"
        raise ValueError(msg)
    return values.astype(np.int64)


def encode(
    counts: NDArray[np.float64] | NDArray[np.uint32],
    covariate: NDArray[np.float64] | None = None,
    *,
    label: NDArray[np.int64] | None = None,
) -> Coded:
    """Code ``counts`` per distinct ``(label, count)`` in Rust.

    Codes ascend by label, then by count.

    Parameters
    ----------
    counts : np.ndarray
        ``(n,)`` non-negative integer counts, or ``(n, 2)`` totals and successes.
    covariate : np.ndarray | None
        Shaped as ``counts``; an observation whose covariate is zero in every
        channel is unobserved and coded ``-1``.
    label : np.ndarray | None
        ``(n,)`` integer labels, one array built by the caller.

    Returns
    -------
    Coded

    Raises
    ------
    ValueError
        If a count is not a non-negative integer below ``2**32``, a shape
        disagrees, or ``label`` is a tuple or not one ``(n,)`` integer array.
    """
    values = _counts(counts)
    given = _covariate(covariate, values)
    wide = values.astype(np.uint64)
    keys = wide if wide.ndim == 1 else (wide[:, 0] << np.uint64(32)) | wide[:, 1]
    labels = _labels(label, values.shape[0])
    observed = None
    if given is not None:
        nonzero = given != 0.0
        observed = np.ascontiguousarray(
            nonzero if nonzero.ndim == 1 else nonzero.any(axis=1)
        )
    distinct, inverse, weight = oxisal.coded_encode(
        np.ascontiguousarray(keys), observed
    )
    code_label = np.zeros(distinct.size, dtype=np.int64)
    if labels is not None:
        # Two passes: the count codes, then (label index, count code) keyed
        # in 64 bits, so a pair's full 64-bit key and a label both fit.
        names, index = np.unique(labels, return_inverse=True)
        seen = inverse >= 0
        joint = np.zeros(inverse.size, dtype=np.uint64)
        joint[seen] = (index[seen].astype(np.uint64) << np.uint64(32)) | inverse[
            seen
        ].astype(np.uint64)
        pairs, inverse, weight = oxisal.coded_encode(
            np.ascontiguousarray(joint), np.ascontiguousarray(seen)
        )
        code_label = names[(pairs >> np.uint64(32)).astype(np.int64)]
        distinct = distinct[(pairs & np.uint64(2**32 - 1)).astype(np.int64)]
    if values.ndim == 1:
        coded = distinct.astype(np.uint32)
    else:
        coded = np.stack(
            [
                (distinct >> np.uint64(32)).astype(np.uint32),
                (distinct & np.uint64(2**32 - 1)).astype(np.uint32),
            ],
            axis=1,
        )
    return Coded(
        label=code_label,
        counts=coded,
        inverse=inverse,
        weight=weight,
        covariate=given,
    )


def _total(
    family: NegativeBinomialEmission,
    codes: NDArray[np.uint32],
    exposure: NDArray[np.float64] | None,
) -> dict[str, NDArray[np.generic]]:
    """The first channel at the distinct counts, as ``coded_log_emission`` takes it."""
    counts = np.ascontiguousarray(codes)
    if exposure is None:
        grid = torch.as_tensor(counts.astype(np.float64))
        table = family.log_density(grid).numpy()
        return {
            "totals": counts,
            "total_table": np.ascontiguousarray(table).reshape(-1),
        }
    return {
        "totals": counts,
        "total_table": np.ascontiguousarray(count_log_factor(family, counts)).reshape(
            -1
        ),
        "exposure": np.ascontiguousarray(exposure),
        "dispersion": np.ascontiguousarray(family.dispersion.detach().numpy()),
        "mean": np.ascontiguousarray(family.mean.detach().numpy()),
    }


def _successes(
    family: BetaBinomialEmission,
    codes: NDArray[np.uint32],
    trials: NDArray[np.float64] | None,
) -> dict[str, NDArray[np.generic]]:
    """The second channel at the distinct successes, as ``coded_log_emission`` takes it."""
    counts = np.ascontiguousarray(codes)
    alpha = family.alpha.detach().numpy()
    if trials is None:
        table = family_log_pmf(
            family,
            counts.astype(np.float64)[:, None],
            family.trials.detach().numpy(),
        )
        return {
            "successes": counts,
            "success_table": np.ascontiguousarray(table).reshape(-1),
        }
    per_observation = checked_counts(trials, "trial count")
    extent = int(per_observation.max()) + 1 if per_observation.size else 1
    # ``V`` and ``W`` are read at each observation's trial count, a covariate,
    # so they span its extent; ``U`` is read at the code.
    tables = trial_tables(family, 1, extent)
    success = scaled_rising_table(alpha, counts.astype(np.float64)).T
    return {
        "successes": counts,
        "success_table": np.ascontiguousarray(success).reshape(-1),
        "trials": per_observation,
        "failure_table": np.ascontiguousarray(tables.failure).reshape(-1),
        "trial_table": np.ascontiguousarray(tables.trial).reshape(-1),
        "log_factorial": log_factorial(extent),
        "log_rate": np.ascontiguousarray(tables.log_rate).reshape(-1),
    }


def _pair_channels(
    family: object,
) -> tuple[NegativeBinomialEmission, BetaBinomialEmission]:
    """An independent pair's two channels, refused as the dense route refuses them."""
    if getattr(family, "joint", False):
        msg = (
            "the joint pair's trial count is the observed total, where a total of "
            "zero is a support point and not an unobserved channel; score it "
            "with log_density"
        )
        raise TypeError(msg)
    total = getattr(family, "total", None)
    successes = getattr(family, "successes", None)
    if not (
        isinstance(total, NegativeBinomialEmission)
        and isinstance(successes, BetaBinomialEmission)
    ):
        msg = (
            f"a coded emission is over a negative binomial, a beta-binomial or "
            f"their independent pair; got {type(family).__name__}"
        )
        raise TypeError(msg)
    return total, successes


def _channel(
    values: NDArray[np.uint32], inverse: NDArray[np.int32]
) -> tuple[NDArray[np.uint32], NDArray[np.int32]]:
    """A channel's distinct values and each observation's row in them, ``-1`` kept."""
    distinct, index = np.unique(values, return_inverse=True)
    if distinct.size == values.size:
        return np.ascontiguousarray(values, dtype=np.uint32), inverse
    # One gather: a trailing -1 is what an unobserved -1 reads.
    lookup = np.append(index, -1).astype(np.int32)
    return np.ascontiguousarray(distinct, dtype=np.uint32), lookup[inverse]


def _shift_factor(
    coded: Coded, shift: NDArray[np.float64] | None
) -> NDArray[np.float64] | None:
    """``exp(shift[label])`` per observation, 1 where unobserved; ``None`` without a shift."""
    if shift is None:
        return None
    values = np.asarray(shift, dtype=np.float64)
    labels = coded.label
    if values.ndim != 1 or (
        labels.size and (labels.min() < 0 or labels.max() >= values.size)
    ):
        msg = (
            f"shift is one (L,) array indexed by label, 0 <= label < L; got "
            f"{values.shape} for labels up to {labels.max() if labels.size else 0}"
        )
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = "every shift must be finite"
        raise ValueError(msg)
    per_code = np.exp(values[labels])
    inverse = coded.inverse
    return np.where(inverse >= 0, per_code[np.maximum(inverse, 0)], 1.0)


def _shifted(
    exposure: NDArray[np.float64] | None, factor: NDArray[np.float64] | None
) -> NDArray[np.float64] | None:
    """The exposure ``c exp(shift[label])``; ``c`` is 1 where absent."""
    if factor is None:
        return exposure
    return factor if exposure is None else exposure * factor


def _coded_arguments(
    family: object, coded: Coded, shift: NDArray[np.float64] | None
) -> dict[str, NDArray[np.generic]]:
    """The kernel's channel arguments for ``family`` over ``coded``, rows per channel."""
    cov = coded.covariate
    pair = coded.counts.ndim == 2
    if isinstance(family, NegativeBinomialEmission | BetaBinomialEmission) and pair:
        msg = "a single-channel family scores (n,) counts, got a pair's (U, 2)"
        raise ValueError(msg)
    factor = _shift_factor(coded, shift)
    if isinstance(family, NegativeBinomialEmission):
        distinct, rows = _channel(coded.counts, coded.inverse)
        return {
            **_total(family, distinct, _shifted(cov, factor)),
            "total_rows": rows,
        }
    if isinstance(family, BetaBinomialEmission):
        distinct, rows = _channel(coded.counts, coded.inverse)
        return {**_successes(family, distinct, cov), "success_rows": rows}
    total, successes = _pair_channels(family)
    if not pair:
        msg = f"a pair's counts are (U, 2), got {coded.counts.shape}"
        raise ValueError(msg)
    totals, total_rows = _channel(coded.counts[:, 0], coded.inverse)
    succ, success_rows = _channel(coded.counts[:, 1], coded.inverse)
    exposure = _shifted(None if cov is None else cov[:, 0], factor)
    return {
        **_total(total, totals, exposure),
        "total_rows": total_rows,
        **_successes(successes, succ, None if cov is None else cov[:, 1]),
        "success_rows": success_rows,
    }


def log_emission(
    family: Family,
    observations: Dense | Coded,
    *,
    shift: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Every observation's log-density under every state, ``(K, n)``, state-major.

    Parameters
    ----------
    family : NegativeBinomialEmission | BetaBinomialEmission | IndependentPair
        The family scored; a pair must be independent.
    observations : Dense | Coded
        A :class:`Coded` is scored from tables at its distinct counts,
        gathered by its inverse; a :class:`Dense` is coded by :func:`encode`
        first, so the two are bitwise equal on the same observations.
    shift : np.ndarray | None
        ``(L,)`` log-rate offsets on the negative binomial, indexed by label:
        ``lambda = c mu exp(shift[label])``. ``None`` is no offset, bitwise.

    Returns
    -------
    np.ndarray
        ``float64``, C-contiguous, ``(K, n)``; an unobserved observation scores 0.

    Raises
    ------
    TypeError
        If the family is not one of those above, or the observations are neither type.
    ValueError
        If a count, a covariate or a shape is refused.
    """
    if isinstance(observations, Dense):
        observations = encode(
            observations.counts, observations.covariate, label=observations.label
        )
    if not isinstance(observations, Coded):
        msg = f"observations are Dense or Coded, got {type(observations).__name__}"
        raise TypeError(msg)
    arguments = _coded_arguments(family, observations, shift)
    n = observations.inverse.size
    out = np.empty(family.n_states * n)
    oxisal.coded_log_emission(family.n_states, observations.inverse, out, **arguments)
    return out.reshape(family.n_states, n)


def log_emission_sum(
    family: Family,
    coded: Coded,
    weights: NDArray[np.float64] | None = None,
    *,
    shift: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """``sum_i w[k, i] log f_k(x_i)``, ``(K,)``: a per-state ``bincount`` over the inverse.

    ``weights`` is ``(n,)``, ``(K, n)`` or ``None`` (each 1); an unobserved
    observation (``-1``) contributes nothing. Without a covariate the sum is
    ``sum_u table[k, u] W[k, u]`` with ``W`` the ``bincount`` of the weights
    per code; with one, per observation. Both are sequential in Rust
    (``coded_weighted_sum``), so the order is stated: bitwise a
    ``np.cumsum`` in that order, and within a reduction-order tolerance of
    the pairwise ``(w * log_emission).sum(axis=1)``. A code of zero total
    weight is skipped, so ``0 * -inf`` never enters. ``shift`` is
    :func:`log_emission`'s.
    """
    k = family.n_states
    w = None if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
    if coded.covariate is None:
        rows = np.arange(coded.weight.size, dtype=np.int32)
        table = log_emission(
            family,
            Coded(coded.label, coded.counts, rows, coded.weight, None),
            shift=shift,
        )
        values, index = table, coded.inverse
    else:
        values = log_emission(family, coded, shift=shift)
        n = coded.inverse.size
        index = np.where(
            coded.inverse >= 0, np.arange(n, dtype=np.int32), np.int32(-1)
        ).astype(np.int32)
    return oxisal.coded_weighted_sum(
        k,
        np.ascontiguousarray(values).reshape(-1),
        np.ascontiguousarray(index),
        None if w is None else w.reshape(-1),
    )


def _gathered(
    per_code: NDArray[np.float64], inverse: NDArray[np.int32]
) -> NDArray[np.float64]:
    """``(K, U)`` per code to ``(K, n)`` per observation; an unobserved one is 0."""
    out = per_code[:, np.maximum(inverse, 0)]
    out[:, inverse < 0] = 0.0
    return np.ascontiguousarray(out)


def _nb_partials(
    family: NegativeBinomialEmission,
    codes: NDArray[np.uint32],
    inverse: NDArray[np.int32],
    exposure: NDArray[np.float64] | None,
    factor: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """``d/dr``, ``d/dmu`` and, under a shift, ``d/dshift`` of the NB log pmf, ``(K, n)``.

    With ``lam = c mu exp(s)`` and ``q = lam / r``: ``d/dr = (psi(r + y) -
    psi(r)) - log1p(q) + (lam - y) / (r + lam)``, ``d/dmu = r (y - lam) / (mu
    (r + lam))`` and ``d/ds = r (y - lam) / (r + lam)``; at ``r = inf`` (the
    Poisson) ``0``, ``y / mu - c exp(s)`` and ``y - lam``. ``factor`` is
    ``exp(shift[label])`` per observation.
    """
    r = family.dispersion.detach().numpy()[:, None]
    mu = family.mean.detach().numpy()[:, None]
    y_code = codes.astype(np.float64)[None, :]
    finite = np.isfinite(r)
    rising = digamma_rising(np.where(finite, r, 1.0), y_code)
    y = _gathered(np.broadcast_to(y_code, rising.shape).copy(), inverse)
    rising = _gathered(rising, inverse)
    c = np.ones(inverse.size) if exposure is None else exposure
    scale = c if factor is None else c * factor
    lam = mu * scale[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = rising - np.log1p(lam / r) + (lam - y) / (r + lam)
        dmu = r * (y - lam) / (mu * (r + lam))
        ds = r * (y - lam) / (r + lam)
    dr = np.where(finite, dr, 0.0)
    dmu = np.where(finite, dmu, y / mu - scale[None, :])
    seen = (inverse >= 0) & (c != 0.0)
    out = {
        "dispersion": np.ascontiguousarray(np.where(seen, dr, 0.0)),
        "mean": np.ascontiguousarray(np.where(seen, dmu, 0.0)),
    }
    if factor is not None:
        ds = np.where(finite, ds, y - lam)
        out["shift"] = np.ascontiguousarray(np.where(seen, ds, 0.0))
    return out


def _bb_partials(
    family: BetaBinomialEmission,
    codes: NDArray[np.uint32],
    inverse: NDArray[np.int32],
    trials: NDArray[np.float64] | None,
) -> dict[str, NDArray[np.float64]]:
    """The beta-binomial's partials: ``alpha`` and ``beta``, or ``rate`` and ``concentration``.

    ``d/da = R(a, z) - R(a + b, n)`` and ``d/db = R(b, n - z) - R(a + b, n)``,
    ``R`` :func:`~sal.emissions.rising.digamma_rising`; with ``a = tau p``,
    ``d/dp = tau (d/da - d/db)`` and ``d/dtau = p d/da + (1 - p) d/db``. At
    ``tau = inf`` (the binomial) ``z / p - (n - z) / (1 - p)`` and ``0``. An
    unobserved observation, or one past its trials, has 0.
    """
    a = family.alpha.detach().numpy()[:, None]
    b = family.beta.detach().numpy()[:, None]
    z = _gathered(
        np.broadcast_to(
            codes.astype(np.float64)[None, :], (a.shape[0], codes.size)
        ).copy(),
        inverse,
    )
    n = (
        np.broadcast_to(family.trials.detach().numpy()[:, None], z.shape)
        if trials is None
        else np.broadcast_to(np.asarray(trials, dtype=np.float64)[None, :], z.shape)
    )
    inside = (inverse >= 0) & (z <= n) & (n > 0)
    zz, nn = np.where(inside, z, 0.0), np.where(inside, n, 0.0)
    limit = np.isinf(a + b)
    safe_a, safe_b = np.where(limit, 1.0, a), np.where(limit, 1.0, b)
    held = digamma_rising(safe_a + safe_b, nn)
    da = digamma_rising(safe_a, zz) - held
    db = digamma_rising(safe_b, nn - zz) - held
    rate = getattr(family, "rate", None)
    if rate is None:
        return {
            "alpha": np.ascontiguousarray(np.where(inside, da, 0.0)),
            "beta": np.ascontiguousarray(np.where(inside, db, 0.0)),
        }
    p = rate.detach().numpy()[:, None]
    tau = a + b
    with np.errstate(invalid="ignore"):
        dp = np.where(limit, zz / p - (nn - zz) / (1.0 - p), tau * (da - db))
        dtau = np.where(limit, 0.0, p * da + (1.0 - p) * db)
    return {
        "rate": np.ascontiguousarray(np.where(inside, dp, 0.0)),
        "concentration": np.ascontiguousarray(np.where(inside, dtau, 0.0)),
    }


def log_emission_partials(
    family: Family,
    coded: Coded,
    *,
    shift: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Each parameter's partial of every observation's log-density, ``{parameter: (K, n)}``.

    The negative binomial gives ``dispersion`` and ``mean``; the
    beta-binomial ``alpha`` and ``beta``, or ``rate`` and ``concentration``
    for :class:`~sal.emissions.RateConcentrationBetaBinomialEmission`; the
    independent pair both channels'. Through
    :func:`~sal.emissions.rising.digamma_rising`, in NumPy: an unobserved
    observation, or successes past their trials, have 0. ``r = inf`` and
    ``tau = inf`` take their limits' partials, and the parameter that is
    infinite has 0. Under a ``shift`` the negative binomial adds ``shift``,
    each observation's partial with respect to its own label's offset;
    summing it per label gives the gradient of a sum.
    """
    cov = coded.covariate
    factor = _shift_factor(coded, shift)
    if isinstance(family, NegativeBinomialEmission):
        return _nb_partials(family, coded.counts, coded.inverse, cov, factor)
    if isinstance(family, BetaBinomialEmission):
        return _bb_partials(family, coded.counts, coded.inverse, cov)
    total, successes = _pair_channels(family)
    return {
        **_nb_partials(
            total,
            coded.counts[:, 0],
            coded.inverse,
            None if cov is None else cov[:, 0],
            factor,
        ),
        **_bb_partials(
            successes,
            coded.counts[:, 1],
            coded.inverse,
            None if cov is None else cov[:, 1],
        ),
    }
