"""A count family's dense log-emission, ``(K, ...)``, from its own tables (issue #1132).

:mod:`sal.emissions.nb` and :mod:`sal.emissions.bb` tabulate the negative
binomial and the beta-binomial by the integers their terms depend on, and
``src/coupled.rs`` completes each observation's covariate term from them per
score. That completion was reachable only inside the coupled E step and field.
:func:`log_emission` returns it for any caller: every observation's
log-density under every state, as ``log_density`` would, through the same
tables and the same Rust arithmetic (``src/dense_emission.rs``).

**Which families.** :class:`~sal.emissions.NegativeBinomialEmission` with an
exposure per observation, :class:`~sal.emissions.BetaBinomialEmission` with a
trial count per observation, either without one, and the independent pair of
the two --- :class:`~sal.emissions.CountPairEmission` with ``joint=False`` or
``sim.count_pairs.IndependentCountPair`` --- whose score is the first
channel's plus the second's. The joint pair is refused: its trial count is the
observed total, and a total of zero is a support point there where the trial
term reads it as unobserved.

**The order is the caller's.** The beta-binomial's nine tabulated terms are
summed in the family's order and each score is its ``log_density`` bit for
bit. The negative binomial's exposure term is completed either in the family's
order, ``A + r ln(r / t) + y ln(mu c / t)`` (:attr:`Order.FAMILY`, two
logarithms a score, 2.3 ulp relative at the ci instance of
``spatio_sequential_counts_covariate``), or as the coupled kernel does,
``B + y ln c - (y + r) ln t`` (:attr:`Order.TABULATED`, one logarithm a
score, 262.9 ulp there). Without an exposure the table is the density itself
and both orders read it. ``order`` has no default: which rounding a caller
takes is theirs to state.

**No derivative is taken.** The result is a NumPy array, and nothing here
tracks a gradient (root ``CLAUDE.md``). A caller differentiating the emission
by finite differences calls this once per perturbed family.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

import numpy as np
import torch

from sal import oxisal
from sal.emissions.base import as_array, split_covariate
from sal.emissions.bb import log_factorial, trial_tables
from sal.emissions.counts import (
    BetaBinomialEmission,
    CountPairEmission,
    NegativeBinomialEmission,
)
from sal.emissions.nb import count_log_factor, exposure_table

#: The largest count a ``uint32`` carries; a larger one would wrap on the cast.
_COUNT_LIMIT = 2**32 - 1


class Order(StrEnum):
    """The order the negative binomial's exposure term is completed in.

    ``FAMILY`` is ``log_density``'s own, ``TABULATED`` the coupled kernel's;
    the module docstring states what each costs and how far each rounds.
    """

    FAMILY = "family"
    TABULATED = "tabulated"


class IndependentPair(Protocol):
    """A negative-binomial total and beta-binomial successes, independent given the state."""

    @property
    def n_states(self) -> int:
        """Hidden states the pair emits from."""
        ...

    @property
    def total(self) -> NegativeBinomialEmission:
        """The first channel."""
        ...

    @property
    def successes(self) -> BetaBinomialEmission | None:
        """The second channel; ``None`` only for a joint pair, which is refused."""
        ...


def _counts(values: np.ndarray, name: str) -> np.ndarray:
    """``values`` as contiguous ``uint32``, refused unless non-negative integers."""
    flat = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if flat.size and not (
        np.isfinite(flat).all()
        and (flat >= 0.0).all()
        and (flat == np.floor(flat)).all()
        and flat.max() <= _COUNT_LIMIT
    ):
        msg = f"every {name} must be a non-negative integer below 2**32"
        raise ValueError(msg)
    return flat.astype(np.uint32)


def _singleton(covariate: np.ndarray, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """A single-channel covariate, ``shape + (1,)`` as ``log_density`` takes it, flat."""
    if covariate.shape != (*shape, 1):
        msg = (
            f"{name} per observation must be shaped {(*shape, 1)}, the "
            f"observations' shape and a trailing singleton, got {covariate.shape}"
        )
        raise ValueError(msg)
    return covariate.reshape(-1)


def _extent(counts: np.ndarray) -> int:
    """One past the largest count, so every count indexes a row."""
    return int(counts.max()) + 1 if counts.size else 1


def _density_table(
    family: NegativeBinomialEmission | BetaBinomialEmission, extent: int
) -> np.ndarray:
    """The family's own ``log_density`` at every count below ``extent``, ``(extent, K)``."""
    grid = torch.arange(extent, dtype=torch.float64)
    return np.ascontiguousarray(family.log_density(grid).numpy()).reshape(-1)


def _total_arguments(
    family: NegativeBinomialEmission,
    totals: np.ndarray,
    exposure: np.ndarray | None,
    order: Order,
) -> dict[str, np.ndarray]:
    """The first channel's table and, under an exposure, its term, as the kernel takes them."""
    counts = _counts(totals, "count")
    extent = _extent(counts)
    if exposure is None:
        return {"totals": counts, "total_table": _density_table(family, extent)}
    grid = torch.arange(extent, dtype=torch.float64)
    table = (
        count_log_factor(family, grid)
        if order is Order.FAMILY
        else exposure_table(family, extent)
    )
    return {
        "totals": counts,
        "total_table": np.ascontiguousarray(table.numpy()).reshape(-1),
        "exposure": np.ascontiguousarray(exposure, dtype=np.float64),
        "dispersion": np.ascontiguousarray(family.dispersion.numpy()),
        "mean": np.ascontiguousarray(family.mean.numpy()),
    }


def _success_arguments(
    family: BetaBinomialEmission, successes: np.ndarray, trials: np.ndarray | None
) -> dict[str, np.ndarray]:
    """The second channel's table and, under a trial count, its term, as the kernel takes them."""
    counts = _counts(successes, "success count")
    extent = _extent(counts)
    if trials is None:
        return {"successes": counts, "success_table": _density_table(family, extent)}
    per_observation = _counts(trials, "trial count")
    trials_extent = _extent(per_observation)
    tables = trial_tables(family, extent, trials_extent)
    return {
        "successes": counts,
        "success_table": np.ascontiguousarray(tables.success.numpy()).reshape(-1),
        "trials": per_observation,
        "failure_table": np.ascontiguousarray(tables.failure.numpy()).reshape(-1),
        "trial_table": np.ascontiguousarray(tables.trial.numpy()).reshape(-1),
        "log_factorial": np.ascontiguousarray(log_factorial(trials_extent).numpy()),
        "log_beta": np.ascontiguousarray(tables.log_beta.numpy()).reshape(-1),
    }


def _arguments(
    family: object,
    observations: np.ndarray,
    covariate: np.ndarray | None,
    order: Order,
) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    """The batch shape and the kernel's channel arguments for ``family``."""
    if isinstance(family, NegativeBinomialEmission):
        shape = observations.shape
        exposure = (
            None if covariate is None else _singleton(covariate, "an exposure", shape)
        )
        return shape, _total_arguments(family, observations, exposure, order)
    if isinstance(family, BetaBinomialEmission):
        shape = observations.shape
        trials = (
            None if covariate is None else _singleton(covariate, "a trial count", shape)
        )
        return shape, _success_arguments(family, observations, trials)
    if isinstance(family, CountPairEmission) and family.joint:
        msg = (
            "the joint pair's trial count is the observed total, where a total of "
            "zero is a support point and not an unobserved channel; score it "
            "with log_density"
        )
        raise TypeError(msg)
    successes = getattr(family, "successes", None)
    total = getattr(family, "total", None)
    if not (
        isinstance(total, NegativeBinomialEmission)
        and isinstance(successes, BetaBinomialEmission)
    ):
        msg = (
            f"a dense emission is over a negative binomial, a beta-binomial or "
            f"their independent pair; got {type(family).__name__}"
        )
        raise TypeError(msg)
    if observations.ndim < 1 or observations.shape[-1] != 2:
        msg = f"a pair's observations end in an axis of two, got {observations.shape}"
        raise ValueError(msg)
    shape = observations.shape[:-1]
    if covariate is not None and covariate.shape != observations.shape:
        msg = (
            f"a pair's covariate is shaped as its observations, {observations.shape}, "
            f"got {covariate.shape}"
        )
        raise ValueError(msg)
    channels = split_covariate(family, covariate)
    exposure = None if channels.exposure is None else channels.exposure.reshape(-1)
    trials = None if channels.trials is None else channels.trials.reshape(-1)
    return shape, {
        **_total_arguments(total, observations[..., 0], exposure, order),
        **_success_arguments(successes, observations[..., 1], trials),
    }


def log_emission(
    family: NegativeBinomialEmission | BetaBinomialEmission | IndependentPair,
    observations: np.ndarray | torch.Tensor,
    covariate: np.ndarray | torch.Tensor | None = None,
    *,
    order: Order,
) -> np.ndarray:
    """Every observation's log-density under every state, state-major.

    ``log_density(observations, covariate)`` with its state axis moved first,
    computed from the tables rather than by ``lgamma`` per score.

    Parameters
    ----------
    family : NegativeBinomialEmission | BetaBinomialEmission | IndependentPair
        The family scored; a pair must be independent.
    observations : np.ndarray | torch.Tensor
        Counts, any shape; a pair's end in an axis of two, the total then the
        successes.
    covariate : np.ndarray | torch.Tensor | None
        As ``log_density`` takes it: the exposure or the trial count per
        observation with a trailing singleton, or a pair's ``(..., 2)``. A
        zero marks a channel unobserved; it scores zero.
    order : Order
        The order the negative binomial's exposure term is completed in.

    Returns
    -------
    np.ndarray
        Shape ``(K, *batch)``, ``batch`` the observations' shape less a
        pair's channel axis; C-contiguous, so each state's scores are one row.

    Raises
    ------
    TypeError
        If the family is not one of those above.
    ValueError
        If a count or trial count is not a non-negative integer below
        ``2**32``, an exposure is negative or not finite, or a shape disagrees.
    """
    values = as_array(observations, torch.float64)
    given = None if covariate is None else as_array(covariate, torch.float64)
    shape, arguments = _arguments(family, values, given, Order(order))
    n_states = family.n_states
    out = np.empty(n_states * int(np.prod(shape, dtype=np.int64)))
    oxisal.dense_log_emission(n_states, order == Order.FAMILY, out, **arguments)
    return out.reshape(n_states, *shape)
