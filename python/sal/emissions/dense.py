"""Shared count helpers for the coded log-emission (issues #1132, #1340).

The dense log-emission, every observation's log-density under every state,
is :func:`sal.emissions.coded.log_emission` over a
:class:`~sal.emissions.coded.Dense` (issue #1340): one implementation, a
dense observation set being scored through its coded form. This module keeps
what that call and its siblings share: :class:`IndependentPair`, the
independent count pair it scores, and :func:`checked_counts`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sal.emissions.counts import (
    BetaBinomialEmission,
    NegativeBinomialEmission,
)

#: The largest count a ``uint32`` carries; a larger one would wrap on the cast.
_COUNT_LIMIT = 2**32 - 1


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


def checked_counts(values: np.ndarray, name: str) -> np.ndarray:
    """``values`` flat as contiguous ``uint32``, refused unless non-negative integers below ``2**32``.

    ``name`` is the count's name in the refusal. The raising sibling of
    :func:`sal.opt.emission_mixture.as_counts`, which returns ``None``
    instead (issue #1341).
    """
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
