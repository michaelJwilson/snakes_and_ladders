"""The seam between an exact evaluation and what stands in for it (issue #308).

Neither a topology nor a lattice appears here: a :class:`Bound` is what a
surrogate claims, a :class:`Surrogate` is any callable that makes the claim,
and :func:`certify` is what holds it to the claim against the exact value.
``snakes_and_ladders.likelihood.surrogate`` builds the analytic bounds on
these and ``snakes_and_ladders.learn.surrogate`` the learned ones, and
``learn/CLAUDE.md`` is why the seam lives above both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import torch


class Bound(StrEnum):
    """What a surrogate claims about the exact value."""

    LOWER = "lower"
    UPPER = "upper"
    POINT = "point"


class Surrogate(Protocol):
    """A stand-in for an exact evaluation, with its claim attached."""

    @property
    def kind(self) -> Bound: ...  # pragma: no cover - protocol

    def __call__(self, structure: object, data: object) -> torch.Tensor:
        """The surrogate's value, a differentiable scalar."""
        ...  # pragma: no cover - protocol


class BoundViolation(ValueError):
    """A surrogate claiming a bound was on the wrong side of the exact value."""


@dataclass(frozen=True)
class Certificate:
    """What certifying a surrogate over a set of structures records.

    Parameters
    ----------
    n_structures : int
        How many were checked.
    violations : int
        Structures on which a claimed bound was on the wrong side; zero for
        an analytic bound, or the certificate would not have been issued.
    worst_gap : float
        The largest ``|surrogate - exact|`` seen.
    mean_gap : float
        The mean of the same.
    """

    n_structures: int
    violations: int
    worst_gap: float
    mean_gap: float


def certify(
    surrogate: Surrogate,
    exact: Callable[[object, object], float],
    structures: Iterable[object],
    data: object,
    *,
    tolerance: float = 1e-9,
    allowed_violation_rate: float = 0.0,
) -> Certificate:
    """Check a surrogate's claim against the exact value on every structure.

    A lower bound may not exceed the exact value by more than ``tolerance``,
    an upper bound may not fall below it, a point prediction claims nothing.
    ``allowed_violation_rate`` is zero for an analytic bound and the
    complement of the nominal coverage for a calibrated one.

    Raises
    ------
    BoundViolation
        If the violation rate exceeds what the claim allows.
    """
    gaps: list[float] = []
    violations = 0
    for structure in structures:
        value = float(surrogate(structure, data))
        truth = float(exact(structure, data))
        gaps.append(abs(value - truth))
        if (surrogate.kind is Bound.LOWER and value > truth + tolerance) or (
            surrogate.kind is Bound.UPPER and value < truth - tolerance
        ):
            violations += 1
    if not gaps:
        msg = "certify needs at least one structure"
        raise ValueError(msg)
    if violations > allowed_violation_rate * len(gaps):
        msg = (
            f"{surrogate.kind} bound violated on {violations} of {len(gaps)} structures, "
            f"above the allowed rate {allowed_violation_rate}"
        )
        raise BoundViolation(msg)
    return Certificate(len(gaps), violations, max(gaps), float(np.mean(gaps)))


__all__ = ["Bound", "BoundViolation", "Certificate", "Surrogate", "certify"]
