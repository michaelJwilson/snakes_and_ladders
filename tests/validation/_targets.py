"""The zero-mean Gaussian targets the BlackJAX comparisons share (issue #963).

One objective, dense or diagonal by the precision's shape, so the oracle, the
goal and the benchmark pair score the same density BlackJAX's script builds.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from snakes_and_ladders.opt.objective import Objective


class GaussianTarget(Objective):
    """``-log N(0, P^-1)`` up to a constant: ``P`` a ``(d, d)`` matrix or a ``(d,)`` diagonal."""

    def __init__(self, precision: np.ndarray) -> None:
        self.precision = torch.as_tensor(precision, dtype=torch.float64)

    @property
    def dimension(self) -> int:
        """``d``."""
        return int(self.precision.shape[0])

    def initial(self) -> torch.Tensor:
        """The mode, which BlackJAX's chain starts from too."""
        return torch.zeros(self.dimension, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The coordinates, already unconstrained."""
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The vector behind ``named``."""
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The negative log density at ``theta``."""
        if self.precision.ndim == 1:
            return 0.5 * (self.precision * theta * theta).sum()
        value: torch.Tensor = 0.5 * theta @ self.precision @ theta
        return value


def dense_precision(dimension: int, seed: int) -> np.ndarray:
    """``A Aᵀ / d + I`` for a standard normal ``A``: eigenvalues from 1 to about 3.5 at d = 10."""
    draw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    return np.asarray(draw @ draw.T / dimension + np.eye(dimension))


def diagonal_precision(dimension: int) -> np.ndarray:
    """Precisions evenly spaced from 1 to 4: a condition number of 4 at every size."""
    return np.linspace(1.0, 4.0, dimension)
