"""The zero-mean Gaussian target the BlackJAX comparisons share (issues #963, #987).

One objective, dense or diagonal by the precision's shape, so the oracle, the
goal, the benchmark pair and `scripts/package.py` score the density BlackJAX's
script builds. Here rather than under `tests/` because `scripts/package.py`
runs it in a fresh interpreter, which imports the package and not the tests.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from sal.opt.objective import Objective


class GaussianTarget(Objective):
    """``-log N(0, P^-1)`` up to a constant: ``P`` a ``(d, d)`` matrix or a ``(d,)`` diagonal."""

    def __init__(self, precision: np.ndarray) -> None:
        self.precision = torch.as_tensor(precision, dtype=torch.float64)

    @property
    def gaussian_precision(self) -> torch.Tensor:
        """``P``, which declares this a :class:`~sal.sample.declared.DeclaredGaussian`."""
        return self.precision

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

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """``P theta``, the closed form ``hmc.gradient_at`` reads (issue #986).

        In NumPy over the tensors' own buffers: a first torch operation in a
        process costs megabytes of resident memory that the product does not.
        """
        precision, point = self.precision.numpy(), theta.detach().numpy()
        if precision.ndim == 1:
            return torch.from_numpy(precision * point)
        return torch.from_numpy(precision @ point)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The negative log density at ``theta``."""
        if self.precision.ndim == 1:
            return 0.5 * (self.precision * theta * theta).sum()
        value: torch.Tensor = 0.5 * theta @ self.precision @ theta
        return value

    def energy(self, x: np.ndarray) -> float:
        """:meth:`__call__` on an array, the value :func:`~sal.opt.objective.energy_of` reads (issue #1011)."""
        precision = self.precision.numpy()
        if precision.ndim == 1:
            return float(0.5 * (precision * x * x).sum())
        return float(0.5 * x @ precision @ x)


def dense_precision(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """``A A^T / d + I`` for a standard normal ``A`` drawn from ``rng``.

    At d = 10 from ``default_rng(963)`` its eigenvalues run from 1.01 to 3.50.
    """
    draw = rng.normal(size=(dimension, dimension))
    return np.asarray(draw @ draw.T / dimension + np.eye(dimension))


def diagonal_precision(dimension: int) -> np.ndarray:
    """Precisions evenly spaced from 1 to 4: a condition number of 4 at every size."""
    return np.linspace(1.0, 4.0, dimension)
