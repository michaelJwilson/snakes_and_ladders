"""The energy families a compiled chain runs, and how an objective declares one (issues #986, #1006).

A torch closure cannot cross the FFI boundary, so a compiled chain runs a
*declared family* rather than an arbitrary objective: an objective opts in by
stating its parameters, never by being recognized. :func:`declared_energy`
is the one place a sampler reads that declaration, and returns the family's
code and parameters as ``oxisal``'s energy kernels take them (``src/energy.rs``).

* :class:`DeclaredGaussian` --- ``U(x) = x' P x / 2``, ``P`` diagonal
  ``(d,)`` or dense ``(d, d)`` symmetric (issue #986).
* :class:`DeclaredMixture` --- a one-channel Gaussian mixture's negative
  log-likelihood of its observations, in ``theta = (k - 1 free weights, k
  means, k log scales)`` (issue #1008).
* :class:`Power` --- an *operator* ``f(x) = x ** k`` elementwise, whose
  :class:`~snakes_and_ladders.sample.expectation.KalmanMean` a compiled
  chain keeps itself rather than handing its draws back (issue #1006).
* :class:`DeclaredRosenbrock` --- ``U(x) = sum_i b (x_{i+1} - x_i^2)^2 +
  (a - x_i)^2`` (``eq:rosenbrock``), the curved valley the non-Gaussian goals
  are set on (issue #1006).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

#: The family codes ``oxisal``'s energy kernels read.
GAUSSIAN, ROSENBROCK, MIXTURE = 0, 1, 2


@runtime_checkable
class DeclaredGaussian(Protocol):
    """An objective that declares itself ``U(x) = x' P x / 2``, a zero-mean Gaussian.

    What :func:`~snakes_and_ladders.sample.hmc.sample` compiles (issue #986).
    ``P`` is ``(d,)`` for a diagonal or ``(d, d)`` symmetric.
    """

    @property
    def gaussian_precision(self) -> torch.Tensor:
        """``P``."""
        ...


@runtime_checkable
class DeclaredRosenbrock(Protocol):
    """An objective that declares itself Rosenbrock's function with constants ``(a, b)``."""

    @property
    def rosenbrock_constants(self) -> tuple[float, float]:
        """``(a, b)``."""
        ...


@runtime_checkable
class DeclaredMixture(Protocol):
    """An objective that declares itself a one-channel Gaussian mixture's negative log-likelihood."""

    @property
    def gaussian_mixture_declaration(self) -> tuple[int, np.ndarray] | None:
        """``(k, observations)``, or ``None`` where the objective is not that family."""
        ...


def declared_energy(objective: object) -> tuple[int, np.ndarray] | None:
    """The family code and its flat ``float64`` parameters, or ``None`` if none is declared."""
    if isinstance(objective, DeclaredMixture):
        declared = objective.gaussian_mixture_declaration
        if declared is not None:
            k, values = declared
            return MIXTURE, np.concatenate(
                ([float(k)], np.asarray(values, float).ravel())
            )
    if isinstance(objective, DeclaredGaussian):
        precision = objective.gaussian_precision.detach().numpy()
        return GAUSSIAN, np.ascontiguousarray(precision, dtype=np.float64).reshape(-1)
    if isinstance(objective, DeclaredRosenbrock):
        return ROSENBROCK, np.asarray(objective.rosenbrock_constants, dtype=np.float64)
    return None


@dataclass(frozen=True)
class Power:
    """The operator ``x ** exponent``, elementwise: ``Power(1)`` is the mean, ``Power(2)`` the second moment.

    A plain callable on the torch route, and declared to a compiled one,
    which evaluates it and filters it in the chain's own loop.
    """

    exponent: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x**self.exponent
