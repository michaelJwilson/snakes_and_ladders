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
* :class:`DeclaredGaussianHmm` --- a Gaussian HMM's negative log-likelihood of
  equal-length sequences, its gradient Fisher's identity over the streamed
  statistics (issue #1008).
* :class:`Power` --- an *operator* ``f(x) = x ** k`` elementwise, whose
  :class:`~sal.sample.expectation.KalmanMean` a compiled
  chain keeps itself rather than handing its draws back (issue #1006).
* :class:`DeclaredRosenbrock` --- ``U(x) = sum_i b (x_{i+1} - x_i^2)^2 +
  (a - x_i)^2`` (``eq:rosenbrock``), the curved valley the non-Gaussian goals
  are set on (issue #1006).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

#: The family codes ``oxisal``'s energy kernels read.
GAUSSIAN, ROSENBROCK, MIXTURE, GAUSSIAN_HMM = 0, 1, 2, 3


@runtime_checkable
class DeclaredGaussian(Protocol):
    """An objective that declares itself ``U(x) = x' P x / 2``, a zero-mean Gaussian.

    What :func:`~sal.sample.hmc.sample` compiles (issue #986).
    ``P`` is ``(d,)`` for a diagonal or ``(d, d)`` symmetric.
    """

    @property
    def gaussian_precision(self) -> ArrayLike:
        """``P``, a constant: an array or a tensor that tracks no gradient."""
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


@runtime_checkable
class DeclaredGaussianHmm(Protocol):
    """An objective that declares itself a Gaussian HMM's negative log-likelihood."""

    @property
    def gaussian_hmm_declaration(self) -> tuple[int, np.ndarray] | None:
        """``(m, observations)``, sequences as rows, or ``None`` where the objective is not that family."""
        ...


def declared_energy(objective: object) -> tuple[int, np.ndarray] | None:
    """The family code and its flat ``float64`` parameters, or ``None`` if none is declared."""
    if isinstance(objective, DeclaredGaussianHmm):
        hmm = objective.gaussian_hmm_declaration
        if hmm is not None:
            m, sequences = hmm
            return GAUSSIAN_HMM, np.concatenate(
                (
                    [float(m), float(sequences.shape[1])],
                    np.asarray(sequences, float).ravel(),
                )
            )
    if isinstance(objective, DeclaredMixture):
        declared = objective.gaussian_mixture_declaration
        if declared is not None:
            k, values = declared
            return MIXTURE, np.concatenate(
                ([float(k)], np.asarray(values, float).ravel())
            )
    if isinstance(objective, DeclaredGaussian):
        precision = np.ascontiguousarray(objective.gaussian_precision, dtype=np.float64)
        return GAUSSIAN, precision.reshape(-1)
    if isinstance(objective, DeclaredRosenbrock):
        return ROSENBROCK, np.asarray(objective.rosenbrock_constants, dtype=np.float64)
    return None


class _Raisable(Protocol):
    """What :class:`Power` applies to: an array or a tensor alike."""

    def __pow__(self, exponent: int, /) -> Self: ...


_R = TypeVar("_R", bound=_Raisable)


@dataclass(frozen=True)
class Power:
    """The operator ``x ** exponent``, elementwise: ``Power(1)`` is the mean, ``Power(2)`` the second moment.

    A plain callable on the torch route, and declared to a compiled one,
    which evaluates it and filters it in the chain's own loop.
    """

    exponent: int

    def __call__(self, x: _R) -> _R:
        return x**self.exponent


@runtime_checkable
class DeclaredJaxEnergy(Protocol):
    """An objective whose energy is a traceable JAX ``(theta, data)`` function (issue #1008).

    What :mod:`sal.sample.hmc.jax` runs a chain on under
    ``jit``; ``None`` where the objective has none at its current backend.
    """

    def jax_energy(self) -> tuple[Callable[[Any, Any], Any], Any] | None:
        """``(energy, data)``, or ``None``."""
        ...


def declared_jax_energy(
    objective: object,
) -> tuple[Callable[[Any, Any], Any], Any] | None:
    """The objective's traceable JAX energy and its data, or ``None`` if it declares none."""
    if isinstance(objective, DeclaredJaxEnergy):
        return objective.jax_energy()
    return None
