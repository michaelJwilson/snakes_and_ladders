"""The kernels a compiled chain runs, and how an objective opts in to one (issues #986, #1006, #1220).

A torch closure cannot cross the FFI boundary, so a compiled chain runs one
of the package's own value+gradient kernels, behind ``src/energy.rs``'s
``Energy`` trait. An objective opts in through
:meth:`SupportedGradient.supported_gradient`, naming the kernel and handing
over its data; it does not restate a family. :func:`declared_energy` is the
one place a sampler reads that, and ``oxisal.SupportedEnergy`` is the one an
objective's own gradient calls, so a chain and a fit evaluate one arithmetic.

* :data:`GAUSSIAN` --- ``U(x) = x' P x / 2``, data ``precision``: ``P``
  diagonal ``(d,)`` or dense ``(d, d)`` symmetric (issue #986).
* :data:`ROSENBROCK` --- ``U(x) = sum_i b (x_{i+1} - x_i^2)^2 + (a - x_i)^2``
  (``eq:rosenbrock``), data ``a`` and ``b`` (issue #1006).
* :data:`GAUSSIAN_MIXTURE` --- a one-channel Gaussian mixture's negative
  log-likelihood in ``theta = (k - 1 free weights, k means, k log scales)``,
  data ``k`` and ``observations``; ``oxisal.gaussian_mixture_gradient``'s
  kernel (issue #1008).
* :data:`GAUSSIAN_HMM` --- a Gaussian HMM's negative log-likelihood of
  equal-length sequences, data ``m`` and ``observations`` with sequences as
  rows; ``oxisal.gaussian_hmm_statistics`` and Fisher's identity (issue #1008).
* :data:`COUNT_MIXTURE` --- a negative binomial, beta-binomial or count-pair
  mixture's negative log-likelihood, data ``k``, ``totals``, ``successes``,
  ``trials`` and ``slots``; ``oxisal.count_mixture_value_and_gradient``'s
  kernel (issue #1136).
* :class:`Power` --- an *operator* ``f(x) = x ** k`` elementwise, whose
  :class:`~sal.sample.expectation.KalmanMean` a compiled
  chain keeps itself rather than handing its draws back (issue #1006).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

#: The kernel names ``src/energy.rs`` builds.
GAUSSIAN = "gaussian"
ROSENBROCK = "rosenbrock"
GAUSSIAN_MIXTURE = "gaussian_mixture"
GAUSSIAN_HMM = "gaussian_hmm"
COUNT_MIXTURE = "count_mixture"

#: What :meth:`SupportedGradient.supported_gradient` returns: a kernel name
#: and its data, arrays and numbers by name.
Supported = tuple[str, Mapping[str, Any]]


@runtime_checkable
class SupportedGradient(Protocol):
    """An objective whose value and gradient one of ``oxisal``'s kernels computes (issue #1220)."""

    def supported_gradient(self) -> Supported | None:
        """``(kernel, data)``, or ``None`` where no kernel is this objective."""
        ...


def declared_energy(objective: object) -> Supported | None:
    """The objective's ``(kernel, data)``, or ``None`` if it supports none."""
    if isinstance(objective, SupportedGradient):
        return objective.supported_gradient()
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
