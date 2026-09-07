"""Temperature schedules: a temperature per step, and nothing about what is tempered.

Three consumers want one and sit in three modules --- the Hamiltonian sampler
in ``snakes_and_ladders.opt``, the Potts move sets in ``snakes_and_ladders.search``, and the
Boltzmann policy in ``snakes_and_ladders.learn``. ``opt`` and ``learn`` may import no
application module and ``search`` may import both, so this is the one
placement all three can reach --- the same argument that puts ``constrain.py``
here rather than beside its callers (issue #267).

**Tempering a likelihood is not tempering an energy.** ``beta * E`` is a
temperature in the physical sense; ``beta * (-log L)`` is a *power posterior*
(Friel & Pettitt, 2008) --- a standard object, and a different one. A chain
annealed over an HMM's objective targets the second, and a reader expecting
the Potts semantics will be wrong about what it converges to. The schedule
serves both and names neither; the consumer says which it is.

**Mirrors ``torch.optim.lr_scheduler``**, so a reader who knows one knows the
other, with one deliberate difference: every schedule here declares its length
and both endpoints, and reaches the final temperature at *exactly* the last
step. A schedule that never quite arrives has no final temperature to check,
and the off-by-one in "reaches at the last step" is the fault no downstream
distributional test would ever localize. Adaptive schedules --- reheating on
a stalled chain, targeting an acceptance rate --- are absent by the same
standing decision that keeps step-size adaptation out of ``hmc.py``.

The policy's learned softmax weight is an inverse temperature too, and it is
**not** put on a schedule: it is the thing the agent learns, and a declared
schedule would remove it. The connection is named here so nobody reintroduces
it as a feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Schedule(Protocol):
    """A positive temperature for each of ``n_steps`` steps.

    ``step`` runs from ``0`` to ``n_steps - 1`` inclusive. Asking outside that
    range is refused rather than clamped: a consumer that asks for step
    ``n_steps`` has run one iteration longer than it declared, and clamping
    would let it.
    """

    @property
    def n_steps(self) -> int:
        """How many steps the schedule covers."""
        ...  # pragma: no cover

    def __call__(self, step: int) -> float:
        """The temperature at ``step``, strictly positive."""
        ...  # pragma: no cover


def _check_temperature(name: str, value: float) -> None:
    if not value > 0.0:
        msg = (
            f"{name} must be a positive temperature, got {value}: at zero every "
            "acceptance ratio is 0 or 1 and the chain is a descent, not a sample"
        )
        raise ValueError(msg)


def _check_length(n_steps: int) -> None:
    if n_steps < 1:
        msg = f"a schedule needs at least one step, got {n_steps}"
        raise ValueError(msg)


def _check_step(step: int, n_steps: int) -> None:
    if not 0 <= step < n_steps:
        msg = (
            f"step {step} is outside a schedule of {n_steps} steps (0 to {n_steps - 1})"
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class Constant:
    """One temperature throughout. ``ConstantLR`` with factor 1.

    The schedule every existing chain and fit is on, so passing it must
    reproduce them exactly; at temperature 1 it is the untempered target.

    Parameters
    ----------
    temperature : float
        The temperature, positive.
    n_steps : int
        Length, ``>= 1``.
    """

    temperature: float
    n_steps: int

    def __post_init__(self) -> None:
        _check_temperature("temperature", self.temperature)
        _check_length(self.n_steps)

    def __call__(self, step: int) -> float:
        _check_step(step, self.n_steps)
        return self.temperature


@dataclass(frozen=True)
class _Interpolated:
    """``start`` at step 0, ``end`` at the last step, some curve between.

    Each subclass supplies the weight ``w(t)`` on ``start`` at fraction ``t``
    of the way through, with ``w(0) = 1`` and ``w(1) = 0`` exactly, and the
    temperature is combined so that both endpoints come out *bitwise*: at
    ``t = 0`` the ``end`` term is multiplied by zero and dropped, and at
    ``t = 1`` the ``start`` term is. A one-step schedule has one temperature,
    so ``start`` and ``end`` must then agree.

    Parameters
    ----------
    start, end : float
        Temperatures at the first and last step, positive.
    n_steps : int
        Length, ``>= 1``.
    """

    start: float
    end: float
    n_steps: int

    def __post_init__(self) -> None:
        _check_temperature("start", self.start)
        _check_temperature("end", self.end)
        _check_length(self.n_steps)
        if self.n_steps == 1 and self.start != self.end:
            msg = (
                f"a one-step schedule has one temperature, but start={self.start} "
                f"and end={self.end} differ"
            )
            raise ValueError(msg)

    def _fraction(self, step: int) -> float:
        _check_step(step, self.n_steps)
        return step / (self.n_steps - 1) if self.n_steps > 1 else 0.0


@dataclass(frozen=True)
class Linear(_Interpolated):
    """Straight line from ``start`` to ``end``. ``LinearLR`` with declared ends.

    ``T(t) = (1 - t) * start + t * end``.
    """

    def __call__(self, step: int) -> float:
        fraction = self._fraction(step)
        return (1.0 - fraction) * self.start + fraction * self.end


@dataclass(frozen=True)
class Exponential(_Interpolated):
    """Geometric from ``start`` to ``end``. ``ExponentialLR`` with a declared end.

    ``T(t) = start ** (1 - t) * end ** t``, so successive temperatures have
    the constant ratio ``(end / start) ** (1 / (n_steps - 1))`` --- the
    ``gamma`` ``ExponentialLR`` would take --- and the endpoints are exact
    because ``pow(x, 0)`` is 1 and ``pow(x, 1)`` is ``x``.

    The schedule simulated annealing is usually run on (Kirkpatrick et al.,
    1983): a constant *factor* per step spends equal effort per decade of
    temperature, where a linear schedule spends almost all of it cold.
    """

    def __call__(self, step: int) -> float:
        fraction = self._fraction(step)
        return math.pow(self.start, 1.0 - fraction) * math.pow(self.end, fraction)


@dataclass(frozen=True)
class Cosine(_Interpolated):
    """Half a cosine from ``start`` to ``end``. ``CosineAnnealingLR`` with ``T_max = n_steps - 1``.

    ``T(t) = w * start + (1 - w) * end`` with ``w = (1 + cos(pi t)) / 2``.
    Slow at both ends and fast in the middle; written with the weight on
    ``start`` rather than in the textbook form ``end + (start - end) * w`` so
    that ``t = 0`` returns ``start`` bitwise rather than ``end + start - end``.
    """

    def __call__(self, step: int) -> float:
        weight = 0.5 * (1.0 + math.cos(math.pi * self._fraction(step)))
        return weight * self.start + (1.0 - weight) * self.end


def temperatures(schedule: Schedule) -> list[float]:
    """Every temperature of ``schedule``, in step order.

    A convenience for tests and for reporting a run; a consumer that steps
    in time asks for one step at a time.
    """
    return [schedule(step) for step in range(schedule.n_steps)]
