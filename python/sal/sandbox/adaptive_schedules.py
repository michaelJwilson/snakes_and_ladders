"""Two declined annealing schedules of issue #1333, conserved finished (issue #1352).

``thermodynamic`` places a ramp at constant thermodynamic speed from a
measured ``sigma_E(T)`` (Nulton & Salamon, 1988); sal measures no
``sigma_E(T)``, so nothing supported can hand it the :class:`ThermodynamicPilot`
it needs. :class:`HuangSchedule` chooses each temperature from the chain as it
runs (Huang, Romeo & Sangiovanni-Vincentelli, 1986), which
:func:`~sal.sample.tune.tune_schedule` cannot rank against a declared ramp.
Both are declined on those grounds (#1338) and kept with the tests that pin
them: equal thermodynamic length per step on the exact ``8 x 8`` Ising
``sigma_E``, and the Huang recurrence bitwise.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from sal.opt.termination import Termination
from sal.sample.loop import Moved, Step, Walked
from sal.sample.schedule import (
    TempSchedule,
    _check_length,
    _check_step,
    _check_temperature,
    _InterpolatedTempSchedule,
)
from sal.track import current

__all__ = [
    "Adaptive",
    "AdaptiveSchedule",
    "AdaptiveWalked",
    "HuangSchedule",
    "ThermodynamicPilot",
    "ThermodynamicTempSchedule",
    "adaptive",
    "anneal_adaptive",
    "thermodynamic",
]


@dataclass(frozen=True)
class ThermodynamicPilot:
    """``sigma_E(T)`` measured on a temperature grid, the input a constant-speed ramp is placed from.

    The thermodynamic length ``L(T) = int sigma_E(T) / T^2 dT`` (Salamon &
    Berry, 1983) is taken by the trapezoid rule at the grid's nodes and
    linearly between them, so it is fixed once the pilot is.

    Parameters
    ----------
    temperatures : tuple[float, ...]
        Strictly increasing, positive, at least two.
    sigma : tuple[float, ...]
        The energy's standard deviation at each, positive.
    """

    temperatures: tuple[float, ...]
    sigma: tuple[float, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.temperatures, dtype=np.float64)
        spread = np.asarray(self.sigma, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 2 or spread.shape != grid.shape:
            msg = (
                "a pilot needs at least two temperatures and one sigma each, got "
                f"{grid.size} and {spread.size}"
            )
            raise ValueError(msg)
        if not (grid[0] > 0.0 and np.all(np.diff(grid) > 0.0)):
            msg = "a pilot's temperatures are positive and strictly increasing"
            raise ValueError(msg)
        if not np.all(spread > 0.0):
            msg = "a pilot's sigma is positive at every temperature"
            raise ValueError(msg)

    def nodes(self) -> np.ndarray:
        """``L`` at each grid temperature, ``0`` at the first."""
        grid = np.asarray(self.temperatures, dtype=np.float64)
        speed = np.asarray(self.sigma, dtype=np.float64) / grid**2
        steps = 0.5 * (speed[1:] + speed[:-1]) * np.diff(grid)
        return np.concatenate(([0.0], np.cumsum(steps)))

    def length(self, temperature: float) -> float:
        """``L(temperature)``, linear between the nodes; refused off the grid."""
        if not self.temperatures[0] <= temperature <= self.temperatures[-1]:
            msg = (
                f"temperature {temperature} is outside the pilot's grid "
                f"[{self.temperatures[0]}, {self.temperatures[-1]}]"
            )
            raise ValueError(msg)
        return float(np.interp(temperature, self.temperatures, self.nodes()))


@dataclass(frozen=True)
class ThermodynamicTempSchedule(_InterpolatedTempSchedule):
    """Constant thermodynamic speed from ``start`` to ``end`` (Nulton & Salamon, 1988).

    ``g(T) = L(T)``, the pilot's thermodynamic length, and ``s = t``: each
    step covers the same length, so the ramp slows where ``sigma_E / T^2``
    peaks. ``T_k`` is ``L``'s inverse, exact on its piecewise-linear form;
    both endpoints are returned bitwise.

    Parameters
    ----------
    pilot : ThermodynamicPilot
        Its grid covers ``start`` and ``end``.
    """

    pilot: ThermodynamicPilot = field(kw_only=True)
    _values: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        first, last = self.pilot.length(self.start), self.pilot.length(self.end)
        fraction = np.arange(self.n_steps) / max(self.n_steps - 1, 1)
        targets = (1.0 - fraction) * first + fraction * last
        values = np.interp(targets, self.pilot.nodes(), self.pilot.temperatures)
        values[0], values[-1] = self.start, self.end
        if self.start == self.end:
            values[:] = self.start
        object.__setattr__(self, "_values", tuple(float(value) for value in values))

    def __call__(self, step: int) -> float:
        _check_step(step, self.n_steps)
        return self._values[step]


def thermodynamic(
    start: float, end: float, n_steps: int, *, pilot: ThermodynamicPilot
) -> TempSchedule:
    """``g(T) = L(T)`` from ``pilot``, ``s = t``: constant thermodynamic speed.

    Was ``ramp.thermodynamic`` and ``ScheduleShape.THERMODYNAMIC`` (#1333).
    """
    return ThermodynamicTempSchedule(start, end, n_steps, pilot=pilot)


@runtime_checkable
class AdaptiveSchedule(Protocol):
    """A temperature chosen from the chain as it runs, rather than declared in advance (issue #1333).

    :func:`anneal_adaptive` runs a block of steps at
    :attr:`start`, hands :meth:`next` the block's energies, and runs the next
    block at what it returns, until it returns ``None`` (converged) or
    :attr:`budget` temperatures have run (budget).
    """

    #: The first temperature.
    start: float
    #: The most temperatures the run visits, ``>= 1``.
    budget: int

    @abstractmethod
    def next(self, temperature: float, energies: Sequence[float]) -> float | None:
        """The temperature after ``temperature``, or ``None`` to stop."""
        ...  # pragma: no cover


@dataclass(frozen=True)
class HuangSchedule(AdaptiveSchedule):
    """``T' = T exp(-rate T / sigma_E(T))`` (Huang, Romeo & Sangiovanni-Vincentelli, 1986).

    ``sigma_E`` is the population standard deviation of the block's
    energies. The step is clamped at ``end``, which the run then visits
    once: :meth:`next` at ``end`` returns ``None``. A block of equal energies
    has ``sigma_E = 0`` and steps to ``end``.

    Parameters
    ----------
    start, end : float
        Positive, ``start >= end``.
    rate : float
        ``lambda > 0``; Huang et al. take ``0.7``.
    budget : int
        The most temperatures visited, ``>= 1``.
    """

    start: float
    end: float
    rate: float
    budget: int

    def __post_init__(self) -> None:
        _check_temperature("start", self.start)
        _check_temperature("end", self.end)
        if self.start < self.end:
            msg = f"Huang's schedule cools, and start={self.start} < end={self.end}"
            raise ValueError(msg)
        if not self.rate > 0.0:
            msg = f"rate must be positive, got {self.rate}"
            raise ValueError(msg)
        _check_length(self.budget)

    def next(self, temperature: float, energies: Sequence[float]) -> float | None:
        if temperature <= self.end:
            return None
        if len(energies) < 2:
            msg = f"sigma_E needs at least two energies, got {len(energies)}"
            raise ValueError(msg)
        sigma = float(np.std(np.asarray(energies, dtype=np.float64)))
        if sigma == 0.0:
            return self.end
        return max(temperature * math.exp(-self.rate * temperature / sigma), self.end)


class Adaptive:
    """:data:`adaptive`: one explicit call per online schedule (issue #1333)."""

    @staticmethod
    def huang(
        start: float, end: float, *, rate: float, budget: int
    ) -> AdaptiveSchedule:
        """Huang, Romeo & Sangiovanni-Vincentelli's ``sigma_E``-driven cooling."""
        return HuangSchedule(start, end, rate, budget)


#: The online schedules: ``adaptive.huang(...)``.
adaptive = Adaptive()


@dataclass(frozen=True, kw_only=True)
class AdaptiveWalked[S](Walked[S]):
    """:class:`~sal.sample.loop.Walked`, with the temperature of each block :func:`anneal_adaptive` ran."""

    temperatures: tuple[float, ...]


def anneal_adaptive[S, C, R](
    step: Step[S, C, R],
    schedule: AdaptiveSchedule,
    start: Moved[S, C],
    rng: R,
    keep: Callable[[S], S],
    *,
    block: int,
) -> AdaptiveWalked[S]:
    """:func:`~sal.sample.loop.anneal` on an :class:`AdaptiveSchedule`: ``block`` steps per temperature (issue #1333).

    Each temperature runs ``block`` steps, and the block's ``block`` energies
    choose the next through ``schedule.next``. The run ends converged when it
    returns ``None``, and on budget after ``schedule.budget`` temperatures;
    the termination counts the temperatures run. The best is kept as
    :func:`~sal.sample.loop.anneal` keeps it.

    Raises
    ------
    ValueError
        If ``block < 2``: ``sigma_E`` needs two energies.
    """
    if block < 2:
        msg = f"a block needs at least two steps to estimate sigma_E, got {block}"
        raise ValueError(msg)
    state, energy, carried, spent = start
    best, best_energy = keep(state), energy
    energies = [energy]
    visited: list[float] = []
    tracked = current()
    temperature: float | None = schedule.start
    index = 0
    while temperature is not None and len(visited) < schedule.budget:
        visited.append(temperature)
        for _ in range(block):
            moved = step(state, energy, carried, temperature, rng)
            state, energy, carried = moved.state, moved.energy, moved.carried
            spent += moved.spent
            energies.append(energy)
            if energy < best_energy:
                best, best_energy = keep(state), energy
            tracked.record(
                index, state=best, temperature=temperature, energy=best_energy
            )
            index += 1
        temperature = schedule.next(temperature, energies[-block:])
    return AdaptiveWalked(
        best=best,
        energy=best_energy,
        final=state,
        energies=tuple(energies),
        spent=spent,
        termination=Termination.after(len(visited), converged=temperature is None),
        temperatures=tuple(visited),
    )
