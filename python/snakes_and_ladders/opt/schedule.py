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
distributional test would ever localize.

**A tempering ladder is chosen from what it measures, not set by hand.** A
ladder is a set of temperatures rather than a sequence in time, and what
makes one right is the exchange acceptance between each neighbouring pair:
near zero the replicas are independent chains and nothing crosses the gap,
near one two temperatures are close enough that one is redundant.
:func:`adapt_ladder` is a warm-up that measures those acceptances, bisects a
gap whose acceptance is below a stated band and removes a temperature both of
whose gaps are above it, until every pair sits inside the band or a budget is
spent; it takes the measurement as a callable so it knows no model, and the
sampler that owns the replicas supplies it (issue #333). Reheating on a
stalled chain --- a schedule in *time* that adapts --- remains absent.

The policy's learned softmax weight is an inverse temperature too, and it is
**not** put on a schedule: it is the thing the agent learns, and a declared
schedule would remove it. The connection is named here so nobody reintroduces
it as a feature.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
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


@dataclass(frozen=True)
class AdaptedLadder:
    """What a ladder warm-up settled on, and what it measured there.

    Parameters
    ----------
    temperatures : tuple[float, ...]
        The ladder, in the order the initial one was given.
    acceptance : tuple[float, ...]
        Exchange acceptance per neighbouring pair on the final ladder, from
        the last measurement --- one fewer than the temperatures.
    within_band : bool
        Whether every entry of ``acceptance`` lies inside the band. False
        means the warm-up stopped on its budget, or on a ladder it could not
        change --- two fixed endpoints whose pair is above the band --- and
        the caller decides whether that ladder is usable.
    rounds : int
        Measurements taken, the last one included.
    replicas_measured : int
        Sum of the ladder's length over every measurement --- the warm-up's
        cost in replica-runs, so a caller that knows the sweeps per
        measurement knows what the ladder cost in sweeps.
    """

    temperatures: tuple[float, ...]
    acceptance: tuple[float, ...]
    within_band: bool
    rounds: int
    replicas_measured: int


def adapt_ladder(
    measure: Callable[[tuple[float, ...]], Sequence[float]],
    ladder: tuple[float, ...],
    band: tuple[float, float],
    max_rounds: int,
    max_replicas: int,
) -> AdaptedLadder:
    """Insert and remove temperatures until every neighbouring pair exchanges within ``band``.

    Each round measures the acceptance of every neighbouring pair on the
    current ladder, then: a pair below the band gets the geometric mean of
    its two temperatures inserted between them; an interior temperature
    both of whose pairs are above the band is removed; and a pair above the
    band beside one inside it has their shared temperature moved halfway
    toward the far end of the pair inside, widening the one and narrowing
    the other. Geometric, because
    the exchange ratio ``eq:exchange`` depends on the temperatures through
    ``beta_i - beta_j``, and for an energy whose variance is set by the
    temperature the acceptance is a function of the ratio of neighbouring
    temperatures rather than their difference. The endpoints are never
    moved: they are the temperatures the caller wants the hottest and
    coldest replica at, and the ladder's job is to connect them.

    The measurement is a callable so this function knows no model and no
    sampler: it is handed a ladder and returns one acceptance per
    neighbouring pair, and whatever it runs to get them is its own. A
    measurement is a Monte Carlo estimate, so a band narrower than its
    noise is a ladder that never settles; the caller chooses a band the
    measurement can resolve, and ``within_band`` on the result says whether
    it did.

    Parameters
    ----------
    measure : Callable[[tuple[float, ...]], Sequence[float]]
        Exchange acceptance per neighbouring pair of a ladder.
    ladder : tuple[float, ...]
        The starting ladder, at least two temperatures, strictly monotone in
        either direction. Its two endpoints are the result's.
    band : tuple[float, float]
        ``(low, high)``, the acceptance every pair is driven into, with
        ``0 < low < high < 1``.
    max_rounds : int
        Measurements to take before stopping, at least 1.
    max_replicas : int
        The most temperatures the ladder may hold; no insertion is made
        past it, so a band the budget cannot reach is reported rather than
        pursued.

    Returns
    -------
    AdaptedLadder

    Raises
    ------
    ValueError
        If the ladder has fewer than two temperatures, is not strictly
        monotone or is not positive, the band is not an interval inside
        ``(0, 1)``, or a budget is below 1.
    """
    _check_ladder(ladder)
    low, high = band
    if not 0.0 < low < high < 1.0:
        msg = f"band must satisfy 0 < low < high < 1, got {band}"
        raise ValueError(msg)
    if max_rounds < 1:
        msg = f"max_rounds must be at least 1, got {max_rounds}"
        raise ValueError(msg)
    if max_replicas < len(ladder):
        msg = (
            f"max_replicas is {max_replicas} but the starting ladder already has "
            f"{len(ladder)} temperatures"
        )
        raise ValueError(msg)

    current = tuple(ladder)
    replicas_measured = 0
    for round_index in range(1, max_rounds + 1):
        acceptance = tuple(float(value) for value in measure(current))
        replicas_measured += len(current)
        if len(acceptance) != len(current) - 1:
            msg = (
                f"measure returned {len(acceptance)} acceptances for a ladder of "
                f"{len(current)}; expected one per neighbouring pair"
            )
            raise ValueError(msg)
        within = all(low <= value <= high for value in acceptance)
        if within or round_index == max_rounds:
            return AdaptedLadder(
                current, acceptance, within, round_index, replicas_measured
            )
        proposal = _revise(current, acceptance, low, high, max_replicas)
        if proposal == current:
            return AdaptedLadder(
                current, acceptance, False, round_index, replicas_measured
            )
        current = proposal
    msg = "unreachable: the loop returns on its last round"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def _revise(
    ladder: tuple[float, ...],
    acceptance: tuple[float, ...],
    low: float,
    high: float,
    max_replicas: int,
) -> tuple[float, ...]:
    """One round of insertions, removals and moves; the endpoints stay.

    In that order. A pair below the band is bisected. An interior
    temperature both of whose pairs are above the band is removed --- never
    two adjacent ones together, since dropping both leaves a gap no
    measurement has seen. A pair above the band with a neighbouring pair
    inside it *moves* their shared temperature halfway, geometrically,
    toward the far end of the neighbouring pair: the high pair widens and
    the neighbouring one narrows, and neither is a temperature the ladder
    can lose. A temperature is moved once per round.
    """
    n = len(ladder)
    room = max_replicas - n
    insert = [False] * (n - 1)
    for index in range(n - 1):
        if acceptance[index] < low and room > 0:
            insert[index] = True
            room -= 1
    keep = [True] * n
    for index in range(1, n - 1):
        if (
            keep[index - 1]
            and acceptance[index - 1] > high
            and acceptance[index] > high
        ):
            keep[index] = False
    value = list(ladder)
    moved = [False] * n
    for index in range(n - 1):
        if not (acceptance[index] > high and keep[index] and keep[index + 1]):
            continue
        # Prefer moving the colder end toward the cold side, then the
        # hotter end toward the hot side; either widens this pair.
        for shared, far in ((index + 1, index + 2), (index, index - 1)):
            if not 0 < shared < n - 1 or moved[shared] or not keep[far]:
                continue
            neighbour = min(shared, far)
            if insert[neighbour] or not low <= acceptance[neighbour] <= high:
                continue
            value[shared] = math.sqrt(ladder[shared] * ladder[far])
            moved[shared] = True
            break
    revised: list[float] = []
    for index in range(n):
        if keep[index]:
            revised.append(value[index])
        if index < n - 1 and insert[index]:
            revised.append(math.sqrt(ladder[index] * ladder[index + 1]))
    return tuple(revised)


def _check_ladder(ladder: tuple[float, ...]) -> None:
    if len(ladder) < 2:
        msg = f"a ladder needs at least two temperatures, got {len(ladder)}"
        raise ValueError(msg)
    for temperature in ladder:
        _check_temperature("every temperature", temperature)
    differences = [b - a for a, b in itertools.pairwise(ladder)]
    if not (all(d > 0 for d in differences) or all(d < 0 for d in differences)):
        msg = (
            f"a ladder must be strictly monotone so its neighbouring pairs are "
            f"its exchanges, got {ladder}"
        )
        raise ValueError(msg)
