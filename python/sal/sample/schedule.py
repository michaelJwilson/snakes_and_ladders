"""Temperature schedules: a temperature per step, and nothing about what is tempered.

Three consumers sit in three modules --- the Hamiltonian sampler in
``sal.opt``, the Potts move sets in
``sal.search``, the Boltzmann policy in
``sal.learn``. ``opt`` and ``learn`` may import no application
module and ``search`` may import both, so this is the one placement all three
reach, the argument that puts ``constrain.py`` here too (issue #267).

**Tempering a likelihood is not tempering an energy.** ``beta * E`` is a
temperature in the physical sense; ``beta * (-log L)`` is a *power posterior*
(Friel & Pettitt, 2008), a different object. The schedule serves both and
names neither; the consumer says which it is.

**Mirrors ``torch.optim.lr_scheduler``**, with one deliberate difference:
every schedule here declares its length and both endpoints, and reaches the
final temperature at *exactly* the last step. A schedule that never quite
arrives has no final temperature to check, and that off-by-one is a fault no
downstream distributional test would localize.

**A schedule a consumer builds at its own length is four numbers.** A
consumer that fixes its step count from a budget is handed a
:class:`ScheduleParams` --- a :class:`ScheduleShape`, both endpoints and the
fraction of steps held at the end --- and builds it; with no hold the build is
the shape's own schedule, so the default consumer's floats do not move
(issue #1038).

**A tempering ladder is chosen from what it measures, not set by hand.** A
ladder is a set of temperatures rather than a sequence in time, and what makes
one right is the exchange acceptance between each neighbouring pair: near zero
nothing crosses the gap, near one one of the two is redundant.
:func:`adapt_ladder` measures those acceptances, bisects a gap below a stated
band and removes a temperature both of whose gaps are above it, until every
pair sits inside the band or a budget is spent. It takes the measurement as a
callable so it knows no model (issue #333). Reheating on a stalled chain --- a
schedule in *time* that adapts --- remains absent.

**An acceptance is a per-pair number, and a round trip is a statement about
the ladder.** A ladder every pair exchanges across can still be one no walker
crosses end to end, so :func:`adapt_ladder_by_round_trips` places a ladder of
the *same length* by the other criterion (Katzgraber, Trebst, Huse & Troyer
2006): the fraction of walkers moving up at each rung, and the rungs
redistributed so the local diffusivity is flat. It is the sibling of
:func:`adapt_ladder` and not its replacement --- the acceptance-placed ladder
is what a caller that asks for neither gets --- and the two answer different
questions with different measurements, which is why the measurement is a
different callable rather than a flag on one (issue #756).

The policy's learned softmax weight is an inverse temperature too, and is
**not** put on a schedule: it is what the agent learns, and a declared
schedule would remove it.
"""

from __future__ import annotations

import bisect
import itertools
import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


@runtime_checkable
class TempSchedule(Protocol):
    """A positive temperature for each of ``n_steps`` steps.

    ``step`` runs from ``0`` to ``n_steps - 1`` inclusive. Asking outside that
    range is refused rather than clamped: a consumer that asks for step
    ``n_steps`` has run one iteration longer than it declared, and clamping
    would let it.
    """

    #: How many steps the schedule covers. Declared as data rather than as a
    #: ``@property``, because a property in a protocol body is a descriptor
    #: every implementer inherits, and it would shadow the dataclass field
    #: :class:`Constant` and :class:`_Interpolated` keep it in (issue #586).
    n_steps: int

    @abstractmethod
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
class ConstantTempSchedule(TempSchedule):
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
class _InterpolatedTempSchedule(TempSchedule):
    """``start`` at step 0, ``end`` at the last step, some curve between.

    Each subclass supplies the weight ``w(t)`` on ``start`` at fraction ``t``
    of the way through, with ``w(0) = 1`` and ``w(1) = 0`` exactly, combined so
    both endpoints come out *bitwise*. A one-step schedule has one
    temperature, so ``start`` and ``end`` must then agree.

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
class LinearTempSchedule(_InterpolatedTempSchedule):
    """Straight line from ``start`` to ``end``. ``LinearLR`` with declared ends.

    ``T(t) = (1 - t) * start + t * end``.
    """

    def __call__(self, step: int) -> float:
        fraction = self._fraction(step)
        return (1.0 - fraction) * self.start + fraction * self.end


@dataclass(frozen=True)
class ExponentialTempSchedule(_InterpolatedTempSchedule):
    """Geometric from ``start`` to ``end``. ``ExponentialLR`` with a declared end.

    ``T(t) = start ** (1 - t) * end ** t``, so successive temperatures have
    the constant ratio ``(end / start) ** (1 / (n_steps - 1))`` ---
    ``ExponentialLR``'s ``gamma`` --- and the endpoints are exact.

    The schedule simulated annealing is usually run on (Kirkpatrick et al.,
    1983): a constant *factor* per step spends equal effort per decade of
    temperature, where a linear schedule spends almost all of it cold.
    """

    def __call__(self, step: int) -> float:
        fraction = self._fraction(step)
        return math.pow(self.start, 1.0 - fraction) * math.pow(self.end, fraction)


@dataclass(frozen=True)
class CosineTempSchedule(_InterpolatedTempSchedule):
    """Half a cosine from ``start`` to ``end``. ``CosineAnnealingLR`` with ``T_max = n_steps - 1``.

    ``T(t) = w * start + (1 - w) * end`` with ``w = (1 + cos(pi t)) / 2``.
    Slow at both ends and fast in the middle; written with the weight on
    ``start`` rather than in the textbook form ``end + (start - end) * w`` so
    that ``t = 0`` returns ``start`` bitwise rather than ``end + start - end``.
    """

    def __call__(self, step: int) -> float:
        weight = 0.5 * (1.0 + math.cos(math.pi * self._fraction(step)))
        return weight * self.start + (1.0 - weight) * self.end


class ScheduleShape(StrEnum):
    """The curve an annealing ramp follows between its two endpoints."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    COSINE = "cosine"


_RAMPS: dict[ScheduleShape, type[_InterpolatedTempSchedule]] = {
    ScheduleShape.EXPONENTIAL: ExponentialTempSchedule,
    ScheduleShape.LINEAR: LinearTempSchedule,
    ScheduleShape.COSINE: CosineTempSchedule,
}


@dataclass(frozen=True)
class HeldTempSchedule(TempSchedule):
    """A ramp, then its final temperature held for ``hold`` further steps.

    The ramp's steps are returned as the ramp returns them, so a hold of zero
    is the ramp itself step for step.

    Parameters
    ----------
    ramp : TempSchedule
        The schedule up to the hold.
    hold : int
        Steps at the ramp's last temperature after it, ``>= 0``.
    """

    ramp: TempSchedule
    hold: int

    def __post_init__(self) -> None:
        if self.hold < 0:
            msg = f"a hold is a count of steps, got {self.hold}"
            raise ValueError(msg)

    @property
    def n_steps(self) -> int:  # type: ignore[override]
        """The ramp's steps and the held ones."""
        return self.ramp.n_steps + self.hold

    def __call__(self, step: int) -> float:
        _check_step(step, self.n_steps)
        return self.ramp(min(step, self.ramp.n_steps - 1))


@dataclass(frozen=True)
class ScheduleParams:
    """An annealing schedule as four numbers, built at whatever length a budget buys.

    A consumer that fixes its step count from a budget cannot be handed a
    :class:`TempSchedule`, whose length is part of it; it is handed these
    and calls :meth:`build`. They are also what a search over schedules
    varies (issue #1038).

    Parameters
    ----------
    shape : ScheduleShape
        The ramp's curve.
    t_start, t_end : float
        Temperatures at the first step and from the ramp's last step on,
        positive.
    hold : float
        Fraction of the steps held at ``t_end`` after the ramp, in
        ``[0, 1)``. The held count is ``floor(hold * n_steps)``.
    """

    shape: ScheduleShape
    t_start: float
    t_end: float
    hold: float = 0.0

    def __post_init__(self) -> None:
        _check_temperature("t_start", self.t_start)
        _check_temperature("t_end", self.t_end)
        if not 0.0 <= self.hold < 1.0:
            msg = f"hold is a fraction of the steps in [0, 1), got {self.hold}"
            raise ValueError(msg)

    def build(self, n_steps: int) -> TempSchedule:
        """The schedule of exactly ``n_steps`` steps.

        With ``hold == 0`` this is the shape's own schedule, the object a
        caller building it by hand gets, so a default consumer's floats are
        unchanged.

        Raises
        ------
        ValueError
            If ``n_steps < 1``, or if the hold leaves a one-step ramp between
            two different temperatures.
        """
        _check_length(n_steps)
        held = math.floor(self.hold * n_steps)
        ramp = _RAMPS[self.shape](self.t_start, self.t_end, n_steps - held)
        return ramp if held == 0 else HeldTempSchedule(ramp, held)


@dataclass(frozen=True)
class LadderTempSchedule(TempSchedule):
    """The temperatures as given, one per step: a ladder is a schedule indexed by rung.

    The second spelling of a schedule this repository carried was a bare
    sequence of temperatures --- ``parallel_tempering(graph, field,
    temperatures, ...)`` and the tempered ensembles --- so a caller who had
    built a :class:`TempSchedule` could not hand it to a tempering. This is
    the sequence as a schedule, and :func:`ladder` reads either spelling into
    the tuple those functions consume, so the values a run sees are the same
    floats whichever way they were written (issue #827).

    Parameters
    ----------
    values : tuple[float, ...]
        Positive temperatures, at least one; ``n_steps`` is their count.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _check_length(len(self.values))
        for value in self.values:
            _check_temperature("temperature", value)

    @property
    def n_steps(self) -> int:  # type: ignore[override]
        """One step per rung."""
        return len(self.values)

    def __call__(self, step: int) -> float:
        _check_step(step, len(self.values))
        return self.values[step]


def ladder(values: TempSchedule | Sequence[float]) -> tuple[float, ...]:
    """A tuple of temperatures from a schedule or from a sequence, the same floats either way.

    A schedule is read step by step; a sequence is read as given. What a
    tempering validates about the ladder --- its length, its sign, its order
    --- it validates on the tuple, so the two spellings are refused on the
    same terms.
    """
    if isinstance(values, TempSchedule):
        return tuple(temperatures(values))
    return tuple(float(value) for value in values)


def beta_ladder(values: TempSchedule | Sequence[float]) -> tuple[float, ...]:
    """A tuple of inverse temperatures from a schedule or from a sequence.

    :func:`ladder` for a consumer whose rungs are ``beta`` rather than ``T``.
    A schedule declares temperatures, so it is read step by step and
    inverted; a sequence is the inverse temperatures as given. The one
    spelling a schedule cannot carry is ``beta = 0``, the infinite
    temperature :func:`_check_temperature` refuses, so a consumer anchored
    there refuses a schedule on its own terms rather than on this one
    (issue #861).
    """
    if isinstance(values, TempSchedule):
        return tuple(1.0 / temperature for temperature in temperatures(values))
    return tuple(float(value) for value in values)


class Monotone(StrEnum):
    """What order a ladder's rungs must be in.

    An order is a requirement of the consumer and not of the ladder: an
    exchange between neighbouring pairs is the same law whichever end is
    cold, so a tempering states :data:`ANY`, a warm-up that bisects a gap
    states :data:`EITHER`, and a consumer that reads ``temperatures[0]`` as
    the coldest states :data:`INCREASING`.
    """

    #: Any order; which pairs are neighbours is what the ladder fixes.
    ANY = "any"
    #: Strictly monotone, in either direction.
    EITHER = "either"
    #: Strictly increasing, coldest first.
    INCREASING = "increasing"


class Quantity(StrEnum):
    """What a rung carries, which is what its sign is checked against."""

    #: A temperature, strictly positive.
    TEMPERATURE = "temperature"
    #: An inverse temperature, non-negative: ``beta = 0`` is a rung.
    BETA = "beta"


#: The count a refusal spells in words. Each of the five validators #861
#: folded spelled its own, and a test reads each, so the temperature ladders
#: keep the word and the beta ladders the digit they were written with.
_COUNTS = {1: "one", 2: "two", 3: "three"}


def check_ladder(
    values: Sequence[float],
    *,
    needed_by: str,
    minimum: int = 2,
    monotone: Monotone = Monotone.ANY,
    quantity: Quantity = Quantity.TEMPERATURE,
    from_zero: bool = False,
) -> tuple[float, ...]:
    """The one validator of a ladder: its length, its sign, its order, its anchor (issue #861).

    Five validators checked four requirements in four combinations ---
    ``schedule``'s and ``tempered``'s ``_check_ladder``, the two temperings'
    inline copies and ``annealed``'s ``_check_betas`` --- and a requirement a
    site did not state was a requirement nothing held it to. Each is an
    argument here, so a consumer says what it needs and reads the refusal
    its need earns.

    Parameters
    ----------
    values : Sequence[float]
        The ladder, already read from its spelling by :func:`ladder` or
        :func:`beta_ladder`.
    needed_by : str
        What refuses a ladder too short, named in that refusal.
    minimum : int
        Rungs the consumer needs. Two, where a ladder is exchanged across;
        one, where the ladder is a sequence of distributions and the first
        is the answer.
    monotone : Monotone
        The order required, as :class:`Monotone` states it.
    quantity : Quantity
        Temperature or inverse temperature, which is the sign required.
    from_zero : bool
        Whether the first rung must be ``beta = 0`` exactly --- the rung
        where ``log Z_0 = n log q`` is exact and an estimate is anchored.

    Returns
    -------
    tuple[float, ...]
        The ladder as floats, in the order given.

    Raises
    ------
    ValueError
        On any requirement the ladder does not meet, naming the requirement
        and the ladder.
    """
    rungs = tuple(float(value) for value in values)
    if len(rungs) < minimum:
        if quantity is Quantity.BETA:
            msg = f"{needed_by} needs at least {minimum} rungs, got {len(rungs)}"
        else:
            count = _COUNTS.get(minimum, str(minimum))
            msg = (
                f"{needed_by} needs at least {count} temperatures, got "
                f"{len(rungs)}: a ladder of one has nothing to exchange"
            )
        raise ValueError(msg)
    if quantity is Quantity.BETA:
        _check_betas(rungs, from_zero=from_zero, monotone=monotone)
        return rungs
    if monotone is Monotone.INCREASING:
        # The sign and the order in one refusal: a consumer that reads the
        # first rung as the coldest is told which of the two it broke by the
        # ladder printed beside them.
        for cold, hot in itertools.pairwise(rungs):
            if not 0.0 < cold < hot:
                msg = (
                    f"temperatures must be positive and increasing, coldest first, "
                    f"got {rungs}"
                )
                raise ValueError(msg)
        return rungs
    for temperature in rungs:
        # The sign of a temperature is checked where every schedule in this
        # module checks it, rather than a sixth time here.
        _check_temperature("every rung", temperature)
    if monotone is Monotone.EITHER:
        differences = [b - a for a, b in itertools.pairwise(rungs)]
        if not (all(d > 0 for d in differences) or all(d < 0 for d in differences)):
            msg = (
                f"a ladder must be strictly monotone so its neighbouring pairs are "
                f"its exchanges, got {rungs}"
            )
            raise ValueError(msg)
    return rungs


def _check_betas(
    rungs: tuple[float, ...], *, from_zero: bool, monotone: Monotone
) -> None:
    """The sign, the anchor and the order of a ladder in ``beta``.

    ``beta = 0`` is the infinite temperature: a rung, and the one a
    :class:`TempSchedule` cannot carry.
    """
    if from_zero and rungs[0] != 0.0:
        msg = (
            f"the ladder must start at beta = 0, got {rungs[0]}: that rung is "
            "where log Z = n log q is exact, and the estimate is anchored on it"
        )
        raise ValueError(msg)
    if rungs[0] < 0.0:
        msg = f"every inverse temperature must be >= 0, got {rungs[0]}"
        raise ValueError(msg)
    if monotone is Monotone.INCREASING and any(
        later <= earlier for earlier, later in itertools.pairwise(rungs)
    ):
        msg = f"the ladder must be strictly increasing in beta, got {rungs}"
        raise ValueError(msg)


def temperatures(schedule: TempSchedule) -> list[float]:
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
        change, and the caller decides whether that ladder is usable.
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
    start: TempSchedule | Sequence[float],
    band: tuple[float, float],
    max_iterations: int,
    max_replicas: int,
) -> AdaptedLadder:
    """Insert and remove temperatures until every neighbouring pair exchanges within ``band``.

    Each round measures the acceptance of every neighbouring pair, then: a
    pair below the band gets the geometric mean of its two temperatures
    inserted between them; an interior temperature both of whose pairs are
    above the band is removed; and a pair above the band beside one inside it
    has their shared temperature moved halfway toward the far end of the pair
    inside. Geometric, because the exchange ratio ``eq:exchange`` depends on
    the temperatures through ``beta_i - beta_j``, so for an energy whose
    variance is set by the temperature the acceptance is a function of their
    ratio rather than their difference. The endpoints are never moved: the
    ladder's job is to connect them.

    The measurement is a callable, so this function knows no model and no
    sampler. It is a Monte Carlo estimate, so a band narrower than its noise
    is a ladder that never settles; ``within_band`` on the result says whether
    the caller's band was reachable.

    Parameters
    ----------
    measure : Callable[[tuple[float, ...]], Sequence[float]]
        Exchange acceptance per neighbouring pair of a ladder.
    start : TempSchedule | Sequence[float]
        The starting ladder, in either spelling and read by :func:`ladder`
        into the same floats: at least two temperatures, strictly monotone in
        either direction. Its two endpoints are the result's.
    band : tuple[float, float]
        ``(low, high)``, the acceptance every pair is driven into, with
        ``0 < low < high < 1``.
    max_iterations : int
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
    rungs = check_ladder(ladder(start), needed_by="a ladder", monotone=Monotone.EITHER)
    low, high = band
    if not 0.0 < low < high < 1.0:
        msg = f"band must satisfy 0 < low < high < 1, got {band}"
        raise ValueError(msg)
    if max_iterations < 1:
        msg = f"max_iterations must be at least 1, got {max_iterations}"
        raise ValueError(msg)
    if max_replicas < len(rungs):
        msg = (
            f"max_replicas is {max_replicas} but the starting ladder already has "
            f"{len(rungs)} temperatures"
        )
        raise ValueError(msg)

    current = rungs
    replicas_measured = 0
    for round_index in range(1, max_iterations + 1):
        acceptance = tuple(float(value) for value in measure(current))
        replicas_measured += len(current)
        if len(acceptance) != len(current) - 1:
            msg = (
                f"measure returned {len(acceptance)} acceptances for a ladder of "
                f"{len(current)}; expected one per neighbouring pair"
            )
            raise ValueError(msg)
        within = all(low <= value <= high for value in acceptance)
        if within or round_index == max_iterations:
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

    In that order. A pair below the band is bisected. An interior temperature
    both of whose pairs are above the band is removed --- never two adjacent
    ones together, since dropping both leaves a gap no measurement has seen. A
    pair above the band with a neighbouring pair inside it *moves* their
    shared temperature halfway, geometrically, toward the far end of the
    neighbouring pair. A temperature is moved once per round.
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


@dataclass(frozen=True)
class FeedbackLadder:
    """What a round-trip warm-up placed, and the circulation it placed it from.

    Parameters
    ----------
    temperatures : tuple[float, ...]
        The ladder, the same length as the one it started from and with the
        same endpoints: the criterion redistributes rungs, it does not buy
        them.
    up_fraction : tuple[float, ...]
        The fraction of labelled visits at each rung made by a walker on its
        way up --- one per temperature, ``1`` at the rung walkers start their
        ascent from and ``0`` at the far end. Measured on the ladder
        :attr:`temperatures` was placed *from*, one round back, where
        :attr:`AdaptedLadder.acceptance` is measured on the ladder it is
        returned with: a placement consumes its measurement where a revision
        stops on one, so a single round here is one measurement and one
        placement rather than a ladder unchanged.
    converged : bool
        Whether the last revision moved every temperature by less than
        ``tolerance``, relatively. False means the warm-up stopped on its
        budget or on a measurement too noisy to settle, and the caller
        decides whether that ladder is usable.
    rounds : int
        Measurements taken, the last one included.
    replicas_measured : int
        Sum of the ladder's length over every measurement --- the warm-up's
        cost in replica-runs, as :attr:`AdaptedLadder.replicas_measured` is.
    """

    temperatures: tuple[float, ...]
    up_fraction: tuple[float, ...]
    converged: bool
    rounds: int
    replicas_measured: int


def adapt_ladder_by_round_trips(
    measure: Callable[[tuple[float, ...]], Sequence[float]],
    start: TempSchedule | Sequence[float],
    tolerance: float,
    max_iterations: int,
) -> FeedbackLadder:
    """Redistribute a ladder of fixed length so a walker's round trip is fastest.

    Feedback-optimized parallel tempering (Katzgraber, Trebst, Huse & Troyer
    2006). ``measure`` returns ``f(T)``, the fraction of labelled visits at
    each rung made by walkers moving up the ladder; it falls from 1 at the
    rung where the ascent is labelled to 0 at the far end, and where it falls
    *steeply* the walkers are held up. The local diffusivity of a walker is
    then flattened by placing the rungs at equal increments of

        ``eta(T) dT = sqrt((-df/dT) / (dT)) dT = sqrt(-df dT)``,

    so an interval across which ``f`` drops by a lot, or which is wide, takes
    more rungs. The endpoints are the caller's and are returned bitwise: the
    ladder's job is to connect them, and only the rungs between are the
    criterion's business.

    ``f`` is measured, so it need not be monotone; it is made non-increasing
    by a running minimum before the placement, rather than refused, because a
    reversal is Monte Carlo noise on a monotone quantity and refusing it would
    reject a ladder for the noise of its own warm-up.

    Parameters
    ----------
    measure : Callable[[tuple[float, ...]], Sequence[float]]
        The up-fraction per rung of a ladder --- one per temperature, all
        finite.
        :func:`sal.sample.tempered.up_fraction` computes it
        from a walker trace.
    start : TempSchedule | Sequence[float]
        The starting ladder, in either spelling and read by :func:`ladder`
        into the same floats: at least three temperatures --- two are the
        endpoints and there is nothing to place --- strictly monotone in
        either direction, all positive.
    tolerance : float
        Relative move, positive: a round whose largest ``|T' / T - 1|`` is
        below it stops the warm-up and reports ``converged``.
    max_iterations : int
        Measurements to take before stopping, at least 1.

    Returns
    -------
    FeedbackLadder

    Raises
    ------
    ValueError
        If the ladder has fewer than three temperatures, is not strictly
        monotone or is not positive; if ``tolerance`` is not positive or
        ``max_iterations`` is below 1; if ``measure`` returns other than one value
        per rung, or a value that is not finite; or if ``f`` is flat over the
        whole ladder, which is a run in which no walker circulated and so
        carries no placement.
    """
    rungs = check_ladder(ladder(start), needed_by="a ladder", monotone=Monotone.EITHER)
    if len(rungs) < 3:
        msg = (
            f"a round-trip placement needs at least three temperatures, got "
            f"{len(rungs)}: the two endpoints are the caller's"
        )
        raise ValueError(msg)
    if not tolerance > 0.0:
        msg = f"tolerance must be positive, got {tolerance}"
        raise ValueError(msg)
    if max_iterations < 1:
        msg = f"max_iterations must be at least 1, got {max_iterations}"
        raise ValueError(msg)

    current = rungs
    replicas_measured = 0
    for round_index in range(1, max_iterations + 1):
        fraction = tuple(float(value) for value in measure(current))
        replicas_measured += len(current)
        if len(fraction) != len(current):
            msg = (
                f"measure returned {len(fraction)} up-fractions for a ladder of "
                f"{len(current)}; expected one per rung"
            )
            raise ValueError(msg)
        if not all(math.isfinite(value) for value in fraction):
            msg = (
                f"every up-fraction must be finite, got {fraction}: a rung no "
                "labelled walker visited is a ladder the placement cannot read"
            )
            raise ValueError(msg)
        proposal = _place(current, fraction)
        moved = max(
            abs(new / old - 1.0) for new, old in zip(proposal, current, strict=True)
        )
        if moved < tolerance or round_index == max_iterations:
            return FeedbackLadder(
                proposal, fraction, moved < tolerance, round_index, replicas_measured
            )
        current = proposal
    msg = "unreachable: the loop returns on its last round"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def _place(ladder: tuple[float, ...], fraction: Sequence[float]) -> tuple[float, ...]:
    """The rungs at equal increments of ``sqrt(-df dT)``, the endpoints kept bitwise.

    ``f`` is made non-increasing by a running minimum, the mass
    ``sqrt(-df dT)`` of each interval is accumulated, and rung ``k`` is placed
    where the accumulation reaches ``k / (n - 1)`` of the total --- by linear
    interpolation inside the interval, where the mass is uniform in ``T``
    because ``eta`` is piecewise constant.
    """
    n = len(ladder)
    monotone = list(itertools.accumulate(fraction, min))
    widths = [abs(ladder[i + 1] - ladder[i]) for i in range(n - 1)]
    mass = [
        math.sqrt(max(monotone[i] - monotone[i + 1], 0.0) * widths[i])
        for i in range(n - 1)
    ]
    total = math.fsum(mass)
    if total == 0.0:
        msg = (
            f"the up-fraction is flat over the whole ladder, {tuple(fraction)}: no "
            "walker circulated, and a placement read from that is read from nothing"
        )
        raise ValueError(msg)
    cumulative = list(itertools.accumulate(mass, initial=0.0))
    placed = [ladder[0]]
    for rung in range(1, n - 1):
        target = total * rung / (n - 1)
        # `target` is strictly inside `(0, total)` and `cumulative` ends at
        # `total`, so this interval exists and carries mass; a run of
        # zero-mass intervals sharing a boundary resolves to the last of them.
        interval = bisect.bisect_right(cumulative, target) - 1
        within = (target - cumulative[interval]) / mass[interval]
        placed.append(
            ladder[interval] + within * (ladder[interval + 1] - ladder[interval])
        )
    placed.append(ladder[-1])
    return tuple(placed)
