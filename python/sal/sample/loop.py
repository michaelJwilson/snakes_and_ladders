"""One annealing loop and one tempering loop, over a :class:`Step` (issue #1218).

Four annealers and four temperings each walked their own schedule, ran their
own exchange and summed their own cost. What differs between them is the
move: a Hamiltonian trajectory, a Potts sweep, a heat-bath sweep over a
factor graph, a Metropolis step over topologies. That move is a
:class:`Step`; the walk down a schedule is :func:`anneal`, the rounds of
replica exchange are :func:`temper`, and each entry point is a constructor of
its step and a reader of what the loop returns.

**A step carries what it scored.** It is handed the state, the energy the
last step left there and an opaque ``carried`` value --- a gradient, or
nothing --- and returns the same three where it landed, so no loop scores a
point twice. An exchange swaps the carried value with the state it belongs
to, since neither depends on the temperature.

**A step charges its own move, in its own unit** (``sample/CLAUDE.md``): the
loop sums the charges and reports :attr:`Step.unit` beside them, and the
charge for scoring a start rides on the start's :class:`Moved`.

**What stays with its problem.** The cluster constructions, the heat-bath
conditionals, an M step and log ``Z`` estimation stay in their modules, as do
three loops that do not fold here: a move between two rungs
(:func:`~sal.sample.potts_mcmc.cluster_tempering`'s Houdayer move), a walker
that moves along the ladder rather than exchanging
(:func:`~sal.sample.annealed.simulated_tempering`), and a sweep whose state
is re-estimated rather than moved
(:func:`~sal.sample.mixture_anneal.anneal_assignments`).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import numpy as np

from sal.cost import Cost
from sal.opt.termination import Termination
from sal.sample.schedule import Annealed, TempSchedule
from sal.track import TrackedOptimization, current


class Moved[S, C](NamedTuple):
    """Where a step left one chain: the state, its energy, what it carries, and the charge.

    Parameters
    ----------
    state : S
        The state after the move.
    energy : float
        Its energy at temperature one, which the loop minimizes and the
        exchange ratio reads.
    carried : C
        What the step scored beside the energy and hands to the next step at
        this state --- ``grad U`` for a Hamiltonian step, ``None`` for a step
        that keeps nothing.
    spent : int
        What the move cost, in the step's :attr:`Step.unit`. On a start, what
        scoring it cost.
    """

    state: S
    energy: float
    carried: C
    spent: int


class Step[S, C, R](Protocol):
    """One move at one temperature, charged in its own unit.

    ``__call__(state, energy, carried, temperature, rng)`` moves ``state``
    once at ``temperature`` on ``rng``, reading ``energy`` and ``carried``
    rather than scoring the state again, and returns the :class:`Moved` it
    left. A step may move ``state`` in place; :meth:`keep` is what the loop
    stores when it keeps a state past the next move.
    """

    @property
    def unit(self) -> Cost:
        """The unit each :class:`Moved` charge is counted in."""
        ...

    def __call__(
        self, state: S, energy: float, carried: C, temperature: float, rng: R, /
    ) -> Moved[S, C]:
        """One move at ``temperature``."""
        ...

    def keep(self, state: S) -> S:
        """A copy of ``state`` no later move changes; the state itself where none can."""
        ...


@dataclass(frozen=True, kw_only=True)
class Walked[S](Annealed[S]):
    """What :func:`anneal` returns: an :class:`~sal.sample.schedule.Annealed` with its energies.

    Parameters
    ----------
    energy : float
        ``best``'s energy; an entry point reports it in its own sign.
    energies : tuple[float, ...]
        The energy at the start and after every step, ``n_steps + 1`` of them.
    """

    energy: float
    energies: tuple[float, ...]


def _state_bytes(state: object) -> int | None:
    """The bytes an array state holds, or ``None`` for a state that is not an array."""
    nbytes = getattr(state, "nbytes", None)
    return None if nbytes is None else int(nbytes)


def anneal[S, C, R](
    step: Step[S, C, R],
    schedule: TempSchedule,
    start: Moved[S, C],
    rng: R,
) -> Walked[S]:
    """Simulated annealing: one ``step`` per schedule entry at its temperature, keeping the lowest energy.

    The best is the *lowest* energy visited, the start included, and a later
    state replaces it only when strictly lower. Each step is recorded into
    the enclosing ``track`` run with the best state and energy so far and the
    step's temperature, and the run's cost once at the end.

    Parameters
    ----------
    step : Step
        The move, at each schedule step's temperature.
    schedule : TempSchedule
        Temperature per step; its length is the number of steps.
    start : Moved
        The start, its energy, what it carries, and what scoring it cost.
    rng : R
        Handed to every step.

    Returns
    -------
    Walked
        The schedule's length as the termination, never converged.
    """
    state, energy, carried, spent = start
    best, best_energy = step.keep(state), energy
    energies = [energy]
    tracked: TrackedOptimization = current()
    for index in range(schedule.n_steps):
        temperature = schedule(index)
        state, energy, carried, cost = step(state, energy, carried, temperature, rng)
        spent += cost
        energies.append(energy)
        if energy < best_energy:
            best, best_energy = step.keep(state), energy
        tracked.record(index, state=best, temperature=temperature, energy=best_energy)
    nbytes = _state_bytes(state)
    if nbytes is not None:
        tracked.record_cost(max(schedule.n_steps - 1, 0), nbytes)
    return Walked(
        best=best,
        energy=best_energy,
        final=state,
        energies=tuple(energies),
        spent=spent,
        unit=step.unit,
        termination=Termination.after(schedule.n_steps, converged=False),
    )


def swap_log_ratio(
    beta_low: float, beta_high: float, energy_low: float, energy_high: float
) -> float:
    """Log acceptance of exchanging the configurations at two temperatures.

    The joint target is the product of the tempered marginals, so the ratio is
    ``(beta_i - beta_j)(E_i - E_j)``: an exchange handing the colder replica
    the lower energy is always accepted. A version omitting this term still
    runs, still mixes, and converges to the wrong distribution ---
    `tests/regression/sample/test_potts_mcmc.py` replaces this function with it
    and asserts the chi-square catches it.
    """
    return (beta_low - beta_high) * (energy_low - energy_high)


@dataclass(eq=False)
class Exchanging[S, C]:
    """The tempering loop's state, read by an observer after every round.

    Parameters
    ----------
    states, energies, carried : list
        Per rung, in the ladder's order, after the round's exchanges.
    best : S
        The lowest-energy state seen at any rung, start included.
    energy : float
        Its energy.
    proposed, accepted : np.ndarray
        Exchanges per adjacent pair so far.
    trace : list[list[int]]
        Per recorded round, the rung of each walker.
    spent : int
        What the rounds and the starts have cost so far, in the steps' unit.
    """

    states: list[S]
    energies: list[float]
    carried: list[C]
    best: S
    energy: float
    proposed: np.ndarray
    accepted: np.ndarray
    trace: list[list[int]] = field(default_factory=list)
    spent: int = 0

    @property
    def walkers(self) -> np.ndarray:
        """The trace as an array, ``(n_recorded, n_replicas)``."""
        return np.array(self.trace, dtype=np.int64).reshape(
            len(self.trace), len(self.states)
        )


@dataclass(frozen=True, kw_only=True)
class Exchanged[S]:
    """What :func:`temper` returns, for an entry point to read into its own result.

    Parameters
    ----------
    best : S
        The lowest-energy state seen at any rung after any round, or a start.
    energy : float
        Its energy.
    swap_acceptance : np.ndarray
        Accepted over proposed exchanges per adjacent pair.
    walkers : np.ndarray
        The rung of each walker at each recorded round,
        ``(n_recorded, n_replicas)``.
    energies : np.ndarray
        Every rung's energy at each recorded round, ``(n_recorded, n_replicas)``.
    keys : tuple[tuple[Hashable, ...], ...]
        Per rung, the key of its state at each recorded round; empty without a
        ``key``.
    scores : dict[Hashable, float]
        Every recorded key's energy; empty without a ``key``.
    spent : int
        The steps' charges and the starts', in ``unit``.
    unit : Cost
        The first rung's step's unit.
    rounds : int
        Rounds run, which a ``stop`` may cut short.
    """

    best: S
    energy: float
    swap_acceptance: np.ndarray
    walkers: np.ndarray
    energies: np.ndarray
    keys: tuple[tuple[Hashable, ...], ...]
    scores: dict[Hashable, float]
    spent: int
    unit: Cost
    rounds: int


def temper[S, C, R](
    steps: Sequence[Step[S, C, R]],
    temperatures: Sequence[float],
    starts: Sequence[Moved[S, C]],
    children: Sequence[R],
    swap: Callable[[float], bool],
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    key: Callable[[S], Hashable] | None = None,
    record: Callable[[Sequence[S]], None] | None = None,
    observe: Callable[[int, Exchanging[S, C]], None] | None = None,
    stop: Callable[[], bool] | None = None,
    ratio: Callable[[float, float, float, float], float] = swap_log_ratio,
) -> Exchanged[S]:
    """Replica exchange: one step per rung per round, then an exchange proposal on every adjacent pair.

    Rung ``r`` moves its state by ``steps[r]`` at ``temperatures[r]`` on
    ``children[r]``: moves belong to rungs and states to walkers, so a state
    handed down the ladder is next moved by the colder rung's step. Each pair
    proposes in ladder order and accepts on ``swap(ratio(beta_i, beta_j,
    E_i, E_j))``.

    **The draw is the caller's and the test is here**, which is
    :mod:`sal.sample.accept`'s own division: ``swap`` is handed the log ratio
    and answers whether the pair exchanges, so a site that draws one NumPy
    uniform only where the ratio is negative and one that draws a torch
    uniform every time are the same loop on their own streams (issue #861).
    ``ratio`` is :func:`swap_log_ratio` unless a caller reads it from its own
    module, where a test replaces it.

    Rounds ``burn_in`` onward are recorded at every ``thin``-th, starting with
    the first. ``record`` is called with the states at each recorded round,
    and ``key`` names each for the result's ``keys`` and ``scores``.
    ``observe`` is called after every round, burn-in included, with the
    loop's state. ``stop`` is asked before every round after the first, and
    ``True`` ends the loop there, so ``burn_in + n_sweeps`` is a ceiling. It
    is for a wall clock and must read no replica's state: a sweep that stops
    on a state-dependent condition biases what it records
    (``sample/CLAUDE.md``).
    """
    n_replicas = len(temperatures)
    betas = [1.0 / temperature for temperature in temperatures]
    states = [start.state for start in starts]
    energies = [start.energy for start in starts]
    lowest = min(range(n_replicas), key=energies.__getitem__)
    run = Exchanging(
        states=states,
        energies=energies,
        carried=[start.carried for start in starts],
        best=steps[lowest].keep(states[lowest]),
        energy=energies[lowest],
        proposed=np.zeros(n_replicas - 1),
        accepted=np.zeros(n_replicas - 1),
        spent=sum(start.spent for start in starts),
    )
    carried = run.carried
    recorded_keys: list[list[Hashable]] = [[] for _ in range(n_replicas)]
    recorded: list[list[float]] = []
    scores: dict[Hashable, float] = {}
    # Which walker sits at each rung. An exchange swaps states between
    # temperatures, so this is what says a *state* crossed the ladder, which
    # the per-pair acceptance cannot.
    at_rung = list(range(n_replicas))
    swept = 0
    for sweep in range(burn_in + n_sweeps):
        if stop is not None and sweep > 0 and stop():
            break
        swept += 1
        for replica in range(n_replicas):
            moved = steps[replica](
                states[replica],
                energies[replica],
                carried[replica],
                temperatures[replica],
                children[replica],
            )
            states[replica], energies[replica], carried[replica] = moved[:3]
            run.spent += moved.spent
        for pair in range(n_replicas - 1):
            log_ratio = ratio(
                betas[pair], betas[pair + 1], energies[pair], energies[pair + 1]
            )
            run.proposed[pair] += 1
            if swap(log_ratio):
                run.accepted[pair] += 1
                upper = pair + 1
                states[pair], states[upper] = states[upper], states[pair]
                energies[pair], energies[upper] = energies[upper], energies[pair]
                carried[pair], carried[upper] = carried[upper], carried[pair]
                at_rung[pair], at_rung[upper] = at_rung[upper], at_rung[pair]
        lowest = min(range(n_replicas), key=energies.__getitem__)
        if energies[lowest] < run.energy:
            run.best, run.energy = steps[lowest].keep(states[lowest]), energies[lowest]
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            if record is not None:
                record(states)
            if key is not None:
                for replica in range(n_replicas):
                    name = key(states[replica])
                    scores[name] = energies[replica]
                    recorded_keys[replica].append(name)
            recorded.append(list(energies))
            rungs = [0] * n_replicas
            for rung, walker in enumerate(at_rung):
                rungs[walker] = rung
            run.trace.append(rungs)
        if observe is not None:
            observe(sweep, run)
    return Exchanged(
        best=run.best,
        energy=run.energy,
        swap_acceptance=run.accepted / run.proposed,
        walkers=run.walkers,
        energies=np.array(recorded),
        keys=tuple(tuple(names) for names in recorded_keys),
        scores=scores,
        spent=run.spent,
        unit=steps[0].unit,
        rounds=swept,
    )
