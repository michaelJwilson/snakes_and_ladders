"""One annealing loop and one tempering loop, over a :data:`Step` (issue #1218).

The annealers and temperings each walked their own schedule, ran their own
exchange and summed their own cost. What differs between them is the move: a
Hamiltonian trajectory, a Potts sweep, a heat-bath sweep over a factor graph,
a Metropolis step over topologies. That move is a :data:`Step`; the walk down
a schedule is :func:`anneal`, the rounds of replica exchange are
:func:`temper`, and each entry point builds its step and reads what the loop
returns.

**A step carries what it scored.** It is handed the state, the energy the
last step left there and an opaque ``carried`` value --- a gradient, or
nothing --- and returns the same three where it landed, so no loop scores a
point twice. An exchange swaps the carried value with the state it belongs
to, since neither depends on the temperature.

**A step charges its own move**, in the unit its entry point reports
(``sample/CLAUDE.md``); the charge for scoring a start rides on the start's
:class:`Moved`. A loop keeps a state past the next move through ``keep``: a
copy where the step moves the state in place.

**What stays with its problem.** The cluster constructions, the heat-bath
conditionals, an M step and log ``Z`` estimation stay in their modules, as do
two loops that do not fold here: a walker that moves along the ladder rather
than exchanging (:func:`~sal.sample.annealed.simulated_tempering`), and a
sweep whose state is re-estimated rather than moved, which scores no start
(:func:`~sal.sample.mixture_anneal.anneal_assignments`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sal.opt.termination import Termination
from sal.sample.schedule import TempSchedule
from sal.track import current


class Moved[S, C](NamedTuple):
    """Where a step left one chain: the state, its energy, what it carries, and the charge.

    ``energy`` is at temperature one, which the loops minimize and the
    exchange ratio reads; ``carried`` is what the step scored beside it ---
    ``grad U`` for a Hamiltonian step, ``None`` for one that keeps nothing;
    ``spent`` is the move's cost, or on a start what scoring it cost.
    """

    state: S
    energy: float
    carried: C
    spent: int


#: One move at one temperature: ``(state, energy, carried, temperature, rng)``
#: to the :class:`Moved` it left, reading ``energy`` and ``carried`` rather
#: than scoring ``state`` again.
type Step[S, C, R] = Callable[[S, float, C, float, R], Moved[S, C]]


@dataclass(frozen=True, kw_only=True)
class Walked[S]:
    """What :func:`anneal` returns: the lowest-energy state and where the chain ended.

    ``energies`` holds the energy at the start and after every step; the
    termination is the schedule's length, never converged.
    """

    best: S
    energy: float
    final: S
    energies: tuple[float, ...]
    spent: int
    termination: Termination


def anneal[S, C, R](
    step: Step[S, C, R],
    schedule: TempSchedule,
    start: Moved[S, C],
    rng: R,
    keep: Callable[[S], S],
) -> Walked[S]:
    """Simulated annealing: one ``step`` per schedule entry at its temperature, keeping the lowest energy.

    The best is the lowest energy visited, the start included, replaced only
    by a strictly lower one. Each step records the best state and energy so
    far and its temperature into the enclosing ``track`` run, and an array
    state its bytes once at the end.
    """
    state, energy, carried, spent = start
    best, best_energy = keep(state), energy
    energies = [energy]
    tracked = current()
    for index in range(schedule.n_steps):
        temperature = schedule(index)
        moved = step(state, energy, carried, temperature, rng)
        state, energy, carried = moved.state, moved.energy, moved.carried
        spent += moved.spent
        energies.append(energy)
        if energy < best_energy:
            best, best_energy = keep(state), energy
        tracked.record(index, state=best, temperature=temperature, energy=best_energy)
    if hasattr(state, "nbytes"):
        tracked.record_cost(max(schedule.n_steps - 1, 0), int(state.nbytes))
    return Walked(
        best=best,
        energy=best_energy,
        final=state,
        energies=tuple(energies),
        spent=spent,
        termination=Termination.after(schedule.n_steps, converged=False),
    )


def anneal_spent[S, C, R](
    step: Step[S, C, R],
    schedule: TempSchedule,
    start: Moved[S, C],
    rng: R,
    keep: Callable[[S], S],
    *,
    budget: int,
) -> Walked[S]:
    """:func:`anneal` until the steps have spent ``budget``, the schedule read at the spent fraction (issue #1344).

    A step whose cost is not known before it runs --- a Wolff step, charged
    by its cluster --- cannot be given a step count that spends a budget.
    Each step here runs at ``schedule(min(n - 1, spent * n // budget))``,
    ``spent`` what the steps before it cost and ``n`` the schedule's length,
    and the run stops at the first step that brings ``spent`` to
    ``budget`` or past it, so it overspends by less than its last step. A
    step of fixed cost ``c`` with ``budget = n * c`` runs step ``k`` at
    index ``k * c * n // (n * c) = k``: :func:`anneal`'s run, bitwise. The
    last temperature runs once ``spent`` reaches ``(n - 1) / n`` of the
    budget, which every run reaches whose steps each cost at most
    ``budget / n``. The termination counts the steps run; the start's
    charge rides on ``spent`` and does not count against ``budget``.

    Raises
    ------
    ValueError
        If ``budget < 1``, or a step charges nothing: such a run never ends.
    """
    if budget < 1:
        msg = f"a budget is at least 1, got {budget}"
        raise ValueError(msg)
    state, energy, carried, charged = start
    best, best_energy = keep(state), energy
    energies = [energy]
    tracked = current()
    n_steps, spent, index = schedule.n_steps, 0, 0
    while spent < budget:
        temperature = schedule(min(n_steps - 1, spent * n_steps // budget))
        moved = step(state, energy, carried, temperature, rng)
        if moved.spent < 1:
            msg = "a step charged nothing, so a budget in its unit is never spent"
            raise ValueError(msg)
        state, energy, carried = moved.state, moved.energy, moved.carried
        spent += moved.spent
        energies.append(energy)
        if energy < best_energy:
            best, best_energy = keep(state), energy
        tracked.record(index, state=best, temperature=temperature, energy=best_energy)
        index += 1
    if hasattr(state, "nbytes"):
        tracked.record_cost(max(index - 1, 0), int(state.nbytes))
    return Walked(
        best=best,
        energy=best_energy,
        final=state,
        energies=tuple(energies),
        spent=charged + spent,
        termination=Termination.after(index, converged=False),
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
class Exchanging[S]:
    """The tempering loop's state, which a hook reads every round and :func:`temper` returns.

    ``states`` and ``energies`` are per rung; ``best`` is the lowest-energy
    state seen at any rung after a round's exchanges, a start included;
    ``proposed`` and ``accepted`` count exchanges per adjacent pair;
    ``trace`` holds each walker's rung per recorded round; ``spent`` sums
    the starts' and the steps' charges over the ``rounds`` run.
    """

    states: list[S]
    energies: list[float]
    best: S
    energy: float
    proposed: np.ndarray
    accepted: np.ndarray
    spent: int
    trace: list[list[int]]
    rounds: int = 0

    @property
    def swap_acceptance(self) -> np.ndarray:
        """Accepted over proposed exchanges per adjacent pair."""
        return np.asarray(self.accepted / self.proposed)

    @property
    def walkers(self) -> np.ndarray:
        """The trace as an array, ``(n_recorded, n_replicas)``."""
        return np.array(self.trace, dtype=np.int64).reshape(-1, len(self.states))


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
    keep: Callable[[S], S],
    record: Callable[[Sequence[S], Sequence[float]], None] | None = None,
    observe: Callable[[int, Exchanging[S]], None] | None = None,
    stop: Callable[[], bool] | None = None,
    ratio: Callable[[float, float, float, float], float] = swap_log_ratio,
    score: Callable[[Sequence[S]], list[float]] | None = None,
    between: Callable[[Exchanging[S]], None] | None = None,
) -> Exchanging[S]:
    """Replica exchange: one step per rung per round, then an exchange proposal on every adjacent pair.

    Rung ``r`` moves its state by ``steps[r]`` at ``temperatures[r]`` on
    ``children[r]``: moves belong to rungs and states to walkers. Each pair
    proposes in ladder order and accepts on ``swap(ratio(beta_i, beta_j,
    E_i, E_j))``: the draw is the caller's and the test is here, so a NumPy
    and a torch stream run the same loop (issue #861). ``ratio`` is
    :func:`swap_log_ratio` unless a caller reads it from its own module,
    where a test replaces it.

    ``score``, where given, scores every rung in one call after the round's
    moves and replaces the steps' energies: one block of Potts labellings
    costs 57 ms over 100 rounds of six rungs at ``spatio_tiling/release``
    where a call per rung costs 92 ms (issue #1218). ``between``, where
    given, may then change any rung's state, energy and ``spent`` before the
    exchanges, for a move across rungs (Houdayer's).

    Rounds ``burn_in`` onward are recorded at every ``thin``-th, the first
    included, through ``record(states, energies)``. ``observe`` is called
    after every round. ``stop`` is asked before every round after the first,
    for a wall clock: it must read no replica's state (``sample/CLAUDE.md``).
    """
    n = len(temperatures)
    betas = [1.0 / temperature for temperature in temperatures]
    states = [start.state for start in starts]
    energies = [start.energy for start in starts]
    carried = [start.carried for start in starts]
    lowest = min(range(n), key=energies.__getitem__)
    best = keep(states[lowest])
    run = Exchanging(
        states,
        energies,
        best,
        energies[lowest],
        np.zeros(n - 1),
        np.zeros(n - 1),
        sum(start.spent for start in starts),
        [],
    )
    # Which walker sits at each rung: what says a *state* crossed the ladder.
    at_rung = list(range(n))
    for sweep in range(burn_in + n_sweeps):
        if stop is not None and sweep > 0 and stop():
            break
        run.rounds += 1
        for r in range(n):
            moved = steps[r](
                states[r], energies[r], carried[r], temperatures[r], children[r]
            )
            states[r], energies[r], carried[r] = (
                moved.state,
                moved.energy,
                moved.carried,
            )
            run.spent += moved.spent
        if score is not None:
            energies[:] = score(states)
        if between is not None:
            between(run)
        for i in range(n - 1):
            run.proposed[i] += 1
            if swap(ratio(betas[i], betas[i + 1], energies[i], energies[i + 1])):
                run.accepted[i] += 1
                states[i], states[i + 1] = states[i + 1], states[i]
                energies[i], energies[i + 1] = energies[i + 1], energies[i]
                carried[i], carried[i + 1] = carried[i + 1], carried[i]
                at_rung[i], at_rung[i + 1] = at_rung[i + 1], at_rung[i]
        lowest = min(range(n), key=energies.__getitem__)
        if energies[lowest] < run.energy:
            run.best, run.energy = keep(states[lowest]), energies[lowest]
        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            if record is not None:
                record(states, energies)
            run.trace.append([at_rung.index(walker) for walker in range(n)])
        if observe is not None:
            observe(sweep, run)
    return run
