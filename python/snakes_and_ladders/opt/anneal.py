"""One annealing driver and one exchange step, over any state a caller can move.

The 0.4.0 audit (``STATUS.md``) found eight entry points running the same two
loops. Four anneal --- one transition per schedule step, keep the best state
seen: :func:`snakes_and_ladders.search.potts_mcmc.anneal_potts` over spins,
:func:`snakes_and_ladders.search.gibbs.anneal_factor_graph` over any factor graph,
:func:`snakes_and_ladders.search.gibbs.anneal_topology` over trees, and
:func:`snakes_and_ladders.opt.hmc.anneal` over a real vector. Four exchange ---
propose to swap the configurations of adjacent replicas and accept on one
ratio: the two ``parallel_tempering`` functions, and the two ensembles of
:mod:`snakes_and_ladders.search.tempered`, which held a third copy of
``_swap_log_ratio``.

What varies between them is the state, the move and the temperature; what
does not is the loop. So the loop is here and the moves stay with their
models: :func:`drive` takes a ``transition`` closure and never learns what it
moves, and :func:`exchange` takes a ``swap`` closure and never learns what it
swaps.

**Home.** ``opt/`` beside ``opt.schedule``, which the same eight already
share, and for the same reason: ``opt`` may import no application module, so
this is the one placement ``search``, ``learn`` and ``opt`` can all reach.

**No generator is passed.** The plan on issue #386 had one; the four
annealers do not agree on what a generator is --- three draw from
:class:`numpy.random.Generator` and one from :class:`torch.Generator` --- and
a driver that took either would have to name both libraries to do nothing
with them. The transition closes over its own source of randomness instead,
which is also what keeps the driver from having any opinion about draw
order.

**A trajectory is always kept.** Two of the four annealers recorded one and
two did not; recording ``n_steps + 1`` floats costs nothing next to the
transitions, and an adapter that does not want it drops it. The alternative
--- a flag --- is a second code path through the one loop the seam exists to
have only once.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

import numpy as np

from snakes_and_ladders.opt.schedule import Schedule

#: The state being annealed. Spins, a factor graph's labelling, a topology or
#: a position vector; the driver only ever hands it back to the transition.
S = TypeVar("S")


class Extremum(StrEnum):
    """Which end of the value axis is the good one.

    Named rather than a ``bool`` because the four annealers split on it ---
    two minimize an energy and two maximize a log-density --- and a bare
    ``True`` at the call site would say nothing about which.
    """

    MINIMUM = "minimum"
    MAXIMUM = "maximum"


@dataclass(frozen=True)
class Annealing(Generic[S]):
    """What one annealing run found, and what it cost.

    Parameters
    ----------
    best : S
        The best state visited, by ``keep``.
    best_value : float
        Its value.
    final : S
        The state the chain ended in, which is not generally the best one:
        the schedule's last steps are still a sampler at its final
        temperature, not a descent.
    trajectory : np.ndarray
        The value after each step, with the starting value first, shape
        ``(n_steps + 1,)``.
    accepted : int
        Proposals accepted, where the transition proposes; ``0`` for a
        transition that always moves, such as a heat-bath sweep.
    """

    best: S
    best_value: float
    final: S
    trajectory: np.ndarray
    accepted: int


def drive(
    start: S,
    value: float,
    transition: Callable[[S, float, float], tuple[S, float, bool]],
    schedule: Schedule,
    *,
    keep: Extremum,
    copy: Callable[[S], S] = lambda state: state,
) -> Annealing[S]:
    """One transition per schedule step, keeping the best state seen.

    Consumers: :func:`snakes_and_ladders.search.potts_mcmc.anneal_potts`,
    :func:`snakes_and_ladders.search.gibbs.anneal_factor_graph`,
    :func:`snakes_and_ladders.search.gibbs.anneal_topology` and
    :func:`snakes_and_ladders.opt.hmc.anneal` --- four copies of this loop before
    issue #386, over spins, a factor graph's labelling, a topology and a real
    vector respectively.

    Parameters
    ----------
    start : S
        The starting state. Drawn by the caller, because who draws the start
        is part of a run's reproducibility and the driver has no generator.
    value : float
        Its value, so the driver never evaluates anything itself. A caller
        that has the number already does not pay for it twice, and one whose
        evaluation is expensive --- a fitted topology --- decides when it
        happens.
    transition : Callable[[S, float, float], tuple[S, float, bool]]
        ``transition(state, value, temperature)`` returns the state after one
        step, its value, and whether a proposal was accepted. It may return
        the state it was given, mutated in place; that is why ``copy``
        exists. The current value is passed as well as the state because a
        Metropolis step needs it to score a proposal, and re-deriving it
        would evaluate a fitted topology twice per step --- the whole cost of
        a walk over trees.
    schedule : Schedule
        Temperature per step. Its length is the budget.
    keep : Extremum
        Whether low or high values are the good ones.
    copy : Callable[[S], S]
        How to take a snapshot of a state worth keeping. The default is
        identity, correct for an immutable state; a transition that mutates
        in place must pass ``np.ndarray.copy``, ``torch.Tensor.clone`` or its
        own equivalent, or ``best`` will alias ``final`` and report the last
        state as the best one.

    Returns
    -------
    Annealing[S]
    """
    state = start
    best, best_value = copy(state), value
    trajectory = [value]
    accepted = 0
    for step in range(schedule.n_steps):
        state, value, was_accepted = transition(state, value, schedule(step))
        accepted += was_accepted
        trajectory.append(value)
        # Strict, so the first state to reach a value is the one kept. Every
        # adapter's committed result was taken under that rule.
        improved = (
            value < best_value if keep is Extremum.MINIMUM else value > best_value
        )
        if improved:
            best, best_value = copy(state), value
    return Annealing(
        best=best,
        best_value=best_value,
        final=state,
        trajectory=np.array(trajectory),
        accepted=accepted,
    )


def swap_log_ratio(
    beta_first: float, beta_second: float, energy_first: float, energy_second: float
) -> float:
    """Log acceptance of exchanging the configurations of two replicas.

    The joint target is the product of the tempered marginals, so the ratio
    is ``(beta_i - beta_j)(E_i - E_j)``: an exchange that hands the colder
    replica the lower energy is always accepted. A version that omits this
    term still runs, still mixes, and converges to the wrong distribution;
    ``tests/regression/search/test_potts_mcmc.py`` replaces it with that
    version and asserts the chi-square catches it.

    Consumers: :func:`exchange`, and through it all four tempering entry
    points. It was written three times before issue #386 --- in
    :mod:`snakes_and_ladders.search.potts_mcmc`, in :mod:`snakes_and_ladders.opt.hmc` in
    temperatures rather than inverse temperatures, and imported privately
    across module lines by :mod:`snakes_and_ladders.search.tempered`, which negated a
    log-density to get an energy at the call site.

    Parameters
    ----------
    beta_first, beta_second : float
        Inverse temperatures of the two replicas.
    energy_first, energy_second : float
        Their energies. A caller holding a log-density passes its negation.

    Returns
    -------
    float
    """
    return (beta_first - beta_second) * (energy_first - energy_second)


def exchange(
    energies: Sequence[float] | np.ndarray,
    betas: Sequence[float],
    draw: Callable[[], float],
    swap: Callable[[int, int], None],
    *,
    draw_first: bool = False,
) -> np.ndarray:
    """One pass of exchange proposals over adjacent replicas.

    Pair ``(r, r + 1)`` is proposed for every ``r`` in order, and an accepted
    exchange is applied before the next pair is scored --- so a configuration
    can travel more than one rung in a pass, which is what all four entry
    points already did and what makes the ladder mix rather than shuffle.

    Consumers: :func:`snakes_and_ladders.search.potts_mcmc.parallel_tempering`,
    :func:`snakes_and_ladders.opt.hmc.parallel_tempering` and, through
    ``snakes_and_ladders.search.tempered._exchange``,
    :func:`snakes_and_ladders.search.tempered.tempered_factor_graph` and
    :func:`snakes_and_ladders.search.tempered.tempered_topologies`.

    Parameters
    ----------
    energies : Sequence[float] | np.ndarray
        One energy per replica, in ladder order. Read, never written: the
        function keeps its own copy and swaps that, so a caller's container
        stays whatever type it is.
    betas : Sequence[float]
        Inverse temperature per replica, in the same order. The order fixes
        which pairs are adjacent; it need not be sorted.
    draw : Callable[[], float]
        A uniform on ``[0, 1)``. A callable rather than a generator for the
        reason the module docstring gives: two of the four consumers draw
        from ``torch`` and two from ``numpy``.
    swap : Callable[[int, int], None]
        Exchange the caller's state and value at two replica indices. Called
        only for an accepted pair, in pair order.
    draw_first : bool
        Whether the uniform is drawn before the ratio is tested, or only
        when the test needs it. This is a difference in the *stream*, not in
        the distribution, and it is the whole reason the flag exists:
        :func:`snakes_and_ladders.opt.hmc.parallel_tempering` draws unconditionally and
        the other three short-circuit, so one setting would move a committed
        chain of one or of three. ``False`` is the majority and the default.

    Returns
    -------
    np.ndarray
        ``1.0`` where the pair's exchange was accepted, shape
        ``(n_replicas - 1,)``.
    """
    energy = [float(value) for value in energies]
    accepted = np.zeros(len(betas) - 1)
    for pair in range(len(betas) - 1):
        log_ratio = swap_log_ratio(
            betas[pair], betas[pair + 1], energy[pair], energy[pair + 1]
        )
        uniform = draw() if draw_first else None
        if log_ratio >= 0.0 or (uniform if uniform is not None else draw()) < math.exp(
            log_ratio
        ):
            accepted[pair] = 1.0
            energy[pair], energy[pair + 1] = energy[pair + 1], energy[pair]
            swap(pair, pair + 1)
    return accepted
