"""Slice sampling over an :class:`~sal.opt.objective.Objective`.

The tuning-free baseline. :func:`~sal.sample.hmc.sample` needs a
step size and a trajectory length, :func:`~sal.sample.langevin.mala`
a step size, and both are wrong silently when they are wrong; slice sampling
(Neal, 2003) needs an initial width ``w`` and adapts to the target within a
single update by stepping out and shrinking. A width off by an order of
magnitude costs evaluations and does not cost correctness, which is what makes
it the baseline a result is read against rather than a competitor for the
fastest number.

**Its cost unit is objective evaluations, not gradients.** It takes none, so
an effective sample per gradient is undefined for it and a comparison against
HMC has to be stated in the unit both spend --- `opt/CLAUDE.md`'s budgets are
counted in evaluations for exactly this reason. Every call this module makes
to the objective is counted and reported on the result, so the number is what
the objective saw and not what the procedure claims.

**An acceptance rate is not reported, because here it is 1 by construction.**
Every point the shrinkage returns lies in the slice, so a rate would read 1.00
on a chain that is mixing and 1.00 on one that is not. What is reported
instead is the evaluations a draw cost and the expansions the stepping-out
spent: a width far below the target's scale shows as expansions, and one far
above it as shrinkages.

**Shrinkage is refused rather than clamped.** The interval always contains the
current point, so the loop terminates in exact arithmetic; in floating point
an interval can collapse onto a point whose density is below the slice, and
that is a defect --- a non-finite objective, or a density that is not the one
the caller thinks --- rather than something to return a value for. It raises.

See Neal (2003), "Slice sampling", §3.1 for the univariate stepping-out and
shrinkage procedures and §4.1 for the multivariate sweeps this composes them
into.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

import torch

from sal.backend import Backend, refuse_backend
from sal.opt.objective import Objective
from sal.sample.chain import Adaptation
from sal.track import TrackedOptimization, current

#: Shrinkages one update may spend before it is refused. The interval halves
#: on average per shrinkage, so 100 is about 30 orders of magnitude of
#: interval width --- far past any width a caller could have meant and far
#: past `float64`'s ability to shrink further. Reaching it means the density
#: is not the one the slice was cut from.
MAX_SHRINKAGES = 100


class SliceDirection(Enum):
    """The line a univariate update runs along.

    ``COORDINATE`` updates one axis at a time (Neal, 2003, §4.1's first
    scheme), which mixes badly on a correlated target and is the cheapest
    thing that works. ``RANDOM`` draws a direction uniformly on the sphere
    --- hit-and-run --- which costs the same per update and does not inherit
    the coordinate system the objective happened to be written in.
    """

    COORDINATE = "coordinate"
    RANDOM = "random"


@dataclass(frozen=True)
class SliceUpdate:
    """One univariate update along one line, and what it spent.

    Parameters
    ----------
    theta : torch.Tensor
        The point the update returned, which lies in the slice:
        ``-objective(theta) >= log_slice``. That is an exact statement about
        the procedure and is what pins it.
    log_slice : float
        ``log y``, the level the slice was cut at: ``-objective(theta_0)``
        plus the log of a uniform.
    evaluations : int
        Objective evaluations the update spent, the current point's included.
    expansions : int
        Stepping-out expansions, on both ends. Reaching ``max_steps_out``
        means the width is far below the target's scale.
    shrinkages : int
        Candidates rejected before one landed in the slice. A width far above
        the target's scale shows here.
    """

    theta: torch.Tensor
    log_slice: float
    evaluations: int
    expansions: int
    shrinkages: int


@dataclass(frozen=True)
class SliceChain:
    """A slice-sampled chain, and what it cost.

    Parameters
    ----------
    draws : torch.Tensor
        Draws in unconstrained coordinates, shape ``(n_samples, dimension)``.
        One draw is one sweep, not one univariate update.
    objective_evaluations : int
        Evaluations spent, burn-in included, so an effective sample size
        divided by it is the cost of a draw in the unit this sampler spends.
    evaluations_per_draw : float
        The same number over the sweeps that produced it, which is what a
        width is diagnosed by.
    expansions_per_draw : float
        Stepping-out expansions a sweep spent, over the whole run.
    shrinkages_per_draw : float
        Shrinkages a sweep spent, over the whole run.
    """

    draws: torch.Tensor
    objective_evaluations: int
    evaluations_per_draw: float
    expansions_per_draw: float
    shrinkages_per_draw: float


def slice_sample(
    objective: Objective,
    generator: torch.Generator,
    n_samples: int,
    *,
    width: float,
    max_steps_out: int,
    direction: SliceDirection = SliceDirection.COORDINATE,
    theta0: torch.Tensor | None = None,
    burn_in: int = 0,
    shrink: bool = True,
    temperature: float = 1.0,
    adaptation: Adaptation | None = None,
    store_chain: bool = True,
    backend: Backend = Backend.PYTHON,
) -> SliceChain:
    """Draw ``n_samples`` sweeps from the density ``exp(-objective / temperature)``.

    A sweep is ``dimension`` univariate updates: along each axis in turn under
    ``COORDINATE``, along that many independently drawn directions under
    ``RANDOM``. Equal counts either way, so the two are comparable at equal
    draws as well as at equal evaluations.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density, as
        :func:`~sal.sample.hmc.sample` reads it.
    generator : torch.Generator
        The stream every uniform, direction and slice level comes from.
    n_samples : int
        Sweeps recorded after burn-in.
    width : float
        The initial interval width ``w``, in the coordinates of ``theta``.
        The one number to set, and the only one: too small buys expansions,
        too large buys shrinkages, neither buys a bias.
    max_steps_out : int
        Neal's ``m``: expansions one update may spend, split at random
        between the two ends so the interval construction stays reversible.
        At least 1.
    direction : SliceDirection
        The lines a sweep runs along.
    theta0 : torch.Tensor | None
        Starting point; ``objective.initial()`` when omitted.
    burn_in : int
        Sweeps discarded before recording.
    shrink : bool
        The shrinkage of Neal's §3.1, on by default. False rejects candidates
        from the interval the stepping-out left, without narrowing it: the
        same stationary distribution at an unbounded cost, which is the
        ablation that says what the shrinkage buys. It is refused past
        :data:`MAX_SHRINKAGES` candidates like any other update.
    temperature : float
        The chain targets ``exp(-objective / temperature)``; the level and
        each comparison divide the energy by it, so at 1 the chain is the
        untempered one bitwise.
    adaptation : Adaptation | None
        Refused unless ``None``: the width is adapted within each update by
        stepping out and shrinking, so a warm-up has no step or metric to
        set. Taken so the samplers share one entry shape (issue #1059).
    store_chain : bool
        Keep the draws. ``False`` keeps none --- ``draws`` has zero rows ---
        and the counters alone are reported.
    backend : Backend
        :data:`~sal.backend.Backend.PYTHON` alone: the torch loop has no
        compiled twin.

    Returns
    -------
    SliceChain

    Raises
    ------
    ValueError
        If ``width`` or ``temperature`` is not positive, ``max_steps_out`` is
        below 1, an ``adaptation`` or a backend other than ``PYTHON`` is
        given, or an update exhausts :data:`MAX_SHRINKAGES`.
    """
    refuse_backend("slice_sample", backend, (Backend.PYTHON,))
    if adaptation is not None:
        msg = (
            "slice_sample takes no adaptation: stepping out and shrinking set "
            "the interval within each update, so a warm-up has nothing to set"
        )
        raise ValueError(msg)
    if not temperature > 0.0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    if width <= 0.0:
        msg = f"width must be positive, got {width}"
        raise ValueError(msg)
    if max_steps_out < 1:
        msg = (
            f"max_steps_out must be at least 1, got {max_steps_out}: an "
            "interval that cannot step out is the initial width, and a slice "
            "wider than it is never reached"
        )
        raise ValueError(msg)

    position = (
        objective.initial().detach().clone()
        if theta0 is None
        else theta0.detach().clone()
    ).to(torch.float64)
    dimension = int(position.shape[0])

    draws = torch.empty(
        (n_samples if store_chain else 0, dimension), dtype=torch.float64
    )
    evaluations = 0
    expansions = 0
    shrinkages = 0
    # One lookup for the chain (`sal.track`), one record per
    # recorded draw: the evaluations spent so far, the unit this sampler is
    # counted in, and the wall beside it (issue #799).
    tracked: TrackedOptimization = current()
    started = time.perf_counter()
    for index in range(n_samples + burn_in):
        for axis in range(dimension):
            update = slice_update(
                objective,
                position,
                _direction(direction, axis, dimension, generator),
                generator,
                width=width,
                max_steps_out=max_steps_out,
                shrink=shrink,
                temperature=temperature,
            )
            position = update.theta
            evaluations += update.evaluations
            expansions += update.expansions
            shrinkages += update.shrinkages
        if index >= burn_in:
            if store_chain:
                draws[index - burn_in] = position
            tracked.record(
                index - burn_in,
                state=position,
                objective_evaluations=float(evaluations),
                wall_s=time.perf_counter() - started,
            )
    tracked.record_cost(max(n_samples - 1, 0), draws.nbytes)

    sweeps = max(n_samples + burn_in, 1)
    return SliceChain(
        draws=draws,
        objective_evaluations=evaluations,
        evaluations_per_draw=evaluations / sweeps,
        expansions_per_draw=expansions / sweeps,
        shrinkages_per_draw=shrinkages / sweeps,
    )


def slice_update(
    objective: Objective,
    position: torch.Tensor,
    direction: torch.Tensor,
    generator: torch.Generator,
    *,
    width: float,
    max_steps_out: int,
    shrink: bool = True,
    temperature: float = 1.0,
) -> SliceUpdate:
    """Neal's (2003, §3.1) stepping-out and shrinkage along one line.

    Public because the property that defines it --- the point returned lies in
    the slice the level was cut at --- is exact, and a test asserting it needs
    the level. A chain cannot report one level a draw: a sweep cuts
    ``dimension`` of them.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density.
    position : torch.Tensor
        The current point, 1-D.
    direction : torch.Tensor
        The line's direction, same shape as ``position``. Unit norm under
        :func:`slice_sample`; any norm scales ``width`` with it.
    generator : torch.Generator
        The stream the level, the interval offset and the candidates come
        from.
    width, max_steps_out, shrink, temperature
        As :func:`slice_sample`.

    Returns
    -------
    SliceUpdate

    Raises
    ------
    ValueError
        If the objective is not finite at ``position`` --- a slice has no
        level to cut then, and `opt/CLAUDE.md` refuses rather than clamps ---
        or if the update spends :data:`MAX_SHRINKAGES` candidates.
    """
    current = float(objective(position.detach()))
    evaluations = 1
    if not math.isfinite(current):
        msg = (
            f"objective is {current} at the current point, so no slice level "
            "exists; a chain cannot be started where the density is not finite"
        )
        raise ValueError(msg)
    # log y = log(f(x) * u) with u uniform on (0, 1), written as a log so a
    # density below `float64`'s smallest normal still cuts a level.
    log_slice = -current / temperature + math.log(
        float(torch.rand(1, generator=generator))
    )

    # Neal's Figure 3: the interval is placed uniformly around the current
    # point and its expansions are split at random between the ends, which is
    # what makes the construction the same for the reverse move.
    lower = -width * float(torch.rand(1, generator=generator))
    upper = lower + width
    left_steps = int(max_steps_out * float(torch.rand(1, generator=generator)))
    right_steps = max_steps_out - 1 - left_steps

    expansions = 0
    while left_steps > 0:
        evaluations += 1
        if -float(objective(position + lower * direction)) / temperature <= log_slice:
            break
        lower -= width
        left_steps -= 1
        expansions += 1
    while right_steps > 0:
        evaluations += 1
        if -float(objective(position + upper * direction)) / temperature <= log_slice:
            break
        upper += width
        right_steps -= 1
        expansions += 1

    shrinkages = 0
    while shrinkages < MAX_SHRINKAGES:
        offset = lower + (upper - lower) * float(torch.rand(1, generator=generator))
        candidate = position + offset * direction
        evaluations += 1
        if -float(objective(candidate)) / temperature >= log_slice:
            return SliceUpdate(
                theta=candidate,
                log_slice=log_slice,
                evaluations=evaluations,
                expansions=expansions,
                shrinkages=shrinkages,
            )
        shrinkages += 1
        if shrink:
            if offset < 0.0:
                lower = offset
            else:
                upper = offset

    msg = (
        f"slice update spent {MAX_SHRINKAGES} candidates on the interval "
        f"[{lower}, {upper}] at level {log_slice} without one in the slice"
        + (
            "; the interval contains the current point, so in exact arithmetic "
            "this cannot happen and the objective is not the density the level "
            "was cut from"
            if shrink
            else "; shrinkage is off, so the interval never narrowed"
        )
    )
    raise ValueError(msg)


def _direction(
    kind: SliceDirection, axis: int, dimension: int, generator: torch.Generator
) -> torch.Tensor:
    """The unit vector one univariate update runs along."""
    if kind is SliceDirection.COORDINATE:
        line = torch.zeros(dimension, dtype=torch.float64)
        line[axis] = 1.0
        return line
    drawn = torch.randn(dimension, generator=generator, dtype=torch.float64)
    unit: torch.Tensor = drawn / drawn.norm()
    return unit
