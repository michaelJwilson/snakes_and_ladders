"""Sampler starts of an HMM, polished by Baum-Welch on the `opt.starts` seam (issue #1172).

:class:`~sal.opt.starts.StartsBenchmark` runs a start, scores what it offers
and hands it to a polisher, over any :class:`~sal.opt.objective.Objective`.
An HMM over any emission family is one:
:class:`~sal.opt.hmm.EmissionHmmObjective` (#1169), whose ``theta`` carries
the initial distribution, the transitions and the emissions, and which
:func:`~sal.opt.starts.polish_by_baum_welch` reads back into a Baum-Welch
fit. So the HMM's starts are entries of :data:`STARTS` --- factories of an
:class:`~sal.opt.initialize.Initializer` from the cell's generator --- and
nothing else: no instance type, no seeding type and no polish of their own,
which is what ties `search.mixture_starts` to the mixture.

**The three samplers run on the objective itself.**
:class:`~sal.sample.initialize.FromChain`,
:class:`~sal.sample.initialize.FromAnnealing` and
:class:`~sal.sample.initialize.FromTempering` take any ``Objective`` and start
where :meth:`~sal.opt.hmm.EmissionHmmObjective.initial` is.
:class:`SampledStart` wraps each so the gradients it spent reach the cell's
``track`` run, where :attr:`~sal.opt.starts.StartTrial.diagnostics` reads
them: a start is compared with a restart at equal evaluations, so what it
spent before the hand-over is a column, not a footnote.

**The objective's own start is the caller's.** ``"quantile"`` is
:class:`~sal.opt.initialize.FromObjective`: the uniform chain at the family
the objective was built with, which is a quantile start when the caller
placed its locations by :func:`~sal.opt.initialize.quantile_locations`
(:func:`gaussian_quantile_start` for a Gaussian family). The samplers and the
restarts start from the same point.

**The step is a grid's, not a warm-up's (issue #1195).** At #1172's step of
3e-2 the chain, annealing and tempering starts reached the truth's basin in
1, 0 and 4 of 10 seeds. At every step from 1e-3 to 3e-2 the median acceptance is at least 0.96
of proposals, so what a larger step costs is distance, not rejection: it
carries the start from the quantile point into another basin. The chain's
own warm-up (:class:`~sal.sample.hmc.Adaptation`, 16 to 48 proposals,
target 0.65 or 0.8) drives the step toward the acceptance target and reaches
the basin in at most 4 of 10 seeds on either instance below, so it is not
used. The steps are instead a grid's, 1e-3 to 3e-2 at 300 passes on two
instances (``test_hmm_starts``' four states and five states), each the
largest step whose median polished gap is within 1 nat of the grid's best on
both: :data:`STEP` 5e-3 for the chain and the ladder, :data:`ANNEALING_STEP`
1e-3. Both were fit at 669 positions; another data size is unmeasured.

**The step is not scaled with temperature.** The fixed step at which a
chain's acceptance falls to 0.65 is 0.063, 0.063, 0.057 and 0.046 at
``T`` = 1, 4, 16 and 64 on four states, 0.074 to 0.041 on five: flat to
falling, where a step scaled by sqrt(T) would rise eightfold. One step serves
every rung, as the momentum rescaling of :mod:`sal.sample.hmc` states.

**Tuned, a sampler start reaches the basin by staying in it.** Its median
start is 2 to 51 nats below the quantile point, against 192 to 305 at 3e-2.
At equal passes every sampler beats restarts (7 of 10 on four states, 0 of 10
on five) and none beats the quantile start (10 of 10 on both): they tie it
on four states and reach 7 or 8 of 10 on five, polishing for 174 to 179
iterations against its 299.

**A unit of every budget here is one pass over the data.** A gradient is
one compiled E step and one backward pass
(:meth:`~sal.opt.hmm.EmissionHmmObjective.value_and_gradient`), a Baum-Welch
iteration one E step and one M step, and a scored start one forward
recursion; :data:`CHARGES` states each start's passes before the polish, so a
comparison at equal evaluations gives each start its polish as the
remainder.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from sal.cost import Cost
from sal.emissions import GaussianEmission
from sal.opt.budget import Budget
from sal.opt.initialize import FromObjective, Initializer, RandomRestart
from sal.opt.initialize import quantile_locations as _quantile_locations
from sal.opt.objective import Objective
from sal.opt.starts import SolverComparison, StartsBenchmark, polish_by_baum_welch
from sal.sample.chain import torch_stream
from sal.sample.initialize import FromAnnealing, FromChain, FromTempering
from sal.sample.schedule import ExponentialTempSchedule
from sal.track import current

#: Leapfrog step of the chain and tempering starts, in unconstrained
#: coordinates, chosen by the grid of #1195 (module note).
STEP = 5.0e-3

#: Leapfrog step of the annealing start, by the same grid.
ANNEALING_STEP = 1.0e-3

#: Leapfrog steps per proposal: a sampler start is charged every gradient.
TRAJECTORY = 4

#: Proposals the chain runs before its one kept draw; no warm-up, since
#: :data:`~sal.sample.initialize.CHAIN_ADAPTATION` alone is 300 proposals.
CHAIN_BURN_IN = 24

#: Proposals the annealing run makes, from the hottest rung to 1.
ANNEAL_STEPS = 24

#: The tempering ladder, coldest first. The hottest rung flattens a barrier
#: of 64 nats to one; on ``test_hmm_starts``' instance Baum-Welch's other
#: basins end 7.5 to 43 nats above the truth's.
TEMPERATURES = (1.0, 4.0, 16.0, 64.0)

#: Rounds of the ladder: ``TEMPERING_ROUNDS * len(TEMPERATURES)`` proposals,
#: the annealing run's count.
TEMPERING_ROUNDS = 6

#: Points :class:`~sal.opt.initialize.RandomRestart` draws, the nominal one
#: excluded, and their spread in unconstrained coordinates.
RESTARTS = 4
RESTART_SCALE = 1.0


@dataclass(frozen=True)
class SampledStart:
    """A sampler's one start, with the gradients it spent recorded into the cell's run.

    Records ``gradients`` and ``acceptance`` (the cold replica's, for a
    ladder) at step 0 of the enclosing ``track`` run, as
    :class:`~sal.search.potts_starts.SolverStart` records its site visits.

    Parameters
    ----------
    sampler : FromChain | FromAnnealing | FromTempering
        The start. A chain offers its last draw: one kept draw is asked of
        it, so the seam polishes one point.
    """

    sampler: FromChain | FromAnnealing | FromTempering

    def starts(self, objective: Objective) -> list[torch.Tensor]:
        """The sampler's point: the chain's last draw, or the best point visited.

        Returns
        -------
        list[torch.Tensor]
            Exactly one start.
        """
        sampler = self.sampler
        tracked = current()
        if isinstance(sampler, FromChain):
            chain = sampler.chain(objective)
            tracked.record(
                0, gradients=float(chain.spent), acceptance=chain.acceptance_rate
            )
            return [chain.draws[-1]]
        if isinstance(sampler, FromAnnealing):
            annealed = sampler.run(objective)
            tracked.record(
                0,
                gradients=float(annealed.spent),
                acceptance=annealed.acceptance_rate,
            )
            return [annealed.best]
        tempered = sampler.run(objective)
        tracked.record(
            0,
            gradients=float(tempered.spent),
            acceptance=float(tempered.acceptance_rate[0]),
        )
        return [tempered.best]


def chain_start(rng: np.random.Generator) -> SampledStart:
    """``FromChain``: :data:`CHAIN_BURN_IN` proposals from the objective's start; its one draw seeds.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromChain(
            1,
            STEP,
            torch_stream(rng),
            n_steps=TRAJECTORY,
            burn_in=CHAIN_BURN_IN,
            adaptation=None,
        )
    )


def annealed_start(rng: np.random.Generator) -> SampledStart:
    """``FromAnnealing``: the best point of :data:`ANNEAL_STEPS` proposals cooling from the hottest rung to 1.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromAnnealing(
            ExponentialTempSchedule(TEMPERATURES[-1], 1.0, ANNEAL_STEPS),
            ANNEALING_STEP,
            torch_stream(rng),
            n_steps=TRAJECTORY,
        )
    )


def tempered_start(rng: np.random.Generator) -> SampledStart:
    """``FromTempering``: the best point any rung of :data:`TEMPERATURES` visits in :data:`TEMPERING_ROUNDS` rounds.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromTempering(
            TEMPERATURES, TEMPERING_ROUNDS, STEP, torch_stream(rng), n_steps=TRAJECTORY
        )
    )


def restart_start(rng: np.random.Generator) -> RandomRestart:
    """``RandomRestart``: :data:`RESTARTS` points at :data:`RESTART_SCALE` around the objective's start, which is not one of them.

    Returns
    -------
    RandomRestart
    """
    return RandomRestart(RESTARTS, RESTART_SCALE, rng, include_nominal=False)


def quantile_start(_rng: np.random.Generator) -> FromObjective:
    """``FromObjective``: the objective's own start, a quantile start where the caller built it so.

    Returns
    -------
    FromObjective
    """
    return FromObjective()


#: Name to start, each a factory of an initializer from the cell's generator.
STARTS: dict[str, Callable[[np.random.Generator], Initializer]] = {
    "quantile": quantile_start,
    "restart": restart_start,
    "chain": chain_start,
    "annealed": annealed_start,
    "tempered": tempered_start,
}

#: Passes over the data each start spends before its polish: a sampler's
#: gradients, ``TRAJECTORY + 1`` per proposal as
#: :func:`~sal.sample.hmc.sample` and its siblings count them, and the points
#: the seam scores, one each.
CHARGES: dict[str, int] = {
    "quantile": 1,
    "restart": RESTARTS,
    "chain": (CHAIN_BURN_IN + 1) * (TRAJECTORY + 1) + 1,
    "annealed": ANNEAL_STEPS * (TRAJECTORY + 1) + 1,
    "tempered": TEMPERING_ROUNDS * len(TEMPERATURES) * (TRAJECTORY + 1) + 1,
}


def gaussian_quantile_start(
    values: np.ndarray | torch.Tensor, n_states: int, variance_floor: float
) -> GaussianEmission:
    """A Gaussian family at ``n_states`` evenly spaced quantiles of ``values``, each at the pooled standard deviation.

    Returns
    -------
    GaussianEmission
    """
    pooled = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    return GaussianEmission(
        _quantile_locations(pooled, n_states),
        torch.full((n_states,), float(pooled.std())),
        variance_floor,
    )


def at_equal_evaluations(
    objective: Objective | Sequence[Objective],
    names: Sequence[str],
    evaluations: int,
    seeds: Sequence[int],
    *,
    reference: Sequence[float] | None = None,
    workers: int = 1,
) -> dict[str, SolverComparison]:
    """Each named start of :data:`STARTS` and its Baum-Welch polish, at ``evaluations`` passes in all.

    One :class:`~sal.opt.starts.StartsBenchmark` per start, its polish budget
    ``evaluations - CHARGES[name]`` Baum-Welch iterations, so a start that
    spends gradients before the hand-over polishes for less; a restart's
    points share theirs (:mod:`sal.opt.starts`). The seam's seeding budget is
    :data:`RESTARTS`, the most points any entry offers.

    Returns
    -------
    dict[str, SolverComparison]
        Per name, in the order given.

    Raises
    ------
    ValueError
        If a start's charge leaves it no polish.
    """
    out: dict[str, SolverComparison] = {}
    for name in names:
        polish = evaluations - CHARGES[name]
        if polish < 1:
            msg = (
                f"{name!r} spends {CHARGES[name]} passes before its polish, "
                f"leaving none of {evaluations}"
            )
            raise ValueError(msg)
        out[name] = StartsBenchmark(
            objective,
            {name: STARTS[name]},
            polish_by_baum_welch,
            seeding_budget=Budget(Cost.EVALUATIONS, RESTARTS),
            polish_budget=Budget(Cost.ITERATIONS, polish),
            seeds=seeds,
            workers=workers,
            reference=reference,
        ).run()
    return out


__all__ = [
    "ANNEALING_STEP",
    "ANNEAL_STEPS",
    "CHAIN_BURN_IN",
    "CHARGES",
    "RESTARTS",
    "RESTART_SCALE",
    "STARTS",
    "STEP",
    "TEMPERATURES",
    "TEMPERING_ROUNDS",
    "TRAJECTORY",
    "SampledStart",
    "annealed_start",
    "at_equal_evaluations",
    "chain_start",
    "gaussian_quantile_start",
    "quantile_start",
    "restart_start",
    "tempered_start",
]
