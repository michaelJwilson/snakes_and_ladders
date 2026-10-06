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

**An adapted sampler does not replace the grid (issue #1208).** Each
sampler's own warm-up at equal passes ---
:data:`ADAPTATION` from :data:`ADAPTED_STEP`, #1172's untuned step: the
chain's 8 proposals before a burn-in of 16, annealing's step re-tuned every 8
proposals along its schedule, tempering's 8 per rung before 2 rounds ---
reaches the truth's basin in 3, 0 and 3 of 10 seeds on four states and 2, 0
and 3 on five, against the grid's 10, 10, 10 and 8, 8, 7. From the grid's own
step of 5e-3 it reaches 3, 0, 2 and 0, 1, 0, and at a target of 0.95 2, 1, 5
and 1, 0, 0. Dual averaging drives the step to the acceptance target, which
here is the distance that carries a start out of the basin, so :data:`STEP`
and :data:`ANNEALING_STEP` stay. The adapted entries are in :data:`STARTS` so
the release experiment holds these counts.

**A tuned step does not replace the grid either (issue #1219).** Each
sampler at ``step_size="auto"`` --- :data:`TUNING`'s pilot over
:data:`TUNING_GRID`, ranked by the lowest energy reached, then half the
fixed-step run --- reaches the truth's basin in 3, 3 and 5 of 10 seeds on
four states at 300 passes, against the grid's 10, 10, 10. The pilot chooses
3e-2 for the chain and the cold rung in 10 of 10 seeds and for annealing in
7: the largest step descends furthest in a short pilot, into another basin,
and the ESJD per gradient ranks it first too (3 and 2 of 10). Neither
criterion reads what the grid read, the polished gap, so :data:`STEP` and
:data:`ANNEALING_STEP` stay; the tuned entries are in :data:`STARTS` so the
release experiment holds these counts.

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
from sal.sample.chain import Adaptation, torch_stream
from sal.sample.initialize import FromAnnealing, FromChain, FromTempering
from sal.sample.schedule import ExponentialTempSchedule
from sal.sample.tune import AUTO, Criterion, StepTuning, TunedStep
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

#: The adapted samplers' warm-up (issue #1208): 8 proposals per window,
#: the least :class:`~sal.sample.hmc.Adaptation` takes, at
#: :data:`~sal.sample.initialize.CHAIN_ADAPTATION`'s target and jitter.
ADAPTATION = Adaptation(warmup=8, target_acceptance=0.65, step_jitter=0.4)

#: Where the adapted samplers' dual averaging starts: #1172's untuned step,
#: so no entry using it reads the grid.
ADAPTED_STEP = 3.0e-2

#: Rounds of the adapted ladder after its warm-up of
#: ``ADAPTATION.warmup`` proposals per rung.
ADAPTED_TEMPERING_ROUNDS = 2

#: The candidate steps of the tuned samplers (issue #1219): 1e-3 to 3e-2 at
#: half a decade, the range of #1195's grid.
TUNING_GRID = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2)

#: Proposals the chain's and the annealing run's pilots give each candidate
#: step, and the ladder's pilots each rung's: the least
#: :func:`~sal.sample.tune.tune_step` takes is two.
PILOT_PROPOSALS = 3
RUNG_PILOT_PROPOSALS = 2

#: The chain's and the annealing run's pilot: the start's gradient and
#: :data:`PILOT_PROPOSALS` per candidate, ranked by the lowest energy reached.
TUNING = StepTuning(
    Budget(Cost.GRADIENTS, 1 + len(TUNING_GRID) * PILOT_PROPOSALS * TRAJECTORY),
    Criterion.LOWEST_ENERGY,
    TUNING_GRID,
)

#: The ladder's pilots, split evenly over :data:`TEMPERATURES`.
TEMPERING_TUNING = StepTuning(
    Budget(
        Cost.GRADIENTS,
        len(TEMPERATURES) * (1 + len(TUNING_GRID) * RUNG_PILOT_PROPOSALS * TRAJECTORY),
    ),
    Criterion.LOWEST_ENERGY,
    TUNING_GRID,
)

#: The tuned runs after their pilots: half the fixed-step runs' proposals,
#: so the chain's and the annealing run's charges are within one pass of
#: their fixed-step twins'.
TUNED_BURN_IN = CHAIN_BURN_IN // 2
TUNED_ANNEAL_STEPS = ANNEAL_STEPS // 2
TUNED_TEMPERING_ROUNDS = 2

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
                0,
                gradients=float(chain.spent),
                acceptance=chain.acceptance_rate,
            )
            _record(chain.tuned)
            return [chain.draws[-1]]
        if isinstance(sampler, FromAnnealing):
            annealed = sampler.run(objective)
            tracked.record(
                0,
                gradients=float(annealed.spent),
                acceptance=annealed.acceptance_rate,
            )
            _record(annealed.tuned)
            return [annealed.best]
        tempered = sampler.run(objective)
        tracked.record(
            0,
            gradients=float(tempered.spent),
            acceptance=float(tempered.acceptance_rate[0]),
        )
        _record(None if tempered.tuned is None else tempered.tuned[0])
        return [tempered.best]


def _record(tuned: TunedStep[torch.Tensor] | None) -> None:
    """A tuned sampler's chosen step and its pilot's gradients, at step 0; nothing for a given step."""
    if tuned is not None:
        current().record(
            0, step_size=tuned.step_size, pilot_gradients=float(tuned.spent)
        )


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


def adapted_chain_start(rng: np.random.Generator) -> SampledStart:
    """``FromChain`` with :data:`ADAPTATION`'s warm-up from :data:`ADAPTED_STEP`, at :func:`chain_start`'s proposals.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromChain(
            1,
            ADAPTED_STEP,
            torch_stream(rng),
            n_steps=TRAJECTORY,
            burn_in=CHAIN_BURN_IN - ADAPTATION.warmup,
            adaptation=ADAPTATION,
        )
    )


def adapted_annealed_start(rng: np.random.Generator) -> SampledStart:
    """``FromAnnealing`` from :data:`ADAPTED_STEP`, its step re-tuned every ``ADAPTATION.warmup`` proposals.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromAnnealing(
            ExponentialTempSchedule(TEMPERATURES[-1], 1.0, ANNEAL_STEPS),
            ADAPTED_STEP,
            torch_stream(rng),
            n_steps=TRAJECTORY,
            adaptation=ADAPTATION,
        )
    )


def adapted_tempered_start(rng: np.random.Generator) -> SampledStart:
    """``FromTempering`` with a warm-up per rung from :data:`ADAPTED_STEP`, then :data:`ADAPTED_TEMPERING_ROUNDS` rounds.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromTempering(
            TEMPERATURES,
            ADAPTED_TEMPERING_ROUNDS,
            ADAPTED_STEP,
            torch_stream(rng),
            n_steps=TRAJECTORY,
            adaptation=ADAPTATION,
        )
    )


def tuned_chain_start(rng: np.random.Generator) -> SampledStart:
    """``FromChain`` at ``step_size="auto"``: :data:`TUNING`'s pilot, then :data:`TUNED_BURN_IN` proposals.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromChain(
            1,
            AUTO,
            torch_stream(rng),
            n_steps=TRAJECTORY,
            burn_in=TUNED_BURN_IN,
            adaptation=None,
            tuning=TUNING,
        )
    )


def tuned_annealed_start(rng: np.random.Generator) -> SampledStart:
    """``FromAnnealing`` at ``step_size="auto"``: :data:`TUNING`'s pilot, then :data:`TUNED_ANNEAL_STEPS` proposals.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromAnnealing(
            ExponentialTempSchedule(TEMPERATURES[-1], 1.0, TUNED_ANNEAL_STEPS),
            AUTO,
            torch_stream(rng),
            n_steps=TRAJECTORY,
            tuning=TUNING,
        )
    )


def tuned_tempered_start(rng: np.random.Generator) -> SampledStart:
    """``FromTempering`` at ``step_size="auto"``: :data:`TEMPERING_TUNING`'s pilots, then :data:`TUNED_TEMPERING_ROUNDS` rounds.

    Returns
    -------
    SampledStart
    """
    return SampledStart(
        FromTempering(
            TEMPERATURES,
            TUNED_TEMPERING_ROUNDS,
            AUTO,
            torch_stream(rng),
            n_steps=TRAJECTORY,
            tuning=TEMPERING_TUNING,
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
    "chain_adapted": adapted_chain_start,
    "annealed_adapted": adapted_annealed_start,
    "tempered_adapted": adapted_tempered_start,
    "chain_tuned": tuned_chain_start,
    "annealed_tuned": tuned_annealed_start,
    "tempered_tuned": tuned_tempered_start,
}

#: Passes over the data each start spends before its polish: a sampler's
#: gradients as :func:`~sal.sample.hmc.sample` and its siblings count them
#: --- ``TRAJECTORY`` per proposal, the first kick's gradient carried, and one
#: at each start of a kernel (issue #1222): one for the chain and the
#: annealing run, one per replica for the ladder, three for an adapted chain
#: (the warm-up's two windows and the chain's) and two per rung for the
#: adapted ladder's warm-up, whose rounds on the metric carry nothing ---
#: a tuned sampler's pilot, its budget exactly (issue #1219), and the points
#: the seam scores, one each.
CHARGES: dict[str, int] = {
    "quantile": 1,
    "restart": RESTARTS,
    "chain": 1 + (CHAIN_BURN_IN + 1) * TRAJECTORY + 1,
    "annealed": 1 + ANNEAL_STEPS * TRAJECTORY + 1,
    "tempered": len(TEMPERATURES) * (1 + TEMPERING_ROUNDS * TRAJECTORY) + 1,
    "chain_adapted": 3 + (CHAIN_BURN_IN + 1) * TRAJECTORY + 1,
    "annealed_adapted": 1 + ANNEAL_STEPS * TRAJECTORY + 1,
    "tempered_adapted": len(TEMPERATURES)
    * (2 + ADAPTATION.warmup * TRAJECTORY + ADAPTED_TEMPERING_ROUNDS * (TRAJECTORY + 1))
    + 1,
    "chain_tuned": TUNING.budget.size + 1 + (TUNED_BURN_IN + 1) * TRAJECTORY + 1,
    "annealed_tuned": TUNING.budget.size + 1 + TUNED_ANNEAL_STEPS * TRAJECTORY + 1,
    "tempered_tuned": TEMPERING_TUNING.budget.size
    + len(TEMPERATURES) * (1 + TUNED_TEMPERING_ROUNDS * TRAJECTORY)
    + 1,
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
    "ADAPTATION",
    "ADAPTED_STEP",
    "ADAPTED_TEMPERING_ROUNDS",
    "ANNEALING_STEP",
    "ANNEAL_STEPS",
    "CHAIN_BURN_IN",
    "CHARGES",
    "PILOT_PROPOSALS",
    "RESTARTS",
    "RESTART_SCALE",
    "RUNG_PILOT_PROPOSALS",
    "STARTS",
    "STEP",
    "TEMPERATURES",
    "TEMPERING_ROUNDS",
    "TEMPERING_TUNING",
    "TRAJECTORY",
    "TUNED_ANNEAL_STEPS",
    "TUNED_BURN_IN",
    "TUNED_TEMPERING_ROUNDS",
    "TUNING",
    "TUNING_GRID",
    "SampledStart",
    "adapted_annealed_start",
    "adapted_chain_start",
    "adapted_tempered_start",
    "annealed_start",
    "at_equal_evaluations",
    "chain_start",
    "gaussian_quantile_start",
    "quantile_start",
    "restart_start",
    "tempered_start",
    "tuned_annealed_start",
    "tuned_chain_start",
    "tuned_tempered_start",
]
