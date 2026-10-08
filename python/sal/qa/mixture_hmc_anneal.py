"""HMC annealing against polished EM restarts on the five-component mixture, at equal evaluations (issue #1378, Part B).

`docs/experiments/004` measured annealing at a hand-set step (0.02) and
ramp (exponential, 8 to 1): it reached the best-known optimum from 1 of 40
starts against restarts' 7 (McNemar p = 0.031). That predates the step
pilot (#1219, ``POLISHED_GAP`` #1251), the Rust anneal loop (#1249) and the
polish from the schedule's best (#1374). This script reruns the question
with all three on the same fixture, starts, budget and polish.

**The fixture** is `mixture/ci`, the one `004` and
`tests/regression/opt/test_opt_mixture_budget.py` use. **The known
answers** are :data:`BEST_KNOWN`, which 16 of 1,000 polished restarts reach
(`004`), re-derived here from restarts at ten times the budget; and the
generating parameters' negative log-likelihood, raw and polished.

**One unit.** An evaluation is one pass over the observations: an EM
iteration, an objective value or a gradient, as in `004`. The anneal's
``spent`` counts its gradients, the step pilot's included; every polish,
the pilot's included, is counted by :class:`_Counted` on the objective.

**The ramp is tuned on held-out starts** (:data:`TUNING_SEED`): every
:data:`RAMP_GRID` candidate anneals from :data:`TUNING_STARTS` starts, ranked
by the starts reaching :data:`BEST_KNOWN`, then the mean polished value.
That search is charged once, not per start, and reported beside the table.

Run as ``python -m sal.qa.mixture_hmc_anneal``; two threads, so the host
keeps two cores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sal.cost import Cost
from sal.emissions import GaussianEmission
from sal.opt.budget import Budget, Comparison, Method, Outcome, compare, restarts
from sal.opt.em import EM
from sal.opt.mixture import (
    GaussianMixtureObjective,
    expectation_maximization,
    uniform_seeds,
)
from sal.opt.starts import PolishedPoint, polish_by_fit
from sal.opt.termination import Termination
from sal.sample.hmc import anneal
from sal.sample.schedule import ScheduleParams, ScheduleShape
from sal.sample.tune import Criterion, StepTuning
from sal.sim import fixtures
from sal.sim.mixture import simulate_mixture

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sal.opt.objective import Objective

#: The fixture `004` measured on: five components 1.5 sd apart, 500 draws.
FIXTURE = ("mixture", "ci")
#: Held equal per start, in passes over the observations (`004`'s budget).
BUDGET = Budget(Cost.EVALUATIONS, 3000)
#: The shared polish: L-BFGS iterations, and the evaluations that bounds
#: (25 line-search calls and a gradient norm per iteration, two at the end).
POLISH_STEPS = 8
POLISH_RESERVE = POLISH_STEPS * 27 + 2
#: The step pilot: its gradients, its candidate steps, and each candidate's
#: polish under ``POLISHED_GAP``, in iterations.
PILOT = Budget(Cost.GRADIENTS, 400)
STEP_GRID = (0.005, 0.01, 0.02, 0.04)
PILOT_POLISH_STEPS = 2
PILOT_POLISH_RESERVE = len(STEP_GRID) * (PILOT_POLISH_STEPS * 27 + 2)
#: Leapfrog steps per proposal, and the gradients one proposal costs.
N_STEPS = 10
PER_PROPOSAL = N_STEPS


def proposals(budget: Budget) -> int:
    """The schedule's proposals: what ``budget`` leaves after the pilot and every polish's reserve, less the chain's opening gradient."""
    return (
        budget.size - PILOT.size - PILOT_POLISH_RESERVE - POLISH_RESERVE - 2
    ) // PER_PROPOSAL


#: The ramps tried on the held-out starts: `004`'s (exponential 8 to 1)
#: and the shapes and endpoints around it.
RAMP_GRID = tuple(
    ScheduleParams(shape, t_start, t_end)
    for shape in (
        ScheduleShape.EXPONENTIAL,
        ScheduleShape.LINEAR,
        ScheduleShape.INVERSE_LINEAR,
    )
    for t_start in (4.0, 16.0, 64.0)
    for t_end in (1.0, 0.25)
)
#: What :func:`tune_ramp` chose on 2026-10-08 (`docs/experiments/030`), so a
#: test reads the ramp without re-running the search.
TUNED_RAMP = ScheduleParams(ScheduleShape.INVERSE_LINEAR, 4.0, 0.25)
#: EM iterations per restart before its polish (`004`'s), and so its cost.
EM_COST = 200
RESTART_COST = EM_COST + POLISH_RESERVE
#: The best-known negative log-likelihood (`004`, 1,000 polished restarts).
BEST_KNOWN = 1111.596410
#: Reaching it: within this, relative.
TOLERANCE = 1e-6
#: A near miss: within this many nats, outside :data:`TOLERANCE`.
NEAR = 0.1
#: Held-out starts the ramp is tuned on, on the stream ``[TUNING_SEED, i]``.
TUNING_SEED = 1378
TUNING_STARTS = 8
#: Starts behind the referee at ten times the budget.
REFEREE_STARTS = 4
REFEREE_BUDGET = Budget(Cost.EVALUATIONS, 10 * BUDGET.size)
#: Reported starts, `004`'s count; start ``i`` draws from ``[0, i]``.
N_STARTS = 40
#: Intra-op threads: two, so the host keeps two cores.
THREADS = 2

METHODS = ("restarts", "anneal")


class _Counted:
    """The objective, counting its calls: each is one value and, by autograd, its gradient."""

    def __init__(self, inner: Objective) -> None:
        self.inner = inner
        self.calls = 0

    def initial(self) -> torch.Tensor:
        return self.inner.initial()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.inner.constrain(theta)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.inner.theta_from(named)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(theta)


@dataclass
class _CountingPolish:
    """:func:`~sal.opt.starts.polish_by_fit`, its evaluations summed into ``calls``.

    A zero-scale probe is refused by the fit; the polish then keeps the
    point it was handed, at its value, as `004` does.
    """

    calls: int = 0

    def __call__(
        self, objective: Objective, theta: torch.Tensor, budget: Budget
    ) -> PolishedPoint:
        counted = _Counted(objective)
        try:
            polished = polish_by_fit(counted, theta, budget)
        except ValueError:
            value = float(objective(theta.detach()))
            counted.calls += 1
            polished = PolishedPoint(
                value=value,
                termination=Termination.after(0, converged=False),
                theta=theta.detach(),
            )
        self.calls += counted.calls
        return polished


@dataclass(frozen=True)
class Fixture:
    """The draws, their objective, and the generating parameters."""

    observations: np.ndarray
    objective: GaussianMixtureObjective
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


def instance(loaded: fixtures.Fixture[Any] | None = None) -> Fixture:
    """`mixture/ci`'s draws and objective, from ``loaded`` where a caller names the fixture."""
    params = (fixtures.fixture(*FIXTURE) if loaded is None else loaded).params
    observations = simulate_mixture(params).observations
    weights = np.asarray(params.weights)
    return Fixture(
        observations,
        GaussianMixtureObjective(observations, len(weights)),
        weights,
        np.asarray(params.components.mean),
        np.asarray(params.components.scale),
    )


def _start(instance: Fixture, rng: np.random.Generator) -> torch.Tensor:
    """`004`'s start: uniform weights, means at distinct observations, pooled scales."""
    centres = np.sort(uniform_seeds(instance.observations, len(instance.weights), rng))
    return instance.objective.theta_from_centres(torch.as_tensor(centres))


def em_then_polish(
    instance: Fixture, budget: Budget, rng: np.random.Generator
) -> Outcome:
    """One restart: EM for the budget less the polish's reserve, then the shared polish."""
    objective = instance.objective
    named = objective.constrain(_start(instance, rng))
    try:
        em = expectation_maximization(
            instance.observations,
            torch.exp(named["log_weight"]),
            GaussianEmission(named["mean"], named["scale"], objective.variance_floor),
            config=replace(EM, max_iterations=budget.size - POLISH_RESERVE),
        )
    except ValueError:
        return Outcome(np.inf, budget.size - POLISH_RESERVE)
    theta = objective.theta_from(
        {
            "log_weight": torch.log(em.weights),
            "mean": em.components.mean,
            "scale": em.components.scale,
        }
    )
    polish = _CountingPolish()
    polished = polish(objective, theta, Budget(Cost.ITERATIONS, POLISH_STEPS))
    value = min(-em.log_likelihood, polished.value)
    return Outcome(value, em.termination.iterations + polish.calls)


@dataclass(frozen=True)
class Annealer:
    """`hmc.anneal` on ``ramp``, ``step_size="auto"``, then the shared polish, inside the budget.

    Picklable, so `opt.budget.compare` may run it on a process pool. The
    outcome's ``detail`` is the run, carrying ``stages`` and ``tuned``.
    """

    ramp: ScheduleParams

    def __call__(
        self, instance: Fixture, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        polish = _CountingPolish()
        tuning = StepTuning(
            PILOT,
            Criterion.POLISHED_GAP,
            STEP_GRID,
            polish,
            Budget(Cost.ITERATIONS, PILOT_POLISH_STEPS),
        )
        start = _start(instance, rng)
        run = anneal(
            instance.objective,
            self.ramp.build(proposals(budget)),
            rng,
            step_size="auto",
            tuning=tuning,
            n_steps=N_STEPS,
            start=start,
            polish=polish,
            polish_budget=Budget(Cost.ITERATIONS, POLISH_STEPS),
        )
        schedule = run.stages[0]
        # The polish's value where it is no worse than the schedule's best,
        # as a restart keeps the lower of EM's and its polish's: a NaN
        # polish (2 of 144 held-out runs before this guard) keeps the schedule's best, and `main` counts them.
        value = run.value if run.value <= schedule.energy else schedule.energy
        return Outcome(value, schedule.spent + polish.calls, run)


@dataclass(frozen=True)
class RampScore:
    """One ramp on the held-out starts."""

    ramp: ScheduleParams
    hits: int
    mean_value: float


def tune_ramp(instance: Fixture) -> tuple[ScheduleParams, tuple[RampScore, ...], float]:
    """The :data:`RAMP_GRID` ramp ranked first on the held-out starts, every score, and the seconds."""
    clock = time.perf_counter()
    scores = []
    for ramp in RAMP_GRID:
        method = Annealer(ramp)
        values = np.array(
            [
                method(instance, BUDGET, np.random.default_rng([TUNING_SEED, i])).value
                for i in range(TUNING_STARTS)
            ]
        )
        hits = int(np.sum(values <= BEST_KNOWN * (1 + TOLERANCE)))
        scores.append(RampScore(ramp, hits, float(values.mean())))
    best = min(scores, key=lambda s: (-s.hits, s.mean_value))
    return best.ramp, tuple(scores), time.perf_counter() - clock


@dataclass(frozen=True)
class Referee:
    """The known answers: restarts at ten times the budget, and the generating parameters."""

    best_of_restarts: float
    at_truth: float
    polished_truth: float


def referee(instance: Fixture, n_starts: int = REFEREE_STARTS) -> Referee:
    """Restarts at :data:`REFEREE_BUDGET` from ``n_starts`` starts, and the truth raw and polished."""
    many = restarts(em_then_polish, RESTART_COST)
    best = min(
        many(
            instance, REFEREE_BUDGET, np.random.default_rng([TUNING_SEED + 1, i])
        ).value
        for i in range(n_starts)
    )
    objective = instance.objective
    truth = objective.theta_from_truth(instance.weights, instance.mean, instance.scale)
    at_truth = float(objective(truth))
    polished = _CountingPolish()(objective, truth, Budget(Cost.ITERATIONS, 100))
    return Referee(best, at_truth, min(at_truth, polished.value))


@dataclass(frozen=True)
class Measurement:
    """One comparison, its seconds per method, and its tuned ramp."""

    comparison: Comparison
    seconds: dict[str, float]
    ramp: ScheduleParams
    stage_values: tuple[float, ...] = field(default=())

    def hits(self) -> dict[str, int]:
        return self.comparison.hits(TOLERANCE, relative=True)

    def paired_p(self) -> float:
        return self.comparison.paired_p("anneal", "restarts", TOLERANCE, relative=True)

    def table(self) -> str:
        """Hits, McNemar p, mean and median gap, spend and seconds per method."""
        n_starts = self.comparison.reference.shape[0]
        hits = self.hits()
        lines = [
            "| method | hits | within 0.1 nat | p vs restarts | mean gap "
            "| median gap | max spent | s per start |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row, name in enumerate(self.comparison.methods):
            gap = self.comparison.best[row] - self.comparison.reference
            p_value = "-" if name == "restarts" else f"{self.paired_p():.3f}"
            lines.append(
                f"| {name} | {hits[name]}/{n_starts} | "
                f"{int(np.sum(gap <= NEAR))}/{n_starts} | {p_value} | "
                f"{gap.mean():.2f} | {np.median(gap):.2f} | "
                f"{int(self.comparison.spent[row].max())} | "
                f"{self.seconds[name] / n_starts:.2f} |"
            )
        return "\n".join(lines)


def measure(instance: Fixture, ramp: ScheduleParams, n_starts: int) -> Measurement:
    """Restarts and the anneal on starts ``[0, i]``, ``i < n_starts``, at :data:`BUDGET`."""
    seconds = dict.fromkeys(METHODS, 0.0)
    methods: dict[str, Method[Fixture]] = {
        "restarts": restarts(em_then_polish, RESTART_COST),
        "anneal": Annealer(ramp),
    }

    def timed(name: str) -> Method[Fixture]:
        def run(item: Fixture, budget: Budget, rng: np.random.Generator) -> Outcome:
            clock = time.perf_counter()
            outcome: Outcome = methods[name](item, budget, rng)
            seconds[name] += time.perf_counter() - clock
            return outcome

        return run

    comparison = compare(
        {name: timed(name) for name in METHODS},
        [instance] * n_starts,
        BUDGET,
        seeds=(0,),
        workers=1,
        known=[BEST_KNOWN] * n_starts,
    )
    stages = tuple(
        float(outcome.detail.stages[0].energy)
        for outcome in comparison.outcomes
        if outcome.detail is not None
    )
    return Measurement(comparison, seconds, ramp, stages)


def main() -> None:
    """Tune the ramp, run the referee and the comparison, and print what `030` reports."""
    torch.set_num_threads(THREADS)
    cell = instance()
    ramp, scores, tuning_seconds = tune_ramp(cell)
    print(f"ramp search: {tuning_seconds:.1f} s over {len(scores)} ramps")
    for score in scores:
        print(
            f"  {score.ramp.shape:>15} {score.ramp.t_start:>5} -> "
            f"{score.ramp.t_end:<5} hits {score.hits}/{TUNING_STARTS} "
            f"mean {score.mean_value:.3f}"
        )
    print(f"chosen: {ramp}")
    clock = time.perf_counter()
    known = referee(cell)
    print(
        f"referee ({time.perf_counter() - clock:.1f} s): best of restarts at "
        f"{REFEREE_BUDGET.size} x {REFEREE_STARTS} {known.best_of_restarts:.6f}, "
        f"truth {known.at_truth:.3f}, polished truth {known.polished_truth:.3f}"
    )
    measurement = measure(cell, ramp, N_STARTS)
    print(measurement.table())
    stages = np.array(measurement.stage_values)
    print(
        f"anneal schedule best before polish: median {np.median(stages):.3f}, "
        f"min {stages.min():.3f}"
    )
    steps = [
        outcome.detail.tuned.step_size
        for outcome in measurement.comparison.outcomes
        if outcome.detail is not None
    ]
    print("steps chosen:", {s: steps.count(s) for s in sorted(set(steps))})
    polishes = [
        outcome.detail.stages[-1]
        for outcome in measurement.comparison.outcomes
        if outcome.detail is not None
    ]
    print(
        f"anneal polishes: {sum(not np.isfinite(p.energy) for p in polishes)} NaN, "
        f"{sum(p.termination.converged for p in polishes)} converged of "
        f"{len(polishes)}"
    )


if __name__ == "__main__":
    main()
