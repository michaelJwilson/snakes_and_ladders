"""HMC annealing as a start for Baum-Welch on a reference count-pair HMM (issue #1378, Part B).

The second fixture of #1378's Part B, declared as `count_pair_hmm/ci`
(:mod:`sal.sim.count_pair_hmm`, issue #1416): seven states over a joint count
pair (:class:`~sal.emissions.CountPairEmission`), a negative-binomial total
and beta-binomial successes, 8,000 positions in one segment. The issue asks for
98% of observations on one level and rare levels at 1--2%, which cannot both
hold of the states; they hold of the levels. Five states share the dominant
total level (98% of positions); two rare states sit on their own total
levels, 1% each, and share the dominant state's success rate, so every
pair of states differs in one channel at least and one triple is equal in
the other. NB dispersion ``1 / r`` is 0.1, BB concentration ``alpha +
beta`` 1.2e4.

**Known answers:** the generating parameters' negative log-likelihood, the
best-known optimum from restarts at ten times the budget, and the
observations the Viterbi path misses after Baum-Welch, scored against the
simulated path under the best one-to-one matching of states.

**The methods** start from one perturbed point per start (the objective's
quantile-placed start plus ``N(0, 1)`` per coordinate): Baum-Welch restarts
from it and from further perturbed points, and `hmc.anneal` from it at
``step_size="auto"``, a ramp from ``T0`` to 1, then Baum-Welch from its
best. An evaluation is one pass over the positions: a gradient or a
Baum-Welch iteration.

**``T0`` against ``|log L| / n``.** The downstream reference tuned ``T0`` to
100 where ``|log L| / n`` is about 10. :data:`T0_MULTIPLES` anneals at
``T0 = c |log L| / n``, ``|log L|`` the truth's, to ask whether one ``c``
serves this fixture and the mixture's (`mixture_hmc_anneal`, ``T0 = 4`` at
``|log L| / n = 2.2``). Reported, not adopted as a default.

Run as ``python -m sal.qa.count_pair_hmm_anneal``; two threads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sal.cost import Cost
from sal.emissions import CountPairEmission
from sal.likelihood.ragged import viterbi
from sal.opt.budget import Budget, Method, Outcome, compare
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.starts import polish_by_baum_welch
from sal.ragged import Ragged
from sal.sample.hmc import anneal
from sal.sample.schedule import InverseLinearTempSchedule
from sal.sample.tune import Criterion, StepTuning
from sal.sim import fixtures
from sal.sim.count_pair_hmm import CountPairHmmParams

#: The seed of the comparison's starts; the instance's is `count_pair_hmm/ci`'s.
SEED = 1378
K = 7
N = 8_000

#: Passes per start: gradients and Baum-Welch iterations alike.
BUDGET = Budget(Cost.EVALUATIONS, 600)
#: Passes of a restart, its Baum-Welch iterations and the start's scoring;
#: the budget buys ``600 // 150 = 4``.
RESTART_ITERATIONS = 150
#: The anneal: leapfrog steps per proposal, its proposals, the step pilot,
#: and the Baum-Welch polish from its best, whose iterations fill the rest.
N_STEPS = 4
N_PROPOSALS = 50
PILOT = Budget(Cost.GRADIENTS, 80)
STEP_GRID = (1e-3, 3e-3, 1e-2)
POLISH_ITERATIONS = BUDGET.size - PILOT.size - N_PROPOSALS * N_STEPS - 2
#: ``T0 = c |log L| / n``.
T0_MULTIPLES = (1.0, 10.0, 100.0)
#: The multiple :func:`measure` reports at: the downstream reference's.
T0_MULTIPLE = 10.0
N_STARTS = 4
#: Starts of the ten-times referee. Run once on 2026-10-08 at one start, 189 s:
#: 63391.554, 40.4 nats above the best of the four shared starts at 1x, so
#: the comparison scores against the best any method found; zero skips it.
REFEREE_STARTS = 0
REFEREE_BUDGET = Budget(Cost.EVALUATIONS, 10 * BUDGET.size)
TOLERANCE = 1e-6
THREADS = 2


@dataclass(frozen=True)
class Fixture:
    """The draws, the simulated path, the objective and the truth's value."""

    data: Ragged
    states: np.ndarray
    objective: EmissionHmmObjective
    at_truth: float

    @property
    def per_position(self) -> float:
        """``|log L| / n`` at the truth."""
        return float(abs(self.at_truth) / self.data.values.shape[0])


def _start(dispersion: float, mean: np.ndarray, rate: np.ndarray) -> CountPairEmission:
    """The joint family a fit starts from, at concentration 100."""
    return CountPairEmission(
        np.full(K, dispersion),
        mean,
        100.0 * rate,
        100.0 * (1.0 - rate),
        None,
        joint=True,
    )


def fixture() -> Fixture:
    """Draw ``count_pair_hmm/ci``, and the objective at a quantile-placed start."""
    params: CountPairHmmParams = fixtures.fixture("count_pair_hmm", "ci").params
    simulated = params.instance()
    values = np.asarray(simulated.observations, dtype=np.float64).reshape(N, 2)
    data = Ragged(values, (N,))
    totals = values[:, 0]
    start = _start(
        params.dispersion,
        np.quantile(totals, np.linspace(0.05, 0.95, K)),
        np.linspace(0.3, 0.7, K),
    )
    objective = EmissionHmmObjective(data, start)
    at = objective.theta_from(
        {
            "log_initial": torch.log(torch.as_tensor(params.occupancy)),
            "log_transition": torch.log(torch.as_tensor(params.transition)),
            **params.components.named_parameters(),
        }
    )
    return Fixture(
        data,
        np.asarray(simulated.states).reshape(-1),
        objective,
        float(objective(at)),
    )


def missed(instance: Fixture, theta: torch.Tensor) -> float:
    """The fraction of positions the Viterbi path at ``theta`` misses, states matched one to one."""
    objective = instance.objective
    with torch.no_grad():
        named = objective.constrain(theta.detach())
        family = objective.components(theta.detach())
        scores = family.log_density(torch.as_tensor(instance.data.values))
    path = viterbi(
        Ragged(scores.numpy(), instance.data.lengths),
        named["log_initial"].numpy(),
        named["log_transition"].numpy(),
    ).path
    confusion = np.zeros((K, K))
    np.add.at(confusion, (instance.states, path), 1.0)
    rows, cols = linear_sum_assignment(-confusion)
    return float(1.0 - confusion[rows, cols].sum() / path.shape[0])


def _perturbed(instance: Fixture, rng: np.random.Generator) -> torch.Tensor:
    base = instance.objective.initial()
    noise = torch.as_tensor(rng.normal(size=base.shape[0]), dtype=base.dtype)
    return base + noise


@dataclass(frozen=True)
class Restarts:
    """Baum-Welch from ``budget // RESTART_ITERATIONS`` perturbed points, the best kept."""

    def __call__(
        self, instance: Fixture, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        best = None
        spent = 0
        for _ in range(budget.size // RESTART_ITERATIONS):
            polished = polish_by_baum_welch(
                instance.objective,
                _perturbed(instance, rng),
                Budget(Cost.ITERATIONS, RESTART_ITERATIONS - 1),
            )
            spent += polished.iterations + 1
            if best is None or polished.value < best.value:
                best = polished
        assert best is not None
        return Outcome(best.value, spent, best.theta)


@dataclass(frozen=True)
class Annealer:
    """`hmc.anneal` from ``T0`` to 1 at ``step_size="auto"``, then Baum-Welch from its best."""

    t0: float

    def __call__(
        self, instance: Fixture, budget: Budget, rng: np.random.Generator
    ) -> Outcome:
        start = _perturbed(instance, rng)
        run = anneal(
            instance.objective,
            InverseLinearTempSchedule(self.t0, 1.0, N_PROPOSALS),
            rng,
            step_size="auto",
            tuning=StepTuning(PILOT, Criterion.LOWEST_ENERGY, STEP_GRID),
            n_steps=N_STEPS,
            start=start,
            polish=polish_by_baum_welch,
            polish_budget=Budget(
                Cost.ITERATIONS,
                budget.size - PILOT.size - N_PROPOSALS * N_STEPS - 2,
            ),
        )
        schedule = run.stages[0]
        value = run.value if run.value <= schedule.energy else schedule.energy
        return Outcome(value, run.spent, run)


def main() -> None:
    """The truth, the referee, the comparison at ``T0 = 10 |log L| / n``, and the ``T0`` sweep on start 0."""
    torch.set_num_threads(THREADS)
    instance = fixture()
    print(f"truth {instance.at_truth:.3f}, |log L| / n {instance.per_position:.3f}")
    clock = time.perf_counter()
    for i in range(REFEREE_STARTS):
        value = Restarts()(
            instance, REFEREE_BUDGET, np.random.default_rng([SEED + 1, i])
        ).value
        print(f"referee {value:.3f} ({time.perf_counter() - clock:.1f} s)")
    seconds = {"restarts": 0.0, "anneal": 0.0}
    methods: dict[str, Method[Fixture]] = {
        "restarts": Restarts(),
        "anneal": Annealer(T0_MULTIPLE * instance.per_position),
    }

    def timed(name: str) -> Method[Fixture]:
        def run(item: Fixture, budget: Budget, rng: np.random.Generator) -> Outcome:
            started = time.perf_counter()
            outcome: Outcome = methods[name](item, budget, rng)
            seconds[name] += time.perf_counter() - started
            return outcome

        return run

    comparison = compare(
        {name: timed(name) for name in methods},
        [instance] * N_STARTS,
        BUDGET,
        seeds=(SEED,),
        workers=1,
    )
    hits = comparison.hits(TOLERANCE, relative=True)
    p_value = comparison.paired_p("anneal", "restarts", TOLERANCE, relative=True)
    for row, name in enumerate(comparison.methods):
        cells = comparison.outcomes[row * N_STARTS : (row + 1) * N_STARTS]
        thetas = [c.detail if name == "restarts" else c.detail.best for c in cells]
        misses = [missed(instance, theta) for theta in thetas]
        gap = comparison.best[row] - comparison.reference
        print(
            f"| {name} | {hits[name]}/{N_STARTS} | {np.median(misses):.4f} "
            f"({min(misses):.4f}-{max(misses):.4f}) | {gap.mean():.2f} | "
            f"{int(comparison.spent[row].max())} | {seconds[name] / N_STARTS:.1f} |"
        )
        if name == "anneal":
            for cell in cells:
                run = cell.detail
                print(
                    f"  step {run.tuned.step_size:.0e} init {run.stages[0].energy:.1f} "
                    f"polished {run.value:.1f} after {run.polish_spent}"
                )
    print(f"McNemar p {p_value:.3f}")
    for multiple in T0_MULTIPLES:
        t0 = multiple * instance.per_position
        outcome = Annealer(t0)(instance, BUDGET, np.random.default_rng([SEED, 0]))
        print(
            f"T0 {t0:.1f} ({multiple:g} |log L|/n): value {outcome.value:.1f}, "
            f"missed {missed(instance, outcome.detail.best):.4f}"
        )


if __name__ == "__main__":
    main()
