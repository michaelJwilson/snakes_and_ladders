"""HMC tuning as an HMM start on the canonical count-pair cell (issue #1390, question 6).

Three measurements on ``count_hmm_reference`` (:mod:`sal.sim.count_hmm_cell`:
8 generating levels fitted by ``K = 7`` states), every start polished by
Baum-Welch at an equal budget of :data:`PASSES`, a pass being one gradient,
one scored draw or one Baum-Welch iteration:

* **(a) What** ``step_size="auto"`` **spends to settle.** The anneal's pilot
  (:class:`~sal.sample.tune.StepTuning`, :attr:`~sal.sample.tune.Criterion.LOWEST_ENERGY`
  over :data:`STEP_GRID`) at :data:`PILOT_PROPOSALS` proposals per candidate;
  a start's step has settled at the smallest pilot from which the choice
  equals the largest pilot's. Against it, the same anneal at each grid step
  given, no pilot.
* **(b) T0 in units of** :func:`scale`, ``|log L| / n``: the negative
  log-likelihood of the 7-state HMM at the start point (chain at ``stay``,
  emissions read off ``K`` positions), summed over the cell and divided by
  its ``n`` positions. The anneal's T0 and the fixed-temperature sampler's T
  are :data:`MULTIPLES` of it; the multiple chosen on stress is re-run on the
  dominant-level and ``delta_log_mean`` variants.
* **(c) Rare levels.** Per run, each non-dominant generating level with
  positions in the cell is recovered when the state the best one-to-one
  matching assigns it holds at least half its positions after Baum-Welch.

The start, the sampled coordinates (``log mu`` and ``logit p`` per state),
the samplers' forms, k-means++ and the matching are #1393's harness
(``docs/experiments/034``), restated here; the start seeds are its, so the
restart arm reproduces its first starts.

Run as ``python -m sal.qa.count_hmm_hmc_tuning stress`` (the stress arms)
and ``python -m sal.qa.count_hmm_hmc_tuning variants <multiple>``; two
worker processes at one thread each.
"""

from __future__ import annotations

import functools
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sal.cost import Cost
from sal.emissions import RateConcentrationCountPairEmission
from sal.fixtures import Scale
from sal.likelihood.ragged import viterbi
from sal.opt.budget import Budget
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.mixture import kmeans_plus_plus
from sal.opt.objective import Restricted, coordinates
from sal.opt.starts import polish_by_baum_welch
from sal.parallel import map_tasks
from sal.ragged import Ragged
from sal.sample.chain import Adaptation
from sal.sample.hmc import anneal, sample
from sal.sample.schedule import InverseLinearTempSchedule
from sal.sample.tune import Criterion, StepTuning
from sal.sim.count_hmm_cell import CountHmm, CountHmmReferenceParams
from sal.sim.fixtures import fixture

#: #1393's start seeds, so the restart arm repeats its first starts.
SEED = 1393
#: Passes per start: gradients, scored draws and Baum-Welch iterations alike.
PASSES = 400
#: Leapfrog steps per proposal, as the downstream sampler takes them.
N_STEPS = 8
#: The anneal's proposals on an inverse-linear ramp from T0 to 1, #1393's.
ANNEAL_PROPOSALS = 16
#: The pilot's candidate steps, #1393's.
STEP_GRID = (3.0e-3, 1.0e-2, 3.0e-2)
#: Pilot proposals per candidate: 2 is #1393's 49 gradients, 8 is 193.
PILOT_PROPOSALS = (2, 4, 8)
#: The fixed-temperature sampler: 8 dual-averaging proposals from
#: :data:`SAMPLE_STEP`, then :data:`DRAWS` draws, the best of which is polished.
ADAPTATION = Adaptation(warmup=8, target_acceptance=0.65, step_jitter=0.4)
SAMPLE_STEP = 1.0e-2
DRAWS = 13
#: T0 (anneal) and T (sampler) in units of :func:`scale`.
MULTIPLES = (1.0, 3.0, 10.0, 30.0, 100.0)
#: The multiple the pilot sweep and the given steps run at.
REFERENCE_MULTIPLE = 10.0
#: The variants the chosen multiple is re-run on.
VARIANTS = ("dominant_50", "dominant_90", "delta_0.1", "delta_0.6")
STRESS_STARTS = 8
VARIANT_STARTS = 4
WORKERS = 2


@dataclass(frozen=True)
class Cell:
    """The drawn cell, its objective at the ``K``-state family and the sampled coordinates."""

    params: CountHmmReferenceParams
    cell: CountHmm
    objective: EmissionHmmObjective
    varied: torch.Tensor

    @property
    def n_states(self) -> int:
        """``K``, the states a fit is given."""
        return self.objective.n_states


@dataclass(frozen=True)
class Result:
    """One start of one arm after Baum-Welch.

    ``recovered`` holds, per generating level in level order, 1 if the state
    matched to it holds at least half its positions, 0 if not, and -1 for a
    level with no positions in the cell. ``step`` is the step the sampler ran
    at (chosen, given or adapted) and ``tuner`` the gradients spent choosing
    it; ``scale`` is :func:`scale` at the start.
    """

    log_likelihood: float
    missed: float
    recovered: tuple[int, ...]
    gradients: int
    tuner: int
    polish_iterations: int
    step: float
    scale: float
    seconds: float


def _family(
    params: CountHmmReferenceParams, mean: np.ndarray, rate: np.ndarray
) -> RateConcentrationCountPairEmission:
    k = mean.shape[0]
    return RateConcentrationCountPairEmission(
        np.full(k, float(params.dispersion[0])),
        mean,
        rate,
        np.full(k, float(params.concentration[0])),
        np.full(k, round(params.trials_mean)),
        joint=False,
    )


@functools.cache
def build(tier: Scale = Scale.STRESS, variant: str | None = None) -> Cell:
    """The registry's cell at ``tier``, or its declared ``variant``, and its ``K``-state objective."""
    params: CountHmmReferenceParams = fixture("count_hmm_reference", tier).params
    if variant is not None:
        params = params.variant(variant)
    cell = params.instance()
    k = params.n_fit_states
    objective = EmissionHmmObjective(
        Ragged(cell.observations, cell.lengths),
        _family(params, np.full(k, params.base_mean), np.full(k, 0.5)),
        covariate=cell.covariate,
    )
    return Cell(params, cell, objective, coordinates(objective, ["mean", "rate"]))


def _features(cell: CountHmm) -> np.ndarray:
    return np.stack(
        [
            cell.observations[:, 0] / cell.covariate[:, 0],
            cell.observations[:, 1] / cell.covariate[:, 1],
        ],
        axis=1,
    )


def _held(instance: Cell, mean: np.ndarray, rate: np.ndarray) -> torch.Tensor:
    """The full ``theta``: the chain at ``stay``, dispersion and concentration declared, these emissions."""
    params = instance.params
    k = instance.n_states
    transition = np.full((k, k), (1.0 - params.stay) / (k - 1))
    np.fill_diagonal(transition, params.stay)
    family = _family(params, np.clip(mean, 0.5, None), np.clip(rate, 0.02, 0.98))
    return instance.objective.theta_from_truth(
        np.full(k, 1.0 / k), transition, **family.named_parameters()
    )


def random_start(instance: Cell, rng: np.random.Generator) -> torch.Tensor:
    """``K`` positions drawn uniformly without replacement, read as the states' means and rates."""
    n = int(instance.cell.observations.shape[0])
    rows = _features(instance.cell)[rng.choice(n, instance.n_states, replace=False)]
    return _held(instance, rows[:, 0], rows[:, 1])


def kmeans_start(instance: Cell, rng: np.random.Generator) -> torch.Tensor:
    """k-means++ centres on the two features, standardized, read back as means and rates."""
    features = _features(instance.cell)
    centre, spread = features.mean(axis=0), features.std(axis=0)
    centres = kmeans_plus_plus((features - centre) / spread, instance.n_states, rng)
    centres = centres * spread + centre
    return _held(instance, centres[:, 0], centres[:, 1])


def scale(instance: Cell, theta: torch.Tensor) -> float:
    """``|log L| / n``: the negative log-likelihood at ``theta`` over the cell's positions."""
    with torch.no_grad():
        value = float(instance.objective(theta))
    return abs(value) / int(instance.cell.observations.shape[0])


def assess(instance: Cell, theta: torch.Tensor) -> tuple[float, tuple[int, ...]]:
    """Missed positions under the best one-to-one matching, and each level's recovery."""
    objective = instance.objective
    cell = instance.cell
    with torch.no_grad():
        named = objective.constrain(theta.detach())
        scores = objective.components(theta.detach()).log_density(
            torch.as_tensor(cell.observations),
            covariate=torch.as_tensor(cell.covariate),
        )
    path = viterbi(
        Ragged(scores.numpy(), cell.lengths),
        named["log_initial"].numpy(),
        named["log_transition"].numpy(),
    ).path
    levels = cell.initial.shape[0]
    confusion = np.zeros((levels, instance.n_states))
    np.add.at(confusion, (cell.states, path), 1.0)
    rows, cols = linear_sum_assignment(-confusion)
    held = confusion.sum(axis=1)
    matched = np.zeros(levels)
    matched[rows] = confusion[rows, cols]
    recovered = tuple(
        -1 if held[level] == 0 else int(matched[level] >= 0.5 * held[level])
        for level in range(levels)
    )
    return float(1.0 - matched.sum() / path.shape[0]), recovered


def _polish(
    instance: Cell,
    theta: torch.Tensor,
    gradients: int,
    tuner: int,
    step: float,
    at: float,
    clock: float,
) -> Result:
    polished = polish_by_baum_welch(
        instance.objective, theta, Budget(Cost.ITERATIONS, PASSES - gradients)
    )
    missed, recovered = assess(instance, polished.theta)
    return Result(
        -float(polished.value),
        missed,
        recovered,
        gradients,
        tuner,
        int(polished.termination.iterations),
        step,
        at,
        time.perf_counter() - clock,
    )


def _scatter(instance: Cell, at: torch.Tensor, reduced: torch.Tensor) -> torch.Tensor:
    full = at.clone()
    full[instance.varied] = reduced.detach()
    return full


def run_restart(instance: Cell, start: torch.Tensor) -> Result:
    """Baum-Welch for the whole budget from ``start``."""
    return _polish(
        instance, start, 0, 0, 0.0, scale(instance, start), time.perf_counter()
    )


def run_anneal(
    instance: Cell,
    start: torch.Tensor,
    multiple: float,
    rng: np.random.Generator,
    *,
    pilot: int | None = PILOT_PROPOSALS[0],
    step: float | None = None,
) -> Result:
    """`hmc.anneal` from ``multiple * scale`` to 1, its step chosen by a pilot or given, then Baum-Welch."""
    clock = time.perf_counter()
    at = scale(instance, start)
    restricted = Restricted(instance.objective, start, instance.varied)
    schedule = InverseLinearTempSchedule(multiple * at, 1.0, ANNEAL_PROPOSALS)
    if step is None:
        assert pilot is not None
        budget = Budget(Cost.GRADIENTS, 1 + len(STEP_GRID) * pilot * N_STEPS)
        run = anneal(
            restricted,
            schedule,
            rng,
            step_size="auto",
            tuning=StepTuning(budget, Criterion.LOWEST_ENERGY, STEP_GRID),
            n_steps=N_STEPS,
            start=restricted.initial(),
        )
        assert run.tuned is not None
        chosen, tuner = run.tuned.step_size, int(run.tuned.spent)
    else:
        run = anneal(
            restricted,
            schedule,
            rng,
            step_size=step,
            n_steps=N_STEPS,
            start=restricted.initial(),
        )
        chosen, tuner = step, 0
    return _polish(
        instance,
        _scatter(instance, start, run.best),
        int(run.spent),
        tuner,
        chosen,
        at,
        clock,
    )


def run_sample(
    instance: Cell, start: torch.Tensor, multiple: float, rng: np.random.Generator
) -> Result:
    """The downstream form at ``T = multiple * scale``: adapt, draw, polish the best draw."""
    clock = time.perf_counter()
    at = scale(instance, start)
    restricted = Restricted(instance.objective, start, instance.varied)
    chain = sample(
        restricted,
        rng,
        DRAWS,
        step_size=SAMPLE_STEP,
        n_steps=N_STEPS,
        start=restricted.initial(),
        temperature=multiple * at,
        adaptation=ADAPTATION,
    )
    with torch.no_grad():
        values = [float(restricted(draw)) for draw in chain.draws]
    best = chain.draws[int(np.argmin(values))]
    assert chain.adapted is not None
    return _polish(
        instance,
        _scatter(instance, start, best),
        int(chain.spent) + DRAWS,
        int(chain.adapted.force_evaluations),
        float(chain.adapted.step_size),
        at,
        clock,
    )


#: One task: the variant (``None`` for stress), the tier, the start, the arm.
type Task = tuple[str | None, Scale, int, str]


def run(task: Task) -> Result:
    """One arm from one shared start; the arm's name says its form and setting."""
    variant, tier, i, arm = task
    instance = build(tier, variant)
    rng = np.random.default_rng([SEED, i, 1])
    if arm == "k-means++":
        clock = time.perf_counter()
        point = kmeans_start(instance, np.random.default_rng([SEED, i, 2]))
        return _polish(instance, point, 0, 0, 0.0, scale(instance, point), clock)
    start = random_start(instance, np.random.default_rng([SEED, i]))
    form, _, setting = arm.partition(" ")
    if form == "restart":
        return run_restart(instance, start)
    if form == "sample":
        return run_sample(instance, start, float(setting.removeprefix("T=")), rng)
    if form == "anneal":
        return run_anneal(instance, start, float(setting.removeprefix("T0=")), rng)
    if form == "pilot":
        pilot = int(setting.removeprefix("p="))
        return run_anneal(instance, start, REFERENCE_MULTIPLE, rng, pilot=pilot)
    if form == "step":
        given = float(setting.removeprefix("h="))
        return run_anneal(instance, start, REFERENCE_MULTIPLE, rng, step=given)
    msg = f"no arm {arm!r}"
    raise ValueError(msg)


def stress_arms() -> tuple[str, ...]:
    """The stress arms: baselines, (b)'s two sweeps, (a)'s pilots and given steps."""
    return (
        "restart",
        "k-means++",
        *(f"anneal T0={m:g}" for m in MULTIPLES),
        *(f"sample T={m:g}" for m in MULTIPLES),
        *(f"pilot p={p}" for p in PILOT_PROPOSALS[1:]),
        *(f"step h={h:g}" for h in STEP_GRID),
    )


def measure(
    arms: tuple[str, ...],
    n_starts: int,
    *,
    variant: str | None = None,
    tier: Scale = Scale.STRESS,
    workers: int = WORKERS,
) -> dict[str, list[Result]]:
    """Every arm from ``n_starts`` shared starts, in parallel over (start, arm)."""
    tasks: list[Task] = [(variant, tier, i, a) for i in range(n_starts) for a in arms]
    results = map_tasks(
        run,
        tasks,
        workers=workers,
        pool="processes" if workers > 1 else "serial",
        intra_op_threads=1,
    )
    table: dict[str, list[Result]] = {arm: [] for arm in arms}
    for (_, _, _, arm), result in zip(tasks, results, strict=True):
        table[arm].append(result)
    return table


def _rare(results: list[Result], levels: range) -> str:
    """Runs recovering each level of ``levels`` present in the cell, as ``hits/runs``."""
    cells = []
    for level in levels:
        marks = [r.recovered[level] for r in results if r.recovered[level] >= 0]
        if marks:
            cells.append(f"L{level} {sum(marks)}/{len(marks)}")
    return ", ".join(cells)


def report(table: dict[str, list[Result]]) -> None:
    """One row per arm: missed, gap to the best reached, passes, step, rare-level recovery."""
    best = max(r.log_likelihood for rs in table.values() for r in rs)
    first = next(iter(table.values()))[0]
    print(f"best reached {best:.1f}; scale at start 0 {first.scale:.3f}")
    print(
        "| arm | missed median (range) | gap to best, median | tuner + grads + BW "
        "| step median | 1% levels | 0.2% levels | s/run |"
    )
    for arm, rs in table.items():
        miss = np.array([r.missed for r in rs]) * 100
        gap = best - np.array([r.log_likelihood for r in rs])
        n_levels = len(rs[0].recovered)
        print(
            f"| {arm} | {np.median(miss):.1f}% ({miss.min():.1f}-{miss.max():.1f}) "
            f"| {np.median(gap):.1f} | {int(np.median([r.tuner for r in rs]))}+"
            f"{int(np.median([r.gradients - r.tuner for r in rs]))}+"
            f"{int(np.median([r.polish_iterations for r in rs]))} "
            f"| {np.median([r.step for r in rs]):.2g} "
            f"| {_rare(rs, range(1, 3))} | {_rare(rs, range(3, n_levels))} "
            f"| {np.mean([r.seconds for r in rs]):.1f} |",
            flush=True,
        )
        print("   steps:", " ".join(f"{r.step:.0e}" for r in rs))
        print("   missed%:", " ".join(f"{100 * r.missed:.1f}" for r in rs))


def settled(table: dict[str, list[Result]]) -> list[int]:
    """Per start, the smallest pilot whose choice equals the largest pilot's, in proposals per candidate."""
    pilots = [
        table[
            f"anneal T0={REFERENCE_MULTIPLE:g}"
            if p == PILOT_PROPOSALS[0]
            else f"pilot p={p}"
        ]
        for p in PILOT_PROPOSALS
    ]
    out = []
    for i in range(len(pilots[0])):
        steps = [rs[i].step for rs in pilots]
        k = len(steps) - 1
        while k > 0 and steps[k - 1] == steps[-1]:
            k -= 1
        out.append(PILOT_PROPOSALS[k])
    return out


def main(argv: list[str]) -> None:
    """``stress`` runs every stress arm; ``variants <multiple>`` the variants at that multiple."""
    torch.set_num_threads(1)
    clock = time.perf_counter()
    if argv[0] == "stress":
        table = measure(stress_arms(), STRESS_STARTS)
        report(table)
        print("settled at proposals per candidate:", settled(table))
    else:
        multiple = float(argv[1])
        arms = (
            "restart",
            "k-means++",
            f"anneal T0={multiple:g}",
            f"sample T={multiple:g}",
        )
        for variant in VARIANTS:
            print(f"## {variant}")
            report(measure(arms, VARIANT_STARTS, variant=variant))
    print(f"wall {time.perf_counter() - clock:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
