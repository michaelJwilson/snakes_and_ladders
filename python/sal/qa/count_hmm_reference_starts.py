"""HMM starts on the canonical count-pair cell, every one polished by Baum-Welch (issue #1393).

#1378's Part B (``docs/experiments/030``) measured `hmc.anneal` as an HMM
start on a seven-state instance of its own (:mod:`sal.qa.count_pair_hmm_anneal`)
that did not match the downstream shape. This module re-measures on the
registry's ``count_hmm_reference`` cell (:mod:`sal.sim.count_hmm_cell`):
8 generating levels fitted by ``K = 7`` states, the dominant level staying
with probability ``1 - 1e-7``, segments of 20, NB totals over an exposure and
beta-binomial successes over per-position trials.

**The sampled coordinates are the downstream caller's.** A sampler varies
only each state's ``log mu`` and ``logit p``
(:class:`~sal.opt.objective.Restricted` over the ``mean`` and ``rate``
blocks of a :class:`~sal.emissions.RateConcentrationCountPairEmission`
start); the chain is held at ``stay = 1 - 1e-7`` with the leaving mass
spread evenly, the initial law uniform, and every dispersion and
concentration at the cell's declared 10 and 100 (equal over levels, so they
say nothing about which level is which). **Baum-Welch then polishes every
start over every parameter**, chain included.

**A start** is ``K`` positions drawn uniformly, read as each state's
``total / exposure`` and ``successes / trials`` (a random restart); the
samplers start there, so each method meets the same ``N_STARTS`` points.
k-means++ (:func:`~sal.opt.mixture.kmeans_plus_plus`) seeds on the same two
features, standardized, and the quantile start places the means at evenly
spaced quantiles of ``total / exposure`` with rates at the pooled rate.

**Budget:** :data:`PASSES` per start, a pass being one gradient, one
Baum-Welch iteration or one scored point. A sampler spends its gradients and
scored draws first; Baum-Welch takes the rest. A restart, k-means++ and the
quantile start give all of it to Baum-Welch.

**Missed** is the Viterbi path at the polished point against the generating
levels, under the best one-to-one matching of the 7 states to the 8 levels
(:func:`~scipy.optimize.linear_sum_assignment` on the ``8 x 7`` confusion):
one level is left unmatched and its positions count as missed. The stress
cell draws no position from level 4, so 0% is attainable.

Run as ``python -m sal.qa.count_hmm_reference_starts``; two threads.
"""

from __future__ import annotations

import time
from collections.abc import Callable
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
from sal.ragged import Ragged
from sal.sample.chain import Adaptation
from sal.sample.hmc import anneal, sample
from sal.sample.schedule import InverseLinearTempSchedule
from sal.sample.tune import Criterion, StepTuning
from sal.sim.count_hmm_cell import CountHmm, CountHmmReferenceParams
from sal.sim.fixtures import baseline, fixture

SEED = 1393
#: Passes per start: gradients, scored draws and Baum-Welch iterations alike.
PASSES = 400
#: Leapfrog steps per proposal, as the downstream sampler takes them.
N_STEPS = 8
#: The downstream fixed-temperature sampler: 8 adaptation proposals, then 13
#: draws, the best of which Baum-Welch polishes.
WARMUP = 8
DRAWS = 13
ADAPTATION = Adaptation(warmup=WARMUP, target_acceptance=0.65, step_jitter=0.4)
#: Where the adaptation's dual averaging starts.
SAMPLE_STEP = 1.0e-2
TEMPERATURES = (10.0, 30.0, 100.0, 300.0)
#: The anneal: a pilot of 2 proposals per candidate step from its start,
#: ranked by the lowest energy, then :data:`ANNEAL_PROPOSALS` on an
#: inverse-linear ramp from ``T0`` to 1; 177 gradients against the fixed
#: sampler's 169 and 13 scored draws.
STEP_GRID = (3.0e-3, 1.0e-2, 3.0e-2)
PILOT = Budget(Cost.GRADIENTS, 1 + len(STEP_GRID) * 2 * N_STEPS)
ANNEAL_PROPOSALS = 16
T0S = (10.0, 100.0, 1000.0)
N_STARTS = 10
#: The downstream forms and the baselines run from every start; the rest of
#: the two sweeps from the first :data:`SWEEP_STARTS`, since a Baum-Welch
#: iteration costs 44 ms at 8,000 positions and the whole is held to ~15 min.
HEADLINE = ("restart", "anneal T0=100", "sample T=100")
SWEEP = ("anneal T0=10", "anneal T0=1000", "sample T=10", "sample T=30", "sample T=300")
SWEEP_STARTS = 4
THREADS = 2


@dataclass(frozen=True)
class Cell:
    """The declared and drawn cell, the full objective, the sampled restriction's indices and the truth's value."""

    params: CountHmmReferenceParams
    cell: CountHmm
    objective: EmissionHmmObjective
    varied: torch.Tensor
    at_truth: float

    @property
    def n_states(self) -> int:
        """``K``, the states a fit is given."""
        return self.objective.n_states


@dataclass(frozen=True)
class Result:
    """One start of one method, after Baum-Welch."""

    log_likelihood: float
    missed: float
    gradients: int
    polish_iterations: int
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


def build(tier: Scale = Scale.STRESS) -> Cell:
    """The registry's cell at ``tier``, its objective at the K = 7 family, and the truth's log-likelihood."""
    params: CountHmmReferenceParams = fixture("count_hmm_reference", tier).params
    cell = params.instance()
    k = params.n_fit_states
    objective = EmissionHmmObjective(
        Ragged(cell.observations, cell.lengths),
        _family(params, np.full(k, params.base_mean), np.full(k, 0.5)),
        covariate=cell.covariate,
    )
    varied = coordinates(objective, ["mean", "rate"])
    at_truth = baseline("count_hmm_reference", tier).value("generating_log_likelihood")
    return Cell(params, cell, objective, varied, float(at_truth))


def _features(cell: CountHmm) -> np.ndarray:
    """Per position: ``total / exposure`` and ``successes / trials``."""
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
    leave = 1.0 - params.stay
    transition = np.full((k, k), leave / (k - 1))
    np.fill_diagonal(transition, params.stay)
    family = _family(
        params,
        np.clip(mean, 0.5, None),
        np.clip(rate, 0.02, 0.98),
    )
    return instance.objective.theta_from_truth(
        np.full(k, 1.0 / k), transition, **family.named_parameters()
    )


def random_start(instance: Cell, rng: np.random.Generator) -> torch.Tensor:
    """``K`` positions drawn uniformly without replacement, read as the states' means and rates."""
    rows = _features(instance.cell)[
        rng.choice(rows_count(instance), instance.n_states, replace=False)
    ]
    return _held(instance, rows[:, 0], rows[:, 1])


def rows_count(instance: Cell) -> int:
    """Positions in the cell."""
    return int(instance.cell.observations.shape[0])


def kmeans_start(instance: Cell, rng: np.random.Generator) -> torch.Tensor:
    """k-means++ centres on the two features, standardized, read back as means and rates."""
    features = _features(instance.cell)
    centre, scale = features.mean(axis=0), features.std(axis=0)
    centres = kmeans_plus_plus((features - centre) / scale, instance.n_states, rng)
    centres = centres * scale + centre
    return _held(instance, centres[:, 0], centres[:, 1])


def quantile_start(instance: Cell) -> torch.Tensor:
    """Means at quantiles 0.05 to 0.95 of ``total / exposure``, every rate the pooled rate."""
    features = _features(instance.cell)
    k = instance.n_states
    mean = np.quantile(features[:, 0], np.linspace(0.05, 0.95, k))
    rate = np.full(
        k, instance.cell.observations[:, 1].sum() / instance.cell.covariate[:, 1].sum()
    )
    return _held(instance, mean, rate)


def missed(instance: Cell, theta: torch.Tensor) -> float:
    """The fraction of positions the Viterbi path misses, 7 states matched one to one onto 8 levels."""
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
    return float(1.0 - confusion[rows, cols].sum() / path.shape[0])


def _polish(
    instance: Cell, theta: torch.Tensor, gradients: int, clock: float
) -> Result:
    polished = polish_by_baum_welch(
        instance.objective, theta, Budget(Cost.ITERATIONS, PASSES - gradients)
    )
    iterations = int(polished.termination.iterations)
    seconds = time.perf_counter() - clock
    return Result(
        -float(polished.value),
        missed(instance, polished.theta),
        gradients,
        iterations,
        seconds,
    )


def _scatter(instance: Cell, at: torch.Tensor, reduced: torch.Tensor) -> torch.Tensor:
    full = at.clone()
    full[instance.varied] = reduced.detach()
    return full


def run_restart(instance: Cell, start: torch.Tensor) -> Result:
    """Baum-Welch for the whole budget from ``start``."""
    return _polish(instance, start, 0, time.perf_counter())


def run_fixed(
    instance: Cell, start: torch.Tensor, temperature: float, rng: np.random.Generator
) -> Result:
    """The downstream form: adapt, draw at ``temperature``, polish the best draw."""
    clock = time.perf_counter()
    restricted = Restricted(instance.objective, start, instance.varied)
    chain = sample(
        restricted,
        rng,
        DRAWS,
        step_size=SAMPLE_STEP,
        n_steps=N_STEPS,
        start=restricted.initial(),
        temperature=temperature,
        adaptation=ADAPTATION,
    )
    with torch.no_grad():
        values = [float(restricted(draw)) for draw in chain.draws]
    best = chain.draws[int(np.argmin(values))]
    return _polish(
        instance, _scatter(instance, start, best), int(chain.spent) + DRAWS, clock
    )


def run_anneal(
    instance: Cell, start: torch.Tensor, t0: float, rng: np.random.Generator
) -> Result:
    """`hmc.anneal` at ``step_size="auto"`` from ``t0`` to 1, then Baum-Welch from its best."""
    clock = time.perf_counter()
    restricted = Restricted(instance.objective, start, instance.varied)
    run = anneal(
        restricted,
        InverseLinearTempSchedule(t0, 1.0, ANNEAL_PROPOSALS),
        rng,
        step_size="auto",
        tuning=StepTuning(PILOT, Criterion.LOWEST_ENERGY, STEP_GRID),
        n_steps=N_STEPS,
        start=restricted.initial(),
    )
    return _polish(instance, _scatter(instance, start, run.best), int(run.spent), clock)


Method = Callable[[Cell, torch.Tensor, np.random.Generator], Result]


def _annealed(t0: float) -> Method:
    return lambda c, s, r: run_anneal(c, s, t0, r)


def _sampled(temperature: float) -> Method:
    return lambda c, s, r: run_fixed(c, s, temperature, r)


def methods() -> dict[str, Method]:
    """Every method that starts from a shared random start, by name."""
    table: dict[str, Method] = {"restart": lambda c, s, _r: run_restart(c, s)}
    for t0 in T0S:
        table[f"anneal T0={t0:g}"] = _annealed(t0)
    for t in TEMPERATURES:
        table[f"sample T={t:g}"] = _sampled(t)
    return table


def measure(
    instance: Cell, n_starts: int, names: tuple[str, ...], *, seeded: bool = True
) -> dict[str, list[Result]]:
    """``names`` from ``n_starts`` shared random starts; with ``seeded``, k-means++ from as many seeds and the quantile start once."""
    table = methods()
    results: dict[str, list[Result]] = {name: [] for name in names}
    for i in range(n_starts):
        start = random_start(instance, np.random.default_rng([SEED, i]))
        for name in names:
            results[name].append(
                table[name](instance, start, np.random.default_rng([SEED, i, 1]))
            )
        if seeded:
            clock = time.perf_counter()
            point = kmeans_start(instance, np.random.default_rng([SEED, i, 2]))
            results.setdefault("k-means++", []).append(
                _polish(instance, point, 0, clock)
            )
    if seeded:
        clock = time.perf_counter()
        results["quantile"] = [_polish(instance, quantile_start(instance), 0, clock)]
    return results


def main() -> None:
    """The table: missed, the gap to the best any run reached and to the truth, passes and wall."""
    torch.set_num_threads(THREADS)
    instance = build()
    results = measure(instance, N_STARTS, HEADLINE)
    results.update(measure(instance, SWEEP_STARTS, SWEEP, seeded=False))
    best = max(r.log_likelihood for rs in results.values() for r in rs)
    print(f"truth {instance.at_truth:.3f}, best reached {best:.3f}")
    print(
        "| method | missed median (min-max) | gap to best, median nats | "
        "gap to truth, median nats | hits within 1 nat | gradients | BW iters | s per start |"
    )
    for name, rs in results.items():
        miss = np.array([r.missed for r in rs]) * 100
        ll = np.array([r.log_likelihood for r in rs])
        print(
            f"| {name} | {np.median(miss):.1f}% ({miss.min():.1f}-{miss.max():.1f}) | "
            f"{np.median(best - ll):.1f} | {np.median(instance.at_truth - ll):.1f} | "
            f"{int((best - ll <= 1.0).sum())}/{len(rs)} | "
            f"{int(np.median([r.gradients for r in rs]))} | "
            f"{int(np.median([r.polish_iterations for r in rs]))} | "
            f"{np.mean([r.seconds for r in rs]):.2f} |",
            flush=True,
        )
        print("  per start missed %:", " ".join(f"{100 * r.missed:.1f}" for r in rs))


if __name__ == "__main__":
    main()
