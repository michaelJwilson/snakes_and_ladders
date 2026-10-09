"""Two solver studies on one stream of problems, through sal's held Rust solvers (issue #1414).

A stream realization carries two problems: a Potts labelling (a per-site field
over labels, a directed adjacency and a coupling scale, a planted labelling)
and a count-pair HMM (counts and successes over exposures and trials, phased
or not, a planted state per row). This module runs sal's solvers on each,
every arm from the same starts and the same seeds, and renders one figure of
two panels and a table.

**The problems.** :func:`from_captured` reads them from
:class:`sal.external.stream.Captured`, a downstream package's own stage
functions run on one realization of its stream; :func:`from_fixtures` builds
them from sal's seeded fixtures (``potts_labelling`` and
``count_hmm_reference``), the source where the downstream package is absent
and the one the CI test pins.

**(a) Labelling.** One :class:`~sal.search.potts_problem.PottsProblem` per
realization, and one held over the directed rows for the downstream path.
:data:`POTTS_ARMS`: the field's argmax; ICM in index order; the downstream
path (worklist descent, the guarded uniform floor, the halved
adjacent-pair merge); ICM with sal's floor; alpha-expansion then ICM; TRW-S's
decode; the anneal under three move sets (single-site heat bath, Wolff then
a Gibbs sweep, Swendsen-Wang then a Gibbs sweep) at ``T0 = J d / 3``
(experiment 032), exponential to ``T0 / 40``, :attr:`Settings.sweeps` sweeps
of site visits; and the whole-label merge alone. Every arm's labelling is
then polished by ICM and the full-gain merge, which is
:attr:`~sal.sample.schedule.Polish.ICM_MERGE`'s stage. A point is the gap to
TRW-S's bound, before and after the polish.

**(b) HMM starts.** :data:`HMM_ARMS` place ``K`` means and success rates: the
run's own start, rows drawn uniformly (restarts), k-means++, per-channel
quantiles, and two Hamiltonian chains on the unphased objective over means
and rates (:func:`~sal.sample.hmc.anneal` from ``10 |log L| / n`` and
:func:`~sal.sample.hmc.sample` at ``|log L| / n``, experiment 038's settings).
Each start is polished by :class:`~sal.opt.hmm.CountPairHmm` or
:class:`~sal.opt.hmm.PhasedCountPairHmm` under the downstream fit's
semantics (:data:`DOWNSTREAM_FIT`), then run on to convergence
(:data:`CONVERGED_FIT`). A point is the log-likelihood below the best any run
reached on its realization, at the fit's own objective.

**Summaries.** Per arm, each realization's median over its starts, then the
median and interquartile range over realizations. Seconds are wall time on
the host the run states.

Run as ``python -m sal.qa.combined_solvers [--source stream|fixtures]``;
``--check TABLE`` compares the untimed table with ``TABLE`` and writes
nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sal.cost import Cost
from sal.emissions import CountPairEmission, RateConcentrationCountPairEmission
from sal.opt.budget import Budget
from sal.opt.em import EmConfig
from sal.opt.hmm import (
    CountPairHmm,
    EmissionHmmObjective,
    MStepSolver,
    PhasedCountPairHmm,
)
from sal.opt.initialize import quantile_locations
from sal.opt.mixture import kmeans_plus_plus
from sal.opt.objective import Restricted, coordinates
from sal.ragged import Ragged
from sal.sample.chain import Adaptation
from sal.sample.potts_mcmc import PottsMove
from sal.sample.schedule import ExponentialTempSchedule, InverseLinearTempSchedule
from sal.sample.tune import Criterion, StepTuning
from sal.search.icm import FloorPolicy, SweepOrder
from sal.search.potts_problem import FloorAt, MergeGain, MergePairs, PottsProblem
from sal.search.spatio_sequential import label_accuracy
from sal.sim.graph import PottsGraph

#: The floor over the sites: the downstream configuration's 50 at 3,000.
FLOOR_SHARE = 50 / 3000
#: ``T0`` in units of ``J d``, the mean coupling per site (experiment 032).
T0_PER_COUPLING = 1.0 / 3.0
#: The anneal ends at ``T0`` over this.
T_END_RATIO = 40.0
#: Every seed derives from this, the realization and the start.
SEED = 1414

#: The downstream fit's starting dispersion and concentration.
START_DISPERSION = 2.0
START_CONCENTRATION = 1000.0
#: The downstream fit: 12 E steps, 2 L-BFGS iterations each, pairs carried.
DOWNSTREAM_FIT = EmConfig(max_iterations=12, tolerance=0.0)
DOWNSTREAM_INNER = 2
#: The second polish, from the first's parameters: L-BFGS to a relative
#: change of 1e-9, under the downstream configuration's declared ceiling on
#: the concentration, which its own optimizer is never handed.
CONVERGED_FIT = EmConfig(max_iterations=200, tolerance=1e-9)
MAX_CONCENTRATION = 5000.0

#: The Hamiltonian starts, experiment 038's: leapfrog steps per proposal,
#: the anneal's proposals and step grid, the pilot's proposals per step, the
#: anneal's and the sampler's temperatures in units of ``|log L| / n``, the
#: sampler's draws, step and adaptation.
N_LEAPFROG = 8
ANNEAL_PROPOSALS = 16
STEP_GRID = (3.0e-3, 1.0e-2, 3.0e-2)
PILOT_PROPOSALS = 2
ANNEAL_MULTIPLE = 10.0
SAMPLE_MULTIPLE = 1.0
DRAWS = 13
SAMPLE_STEP = 1.0e-2
ADAPTATION = Adaptation(warmup=8, target_acceptance=0.65, step_jitter=0.4)


@dataclass(frozen=True)
class Settings:
    """The cost knobs: anneal sweeps, the stochastic starts per arm, and whether the chains run."""

    sweeps: int = 4000
    potts_starts: int = 3
    hmm_seeds: int = 2
    chains: bool = True


#: The settings the stream study runs at, and the CI test's.
STREAM = Settings()
CI = Settings(sweeps=200, potts_starts=2, hmm_seeds=1, chains=False)


@dataclass(frozen=True)
class LabellingProblem:
    """A Potts labelling: field ``(sites, labels)`` as a log-weight, directed CSR rows scaled by ``scale``."""

    field: np.ndarray
    planted: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    scale: float

    @property
    def n_sites(self) -> int:
        """Sites."""
        return int(self.field.shape[0])

    @property
    def n_labels(self) -> int:
        """Labels."""
        return int(self.field.shape[1])


@dataclass(frozen=True)
class CountPairProblem:
    """A count-pair HMM fit: rows, segments, the chain's stay, a start and the planted states.

    ``log_switch`` is each row's log phase-switch probability, read only
    where ``phased``; ``max_rdr`` masks a row whose count over its exposure
    exceeds it (its exposure set to zero, the count unobserved).
    """

    counts: np.ndarray
    successes: np.ndarray
    exposure: np.ndarray
    trials: np.ndarray
    lengths: tuple[int, ...]
    n_states: int
    stay: float
    start_mean: np.ndarray
    start_rate: np.ndarray
    truth: np.ndarray
    phased: bool
    log_switch: np.ndarray
    max_rdr: float


@dataclass(frozen=True)
class Realization:
    """One realization's two problems and where they came from."""

    name: str
    labelling: LabellingProblem
    hmm: CountPairProblem


def from_captured(captured: Any) -> Realization:
    """The two problems of a :class:`sal.external.stream.Captured` realization."""
    labelling = LabellingProblem(
        np.asarray(captured["potts_field"], dtype=np.float64),
        np.asarray(captured["potts_planted"], dtype=np.int64),
        np.asarray(captured["potts_indptr"], dtype=np.int64),
        np.asarray(captured["potts_indices"], dtype=np.int64),
        np.asarray(captured["potts_weights"], dtype=np.float64),
        float(captured["potts_spatial_weight"]),
    )
    hmm = CountPairProblem(
        counts=np.asarray(captured["hmm_counts"], dtype=np.int64),
        successes=np.asarray(captured["hmm_successes"], dtype=np.int64),
        exposure=np.asarray(captured["hmm_exposure"], dtype=np.float64),
        trials=np.asarray(captured["hmm_trials"], dtype=np.float64),
        lengths=tuple(int(n) for n in captured["hmm_lengths"]),
        n_states=int(captured["hmm_n_states"]),
        stay=float(captured["hmm_t"]),
        start_mean=np.exp(np.asarray(captured["hmm_init_log_mu"]).ravel()),
        start_rate=np.asarray(captured["hmm_init_p_binom"]).ravel(),
        truth=np.asarray(captured["hmm_truth"], dtype=np.int64),
        phased=bool(captured["hmm_phased"]),
        log_switch=np.asarray(captured["hmm_log_switch"], dtype=np.float64),
        max_rdr=float(captured["hmm_max_rdr"]),
    )
    return Realization(f"r{captured.realization}", labelling, hmm)


def from_fixtures(tier: str = "ci") -> Realization:
    """The ``potts_labelling`` and ``count_hmm_reference`` cells at ``tier``, as one realization.

    The labelling's adjacency is its graph's symmetric rows at unit scale; the
    HMM is unphased, unmasked, and starts at
    :func:`sal.qa.hmm_fit_semantics.start`.
    """
    from sal.qa.hmm_fit_semantics import start
    from sal.sim.fixtures import fixture

    cell = fixture("potts_labelling", tier).params.instance()
    offsets, neighbours, couplings = cell.graph.compressed_adjacency()
    labelling = LabellingProblem(
        np.asarray(cell.field, dtype=np.float64),
        np.asarray(cell.planted, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(neighbours, dtype=np.int64),
        np.asarray(couplings, dtype=np.float64),
        1.0,
    )
    params = fixture("count_hmm_reference", tier).params
    counts = params.instance()
    _, _, family = start(counts, params.n_fit_states, params.stay)
    hmm = CountPairProblem(
        counts=np.asarray(counts.observations[:, 0], dtype=np.int64),
        successes=np.asarray(counts.observations[:, 1], dtype=np.int64),
        exposure=np.asarray(counts.covariate[:, 0], dtype=np.float64),
        trials=np.asarray(counts.covariate[:, 1], dtype=np.float64),
        lengths=tuple(int(n) for n in counts.lengths),
        n_states=params.n_fit_states,
        stay=float(params.stay),
        start_mean=family.total.mean.numpy(),
        start_rate=family.rate.numpy(),
        truth=np.asarray(counts.states, dtype=np.int64),
        phased=False,
        log_switch=np.zeros(counts.states.size),
        max_rdr=float("inf"),
    )
    return Realization(f"fixtures-{tier}", labelling, hmm)


@dataclass(frozen=True)
class Run:
    """One arm from one start: before and after its polish.

    ``value`` is the gap to the bound (a) or the log-likelihood (b);
    ``missed`` the share of sites or rows unlike the planted ones under the
    best renaming, in percent; ``seconds`` the wall time to that point.
    """

    panel: str
    realization: str
    arm: str
    start: int
    stage: str
    value: float
    missed: float
    seconds: float


# --------------------------------------------------------------------------
# (a) Labelling
# --------------------------------------------------------------------------


@dataclass
class _Held:
    """The labelling held twice, its bound, its floor and the anneal's schedule."""

    problem: LabellingProblem
    graph: PottsGraph = field(init=False)
    held: PottsProblem = field(init=False)
    directed: PottsProblem = field(init=False)
    floor: int = field(init=False)
    bound: float = field(init=False)
    decoded: np.ndarray = field(init=False)
    trws_seconds: float = field(init=False)
    visits: int = field(init=False)
    t0: float = field(init=False)

    def __post_init__(self) -> None:
        p = self.problem
        self.graph = PottsGraph.from_directed_csr(
            p.indptr, p.indices, p.weights, scale=p.scale
        )
        self.held = PottsProblem(self.graph, p.field)
        self.directed = PottsProblem.from_directed_csr(
            p.indptr, p.indices, p.weights, p.field, scale=p.scale
        )
        self.floor = max(1, round(FLOOR_SHARE * p.n_sites))
        clock = time.perf_counter()
        bounded = self.held.trws()
        self.trws_seconds = time.perf_counter() - clock
        self.bound = float(bounded.bound)
        self.decoded = np.asarray(bounded.labelling)
        self.visits = self.graph.n_nodes + 2 * len(self.graph.edges)
        _, _, couplings = self.graph.endpoints
        coupling = float(np.sum(couplings)) * 2.0 / self.graph.n_nodes
        self.t0 = T0_PER_COUPLING * coupling


def _anneal(
    move: PottsMove,
) -> Callable[[_Held, np.ndarray, np.random.Generator, Settings], np.ndarray]:
    def solve(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
        schedule = ExponentialTempSchedule(
            work.t0, work.t0 / T_END_RATIO, settings.sweeps
        )
        run = work.held.anneal(
            schedule,
            rng,
            start=start,
            move=move,
            budget=Budget(Cost.SITE_VISITS, settings.sweeps * work.visits),
        )
        return np.asarray(run.labelling)

    return solve


def _downstream(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
    del settings
    descended = work.directed.icm(
        rng,
        start=start,
        order=SweepOrder.WORKLIST,
        min_sites=work.floor,
        policy=FloorPolicy.UNIFORM,
        floor_at=FloorAt.EPOCH_GUARDED,
    )
    merged = work.directed.merge(
        np.asarray(descended.labelling),
        gain=MergeGain.HALVED,
        pairs=MergePairs.ADJACENT,
    )
    return np.asarray(merged.labelling)


def _argmax(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
    del start, rng, settings
    return np.asarray(work.held.argmax().labelling)


def _icm(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
    del settings
    return np.asarray(work.held.icm(rng, start=start).labelling)


def _floored(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
    del settings
    return np.asarray(work.held.icm(rng, start=start, min_sites=work.floor).labelling)


def _expansion(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
    del rng, settings
    return np.asarray(work.held.alpha_expansion(start=start, then_icm=True).labelling)


def _decoded(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
    del start, rng, settings
    return work.decoded


def _merged(work: _Held, start: np.ndarray, rng: np.random.Generator, settings: Settings) -> np.ndarray:  # fmt: skip
    del rng, settings
    return np.asarray(work.held.merge(start).labelling)


#: Every labelling arm by name: ``(held, start, rng, settings)`` to a labelling.
POTTS_ARMS: dict[
    str, Callable[[_Held, np.ndarray, np.random.Generator, Settings], np.ndarray]
] = {
    "argmax": _argmax,
    "icm": _icm,
    "downstream icm+floor+merge": _downstream,
    "icm+floor": _floored,
    "ae+icm": _expansion,
    "trws": _decoded,
    "anneal heat bath": _anneal(PottsMove.SINGLE_SITE),
    "anneal wolff+gibbs": _anneal(PottsMove.WOLFF),
    "anneal sw+gibbs": _anneal(PottsMove.SWENDSEN_WANG),
    "merge": _merged,
}

#: Arms whose output does not read the start: run once per realization.
START_FREE = frozenset({"argmax", "trws"})


def polish(work: _Held, labelling: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ICM in index order, then the full-gain merge: :attr:`~sal.sample.schedule.Polish.ICM_MERGE`'s stage."""
    descended = work.held.icm(rng, start=labelling)
    return np.asarray(work.held.merge(np.asarray(descended.labelling)).labelling)


def potts_starts(work: _Held, realization: int, settings: Settings) -> list[np.ndarray]:
    """The field's argmax, then uniform labellings, one per further start."""
    p = work.problem
    starts = [np.argmax(p.field, axis=1).astype(np.int64)]
    for index in range(1, settings.potts_starts):
        rng = np.random.default_rng([SEED, realization, index, 0])
        starts.append(rng.integers(0, p.n_labels, p.n_sites))
    return starts


def potts_runs(
    found: Realization, index: int, settings: Settings, arms: Sequence[str] = tuple(POTTS_ARMS)
) -> list[Run]:  # fmt: skip
    """Every labelling arm from every start, before and after the polish."""
    work = _Held(found.labelling)
    p = work.problem
    runs = []

    def unlike(labelling: np.ndarray) -> float:
        return 100.0 * (1.0 - label_accuracy(labelling, p.planted, p.n_labels))

    for start_index, start in enumerate(potts_starts(work, index, settings)):
        for arm in arms:
            if arm in START_FREE and start_index > 0:
                continue
            rng = np.random.default_rng([SEED, index, start_index, 1])
            clock = time.perf_counter()
            raw = POTTS_ARMS[arm](work, start.copy(), rng, settings)
            seconds = time.perf_counter() - clock
            if arm == "trws":
                seconds = work.trws_seconds
            clock = time.perf_counter()
            polished = polish(work, raw, rng)
            after = seconds + time.perf_counter() - clock
            for stage, labelling, wall in (("before", raw, seconds), ("after", polished, after)):  # fmt: skip
                runs.append(
                    Run(
                        "a",
                        found.name,
                        arm,
                        start_index,
                        stage,
                        work.held.energy(labelling) - work.bound,
                        unlike(labelling),
                        wall,
                    )  # fmt: skip
                )
    return runs


# --------------------------------------------------------------------------
# (b) HMM starts
# --------------------------------------------------------------------------


def _masked_exposure(p: CountPairProblem) -> np.ndarray:
    """The exposure the fit reads: zero above ``max_rdr``, rounded where the model is phased."""
    exposure = p.exposure.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p.counts / exposure
    exposure[np.nan_to_num(ratio, nan=0.0) > p.max_rdr] = 0.0
    # The downstream's coded phased emission reads integer exposures.
    return np.round(exposure) if p.phased else exposure


def _features(p: CountPairProblem) -> np.ndarray:
    """``(rows, 2)`` depth ratio and success share over the rows both are observed on."""
    kept = (p.exposure > 0) & (p.trials > 0)
    return np.stack(
        [p.counts[kept] / p.exposure[kept], p.successes[kept] / p.trials[kept]], axis=1
    )


def _clipped(mean: np.ndarray, rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(mean, 0.05, None), np.clip(rate, 0.02, 0.98)


def _model(p: CountPairProblem, mean: np.ndarray, rate: np.ndarray) -> CountPairHmm | PhasedCountPairHmm:  # fmt: skip
    """The held fit at a start, under the downstream semantics: tied, chain held, masked exposures."""
    k = p.n_states
    observations = Ragged(
        np.ascontiguousarray(np.stack([p.counts, p.successes], 1)), p.lengths
    )
    covariate = Ragged(
        np.ascontiguousarray(np.stack([_masked_exposure(p), p.trials], 1)), p.lengths
    )
    family = CountPairEmission(
        np.full(k, START_DISPERSION),
        mean,
        rate * START_CONCENTRATION,
        (1.0 - rate) * START_CONCENTRATION,
        np.ones(k, dtype=np.int64),
        joint=False,
    )
    transition = np.full((k, k), (1.0 - p.stay) / (k - 1))
    np.fill_diagonal(transition, p.stay)
    log_transition = np.log(transition)
    if p.phased:
        # The switch at row t - 1 enters the step into row t.
        switch = np.concatenate([[0.5], np.exp(p.log_switch)[:-1]])
        return PhasedCountPairHmm(
            observations, np.full(2 * k, -np.log(2 * k)), log_transition, family,
            covariate=covariate, switch=switch, tied=True, fit_initial=False,
            fit_transition=False,
        )  # fmt: skip
    return CountPairHmm(
        observations, np.full(k, -np.log(k)), log_transition, family,
        covariate=covariate, tied=True, fit_initial=False, fit_transition=False,
    )  # fmt: skip


def missed(p: CountPairProblem, model: CountPairHmm | PhasedCountPairHmm) -> float:
    """Rows whose posterior-argmax copy state is not the planted state under the best matching, in percent."""
    label = model.posteriors().log_posterior.argmax(axis=1)
    if p.phased:
        label = label // 2
    truth = p.truth
    confusion = np.zeros((p.n_states, int(truth.max()) + 1), dtype=np.int64)
    np.add.at(confusion, (label, truth), 1)
    rows, cols = linear_sum_assignment(-confusion)
    return 100.0 * float(truth.size - confusion[rows, cols].sum()) / truth.size


class _Chains:
    """The unphased objective over means and rates the Hamiltonian starts move on."""

    def __init__(self, p: CountPairProblem) -> None:
        k = p.n_states
        self.problem = p
        family = RateConcentrationCountPairEmission(
            np.full(k, START_DISPERSION),
            np.full(k, float(np.median(_features(p)[:, 0]))),
            np.full(k, 0.5),
            np.full(k, START_CONCENTRATION),
            np.full(k, max(1, round(float(np.median(p.trials))))),
            joint=False,
        )
        exposure = _masked_exposure(replace(p, phased=False))
        self.objective = EmissionHmmObjective(
            Ragged(
                np.ascontiguousarray(np.stack([p.counts, p.successes], 1)), p.lengths
            ),
            family,
            covariate=np.ascontiguousarray(np.stack([exposure, p.trials], 1)),
        )
        self.varied = coordinates(self.objective, ["mean", "rate"])

    def theta(self, mean: np.ndarray, rate: np.ndarray) -> torch.Tensor:
        p = self.problem
        k = p.n_states
        transition = np.full((k, k), (1.0 - p.stay) / (k - 1))
        np.fill_diagonal(transition, p.stay)
        family = RateConcentrationCountPairEmission(
            np.full(k, START_DISPERSION), mean, rate, np.full(k, START_CONCENTRATION),
            np.full(k, max(1, round(float(np.median(p.trials))))), joint=False,
        )  # fmt: skip
        return self.objective.theta_from_truth(
            np.full(k, 1.0 / k), transition, **family.named_parameters()
        )

    def scale(self, theta: torch.Tensor) -> float:
        with torch.no_grad():
            value = float(self.objective(theta))
        return abs(value) / int(self.problem.counts.size)

    def read(
        self, at: torch.Tensor, reduced: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        full = at.clone()
        full[self.varied] = reduced.detach()
        with torch.no_grad():
            named = self.objective.constrain(full)
        return named["mean"].numpy().copy(), named["rate"].numpy().copy()


def _run_start(
    p: CountPairProblem, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    del rng
    return p.start_mean, p.start_rate


def _restart(
    p: CountPairProblem, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    rows = _features(p)
    picked = rows[rng.choice(rows.shape[0], p.n_states, replace=False)]
    return picked[:, 0], picked[:, 1]


def _kmeans(
    p: CountPairProblem, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    rows = _features(p)
    centre, spread = rows.mean(axis=0), rows.std(axis=0)
    centres = (
        kmeans_plus_plus((rows - centre) / spread, p.n_states, rng) * spread + centre
    )
    return centres[:, 0], centres[:, 1]


def _quantile(
    p: CountPairProblem, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    del rng
    placed = quantile_locations(
        torch.as_tensor(_features(p)), p.n_states, dim=0
    ).numpy()
    return placed[:, 0], placed[:, 1]


def _annealed(
    p: CountPairProblem, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    chains = _Chains(p)
    at = chains.theta(*_clipped(*_restart(p, rng)))
    restricted = Restricted(chains.objective, at, chains.varied)
    from sal.sample.hmc import anneal

    run = anneal(
        restricted,
        InverseLinearTempSchedule(
            ANNEAL_MULTIPLE * chains.scale(at), 1.0, ANNEAL_PROPOSALS
        ),
        rng,
        step_size="auto",
        tuning=StepTuning(
            Budget(Cost.GRADIENTS, 1 + len(STEP_GRID) * PILOT_PROPOSALS * N_LEAPFROG),
            Criterion.LOWEST_ENERGY,
            STEP_GRID,
        ),
        n_steps=N_LEAPFROG,
        start=restricted.initial(),
    )
    return chains.read(at, run.best)


def _sampled(
    p: CountPairProblem, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    chains = _Chains(p)
    at = chains.theta(*_clipped(*_restart(p, rng)))
    restricted = Restricted(chains.objective, at, chains.varied)
    from sal.sample.hmc import sample

    chain = sample(
        restricted,
        rng,
        DRAWS,
        step_size=SAMPLE_STEP,
        n_steps=N_LEAPFROG,
        start=restricted.initial(),
        temperature=SAMPLE_MULTIPLE * chains.scale(at),
        adaptation=ADAPTATION,
    )
    with torch.no_grad():
        values = [float(restricted(draw)) for draw in chain.draws]
    return chains.read(at, chain.draws[int(np.argmin(values))])


#: Every start by name: ``(problem, rng)`` to ``K`` means and rates.
HMM_ARMS: dict[
    str,
    Callable[[CountPairProblem, np.random.Generator], tuple[np.ndarray, np.ndarray]],
] = {  # fmt: skip
    "run start": _run_start,
    "restart": _restart,
    "k-means++": _kmeans,
    "quantile": _quantile,
    "hmc anneal": _annealed,
    "hmc sample": _sampled,
}
#: Starts that draw nothing: run once per realization.
DETERMINISTIC = frozenset({"run start", "quantile"})
#: Starts that run the Hamiltonian chains.
CHAINS = frozenset({"hmc anneal", "hmc sample"})


def hmm_runs(
    found: Realization, index: int, settings: Settings, arms: Sequence[str] = tuple(HMM_ARMS)
) -> list[Run]:  # fmt: skip
    """Every start, polished by the downstream fit and then to convergence; log-likelihoods, not yet gaps."""
    p = found.hmm
    torch.manual_seed(SEED)
    runs = []
    for arm in arms:
        if arm in CHAINS and not settings.chains:
            continue
        seeds = 1 if arm in DETERMINISTIC else settings.hmm_seeds
        for seed in range(seeds):
            rng = np.random.default_rng([SEED, index, seed, 2])
            clock = time.perf_counter()
            mean, rate = _clipped(*HMM_ARMS[arm](p, rng))
            model = _model(p, mean, rate)
            started = time.perf_counter() - clock
            runs.append(Run("b", found.name, arm, seed, "start",
                            model.log_likelihood(), missed(p, model), started))  # fmt: skip
            clock = time.perf_counter()
            model.fit(DOWNSTREAM_FIT, solver=MStepSolver.LBFGS,
                      inner_iterations=DOWNSTREAM_INNER, carry=True)  # fmt: skip
            first = started + time.perf_counter() - clock
            runs.append(Run("b", found.name, arm, seed, "downstream fit",
                            model.log_likelihood(), missed(p, model), first))  # fmt: skip
            clock = time.perf_counter()
            model.fit(CONVERGED_FIT, solver=MStepSolver.LBFGS,
                      max_concentration=MAX_CONCENTRATION)  # fmt: skip
            second = first + time.perf_counter() - clock
            runs.append(Run("b", found.name, arm, seed, "converged",
                            model.log_likelihood(), missed(p, model), second))  # fmt: skip
    best = max(r.value for r in runs)
    return [replace(r, value=best - r.value) for r in runs]


def study(found: Realization, index: int, settings: Settings) -> list[Run]:
    """Both panels on one realization."""
    return potts_runs(found, index, settings) + hmm_runs(found, index, settings)


# --------------------------------------------------------------------------
# Summary, table, figure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """One arm at one stage: median and quartiles over realizations of each realization's median."""

    panel: str
    arm: str
    stage: str
    realizations: int
    value: tuple[float, float, float]
    missed: tuple[float, float, float]
    seconds: tuple[float, float, float]


def summarize(runs: Sequence[Run]) -> list[Summary]:
    """Per arm and stage, in first-seen order."""
    keys: dict[tuple[str, str, str], dict[str, list[Run]]] = {}
    for run in runs:
        keys.setdefault((run.panel, run.arm, run.stage), {}).setdefault(run.realization, []).append(run)  # fmt: skip

    def spread(values: list[float]) -> tuple[float, float, float]:
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        return float(q2), float(q1), float(q3)

    out = []
    for (panel, arm, stage), by in keys.items():
        medians = [
            (
                float(np.median([r.value for r in rs])),
                float(np.median([r.missed for r in rs])),
                float(np.median([r.seconds for r in rs])),
            )
            for rs in by.values()
        ]
        columns = list(zip(*medians, strict=True))
        out.append(
            Summary(
                panel,
                arm,
                stage,
                len(by),
                spread(list(columns[0])),
                spread(list(columns[1])),
                spread(list(columns[2])),
            )  # fmt: skip
        )
    return out


def _shown(spread: tuple[float, float, float], form: str) -> str:
    """``median [q1, q3]``, a negative zero printed as zero."""
    median, low, high = (format(v, form).replace("-0.00", "0.00") for v in spread)
    return f"{median} [{low}, {high}]"


def table(summaries: Sequence[Summary], *, timed: bool = True) -> str:
    """The summaries as markdown; ``timed=False`` drops the wall columns, which ``--check`` compares."""
    head = "| panel | arm | stage | n | gap, median [IQR] | missed %, median [IQR] |"
    rule = "| --- | --- | --- | --- | --- | --- |"
    if timed:
        head += " seconds, median [IQR] |"
        rule += " --- |"
    lines = [head, rule]
    for s in summaries:
        row = (
            f"| {s.panel} | {s.arm} | {s.stage} | {s.realizations} | "
            f"{_shown(s.value, '.2f')} | {_shown(s.missed, '.1f')} |"
        )
        if timed:
            row += f" {_shown(s.seconds, '.3g')} |"
        lines.append(row)
    return "\n".join(lines) + "\n"


#: Ten distinguishable hues (matplotlib's tab10), one per arm; markers repeat every eight.
COLOURS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)  # fmt: skip


def figure(summaries: Sequence[Summary], title: str) -> Any:
    """The two panels: gap against wall time, median and IQR, a point per arm and stage."""
    import matplotlib.pyplot as plt

    from sal.qa.style import INK_MUTED, letter_style

    stages = {"a": ("before", "after"), "b": ("downstream fit", "converged")}
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.2), constrained_layout=True)
        for ax, panel in zip(axes, ("a", "b"), strict=True):
            arms = list(dict.fromkeys(s.arm for s in summaries if s.panel == panel))
            for i, arm in enumerate(arms):
                colour = COLOURS[i % len(COLOURS)]
                marker = "osD^v<>p"[i % 8]
                points = [
                    next(s for s in summaries if (s.panel, s.arm, s.stage) == (panel, arm, stage))
                    for stage in stages[panel]
                ]  # fmt: skip
                xs = [s.seconds[0] for s in points]
                ys = [s.value[0] for s in points]
                ax.plot(xs, ys, color=colour, lw=0.8, alpha=0.6)
                for j, s in enumerate(points):
                    ax.errorbar(
                        s.seconds[0], s.value[0],
                        xerr=[[s.seconds[0] - s.seconds[1]], [s.seconds[2] - s.seconds[0]]],
                        yerr=[[s.value[0] - s.value[1]], [s.value[2] - s.value[0]]],
                        fmt=marker, color=colour, mfc=colour if j else "white",
                        ms=6, lw=0.7, capsize=0,
                        label=f"{arm} ({points[0].missed[0]:.1f}% → {points[1].missed[0]:.1f}%)"
                        if j else None,
                    )  # fmt: skip
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("wall time, s (log)")
            ax.grid(True, which="major", color="#e5e5e5", lw=0.5)
            ax.set_ylim(bottom=-0.05)
            ax.legend(
                fontsize=6.5,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=2,
                title="arm (missed %: open \u2192 filled)",
                title_fontsize=7,
            )
        axes[0].set_ylabel("nats above the TRW-S bound (symlog)")
        axes[0].set_title(
            "(a) labelling: open before polish, filled after ICM + merge", fontsize=8
        )
        axes[1].set_ylabel("log L below the best run (symlog)")
        axes[1].set_title(
            "(b) HMM starts: open downstream fit, filled converged", fontsize=8
        )
        fig.suptitle(title, fontsize=8, color=INK_MUTED)
    return fig


#: One realization's work: the source, the realization, the capture cache, the runs' directory, the settings.
Task = tuple[str, int, str, str, Settings]


def _run_one(task: Task) -> list[dict[str, Any]]:
    """One realization's runs; a stream realization's are kept in ``runs`` and read back from there."""
    source, realization, cache, runs, settings = task
    torch.set_num_threads(1)
    kept = Path(runs) / f"{source}-r{realization:03d}.json"
    if source == "stream" and kept.is_file():
        found_runs: list[dict[str, Any]] = json.loads(kept.read_text())
        return found_runs
    if source == "stream":
        from sal.external.protocol import load
        from sal.external.stream import MANIFEST, Captured, cached

        found = from_captured(
            Captured(load(cached(Path(cache), MANIFEST, realization)))
        )
    else:
        found = from_fixtures(source.removeprefix("fixtures-"))
    done = [asdict(r) for r in study(found, realization, settings)]
    if source == "stream":
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(done))
    return done


def main(argv: list[str] | None = None) -> int:
    """Run the study, write the PNG and the table; or with ``--check``, compare the untimed table."""
    parser = argparse.ArgumentParser(prog="python -m sal.qa.combined_solvers")
    parser.add_argument("--source", default="fixtures-ci",
                        help="'stream' (the cached captures) or 'fixtures-<tier>'")  # fmt: skip
    parser.add_argument(
        "--realizations", default="5:30", help="first:stop, for the stream"
    )
    parser.add_argument("--cache", default=".cache/stream")
    parser.add_argument("--out", default=".cache/combined_solvers")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--check", type=Path, default=None)
    arguments = parser.parse_args(argv)
    settings = STREAM if arguments.source == "stream" else CI
    out = Path(arguments.out)
    if arguments.source == "stream":
        first, stop = (int(v) for v in arguments.realizations.split(":"))
        tasks: list[Task] = [
            ("stream", k, arguments.cache, str(out / "runs"), settings)
            for k in range(first, stop)
        ]
    else:
        tasks = [(arguments.source, 0, arguments.cache, str(out / "runs"), settings)]
    from sal.parallel import map_tasks

    results = map_tasks(
        _run_one, tasks, workers=arguments.workers,
        pool="processes" if arguments.workers > 1 else "serial", intra_op_threads=1,
    )  # fmt: skip
    runs = [Run(**r) for rs in results for r in rs]
    summaries = summarize(runs)
    if arguments.check is not None:
        expected = arguments.check.read_text()
        got = table(summaries, timed=False)
        if got != expected:
            sys.stdout.write(got)
            return 1
        return 0
    out.mkdir(parents=True, exist_ok=True)
    stem = f"combined_solvers_{arguments.source}"
    (out / f"{stem}.json").write_text(json.dumps([asdict(r) for r in runs]))
    (out / f"{stem}.md").write_text(table(summaries))
    (out / f"{stem}.untimed.md").write_text(table(summaries, timed=False))
    fig = figure(summaries, f"{len(tasks)} realizations, source {arguments.source}")
    fig.savefig(out / f"{stem}.png", dpi=200)
    sys.stdout.write(table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
