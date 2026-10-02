"""Sampler starts of an HMM, polished by Baum-Welch on the `opt.starts` seam (issue #1172).

The instance is a four-state Gaussian HMM whose means are one unit apart at
scale 0.7, twelve segments of 30 to 89 positions (669 in all), drawn from a
seeded truth. Its likelihood is multimodal: Baum-Welch from twenty points
drawn around the quantile start ends at least 7.5 nats above the truth's
basin from 15 of them. The reference is Baum-Welch run from the truth for
1,000 iterations, and a start succeeds when its polish ends within
:data:`TOLERANCE` of it.

Referees: the polisher's segmented and covariate routes are the entry point
they call, bitwise (`smoke`, `patch`); each sampler's recorded gradients are the
charge :data:`~sal.search.hmm_starts.CHARGES` declares (`analytic`); the
quantile start polished by Baum-Welch reaches the truth's basin in more of
ten seeds than random restarts at equal passes (`end2end`); and every start's
success count, gaps, passes and seconds are measured at release
(`experiment`).
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.cost import Cost
from sal.emissions import GaussianEmission, NegativeBinomialEmission
from sal.opt.budget import Budget
from sal.opt.em import EM
from sal.opt.hmm import (
    EmissionHmmObjective,
    baum_welch_family,
    family_start,
)
from sal.opt.starts import StartsBenchmark, polish_by_baum_welch
from sal.ragged import Ragged
from sal.search.hmm_starts import (
    CHARGES,
    STARTS,
    at_equal_evaluations,
    gaussian_quantile_start,
)
from sal.sim.hmm import HmmParams, simulate_sequences

#: States, and the truth's means and scale.
N_STATES = 4
MEANS = np.arange(N_STATES, dtype=float)
SCALE = 0.7

#: Probability of staying in a state; the rest is spread evenly.
STAY = 0.9

#: Seeds of the generating draw and of the segment lengths.
SEED = 1172

#: Passes over the data each start is given, its seeding and polish together.
EVALUATIONS = 300

#: Nats above the reference a polish may end and count as reaching the truth's
#: basin. The nearest other basin measured is 7.55 above; a restart's polish
#: still converging ended 0.9 above at 74 iterations.
TOLERANCE = 2.0

#: Cells per start.
SEEDS = tuple(range(10))


def _transition() -> np.ndarray:
    off = (1.0 - STAY) / (N_STATES - 1)
    return np.full((N_STATES, N_STATES), off) + np.eye(N_STATES) * (STAY - off)


def _instance() -> tuple[EmissionHmmObjective, float]:
    """The objective at a quantile start, and Baum-Welch's value from the truth."""
    lengths = tuple(
        int(one) for one in np.random.default_rng(0).integers(30, 90, size=12)
    )
    initial = np.full(N_STATES, 1.0 / N_STATES)
    truth = GaussianEmission(MEANS, np.full(N_STATES, SCALE), 1e-6)
    data = simulate_sequences(
        HmmParams(N_STATES, lengths, initial, _transition(), truth, SEED, 0.0)
    )
    objective = EmissionHmmObjective(
        data.batch, gaussian_quantile_start(data.batch.values, N_STATES, 1e-6)
    )
    at_truth = objective.theta_from(
        {
            "log_initial": torch.log(torch.as_tensor(initial)),
            "log_transition": torch.log(torch.as_tensor(_transition())),
            **truth.named_parameters(),
        }
    )
    reference = polish_by_baum_welch(
        objective, at_truth, Budget(Cost.ITERATIONS, 1000)
    ).value
    return objective, reference


def _counts() -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Counts and an exposure on three segments of unequal length."""
    rng = np.random.default_rng([SEED, 1])
    lengths = (30, 45, 25)
    exposure = rng.uniform(0.5, 2.0, size=sum(lengths))
    counts = rng.poisson(5.0 * exposure)
    return counts, exposure, lengths


@pytest.mark.smoke
@pytest.mark.patch
def test_the_polisher_hands_a_segmented_objective_and_its_covariate_to_baum_welch() -> (
    None
):
    counts, exposure, lengths = _counts()
    start = family_start(NegativeBinomialEmission, counts[:30], 2)
    objective = EmissionHmmObjective(
        Ragged(counts, lengths), start, covariate=exposure[:, None]
    )
    theta = objective.initial() + torch.linspace(
        -0.3, 0.2, objective.n_parameters, dtype=torch.float64
    )
    polished = polish_by_baum_welch(objective, theta, Budget(Cost.ITERATIONS, 20))
    named = objective.constrain(theta)
    direct = baum_welch_family(
        Ragged(counts, lengths),
        named["log_initial"],
        named["log_transition"],
        objective.components(theta),
        config=replace(EM, max_iterations=20),
        covariate=Ragged(exposure[:, None], lengths),
    )
    assert polished.value == -direct.log_likelihood
    assert polished.termination == direct.termination


@pytest.mark.smoke
@pytest.mark.patch
def test_a_rectangular_batch_polishes_as_baum_welch_on_the_rectangle_does() -> None:
    # Equal segments through `EmissionHmmObjective`, handed over as a Ragged
    # batch, against Baum-Welch on the same rectangle handed over as an array
    # with its covariate: the route the adapter took for the per-family
    # objectives before #1172, which every HMM objective now leaves (#1189).
    counts, exposure, _ = _counts()
    rectangle, rate = counts[:90].reshape(3, 30), exposure[:90].reshape(3, 30)
    objective = EmissionHmmObjective(
        rectangle,
        family_start(NegativeBinomialEmission, rectangle, 2),
        covariate=rate[..., None],
    )
    theta = objective.initial()
    named = objective.constrain(theta)
    ours = polish_by_baum_welch(objective, theta, Budget(Cost.ITERATIONS, 15))
    theirs = baum_welch_family(
        rectangle,
        named["log_initial"],
        named["log_transition"],
        objective.components(theta),
        config=replace(EM, max_iterations=15),
        covariate=rate[..., None],
    )
    assert ours.value == -theirs.log_likelihood
    assert ours.termination == theirs.termination


@pytest.mark.analytic
def test_each_sampler_records_the_gradients_its_charge_declares() -> None:
    # One cell per start on the instance at a two-iteration polish: the
    # gradients each sampler recorded, plus the one point the seam scores,
    # are `CHARGES`, and a restart's charge is the points it offered.
    objective, _ = _instance()
    result = StartsBenchmark(
        objective,
        STARTS,
        polish_by_baum_welch,
        seeding_budget=Budget(Cost.EVALUATIONS, CHARGES["restart"]),
        polish_budget=Budget(Cost.ITERATIONS, 8),
        seeds=(0,),
        workers=1,
    ).run()
    for name in ("chain", "annealed", "tempered"):
        (trial,) = result.trials(name)
        assert trial.diagnostics["gradients"] + 1 == CHARGES[name], name
    (restart,) = result.trials("restart")
    assert restart.handover + 1 == CHARGES["restart"]


@pytest.mark.end2end
def test_the_quantile_start_reaches_the_truths_basin_more_often_than_restarts() -> None:
    # At 300 passes each: the quantile start's one polish of 299 iterations,
    # and four restarts sharing 296. The quantile start draws nothing, so its
    # ten cells are one; it is run at one seed. Measured: quantile 10 of 10,
    # restarts 7 of 10.
    objective, reference = _instance()
    runs = at_equal_evaluations(
        objective, ("restart",), EVALUATIONS, SEEDS, reference=[reference]
    )
    runs.update(
        at_equal_evaluations(
            objective, ("quantile",), EVALUATIONS, SEEDS[:1], reference=[reference]
        )
    )
    reached = {
        name: sum(trial.value - reference <= TOLERANCE for trial in run.trials(name))
        for name, run in runs.items()
    }
    assert reached["quantile"] == 1
    assert reached["restart"] < len(SEEDS)


#: Success counts of ten seeds at :data:`EVALUATIONS`, measured on the
#: reference host (#1172). Every sampler starts at the quantile point; under
#: the declared step each leaves its basin for a worse one more often than
#: four restarts miss the truth's.
MEASURED = {"quantile": 10, "restart": 7, "chain": 1, "annealed": 0, "tempered": 4}


@pytest.mark.release
@pytest.mark.experiment
def test_every_start_at_equal_passes_reaches_the_truths_basin_as_measured() -> None:
    objective, reference = _instance()
    began = time.perf_counter()
    runs = at_equal_evaluations(
        objective, tuple(STARTS), EVALUATIONS, SEEDS, reference=[reference]
    )
    seconds = time.perf_counter() - began
    rows = []
    for name, run in runs.items():
        trials = run.trials(name)
        (reading,) = run.readings()
        rows.append(
            (
                name,
                sum(trial.value - reference <= TOLERANCE for trial in trials),
                float(np.median([trial.seeded_value - reference for trial in trials])),
                float(np.median([trial.value - reference for trial in trials])),
                CHARGES[name] + max(trial.spent for trial in trials),
                reading.seeding_seconds + reading.polish_seconds,
            )
        )
    print(
        f"\n{'start':<10}{'reached':>8}{'start gap':>11}{'gap':>8}{'passes':>8}{'s':>7}"
    )
    for name, reached, start_gap, gap, passes, wall in rows:
        print(
            f"{name:<10}{reached:>8}{start_gap:>11.1f}{gap:>8.2f}{passes:>8}{wall:>7.2f}"
        )
    print(f"total {seconds:.1f} s")
    assert {name: reached for name, reached, *_ in rows} == MEASURED
    for _, _, _, _, passes, _ in rows:
        assert passes <= EVALUATIONS
