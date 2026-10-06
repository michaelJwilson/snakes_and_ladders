"""Sampler starts of an HMM, polished by Baum-Welch on the `opt.starts` seam (issue #1172).

The instance is a four-state Gaussian HMM whose means are one unit apart at
scale 0.7, twelve segments of 30 to 89 positions (669 in all), drawn from a
seeded truth; the harder instance has five states on the same segments
(#1195), and four restarts reach its truth's basin in none of ten seeds. Its likelihood is multimodal: Baum-Welch from twenty points
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

#: States of #1172's instance and of the harder one (#1195), and the truth's
#: scale; the means are one unit apart.
N_STATES = 4
HARDER = 5
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


def _transition(n_states: int) -> np.ndarray:
    off = (1.0 - STAY) / (n_states - 1)
    return np.full((n_states, n_states), off) + np.eye(n_states) * (STAY - off)


def _instance(n_states: int = N_STATES) -> tuple[EmissionHmmObjective, float]:
    """The objective at a quantile start, and Baum-Welch's value from the truth."""
    lengths = tuple(
        int(one) for one in np.random.default_rng(0).integers(30, 90, size=12)
    )
    initial = np.full(n_states, 1.0 / n_states)
    means = np.arange(n_states, dtype=float)
    truth = GaussianEmission(means, np.full(n_states, SCALE), 1e-6)
    data = simulate_sequences(
        HmmParams(n_states, lengths, initial, _transition(n_states), truth, SEED, 0.0)
    )
    objective = EmissionHmmObjective(
        data.batch, gaussian_quantile_start(data.batch.values, n_states, 1e-6)
    )
    at_truth = objective.theta_from(
        {
            "log_initial": torch.log(torch.as_tensor(initial)),
            "log_transition": torch.log(torch.as_tensor(_transition(n_states))),
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
    for name in (
        "chain",
        "annealed",
        "tempered",
        "chain_adapted",
        "annealed_adapted",
        "tempered_adapted",
        "chain_tuned",
        "annealed_tuned",
        "tempered_tuned",
    ):
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


#: Success counts of ten seeds at :data:`EVALUATIONS`, per instance, measured
#: on the reference host at the steps of #1195; at #1172's untuned step of
#: 3e-2 the four-state counts were chain 1, annealed 0 and tempered 4. Every
#: sampler starts at the quantile point and reaches the truth's basin by
#: staying near it: its median start is 2 to 51 nats below the quantile
#: point's, against 192 to 305 at 3e-2.
#: The adapted entries are each sampler's own warm-up from 3e-2 at equal
#: passes (#1208), and reach fewer seeds than the grid's steps on both.
MEASURED = {
    N_STATES: {
        "quantile": 10,
        "restart": 7,
        "chain": 10,
        "annealed": 10,
        "tempered": 10,
        "chain_adapted": 3,
        "annealed_adapted": 0,
        "tempered_adapted": 3,
    },
    HARDER: {
        "quantile": 10,
        "restart": 0,
        "chain": 8,
        "annealed": 8,
        "tempered": 7,
        "chain_adapted": 2,
        "annealed_adapted": 0,
        "tempered_adapted": 3,
    },
}


@pytest.mark.release
@pytest.mark.experiment
@pytest.mark.parametrize("n_states", [N_STATES, HARDER])
def test_every_start_at_equal_passes_reaches_the_truths_basin_as_measured(
    n_states: int,
) -> None:
    objective, reference = _instance(n_states)
    began = time.perf_counter()
    runs = at_equal_evaluations(
        objective, tuple(MEASURED[n_states]), EVALUATIONS, SEEDS, reference=[reference]
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
        f"\n{n_states} states\n"
        f"{'start':<17}{'reached':>8}{'start gap':>11}{'gap':>8}{'passes':>8}{'s':>7}"
    )
    for name, reached, start_gap, gap, passes, wall in rows:
        print(
            f"{name:<17}{reached:>8}{start_gap:>11.1f}{gap:>8.2f}{passes:>8}{wall:>7.2f}"
        )
    print(f"total {seconds:.1f} s")
    counts = {name: count for name, count, *_ in rows}
    assert counts == MEASURED[n_states]
    # The decision of #1208: the grid's steps stay while an adapted sampler
    # reaches fewer seeds than its hand-tuned twin.
    for sampler in ("chain", "annealed", "tempered"):
        assert counts[f"{sampler}_adapted"] < counts[sampler], sampler
    for _, _, _, _, passes, _ in rows:
        assert passes <= EVALUATIONS


#: The tuned entries' success counts of ten seeds on four states (issue
#: #1219), one pass on the development host: the pilot chose 3e-2 for the
#: chain and the cold rung in every seed. Five states is unmeasured.
TUNED = {"chain_tuned": 3, "annealed_tuned": 3, "tempered_tuned": 5}


@pytest.mark.release
@pytest.mark.experiment
def test_a_tuned_step_reaches_the_truths_basin_in_fewer_seeds_than_the_grid() -> None:
    objective, reference = _instance(N_STATES)
    runs = at_equal_evaluations(
        objective, tuple(TUNED), EVALUATIONS, SEEDS, reference=[reference]
    )
    counts = {
        name: sum(trial.value - reference <= TOLERANCE for trial in run.trials(name))
        for name, run in runs.items()
    }
    assert counts == TUNED
    # The decision of #1219: the grid's steps stay while a tuned sampler
    # reaches fewer than the issue's bar of 9 of 10.
    for sampler in ("chain", "annealed", "tempered"):
        assert counts[f"{sampler}_tuned"] < 9 <= MEASURED[N_STATES][sampler], sampler
