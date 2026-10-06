"""Step adaptation for the Hamiltonian ladder and annealing run, and per-rung starts (issue #1208).

Referees: on the correlated Gaussian every rung's acceptance after its own
warm-up lands within 0.05 of the target, and every rung's marginal variance
is the tempered closed form ``T * Sigma_ii`` (`oracle`); without an
``adaptation`` both entry points reproduce the base commit's runs bitwise
(`snapshot`); the warm-up's gradients are in ``spent`` (`analytic`); the
annealing heuristic's steps on a Gaussian at unit mass do not depend on the
temperature, since a tempered Gaussian is the untempered one rescaled
(`analytic`), and land near the target on a constant schedule (`smoke`); and per-rung starts are the shared start when every rung is
given the same point (`smoke`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sal.sample.hmc import Adaptation, anneal, leapfrog, parallel_tempering
from sal.sample.schedule import ConstantTempSchedule, ExponentialTempSchedule

from tests._posteriors import GAUSSIAN
from tests._scale import at_scale

TARGET = 0.65
ADAPTATION = Adaptation(warmup=300, target_acceptance=TARGET, step_jitter=0.4)
LADDER = (1.0, 4.0, 16.0)
COVARIANCE = np.array([[2.0, 0.6], [0.6, 0.5]])

#: Pooled acceptance's distance from the target, the issue's band.
ACCEPTANCE_BAND = 0.05

#: Relative error of a rung's pooled marginal variance: Monte Carlo error of
#: a correlated chain at the CI size, where T = 16 realized 0.21.
VARIANCE_TOLERANCE = 0.3


@pytest.mark.oracle
@at_scale("size", ci=(5, 300), stress=(20, 1000))
def test_each_rung_lands_at_the_target_acceptance_and_samples_its_tempered_gaussian(
    size: tuple[int, int],
) -> None:
    # Per-rung warm-up of 300 proposals from a step of 0.1, then the rounds.
    # Measured at CI size (5 seeds, 300 rounds): pooled acceptance 0.641,
    # 0.665, 0.634; variance ratios 0.975 to 1.206. At stress size (20 seeds,
    # 1,000 rounds): 0.644, 0.636, 0.632; ratios 1.009 to 1.028. The steps
    # fall with T (1.06, 0.54, 0.22 on seed 0): the mass is the warm-up's
    # tempered variance, which raises each coordinate's frequency by sqrt(T).
    n_seeds, n_rounds = size
    runs = [
        parallel_tempering(
            GAUSSIAN,
            LADDER,
            torch.Generator().manual_seed(seed),
            n_rounds,
            step_size=0.1,
            n_steps=5,
            adaptation=ADAPTATION,
        )
        for seed in range(n_seeds)
    ]
    pooled = np.mean([run.acceptance_rate.numpy() for run in runs], axis=0)
    positions = np.concatenate([run.positions.numpy() for run in runs])
    for rung, temperature in enumerate(LADDER):
        assert abs(pooled[rung] - TARGET) < ACCEPTANCE_BAND, (temperature, pooled)
        ratio = positions[:, rung].var(axis=0) / (temperature * np.diag(COVARIANCE))
        np.testing.assert_allclose(ratio, 1.0, atol=VARIANCE_TOLERANCE)
    for run in runs:
        assert run.adapted is not None
        assert len(run.adapted) == len(LADDER)


#: The runs below on the base commit a764e71e, before `adaptation` and
#: per-rung starts existed: the positions' sum, the best value, the per-rung
#: acceptance and the swap acceptance; for annealing the final point's sum,
#: the best value and the acceptance.
TEMPERED_PIN = (
    "-0x1.cb90e25bd800ap+7",
    "0x1.74451d62fd6f8p-4",
    [1.0, 0.925, 1.0],
    [0.45, 0.25],
)
ANNEALED_PIN = ("-0x1.b6b98779340b2p-1", "0x1.c4f3d10ad6c60p-8", 0.975)


@pytest.mark.smoke
@pytest.mark.snapshot
def test_without_adaptation_both_runs_are_the_base_commits_bitwise() -> None:
    tempered = parallel_tempering(
        GAUSSIAN,
        (1.0, 3.0, 9.0),
        torch.Generator().manual_seed(1208),
        40,
        step_size=0.3,
        n_steps=5,
    )
    annealed = anneal(
        GAUSSIAN,
        ExponentialTempSchedule(9.0, 1.0, 40),
        torch.Generator().manual_seed(1208),
        step_size=0.3,
        n_steps=5,
    )
    assert (
        float(tempered.positions.sum()).hex(),
        float(tempered.value).hex(),
        tempered.acceptance_rate.tolist(),
        tempered.swap_acceptance.tolist(),
    ) == TEMPERED_PIN
    assert tempered.adapted is None
    assert (
        float(annealed.final.sum()).hex(),
        float(annealed.value).hex(),
        annealed.acceptance_rate,
    ) == ANNEALED_PIN
    assert annealed.step_sizes is None


@pytest.mark.analytic
def test_the_ladder_charges_every_rungs_warm_up() -> None:
    adaptation = Adaptation(warmup=16, target_acceptance=TARGET, step_jitter=0.4)
    run = parallel_tempering(
        GAUSSIAN,
        LADDER,
        torch.Generator().manual_seed(3),
        10,
        step_size=0.1,
        n_steps=5,
        adaptation=adaptation,
    )
    per_proposal = leapfrog.force_evaluations(5)
    assert run.adapted is not None
    assert run.spent == (10 + 16) * len(LADDER) * per_proposal
    assert all(report.force_evaluations == 16 * per_proposal for report in run.adapted)
    assert len({report.step_size for report in run.adapted}) == len(LADDER)


@pytest.mark.analytic
def test_the_annealing_steps_do_not_depend_on_a_gaussians_temperature() -> None:
    # At unit mass a tempered Gaussian is the untempered one stretched by
    # sqrt(T) about its mean, and so is every trajectory: the momentum is
    # drawn N(0, T) from the same normals, the energy error scales by T and
    # the acceptance divides it by T. From a start stretched alike, each
    # window's step is the same; at T = 4 the stretch is 2, a power of two,
    # so every product is exact and the steps agree bitwise.
    mean = torch.tensor([1.0, -2.0], dtype=torch.float64)
    away = torch.tensor([-1.0, 0.5], dtype=torch.float64)
    adaptation = Adaptation(warmup=50, target_acceptance=TARGET, step_jitter=0.4)
    steps = []
    for temperature in (1.0, 4.0):
        run = anneal(
            GAUSSIAN,
            ConstantTempSchedule(temperature, 200),
            torch.Generator().manual_seed(11),
            step_size=0.1,
            n_steps=5,
            start=mean + math.sqrt(temperature) * away,
            adaptation=adaptation,
        )
        assert run.step_sizes is not None
        assert len(run.step_sizes) == 200 // 50
        steps.append(run.step_sizes)
    assert steps[1] == steps[0]


@pytest.mark.smoke
def test_the_annealing_heuristic_lands_near_its_target_on_a_constant_schedule() -> None:
    # Three seeds of 600 proposals at T = 1, re-tuned every 200: pooled
    # acceptance 0.622, the windows' iterates included.
    adaptation = Adaptation(warmup=200, target_acceptance=TARGET, step_jitter=0.4)
    runs = [
        anneal(
            GAUSSIAN,
            ConstantTempSchedule(1.0, 600),
            torch.Generator().manual_seed(seed),
            step_size=0.1,
            n_steps=5,
            adaptation=adaptation,
        )
        for seed in range(3)
    ]
    pooled = float(np.mean([run.acceptance_rate for run in runs]))
    assert abs(pooled - TARGET) < ACCEPTANCE_BAND, pooled


@pytest.mark.smoke
def test_one_start_per_rung_is_the_shared_start_when_every_rung_has_it() -> None:
    point = torch.tensor([0.5, -1.0], dtype=torch.float64)
    shared, per_rung = (
        parallel_tempering(
            GAUSSIAN,
            LADDER,
            torch.Generator().manual_seed(7),
            20,
            step_size=0.3,
            n_steps=5,
            start=start,
        )
        for start in (point, [point.clone() for _ in LADDER])
    )
    assert torch.equal(shared.positions, per_rung.positions)
    assert torch.equal(shared.best, per_rung.best)
    # A rung started at the mode makes the mode's value the best seen.
    mode = torch.tensor([1.0, -2.0], dtype=torch.float64)
    seeded = parallel_tempering(
        GAUSSIAN,
        LADDER,
        torch.Generator().manual_seed(7),
        1,
        step_size=0.3,
        start=[point, point, mode],
    )
    assert seeded.value <= float(GAUSSIAN(mode))
    with pytest.raises(ValueError, match="2 points for a ladder of 3 rungs"):
        parallel_tempering(
            GAUSSIAN,
            LADDER,
            torch.Generator().manual_seed(7),
            1,
            step_size=0.3,
            start=[point, point],
        )
