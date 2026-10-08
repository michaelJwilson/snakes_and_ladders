"""Compiled annealing and tempering: the shared loops over an ``oxisal.HmcWalk`` step (issue #1249).

Referees: on the correlated Gaussian every compiled rung at ``T`` = 1, 4, 16
samples ``T * Sigma`` (`oracle`), unadapted and after each rung's own
warm-up; annealing on a constant schedule is the compiled chain at that
temperature draw for draw, on the Gaussian and on Rosenbrock (`smoke`, `patch`); and
an exchange moves the carried ``U`` with its position, so the reported value
is the kernel's at the best point bitwise, at the torch route's charge
(`analytic`). The compiled walk draws from its own ChaCha8 stream, so the
torch route is matched in distribution, not draw for draw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.backend import Backend
from sal.opt.testfunctions import Rosenbrock
from sal.sample import hmc
from sal.sample.hmc import Adaptation, anneal, leapfrog, parallel_tempering, sample
from sal.sample.schedule import ConstantTempSchedule
from sal.validation.gaussian import GaussianTarget

COVARIANCE = np.array([[2.0, 0.6], [0.6, 0.5]])
GAUSSIAN = GaussianTarget(np.linalg.inv(COVARIANCE))
LADDER = (1.0, 4.0, 16.0)

#: Relative error of a rung's marginal variance over 3,500 recorded rounds:
#: three seeds realized 0.051 at most (ratios 0.987 to 1.051), and eight
#: adapted seeds of 1,500 rounds 0.016.
VARIANCE_TOLERANCE = 0.12

#: A rung's mean in its own standard deviations: realized 0.022 at most.
MEAN_TOLERANCE = 0.08

#: Pooled acceptance's distance from the target after each rung's warm-up,
#: `test_hmc_tempering_adaptation`'s band.
ACCEPTANCE_BAND = 0.05


@pytest.fixture
def walks(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every ``oxisal.HmcWalk`` built while the test runs: what says the compiled route ran."""
    built: list[Any] = []
    make = oxisal.HmcWalk

    def counted(*args: Any) -> Any:
        walk = make(*args)
        built.append(walk)
        return walk

    monkeypatch.setattr(oxisal, "HmcWalk", counted)
    return built


def _assert_tempered_moments(positions: np.ndarray) -> None:
    for rung, temperature in enumerate(LADDER):
        spread = np.sqrt(temperature * np.diag(COVARIANCE))
        mean = positions[:, rung].mean(axis=0) / spread
        ratio = positions[:, rung].var(axis=0) / spread**2
        assert np.abs(mean).max() < MEAN_TOLERANCE, (temperature, mean)
        np.testing.assert_allclose(ratio, 1.0, atol=VARIANCE_TOLERANCE)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1])
def test_each_compiled_rung_samples_its_tempered_gaussian(
    walks: list[Any], seed: int
) -> None:
    run = parallel_tempering(
        GAUSSIAN,
        LADDER,
        torch.Generator().manual_seed(seed),
        4_000,
        step_size=0.3,
        n_steps=8,
    )
    assert len(walks) == len(LADDER)
    _assert_tempered_moments(run.positions[500:].numpy())
    # Exchanges happen, so the moments are of the ladder and not of three
    # independent chains: 0.39 to 0.41 realized.
    assert bool((run.swap_acceptance > 0.3).all()), run.swap_acceptance


@pytest.mark.oracle
def test_each_compiled_rung_warms_up_at_its_own_temperature(walks: list[Any]) -> None:
    # Each walk runs `Adaptation`'s two windows on construction at its rung's
    # temperature. Pooled over eight seeds of 1,500 rounds: acceptance 0.638,
    # 0.629, 0.650; variance ratios 0.984 to 1.014. At 300 rounds the T = 16
    # rung realized 1.18, Monte Carlo error that 20 seeds of 3,000 bring to
    # 1.029, so the rounds are what the tolerance is held at.
    adaptation = Adaptation(warmup=300, target_acceptance=0.65, step_jitter=0.4)
    runs = [
        parallel_tempering(
            GAUSSIAN,
            LADDER,
            torch.Generator().manual_seed(seed),
            1_500,
            step_size=0.1,
            n_steps=5,
            adaptation=adaptation,
        )
        for seed in range(8)
    ]
    assert len(walks) == 8 * len(LADDER)
    pooled = np.mean([run.acceptance_rate.numpy() for run in runs], axis=0)
    assert np.abs(pooled - 0.65).max() < ACCEPTANCE_BAND, pooled
    _assert_tempered_moments(np.concatenate([run.positions.numpy() for run in runs]))
    for run in runs:
        assert run.adapted is not None
        # The walk carries `grad U` across its metric: the start once, then
        # `n_steps` per proposal, in the warm-up and the rounds alike.
        per_warmup = 1 + 300 * leapfrog.force_evaluations(5, carried=True)
        assert all(report.force_evaluations == per_warmup for report in run.adapted)
        assert run.spent == len(LADDER) * (per_warmup + 1_500 * 5)


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize(
    ("objective", "temperature", "step_size"),
    [(GAUSSIAN, 3.0, 0.2), (Rosenbrock(dimension=2), 0.5, 0.01)],
    ids=["gaussian", "rosenbrock"],
)
def test_a_constant_schedule_is_the_compiled_chain_draw_for_draw(
    walks: list[Any], objective: Any, temperature: float, step_size: float
) -> None:
    # One `advance_at(1, T)` per schedule step is one `advance(n)` at `T`:
    # the walk is seeded alike and the loops share no state, bitwise.
    chain = sample(
        objective,
        torch.Generator().manual_seed(5),
        200,
        step_size=step_size,
        n_steps=10,
        temperature=temperature,
    )
    annealed = anneal(
        objective,
        ConstantTempSchedule(temperature, 200),
        torch.Generator().manual_seed(5),
        step_size=step_size,
        n_steps=10,
    )
    assert len(walks) == 2
    # The best visited is no worse than any draw the chain made (#1374
    # removed `final`, which this compared with the last draw).
    assert annealed.value <= min(float(objective(draw)) for draw in chain.draws)
    assert annealed.acceptance_rate == chain.acceptance_rate
    assert annealed.acceptance_rate > 0.0
    assert annealed.spent == chain.spent == 1 + 200 * 10


@pytest.mark.analytic
def test_an_exchange_carries_the_energy_with_the_position(walks: list[Any]) -> None:
    # The best energy is read from what the rungs carried through every
    # exchange; the kernel at the best point is the same arithmetic, so a
    # carried value left behind by a swap would differ here.
    run = parallel_tempering(
        GAUSSIAN,
        LADDER,
        torch.Generator().manual_seed(1249),
        300,
        step_size=0.3,
        n_steps=8,
    )
    kernel = oxisal.SupportedEnergy(*GAUSSIAN.supported_gradient(), 2)
    value, _ = kernel.value_and_gradient(run.best.numpy())
    assert len(walks) == len(LADDER)
    assert int(run.swap_acceptance.sum() * 300) > 0
    assert run.value == value
    # The torch route's charge: each rung's start once, then `n_steps` per
    # proposal (issue #1222).
    assert run.spent == len(LADDER) * (1 + 300 * 8)
    again = parallel_tempering(
        GAUSSIAN,
        LADDER,
        torch.Generator().manual_seed(1249),
        300,
        step_size=0.3,
        n_steps=8,
    )
    assert torch.equal(run.positions, again.positions)
    assert np.array_equal(run.walkers, again.walkers)


@pytest.mark.smoke
def test_the_python_backend_and_other_integrators_keep_the_torch_route(
    walks: list[Any],
) -> None:
    parallel_tempering(
        GAUSSIAN,
        LADDER,
        torch.Generator().manual_seed(0),
        3,
        step_size=0.3,
        n_steps=2,
        backend=Backend.PYTHON,
    )
    anneal(
        GAUSSIAN,
        ConstantTempSchedule(1.0, 3),
        torch.Generator().manual_seed(0),
        step_size=0.3,
        n_steps=2,
        integrator=hmc.yoshida,
    )
    assert walks == []
    with pytest.raises(ValueError, match="hmc.anneal"):
        anneal(
            GAUSSIAN,
            ConstantTempSchedule(1.0, 3),
            torch.Generator().manual_seed(0),
            step_size=0.3,
            backend=Backend.TORCH,
        )
