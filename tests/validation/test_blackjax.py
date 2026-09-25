"""`sample.hmc` against BlackJAX, run in a subprocess (issue #963).

BlackJAX shares no code with `sample/hmc.py`. Leapfrog and Yoshida, 25 steps
on a dense 10-D Gaussian: endpoint within 1e-12 of
`generate_euclidean_integrator`. Chains of 5,000 at step 0.15, seven steps:
means within 4 standard errors at each chain's ESS, acceptance within 0.01
(off the half-period: at 0.2 x 10 the chain is antithetic and ESS negative,
#984). Random walk (#1006): `rmh` against `metropolis.replay` on shared
increments and uniforms, draw for draw at 1e-10 on a Gaussian and on
Rosenbrock, acceptance within 0.02. Runtime goals: `test_goals.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.opt.testfunctions import Rosenbrock
from sal.sample import hmc, langevin, metropolis
from sal.validation import blackjax
from sal.validation.gaussian import (
    GaussianTarget,
    dense_precision,
    diagonal_precision,
)

from tests._frameworks import requires

pytestmark = [
    pytest.mark.validation,
    requires("blackjax"),
]

#: The dense target's dimension and the seed its precision is drawn from.
DIMENSION, SEED = 10, 963


@pytest.mark.oracle
@pytest.mark.parametrize("integrator", [hmc.leapfrog, hmc.yoshida], ids=str)
def test_the_integrator_is_blackjaxs_step_for_step(integrator: hmc.Integrator) -> None:
    precision = dense_precision(DIMENSION, np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    position, momentum = rng.normal(size=DIMENSION), rng.normal(size=DIMENSION)
    ours = integrator(
        GaussianTarget(precision),
        torch.as_tensor(position),
        torch.as_tensor(momentum),
        0.1,
        25,
    )
    theirs = blackjax.integrate(
        precision, position, momentum, 0.1, 25, integrator.weights
    )
    np.testing.assert_allclose(ours.position.numpy(), theirs.position, atol=1e-12)
    np.testing.assert_allclose(ours.momentum.numpy(), theirs.momentum, atol=1e-12)
    assert np.abs(theirs.position - position).max() > 0.1


@pytest.mark.experiment
def test_both_chains_centre_on_the_mean_and_accept_alike() -> None:
    precision = dense_precision(DIMENSION, np.random.default_rng(SEED))
    variance = np.diag(np.linalg.inv(precision))
    ours = hmc.sample(
        GaussianTarget(precision),
        torch.Generator().manual_seed(SEED),
        5_000,
        step_size=0.15,
        n_steps=7,
    )
    theirs = blackjax.sample(precision, np.zeros(DIMENSION), 0.15, 7, 5_000, SEED)
    for draws in (ours.theta.numpy(), theirs.draws):
        size = hmc.effective_sample_size(torch.as_tensor(draws)).numpy()
        assert size.min() > 100.0
        error = np.sqrt(variance / size)
        assert np.abs(draws.mean(axis=0)).max() < 4.0 * error.max()
    assert abs(ours.acceptance_rate - theirs.acceptance) < 0.01


@pytest.mark.smoke
def test_blackjax_s_chain_free_run_is_the_same_chain_unstored() -> None:
    # Issue #997: with no draw kept the scan runs the same transitions on the
    # same keys, so the acceptance is the stored run's exactly.
    precision = np.linspace(1.0, 4.0, 20)
    stored, free = (
        blackjax.sample(precision, np.zeros(20), 0.3, 5, 200, 997, store_chain=keep)
        for keep in (True, False)
    )
    assert stored.draws.shape == (200, 20)
    assert free.draws.shape == (0, 20)
    assert free.acceptance == stored.acceptance


@pytest.mark.oracle
def test_mala_accepts_as_blackjaxs_mala_does() -> None:
    # Issue #997: `langevin.mala` at step h and `blackjax.mala` at
    # epsilon = h^2 / 2 make the same proposal, so on one Gaussian their
    # acceptance rates agree to the chains' sampling error. The streams
    # differ; 5,000 transitions put a rate near 0.9 within about 0.01.
    dimension = 100
    precision = diagonal_precision(dimension)
    step = 1.6 / dimension**0.25
    ours = langevin.mala(
        GaussianTarget(precision),
        torch.Generator().manual_seed(997),
        5_000,
        step_size=step,
        store_chain=False,
    )
    theirs = blackjax.mala(
        precision, np.zeros(dimension), step, 5_000, 997, store_chain=False
    )
    assert 0.3 < theirs.acceptance < 0.97
    assert abs(ours.acceptance_rate - theirs.acceptance) < 0.03


@pytest.mark.oracle
@pytest.mark.parametrize("target", ["gaussian", "rosenbrock"])
def test_the_random_walk_is_blackjaxs_rmh_draw_for_draw(target: str) -> None:
    # Issue #1006: BlackJAX's `build_rmh` kernel on the increments given, and
    # `metropolis.replay` on the same increments and the uniforms BlackJAX's
    # acceptance compared, run the same chain; measured agreement is exact.
    rng = np.random.default_rng(1006)
    if target == "rosenbrock":
        objective: object = Rosenbrock(10)
        options: dict[str, object] = {"rosenbrock": (1.0, 100.0)}
        start, step = np.full(10, -1.2), 0.02
    else:
        precision = diagonal_precision(50)
        objective, options = GaussianTarget(precision), {"precision": precision}
        start, step = np.zeros(50), 0.2
    increments = step * rng.normal(size=(2_000, start.size))
    theirs = blackjax.replay(start, increments, 1006, **options)  # type: ignore[arg-type]
    ours = metropolis.replay(objective, start, 1.0, increments, theirs.uniforms)  # type: ignore[arg-type]
    np.testing.assert_allclose(ours.draws, theirs.draws, rtol=0, atol=1e-10)
    assert 0.05 < ours.accepted.mean() < 0.95


@pytest.mark.experiment
def test_the_random_walk_accepts_as_blackjaxs_does() -> None:
    # Issue #1006: the same proposal scale on the same target, different
    # streams; 20,000 transitions put a rate near 0.34 within about 0.01.
    precision = diagonal_precision(100)
    ours = metropolis.random_walk(
        GaussianTarget(precision),
        np.random.default_rng(1006),
        20_000,
        step_size=0.12,
        store_chain=False,
    )
    theirs = blackjax.random_walk(
        np.zeros(100), 0.12, 20_000, 1006, precision=precision, store_chain=False
    )
    assert abs(ours.acceptance_rate - theirs.acceptance) < 0.02


@pytest.mark.oracle
@pytest.mark.parametrize("integrator", [hmc.leapfrog, hmc.yoshida], ids=str)
def test_the_integrator_is_blackjaxs_on_rosenbrock(integrator: hmc.Integrator) -> None:
    # Issue #1008: a non-Gaussian force, the curved valley's, step for step.
    rng = np.random.default_rng(1008)
    position, momentum = rng.normal(size=10) * 0.3, rng.normal(size=10)
    ours = integrator(
        Rosenbrock(10), torch.as_tensor(position), torch.as_tensor(momentum), 0.002, 25
    )
    theirs = blackjax.integrate(
        None,
        position,
        momentum,
        0.002,
        25,
        integrator.weights,
        rosenbrock=(1.0, 100.0),
    )
    np.testing.assert_allclose(ours.position.numpy(), theirs.position, atol=1e-12)
    np.testing.assert_allclose(ours.momentum.numpy(), theirs.momentum, atol=1e-10)
    assert np.abs(theirs.position - position).max() > 0.01
