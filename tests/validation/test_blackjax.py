"""`sample.hmc` against BlackJAX, run in a subprocess (issue #963).

BlackJAX shares no code with `sample/hmc.py` and runs on JAX, which the
package does not take on. Referees:

- the integrators: leapfrog and the Yoshida triple jump, 25 steps from one
  phase-space point on a dense 10-dimensional Gaussian, the endpoint's
  position and momentum within 1e-12 of BlackJAX's `generate_euclidean_integrator`
  on the same composition;
- the chains: 5,000 transitions each at step 0.15 and seven leapfrog steps,
  both means within 4 Monte Carlo standard errors of zero, the error read at
  each chain's own effective size, and the two acceptance rates within 0.01.

The trajectory length stays off the target's half-period: at step 0.2 and ten
steps the chain is antithetic, and `effective_sample_size` returns a negative
size there (#984).

The runtime goal BlackJAX sets is in `test_goals.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.validation import blackjax
from snakes_and_ladders.validation.gaussian import GaussianTarget, dense_precision
from snakes_and_ladders.validation.runner import available

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not available("blackjax"), reason="BlackJAX is the validation-blackjax extra"
    ),
]

#: The dense target's dimension and the seed its precision is drawn from.
DIMENSION, SEED = 10, 963


@pytest.mark.oracle
@pytest.mark.parametrize("integrator", [hmc.leapfrog, hmc.yoshida], ids=str)
def test_the_integrator_is_blackjaxs_step_for_step(integrator: hmc.Integrator) -> None:
    precision = dense_precision(DIMENSION, SEED)
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
    precision = dense_precision(DIMENSION, SEED)
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
