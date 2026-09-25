"""`opt.mixture.expectation_maximization` against scikit-learn's, run in a subprocess (issue #975).

scikit-learn's `GaussianMixture` with a diagonal covariance and no
regularization is an independent implementation of the same EM on a
one-dimensional mixture. Referees:

- one and ten iterations from one start on 5,000 draws of three components:
  every weight, mean and standard deviation within 1e-11;
- the same on `mixture/ci.yaml` (five components, 500 draws).

The runtime goal scikit-learn sets is in `test_goals.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sal.emissions import GaussianEmission
from sal.fixtures import load_params
from sal.opt.em import EmConfig
from sal.opt.mixture import (
    expectation_maximization,
    mixture_log_likelihood,
)
from sal.sim.mixture import MixtureParams, simulate_mixture
from sal.validation import scikit_learn

from tests._fixtures import FIXTURES_DIR
from tests._frameworks import requires
from tests._rows import every_value

pytestmark = [
    pytest.mark.validation,
    requires("scikit_learn"),
]

FIXTURE: Path = FIXTURES_DIR / "mixture" / "ci.yaml"

#: The float64 tolerance two implementations of one recursion are held to.
ATOL = 1e-11

#: The start: away from the draws' generating values below.
WEIGHTS = np.array([0.3, 0.3, 0.4])
MEAN = np.array([-3.0, 0.5, 4.0])
SCALE = np.array([1.2, 1.0, 1.3])


def _draws(n_samples: int) -> np.ndarray:
    """Three components at -4, 0 and 5, deviations 1, 1.5 and 1, weighted 0.3, 0.3, 0.4."""
    rng = np.random.default_rng(975)
    component = rng.choice(3, size=n_samples, p=[0.3, 0.3, 0.4])
    centre = np.array([-4.0, 0.0, 5.0])[component]
    spread = np.array([1.0, 1.5, 1.0])[component]
    return np.asarray(rng.normal(centre, spread))


def _ours(
    observations: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = expectation_maximization(
        observations,
        torch.as_tensor(weights),
        GaussianEmission(mean, scale, 1e-12),
        config=EmConfig(max_iterations=n_iter, tolerance=-np.inf),
    )
    return (
        fit.weights.numpy(),
        np.asarray(fit.components.mean).ravel(),
        np.asarray(fit.components.scale).ravel(),
    )


def _agree(
    observations: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    n_iter: int,
) -> None:
    theirs = scikit_learn.expectation_maximization(
        observations, weights, mean, scale, n_iter
    )
    ours = _ours(observations, weights, mean, scale, n_iter)
    assert theirs.iterations == n_iter
    for mine, other in zip(
        ours, (theirs.weights, theirs.mean, theirs.scale), strict=False
    ):
        np.testing.assert_allclose(mine, other, rtol=0.0, atol=ATOL)
    assert np.abs(theirs.mean - mean).max() > 1e-3


@pytest.mark.oracle
def test_em_is_scikit_learns_iteration_for_iteration() -> None:
    def check(n_iter: int) -> None:
        _agree(_draws(5_000), WEIGHTS, MEAN, SCALE, n_iter)

    every_value([1, 10], check)


@pytest.mark.oracle
def test_em_is_scikit_learns_on_the_fixture() -> None:
    params = load_params(FIXTURE, MixtureParams)
    observations = simulate_mixture(params).observations
    n_components = params.weights.size
    start_mean = np.linspace(-2.5, 2.5, n_components)
    _agree(
        observations,
        np.full(n_components, 1.0 / n_components),
        start_mean,
        np.full(n_components, 1.3),
        10,
    )


@pytest.mark.oracle
def test_the_mixture_log_likelihood_is_scikit_learns_score() -> None:
    # Issue #997: the streamed, gradient-free log-likelihood against the sum
    # of scikit-learn's `score_samples` at the same parameters.
    rng = np.random.default_rng(997)
    values = rng.normal(0.0, 3.0, 20_000)
    weights, mean, scale = (
        np.array([0.3, 0.3, 0.4]),
        np.array([-3.0, 0.5, 4.0]),
        np.array([1.2, 1.0, 1.3]),
    )
    ours = mixture_log_likelihood(
        torch.as_tensor(values),
        torch.log(torch.as_tensor(weights)),
        GaussianEmission(mean, scale, 1e-12),
    )
    theirs = scikit_learn.score(values, weights, mean, scale)
    np.testing.assert_allclose(float(ours), theirs.log_likelihood, rtol=1e-11)
