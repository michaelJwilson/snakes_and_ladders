"""Values and gradients of the HMC objectives against JAX's, run in a subprocess (issue #991).

JAX's reverse mode shares no code with PyTorch's autograd. Referees: at 100
points each, the value and the gradient `hmc.gradient_at` takes agree with
`jax.value_and_grad` within 1e-12 relative, on the diagonal Gaussian at
d = 10 and 10³, the dense one at d = 10, and the three-component mixture
likelihood at n = 10⁴ observations. `torch.func.grad_and_value` and the
closed form `P x` agree with it too, so the three torch routes the benchmark
times compute one gradient.

The runtime and memory goals JAX sets are in `test_goals.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.mixture import GaussianMixtureObjective
from snakes_and_ladders.sample.hmc import gradient_at
from snakes_and_ladders.validation import jax
from snakes_and_ladders.validation.gaussian import (
    GaussianTarget,
    dense_precision,
    diagonal_precision,
)
from snakes_and_ladders.validation.runner import available, package

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(not available("jax"), reason="JAX is the validation-jax extra"),
]

#: Relative agreement of two reverse modes summing in different orders.
RTOL = 1e-12


def _compare(target: object, points: np.ndarray, theirs: jax.Gradients) -> None:
    for row, value, gradient in zip(
        points, theirs.values, theirs.gradients, strict=True
    ):
        point = torch.as_tensor(row)
        ours = gradient_at(target, point)  # type: ignore[arg-type]
        scale = max(1.0, float(np.abs(gradient).max()))
        np.testing.assert_allclose(ours.numpy(), gradient, rtol=0.0, atol=RTOL * scale)
        mine = float(target(point))  # type: ignore[operator]
        assert mine == pytest.approx(float(value), rel=RTOL)


@pytest.mark.oracle
@pytest.mark.parametrize("dimension", [10, 1_000])
def test_the_diagonal_gaussian_differentiates_as_jax_does(dimension: int) -> None:
    precision = diagonal_precision(dimension)
    points = np.random.default_rng(991).normal(size=(100, dimension))
    _compare(
        GaussianTarget(precision), points, jax.gradients(points, precision=precision)
    )


@pytest.mark.oracle
def test_the_dense_gaussian_differentiates_as_jax_does() -> None:
    precision = dense_precision(10, np.random.default_rng(963))
    points = np.random.default_rng(9910).normal(size=(100, 10))
    _compare(
        GaussianTarget(precision), points, jax.gradients(points, precision=precision)
    )


def _mixture(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(975)
    component = rng.choice(3, size=n, p=[0.3, 0.3, 0.4])
    observations = rng.normal(
        np.array([-4.0, 0.0, 5.0])[component], np.array([1.0, 1.5, 1.0])[component]
    )
    centre = np.array([0.0, 0.3, -3.0, 0.5, 4.0, 0.2, 0.0, 0.3])
    points = centre + 0.05 * np.random.default_rng(991).normal(size=(100, 8))
    return observations, points


@pytest.mark.oracle
def test_the_mixture_likelihood_differentiates_as_jax_does() -> None:
    observations, points = _mixture(10_000)
    theirs = jax.gradients(points, observations=observations, n_components=3)
    _compare(GaussianMixtureObjective(observations, 3), points, theirs)


@pytest.mark.oracle
@pytest.mark.parametrize("route", ["func", "closed"])
def test_every_torch_route_the_benchmark_times_is_one_gradient(route: str) -> None:
    precision = diagonal_precision(100)
    points = np.random.default_rng(9911).normal(size=(20, 100))
    theirs = jax.gradients(points, precision=precision)
    ours = package(
        "gradient",
        {"precision": precision, "points": points, "route": np.asarray(route)},
    )
    np.testing.assert_allclose(ours.outputs["gradients"], theirs.gradients, atol=1e-12)
