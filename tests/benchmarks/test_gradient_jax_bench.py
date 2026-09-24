"""The gradient HMC spends, beside JAX's, per point and per target (issue #991).

Agreement is pinned in `tests/validation/test_jax.py` and the goals in
`test_goals.py`. Each row reads, in fresh interpreters, the median seconds
per point and the peak resident memory over 100 points of three torch routes
--- `hmc.gradient_at` (autograd, the one HMC calls), `torch.func.grad_and_value`,
and for a Gaussian the closed form `P x` --- beside JAX's `value_and_grad`
eagerly, under `jit`, and under `jit(vmap)` over the 100 points at once.
Measured, not timed by pytest-benchmark: every figure is its script's own.

Sizes: the diagonal Gaussian at d = 10 (gate), 10^3 and 10^4, the dense one
at d = 10 and 10^3, and the three-component mixture at n = 10^5.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.validation import jax
from snakes_and_ladders.validation.gaussian import dense_precision, diagonal_precision
from snakes_and_ladders.validation.runner import package

from tests._frameworks import requires

pytestmark = requires("jax")

CASES = [
    "diagonal 10",
    "diagonal 1000",
    "diagonal 10000",
    "dense 10",
    "dense 1000",
    "mixture 100000",
]


def _case(case: str) -> dict[str, np.ndarray]:
    kind, size = case.split()
    rng = np.random.default_rng(991)
    if kind == "diagonal":
        return {
            "precision": diagonal_precision(int(size)),
            "points": rng.normal(size=(100, int(size))),
        }
    if kind == "dense":
        return {
            "precision": dense_precision(int(size), np.random.default_rng(963)),
            "points": rng.normal(size=(100, int(size))),
        }
    draw = np.random.default_rng(975)
    component = draw.choice(3, size=int(size), p=[0.3, 0.3, 0.4])
    observations = draw.normal(
        np.array([-4.0, 0.0, 5.0])[component], np.array([1.0, 1.5, 1.0])[component]
    )
    centre = np.array([0.0, 0.3, -3.0, 0.5, 4.0, 0.2, 0.0, 0.3])
    return {
        "observations": observations,
        "n_components": np.asarray(3),
        "points": centre + 0.05 * rng.normal(size=(100, 8)),
    }


@pytest.mark.parametrize("case", CASES)
def test_gradient_routes_beside_jax_benchmark(
    benchmark: BenchmarkFixture, case: str
) -> None:
    inputs = _case(case)
    routes = ["autograd", "func"] + (["closed"] if "precision" in inputs else [])
    for route in routes:
        runs = [
            package("gradient", {**inputs, "route": np.asarray(route)})
            for _ in range(3)
        ]
        benchmark.extra_info[f"{route}_s"] = float(
            np.median([float(run.outputs["per_point"]) for run in runs])
        )
        benchmark.extra_info[f"{route}_peak_bytes"] = float(
            np.median([run.peak_bytes for run in runs])
        )
    keywords = (
        {"precision": inputs["precision"]}
        if "precision" in inputs
        else {"observations": inputs["observations"], "n_components": 3}
    )
    theirs = benchmark.pedantic(  # type: ignore[no-untyped-call]
        lambda: [jax.gradients(inputs["points"], **keywords) for _ in range(3)],
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["jax_eager_s"] = float(
        np.median([t.eager_seconds for t in theirs])
    )
    benchmark.extra_info["jax_jit_s"] = float(np.median([t.seconds for t in theirs]))
    benchmark.extra_info["jax_batch_per_point_s"] = float(
        np.median([t.batch_seconds for t in theirs]) / 100
    )
    benchmark.extra_info["jax_peak_bytes"] = float(
        np.median([t.peak_bytes for t in theirs])
    )

    assert theirs[0].gradients.shape == inputs["points"].shape
