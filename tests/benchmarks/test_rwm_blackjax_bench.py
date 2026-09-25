"""`sample.metropolis.random_walk` beside BlackJAX's random walk, per target (issue #1006).

The draw-for-draw oracle is in `tests/validation/test_blackjax.py`, and the
goals in `test_goals.py` read their BlackJAX figures from this pair. Each row
times 1,000 transitions of the package's compiled chain and records beside it
BlackJAX's `additive_step_random_walk` at the same scale, compiled,
compilation excluded, the median of `REPEATS` subprocess runs, with each
side's peak resident memory read in a fresh interpreter. The work per
transition is one energy evaluation on both sides, so the pair is a runtime
ratio. Targets: `diagonal_precision(d)` at d = 10², 10³, 10⁴ from the origin
at step 1.2 / sqrt(d), and Rosenbrock (a = 1, b = 100) at d = 10, 100 from
-1.2 at 0.06 / sqrt(d).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from sal.opt.testfunctions import Rosenbrock
from sal.sample import metropolis
from sal.validation import blackjax
from sal.validation.gaussian import GaussianTarget, diagonal_precision

from tests._frameworks import requires
from tests.validation._goals import median_package

pytestmark = requires("blackjax")

#: Subprocess runs whose median BlackJAX's figure is.
REPEATS = 3

N_DRAWS = 1_000

TARGETS = [("gaussian", d) for d in (100, 1_000, 10_000)] + [
    ("rosenbrock", d) for d in (10, 100)
]


@pytest.mark.parametrize(("family", "dimension"), TARGETS)
def test_random_walk_beside_blackjax_benchmark(
    benchmark: BenchmarkFixture, family: str, dimension: int
) -> None:
    if family == "gaussian":
        precision = diagonal_precision(dimension)
        objective: GaussianTarget | Rosenbrock = GaussianTarget(precision)
        start, step = np.zeros(dimension), 1.2 / np.sqrt(dimension)
        options: dict[str, object] = {"precision": precision}
        target = {"target": np.asarray(0), "precision": precision}
    else:
        objective = Rosenbrock(dimension)
        start, step = np.full(dimension, -1.2), 0.06 / np.sqrt(dimension)
        options = {"rosenbrock": (1.0, 100.0)}
        target = {"target": np.asarray(1), "constants": np.asarray([1.0, 100.0])}
    runs = [
        blackjax.random_walk(start, step, N_DRAWS, 1006, **options)  # type: ignore[arg-type]
        for _ in range(REPEATS)
    ]
    benchmark.extra_info["blackjax_s"] = float(np.median([run.seconds for run in runs]))
    benchmark.extra_info["blackjax_acceptance"] = runs[0].acceptance
    benchmark.extra_info["blackjax_peak_bytes"] = float(
        np.median([run.peak_bytes for run in runs])
    )
    inputs = {
        **target,
        "position": start,
        "step_size": np.asarray(step),
        "n_draws": np.asarray(N_DRAWS),
        "seed": np.asarray(1006),
        "warmup": np.asarray(0),
        "store_chain": np.asarray(True),
    }
    benchmark.extra_info["package_peak_bytes"] = median_package(
        "random_walk_sample", inputs, "peak_bytes", repeats=REPEATS
    )
    chain = benchmark.pedantic(  # type: ignore[no-untyped-call]
        lambda: metropolis.random_walk(
            objective,
            np.random.default_rng(1006),
            N_DRAWS,
            step_size=step,
            theta0=torch.as_tensor(start),
        ),
        rounds=REPEATS,
        iterations=1,
    )
    assert 0.05 < chain.acceptance_rate < 0.95
