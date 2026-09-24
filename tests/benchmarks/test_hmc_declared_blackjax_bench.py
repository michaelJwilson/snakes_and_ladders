"""`sample.hmc.sample` on declared targets beside BlackJAX's HMC, with and without warm-up (issue #1008).

The integrator oracles are in `tests/validation/test_blackjax.py`, and the
goals in `test_goals.py` read their BlackJAX figures from this pair. Each row
times 1,000 transitions of ten leapfrog steps on the package's compiled chain
and records beside it BlackJAX's compiled chain at the same step, compilation
excluded, the median of `REPEATS` subprocess runs, with each side's peak
resident memory read in a fresh interpreter. The warm-up rows run 500 steps of
`blackjax.window_adaptation` against the package's `Adaptation(500, 0.65,
0.0)`, each followed by the same 1,000 transitions. Targets: Rosenbrock
(a = 1, b = 100) at d = 10, 100 from -1.2 at 0.01 / sqrt(d / 10), and
`diagonal_precision(d)` at d = 10², 10³, 10⁴ from the origin at
0.9 / (2 d^(1/4)) with the warm-up.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.validation import blackjax
from snakes_and_ladders.validation.gaussian import diagonal_precision
from snakes_and_ladders.validation.runner import available, package

pytestmark = pytest.mark.skipif(
    not available("blackjax"), reason="BlackJAX is the validation-blackjax extra"
)

REPEATS = 3

CELLS = [("rosenbrock", d, w) for d in (10, 100) for w in (0, 500)] + [
    ("gaussian", d, 500) for d in (100, 1_000, 10_000)
]


@pytest.mark.parametrize(("family", "dimension", "warmup"), CELLS)
def test_hmc_on_a_declared_target_beside_blackjax_benchmark(
    benchmark: BenchmarkFixture, family: str, dimension: int, warmup: int
) -> None:
    if family == "rosenbrock":
        start, step = np.full(dimension, -1.2), 0.01 / np.sqrt(dimension / 10)
        options: dict[str, object] = {"rosenbrock": (1.0, 100.0)}
        target = {"target": np.asarray(1), "constants": np.asarray([1.0, 100.0])}
    else:
        precision = diagonal_precision(dimension)
        start, step = np.zeros(dimension), 0.9 / (2.0 * dimension**0.25)
        options = {"precision": precision}
        target = {"target": np.asarray(0), "precision": precision}
    if warmup:
        runs = [
            blackjax.adapted_sample(
                start, step, 10, warmup, 1_000, 1008, 0.65, **options
            ).chain  # type: ignore[arg-type]
            for _ in range(REPEATS)
        ]
    else:
        runs = [
            blackjax.sample(
                options.get("precision"),  # type: ignore[arg-type]
                start,
                step,
                10,
                1_000,
                1008,
                rosenbrock=options.get("rosenbrock"),  # type: ignore[arg-type]
            )
            for _ in range(REPEATS)
        ]
    benchmark.extra_info["blackjax_s"] = float(np.median([run.seconds for run in runs]))
    benchmark.extra_info["blackjax_peak_bytes"] = float(
        np.median([run.peak_bytes for run in runs])
    )
    inputs = {
        **target,
        "position": start,
        "step_size": np.asarray(step),
        "n_steps": np.asarray(10),
        "n_draws": np.asarray(1_000),
        "seed": np.asarray(1008),
        "warmup": np.asarray(warmup),
        "target_acceptance": np.asarray(0.65),
        "store_chain": np.asarray(True),
    }
    ours = benchmark.pedantic(  # type: ignore[no-untyped-call]
        lambda: [package("hmc_declared", inputs) for _ in range(REPEATS)],
        rounds=1,
        iterations=1,
    )
    benchmark.extra_info["package_s"] = float(np.median([run.seconds for run in ours]))
    benchmark.extra_info["package_peak_bytes"] = float(
        np.median([run.peak_bytes or 0 for run in ours])
    )
