"""`sample.hmc` beside BlackJAX's HMC, per transition, on one target per size (issue #963).

The integrators are pinned in `tests/validation/test_blackjax.py`, and the
runtime goals in `test_goals.py`. Each row times 1,000 transitions of the
package's HMC at unit mass, ten leapfrog steps each, and records beside it
BlackJAX's compiled chain on the same target at the same step and length,
compilation excluded, the median of `REPEATS` subprocess runs, and the peak
resident memory each side's chain adds, each read in a fresh interpreter
(#987). The work per
transition is identical by construction, so the pair is a runtime ratio.

The target is a diagonal Gaussian with precisions from 1 to 4 at every size;
the step is 0.9 / (2 d^(1/4)), which holds the acceptance near 0.95 as d
grows. Sizes: d = 10 is the gate size, and d = 10², 10³ and 10⁴ the stress
sizes a speedup is read at (root `CLAUDE.md`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.validation import blackjax
from snakes_and_ladders.validation.gaussian import GaussianTarget, diagonal_precision

from tests._frameworks import requires
from tests.validation._goals import median_package

pytestmark = requires("blackjax")

#: Subprocess runs whose median BlackJAX's figure is.
REPEATS = 3

#: Transitions per timed chain, and leapfrog steps per transition.
N_DRAWS, N_STEPS = 1_000, 10


def step_size(dimension: int) -> float:
    """The step at ``dimension``: 0.9 over the largest scale, shrunk as ``d^(1/4)``."""
    return float(0.9 / (2.0 * dimension**0.25))


@pytest.mark.parametrize("dimension", [10, 100, 1_000, 10_000])
def test_hmc_beside_blackjax_benchmark(
    benchmark: BenchmarkFixture, dimension: int
) -> None:
    precision = diagonal_precision(dimension)
    step = step_size(dimension)
    runs = [
        blackjax.sample(precision, np.zeros(dimension), step, N_STEPS, N_DRAWS, 963)
        for _ in range(REPEATS)
    ]
    benchmark.extra_info["blackjax_s"] = float(np.median([run.seconds for run in runs]))
    benchmark.extra_info["blackjax_acceptance"] = runs[0].acceptance
    benchmark.extra_info["blackjax_peak_bytes"] = float(
        np.median([run.peak_bytes for run in runs])
    )
    inputs = {
        "precision": precision,
        "step_size": np.asarray(step),
        "n_steps": np.asarray(N_STEPS),
        "n_draws": np.asarray(N_DRAWS),
        "seed": np.asarray(963),
    }
    benchmark.extra_info["package_peak_bytes"] = median_package(
        "hmc_sample", inputs, "peak_bytes", repeats=REPEATS
    )

    chain = benchmark.pedantic(  # type: ignore[no-untyped-call]
        lambda: hmc.sample(
            GaussianTarget(precision),
            torch.Generator().manual_seed(963),
            N_DRAWS,
            step_size=step,
            n_steps=N_STEPS,
        ),
        rounds=3,
        iterations=1,
    )
    benchmark.extra_info["acceptance"] = chain.acceptance_rate

    assert 0.0 < chain.acceptance_rate <= 1.0
