"""HMC starts against a Baum-Welch restart on `count_hmm_reference/ci`, every one polished by Baum-Welch (issue #1393).

`docs/experiments/034` runs the stress cell (8,000 positions) from ten
shared starts. This pins, on the CI cell's first start, the reading against
#1393's 48.7%: the tuned anneal, polished by Baum-Welch, reaches the
Baum-Welch restart's optimum from the same start.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from sal.fixtures import Scale
from sal.qa import count_hmm_reference_starts as study


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The suite's one thread per process; restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.experiment
def test_the_tuned_anneal_reaches_the_restarts_optimum_on_the_ci_cell() -> None:
    # Realized on start [1393, 0] at one thread, 2.5 s: both reach
    # -7610.8566 and miss 14.0% (the CI cell's rare levels hold 20 to 40 of
    # 1,000 positions); #1393's reading was the anneal 2,300 nats short.
    instance = study.build(Scale.CI)
    start = study.random_start(instance, np.random.default_rng([study.SEED, 0]))
    restart = study.run_restart(instance, start)
    annealed = study.run_anneal(
        instance, start, 100.0, np.random.default_rng([study.SEED, 0, 1])
    )

    for result in (restart, annealed):
        assert result.gradients + result.polish_iterations <= study.PASSES
    assert annealed.log_likelihood == pytest.approx(restart.log_likelihood, rel=1e-9)
    assert annealed.missed == pytest.approx(restart.missed, abs=1e-12)
