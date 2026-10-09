"""Tuned HMC annealing against polished EM restarts on `mixture/ci`, at CI size (issue #1378, Part B).

`docs/experiments/030` runs 40 starts: restarts reach the best-known optimum
from 7, the anneal from 5 (McNemar p = 0.774), mean gaps 2.68 and 5.37
nats. This pins the direction of the gap on the first three starts.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from sal.qa import mixture_hmc_anneal as study
from sal.sim.fixtures import fixture


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The suite's one thread per process; restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.experiment
def test_restarts_reach_further_than_the_tuned_anneal_on_three_starts() -> None:
    # Realized on starts [0, 0..2] at one thread, 6.7 s: restarts' mean gap
    # 1.95 nats (1113.548 from all three), the anneal's 4.97.
    measurement = study.measure(
        study.instance(fixture("mixture", "ci")), study.TUNED_RAMP, 3
    )
    gaps = measurement.comparison.mean_gap()
    margin = study.TOLERANCE * study.BEST_KNOWN
    assert float(measurement.comparison.best.min()) >= study.BEST_KNOWN - margin
    assert int(measurement.comparison.spent.max()) <= study.BUDGET.size
    assert gaps["restarts"] < gaps["anneal"], gaps
