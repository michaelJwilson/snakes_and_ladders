"""HMC tuning as an HMM start on `count_hmm_reference/ci`, every start polished by Baum-Welch (issue #1390, question 6).

`docs/experiments/038` runs the stress cell (8,000 positions) from eight
shared starts. This pins, on the CI cell's first start, the three readings it
reports: what the step pilot spends, T0 stated in `|log L| / n`, and the
rare levels recovered after Baum-Welch.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from sal.fixtures import Scale
from sal.qa import count_hmm_hmc_tuning as study
from sal.sim.fixtures import fixture


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The suite's one thread per process; restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.experiment
def test_the_tuned_anneal_at_ten_scales_recovers_the_restarts_rare_levels() -> None:
    # Realized on start [1393, 0] at one thread, 2.4 s: both reach -7610.8566
    # and miss 14.0%, recovering levels 1, 2, 3 and 7; the pilot chose 3e-3
    # for 49 of the 400 passes, and |log L| / n at the start is 7.761.
    # The registry's CI cell, named literally for the problem-marker scan.
    declared = fixture("count_hmm_reference", "ci")
    # The study's cell at the same tier, checked to be that draw.
    instance = study.build(Scale.CI)
    assert instance.params.observations_digest == declared.params.observations_digest
    # The first shared start, as the stress run draws it.
    start = study.random_start(instance, np.random.default_rng([study.SEED, 0]))
    # The baseline: Baum-Welch for the whole budget.
    restart = study.run_restart(instance, start)
    # The anneal from 10 x |log L| / n, its step chosen by a 2-proposal pilot.
    annealed = study.run_anneal(
        instance, start, 10.0, np.random.default_rng([study.SEED, 0, 1])
    )

    # The scale is the start's negative log-likelihood per position.
    n = instance.cell.observations.shape[0]
    with torch.no_grad():
        at_start = float(instance.objective(start))
    assert annealed.scale == pytest.approx(abs(at_start) / n, rel=1e-12)
    assert annealed.scale == pytest.approx(7.761, abs=1e-3)
    # The pilot's charge: 1 + 3 candidates x 2 proposals x 8 steps.
    assert annealed.tuner == 1 + len(study.STEP_GRID) * 2 * study.N_STEPS
    for result in (restart, annealed):
        assert result.gradients + result.polish_iterations <= study.PASSES
    assert annealed.log_likelihood == pytest.approx(restart.log_likelihood, rel=1e-9)
    # Both recover the two 1% levels as their own states; level 4 holds none.
    assert annealed.recovered == restart.recovered
    assert annealed.recovered[1:3] == (1, 1)
    assert annealed.recovered[4] == -1
