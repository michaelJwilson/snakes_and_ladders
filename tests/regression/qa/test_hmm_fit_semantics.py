"""sal's Baum-Welch semantics against a downstream caller's on `count_hmm_reference/ci` (issue #1412).

`docs/experiments/039` runs both to their relative tolerance: per-state
dispersion reaches the higher log-likelihood, -7615.48 against -7621.95, and
misses 123 of 1,000 positions against the shared dispersion's 40. This pins
both directions at 20 iterations each.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch
from sal.qa import hmm_fit_semantics as study
from sal.sim.fixtures import fixture


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The suite's one thread per process; restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.experiment
def test_per_state_dispersion_fits_higher_and_recovers_fewer_levels() -> None:
    params = fixture("count_hmm_reference", "ci").params
    cell = params.instance()
    own, downstream = (
        study.fit(
            cell,
            params.n_fit_states,
            replace(semantics, max_iterations=20),
            params.stay,
        )
        for semantics in (study.SAL, study.DOWNSTREAM)
    )
    # NB the tied fit holds one value across states; the untied one does not
    assert float(downstream.dispersion.std()) == pytest.approx(0.0, abs=1e-12)
    assert float(downstream.concentration.std()) == pytest.approx(0.0, abs=1e-9)
    assert own.log_likelihood > downstream.log_likelihood
    assert downstream.missed < own.missed
