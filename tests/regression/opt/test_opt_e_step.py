"""One log-density pass for both halves of an E step (issue #924).

`opt.mixture.e_step` returns what `mixture_log_likelihood` and
`responsibilities` return, from one evaluation of the components'
log-density. The referee is the two functions themselves, bitwise, on a
continuous family and on the count pair, the two EM entry points' families.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import CountPairEmission, GaussianEmission
from sal.opt.mixture import (
    e_step,
    mixture_log_likelihood,
    responsibilities_torch,
)
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture


@pytest.mark.analytic
def test_the_e_step_is_both_functions_bitwise_on_the_count_pair() -> None:
    params = fixture("emission_mixture", "stress").params
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    values = torch.as_tensor(
        simulate_emission_mixture(params).observations, dtype=torch.float64
    )
    log_weight = torch.log(torch.as_tensor(params.weights, dtype=torch.float64))
    evidence, posterior = e_step(values, log_weight, truth)
    assert torch.equal(evidence, mixture_log_likelihood(values, log_weight, truth))
    assert torch.equal(posterior, responsibilities_torch(values, log_weight, truth))


@pytest.mark.analytic
def test_the_e_step_is_both_functions_bitwise_on_a_gaussian() -> None:
    rng = np.random.default_rng(924)
    values = torch.as_tensor(
        np.concatenate([rng.normal(-2.0, 1.0, 400), rng.normal(3.0, 0.5, 600)]),
        dtype=torch.float64,
    )
    family = GaussianEmission(
        torch.tensor([-1.0, 2.0], dtype=torch.float64),
        torch.tensor([1.5, 1.0], dtype=torch.float64),
        1e-12,
    )
    log_weight = torch.log(torch.tensor([0.3, 0.7], dtype=torch.float64))
    evidence, posterior = e_step(values, log_weight, family)
    # The torch route is the one `e_step` shares; the streamed default is
    # pinned to it at a tolerance in test_opt_mixture.py (#997).
    assert torch.equal(
        evidence,
        mixture_log_likelihood(values, log_weight, family, backend=Backend.PYTHON),
    )
    assert torch.equal(posterior, responsibilities_torch(values, log_weight, family))


@pytest.mark.analytic
def test_lgamma_on_the_distinct_counts_is_the_direct_form_bitwise() -> None:
    # The count families' log-densities take lgamma(counts + shift) on the
    # distinct counts and gather; elementwise, so the direct form bitwise,
    # and the direct form wherever the shapes or autodiff rule the gather out.
    from sal.emissions.counts import lgamma_shifted

    rng = np.random.default_rng(924)
    counts = torch.as_tensor(rng.integers(0, 300, (3_000, 1)), dtype=torch.float64)
    shift = torch.as_tensor(rng.random(10) * 50.0, dtype=torch.float64)
    assert torch.equal(lgamma_shifted(counts, shift), torch.lgamma(counts + shift))
    per_observation = torch.as_tensor(rng.random((3_000, 10)), dtype=torch.float64)
    assert torch.equal(
        lgamma_shifted(counts, per_observation),
        torch.lgamma(counts + per_observation),
    )
    tracked = shift.clone().requires_grad_(True)
    assert torch.equal(
        lgamma_shifted(counts, tracked).detach(), torch.lgamma(counts + shift)
    )
