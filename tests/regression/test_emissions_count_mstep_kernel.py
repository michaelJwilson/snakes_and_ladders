"""The compiled count M step against the batched torch solves it replaces as the default (issue #922).

`src/count_mstep.rs` runs each state's bisection in Rust and evaluates every
``sum_u w_u (digamma(u + x) - digamma(x))`` as ``sum_j T_j / (x + j)``, the
tails of the weights, with no special function. The batched torch solves
(`_solve_dispersion_batched`, #918; `_solve_beta_binomial_batched`, #892) stay
as the oracle. A bisection's answer is set by the signs of its scores, so the
two agree bitwise unless a score within rounding of zero flips sign; the pin
is #648's 2e-06 floor, and the difference each draw measures is stated.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal import emissions
from sal.backend import Backend
from sal.emissions import BetaBinomialEmission, CountPairEmission, mstep
from sal.opt.mixture import responsibilities_torch
from sal.search.projection import flatten, project
from sal.sim.count_pairs import binned_model
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture

#: The declared floor for a reordered sum through a bisection (#648).
FLOOR = 2e-06


def _mixture(tier: str) -> tuple[torch.Tensor, torch.Tensor, CountPairEmission]:
    params = fixture("emission_mixture", tier).params
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    values = torch.as_tensor(
        simulate_emission_mixture(params).observations, dtype=torch.float64
    )
    posterior = responsibilities_torch(
        values, torch.log(torch.as_tensor(params.weights, dtype=torch.float64)), truth
    )
    return values, posterior, truth


def _starts(
    family: BetaBinomialEmission | CountPairEmission,
) -> tuple[list[float], list[float]]:
    named = family.named_parameters()
    totals = [float(a + b) for a, b in zip(named["alpha"], named["beta"], strict=True)]
    rates = [float(a) / t for a, t in zip(named["alpha"], totals, strict=True)]
    return rates, totals


def _relative(first: list[float], second: list[float]) -> float:
    return max(abs(a - b) / abs(b) for a, b in zip(first, second, strict=True))


@pytest.mark.analytic
def test_the_tails_sum_is_the_digamma_difference() -> None:
    # sum_u w_u (digamma(u + x) - digamma(x)) = sum_j T_j / (x + j), an
    # identity for integer u; the two sum in different orders, so they agree
    # to a float64 reduction's rounding.
    rng = np.random.default_rng(922)
    counts = torch.as_tensor(rng.integers(0, 60, 500), dtype=torch.float64)
    weights = torch.as_tensor(rng.random((2, 500)), dtype=torch.float64)
    tails = mstep.weight_tails(counts, weights)
    for x in (0.03, 1.7, 250.0):
        by_tails = (tails / (x + np.arange(tails.shape[1]))).sum(axis=1)
        by_digamma = (
            weights
            * (
                torch.digamma(counts + x)
                - torch.digamma(torch.tensor(x, dtype=torch.float64))
            )
        ).sum(dim=1)
        np.testing.assert_allclose(by_tails, by_digamma.numpy(), rtol=1e-12)


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("tier", ["ci", "stress"])
def test_the_kernel_is_the_batched_solves_on_the_emission_mixture(tier: str) -> None:
    # Measured: the beta-binomial bitwise on both tiers; the negative
    # binomial bitwise on ci and within 1.2e-12 relative on stress, the
    # boundary decisions and iteration counts the oracle's.
    values, posterior, truth = _mixture(tier)
    totals, successes = values[:, 0], values[:, 1]
    means = [float(m) for m in (posterior.T @ totals) / posterior.sum(dim=0)]
    rust = mstep.solve_dispersion_rust(totals, posterior, means)
    oracle = mstep.solve_dispersion_batched(totals, posterior, means)
    assert [r.at_boundary for r in rust] == [o.at_boundary for o in oracle]
    assert [r.iterations for r in rust] == [o.iterations for o in oracle]
    assert _relative([r.value for r in rust], [o.value for o in oracle]) < FLOOR
    rates, concentrations = _starts(truth)
    rust_bb = mstep.solve_beta_binomial_rust(
        successes, posterior, totals, rates, concentrations
    )
    oracle_bb = mstep.solve_beta_binomial_batched(
        successes, posterior, totals, rates, concentrations
    )
    assert [r.iterations for r in rust_bb] == [o.iterations for o in oracle_bb]
    assert [r.at_boundary for r in rust_bb] == [o.at_boundary for o in oracle_bb]
    assert _relative([r.alpha for r in rust_bb], [o.alpha for o in oracle_bb]) < FLOOR
    assert _relative([r.beta for r in rust_bb], [o.beta for o in oracle_bb]) < FLOOR


@pytest.mark.oracle
@pytest.mark.backend
def test_the_kernel_takes_a_trial_count_per_component() -> None:
    # The projection's success channel: 100 components, one trial count each,
    # so the failure tails differ per component. Measured bitwise.
    params = binned_model(fixture("spatio_sequential_counts", "ci").params.model, 1)
    instance = project(params, 400, np.random.default_rng([922, 1]))
    truth = flatten(params)
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    k = truth.n_states
    posterior = responsibilities_torch(
        values, torch.full((k,), -float(np.log(k)), dtype=torch.float64), truth
    )
    channel = truth._successes
    assert isinstance(channel, BetaBinomialEmission)
    trials = [float(t) for t in channel.trials]
    rates, concentrations = _starts(channel)
    rust = mstep.solve_beta_binomial_rust(
        values[:, 1], posterior, trials, rates, concentrations
    )
    oracle = mstep.solve_beta_binomial_batched(
        values[:, 1], posterior, trials, rates, concentrations
    )
    assert _relative([r.alpha for r in rust], [o.alpha for o in oracle]) < FLOOR
    assert _relative([r.beta for r in rust], [o.beta for o in oracle]) < FLOOR


@pytest.mark.smoke
def test_the_m_step_backend_selects_the_route(monkeypatch: pytest.MonkeyPatch) -> None:
    values, posterior, truth = _mixture("ci")
    assert emissions.M_STEP_BACKEND is Backend.RUST
    compiled = truth.reestimate(values, posterior).emissions
    monkeypatch.setattr(emissions, "M_STEP_BACKEND", Backend.PYTHON)
    # The package forwards the setting to the module that reads it (#1010);
    # a copy would leave the solves compiled and compare them to themselves.
    assert mstep.M_STEP_BACKEND is Backend.PYTHON
    batched = truth.reestimate(values, posterior).emissions
    for name, tensor in compiled.named_parameters().items():
        reference = batched.named_parameters()[name]
        assert float(((tensor - reference).abs() / reference.abs()).max()) < FLOOR
