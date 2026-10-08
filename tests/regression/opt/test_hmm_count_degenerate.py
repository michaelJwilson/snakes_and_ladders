"""Two count-fit degeneracies end Baum-Welch with `Stop.DEGENERATE`, not a raise (issue #1346).

- **Negative-binomial dispersion at its floor:** a state whose dispersion
  solve lands at or under `DISPERSION_FLOOR` is reported degenerate and ends
  the fit with the last valid parameters, naming the state. The input that
  underflowed the bracket to `r = 0` and raised on both M-step routes now
  returns.
- **Beta-binomial under two mean trials:** the state holds its concentration
  and re-estimates its rate; at one trial that rate is the Bernoulli maximum,
  the posterior-weighted mean.

Thresholds, declared before the run: the floor is `float64` epsilon; the
planted start's dispersion is 1e-30, fourteen decades under it; the rate's
bisection tolerance is 1e-10, so the Bernoulli rate is pinned at 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import BetaBinomialEmission, NegativeBinomialEmission, mstep
from sal.opt.em import EmConfig
from sal.opt.hmm.estimation import baum_welch_family
from sal.opt.termination import Stop

#: The dispersion planted on state 0's start, under the floor.
PLANTED = 1e-30

#: The Bernoulli rate's tolerance: ten times the bisection's own 1e-10.
RATE_TOLERANCE = 1e-9

BUDGET = EmConfig(max_iterations=200)


def _uniform(m: int) -> tuple[torch.Tensor, torch.Tensor]:
    log_initial = torch.log(torch.full((m,), 1.0 / m, dtype=torch.float64))
    log_transition = torch.log(torch.full((m, m), 1.0 / m, dtype=torch.float64))
    return log_initial, log_transition


def _zeros_and_counts() -> np.ndarray:
    """State 0 emits zeros, state 1 a negative binomial of mean 40: 8 chains of 200."""
    rng = np.random.default_rng(1346)
    labels = rng.integers(0, 2, (8, 200))
    draws = rng.negative_binomial(5.0, 5.0 / 45.0, labels.shape)
    return np.where(labels == 0, 0, draws).astype(np.float64)


@pytest.mark.analytic
@pytest.mark.hmm
def test_the_floor_is_float64_epsilon() -> None:
    assert mstep.DISPERSION_FLOOR == float(np.finfo(np.float64).eps)


@pytest.mark.bug
@pytest.mark.hmm
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_an_underflowing_dispersion_is_degenerate_not_raised(
    backend: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    # State 0 holds one count at weight 1e-320: its identifiable bound is
    # subnormal and the bracket's lower end underflowed to 0, which raised
    # `ParameterDomainError` (Rust) and `math domain error` (torch) before.
    monkeypatch.setattr(mstep, "M_STEP_BACKEND", backend)
    counts = np.r_[np.zeros(99), 3.0]
    posterior = np.full((100, 2), 0.5)
    posterior[-1] = [1e-320, 1.0]
    step = NegativeBinomialEmission([1.0, 1.0], [1.0, 1.0]).reestimate(
        counts, posterior
    )
    assert not step.converged
    assert step.degenerate == (0,)
    assert float(step.components.dispersion[0]) == mstep.DISPERSION_FLOOR


@pytest.mark.end2end
@pytest.mark.hmm
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_a_planted_zero_dispersion_ends_degenerate_with_the_start(
    backend: Backend,
) -> None:
    # Started at r = 1e-30 on the all-zero state, the first M step solves under
    # the floor; the fit returns the start, naming state 0.
    start = NegativeBinomialEmission([PLANTED, 1.0], [1e-3, 30.0])
    fit = baum_welch_family(
        _zeros_and_counts(), *_uniform(2), start, BUDGET, backend=backend
    )
    assert fit.termination.reason is Stop.DEGENERATE
    assert fit.termination.iterations == 1
    assert fit.unsettled is not None
    assert fit.unsettled.states == (0,)
    assert torch.equal(fit.components.dispersion, start.dispersion)
    assert torch.equal(fit.components.mean, start.mean)


@pytest.mark.analytic
@pytest.mark.hmm
def test_a_bracket_above_the_floor_is_unchanged() -> None:
    # The floor moves the lower end only where `upper * 1e-9` is under it.
    for upper in (2.3e-7, 1.0, 1e6):
        assert mstep._dispersion_lower(upper) == upper * 1e-9


@pytest.mark.analytic
@pytest.mark.hmm
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
def test_a_beta_binomial_at_one_trial_holds_its_concentration(
    backend: Backend,
) -> None:
    # At one trial the beta-binomial is a Bernoulli: the rate is identified and
    # is the posterior-weighted mean; the concentration is not, and is held.
    rng = np.random.default_rng(1346)
    labels = rng.integers(0, 2, (6, 150))
    successes = (rng.random(labels.shape) < np.where(labels == 0, 0.2, 0.8)).astype(
        np.float64
    )
    start = BetaBinomialEmission([1.0, 1.0], [1.0, 3.0], [3.0, 1.0])
    fit = baum_welch_family(successes, *_uniform(2), start, BUDGET, backend=backend)
    assert fit.termination.reason is Stop.DEGENERATE
    assert fit.unsettled is not None
    assert fit.unsettled.states == (0, 1)
    assert torch.equal(fit.components.concentration, start.concentration)
    # The rate is the Bernoulli maximum at the fit's last posterior.
    step = start.reestimate(successes.reshape(-1), np.full((successes.size, 2), 0.5))
    assert step.degenerate == (0, 1)
    rate = step.components.alpha / step.components.concentration
    expected = float(successes.mean())
    assert torch.allclose(
        rate, torch.full_like(rate, expected), rtol=0.0, atol=RATE_TOLERANCE
    )
