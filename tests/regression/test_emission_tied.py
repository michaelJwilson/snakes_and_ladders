"""One dispersion, or one concentration, shared by every state (issue #933, R3).

`NegativeBinomialEmission(..., tied=True)` solves one ``r`` on the score
summed over the states; `BetaBinomialEmission(..., tied=True)` one ``a + b``,
each state keeping its rate. Three referees: at one state the tied solve is
the untied one; the tied answer is the maximum of the likelihood itself,
found by `scipy.optimize` through `log_density` with no score in sight; and a
draw with one planted dispersion recovers it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sal.emissions import (
    BetaBinomialEmission,
    NegativeBinomialEmission,
    mstep,
)
from sal.sim.count_pairs import IndependentCountPair
from scipy.optimize import minimize, minimize_scalar

from tests._rows import every_value

#: The declared floor for a reordered sum through a bisection (#648).
FLOOR = 2e-06
N_OBS = 1_500


def _counts(seed: int, exposure: bool) -> dict[str, torch.Tensor | None]:
    """Negative-binomial counts from three states sharing ``r = 3``, and a posterior."""
    rng = np.random.default_rng([933, 3, seed])
    states = rng.integers(0, 3, N_OBS)
    offsets = rng.uniform(0.5, 2.0, N_OBS) if exposure else np.ones(N_OBS)
    means = np.array([2.0, 8.0, 25.0])[states] * offsets
    counts = rng.negative_binomial(3.0, 3.0 / (3.0 + means))
    posterior = np.full((N_OBS, 3), 0.05)
    posterior[np.arange(N_OBS), states] = 0.9
    return {
        "counts": torch.as_tensor(counts, dtype=torch.float64),
        "posterior": torch.as_tensor(posterior),
        "exposure": torch.as_tensor(offsets).unsqueeze(-1) if exposure else None,
    }


def _successes(seed: int) -> dict[str, torch.Tensor]:
    """Beta-binomial successes out of 40 from three states sharing ``a + b = 12``."""
    rng = np.random.default_rng([933, 4, seed])
    states = rng.integers(0, 3, N_OBS)
    rate = rng.beta(
        12.0 * np.array([0.2, 0.5, 0.8])[states],
        12.0 * np.array([0.8, 0.5, 0.2])[states],
    )
    posterior = np.full((N_OBS, 3), 0.05)
    posterior[np.arange(N_OBS), states] = 0.9
    return {
        "successes": torch.as_tensor(rng.binomial(40, rate), dtype=torch.float64),
        "posterior": torch.as_tensor(posterior),
    }


def _nb_start(tied: bool = True) -> NegativeBinomialEmission:
    return NegativeBinomialEmission([1.0, 1.0, 1.0], [1.0, 5.0, 20.0], tied=tied)


def _bb_start(tied: bool = True) -> BetaBinomialEmission:
    return BetaBinomialEmission([40.0] * 3, [1.0, 2.5, 4.0], [4.0, 2.5, 1.0], tied=tied)


@pytest.mark.critical
@pytest.mark.oracle
def test_one_tied_state_is_the_untied_solve() -> None:
    # At K = 1 the summed score is the one score, so the tied solve is the
    # per-state one; the digamma half is summed on the distinct counts here and
    # per observation there, so the pin is #648's floor.
    def check(exposure: bool) -> None:
        data = _counts(0, exposure)
        weights = data["posterior"][:, :1]  # type: ignore[index]
        values = data["counts"]
        assert values is not None
        offsets = None if data["exposure"] is None else data["exposure"].reshape(-1)
        mean = (weights.T @ values) / (
            weights.sum(0) if offsets is None else weights.T @ offsets
        )
        tied = mstep.solve_dispersion_tied(values, weights, mean, offsets)
        rate = float(mean[0]) if offsets is None else offsets * float(mean[0])
        untied = mstep.solve_dispersion(values, weights[:, 0], rate)
        assert abs(tied.value - untied.value) / untied.value < FLOOR
        assert tied.at_boundary == untied.at_boundary

    every_value([False, True], check)


@pytest.mark.critical
@pytest.mark.oracle
def test_one_tied_beta_binomial_state_is_the_untied_solve_bitwise() -> None:
    # One state's summed concentration score is its score, term for term.
    data = _successes(0)
    weights = data["posterior"][:, :1]
    tied = mstep.solve_beta_binomial_tied(
        data["successes"], weights, [40.0], [0.3], 5.0
    )
    untied = mstep.solve_beta_binomial(data["successes"], weights[:, 0], 40.0, 0.3, 5.0)
    assert float(tied.alpha[0]) == untied.alpha
    assert float(tied.beta[0]) == untied.beta
    assert tied.iterations == untied.iterations


@pytest.mark.critical
@pytest.mark.oracle
def test_the_tied_dispersion_maximizes_the_likelihood() -> None:
    # The referee never sees a score: it maximizes the weighted log-density
    # over log r at the profiled means, which are the means' MLE at any r.
    def check(exposure: bool) -> None:
        data = _counts(1, exposure)
        counts, posterior = data["counts"], data["posterior"]
        assert counts is not None
        assert posterior is not None
        fitted = _nb_start().reestimate(counts, posterior, data["exposure"]).emissions
        assert fitted.tied
        assert bool((fitted.dispersion == fitted.dispersion[0]).all())

        def negative(log_r: float) -> float:
            family = NegativeBinomialEmission([math.exp(log_r)] * 3, fitted.mean)
            return -float(
                (posterior * family.log_density(counts, data["exposure"])).sum()
            )

        best = minimize_scalar(
            negative, bounds=(-3.0, 5.0), method="bounded", options={"xatol": 1e-10}
        )
        assert (
            abs(float(fitted.dispersion[0]) - math.exp(best.x)) / math.exp(best.x)
            < 1e-6
        )

    every_value([False, True], check)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_tied_concentration_maximizes_the_likelihood() -> None:
    # Four free parameters --- three rates and one log concentration ---
    # maximized through `log_density` by L-BFGS-B.
    data = _successes(1)
    successes, posterior = data["successes"], data["posterior"]
    fitted = _bb_start().reestimate(successes, posterior).emissions
    assert fitted.tied

    def negative(point: np.ndarray) -> float:
        rates, total = point[:3], math.exp(point[3])
        family = BetaBinomialEmission([40.0] * 3, rates * total, (1.0 - rates) * total)
        return -float((posterior * family.log_density(successes)).sum())

    start = np.array([0.3, 0.5, 0.7, math.log(5.0)])
    best = minimize(
        negative,
        start,
        method="L-BFGS-B",
        bounds=[(1e-3, 1 - 1e-3)] * 3 + [(-2.0, 6.0)],
        options={"ftol": 1e-15, "gtol": 1e-10},
    )
    concentration = fitted.alpha + fitted.beta
    np.testing.assert_allclose(float(concentration[0]), math.exp(best.x[3]), rtol=1e-4)
    np.testing.assert_allclose(
        (fitted.alpha / concentration).numpy(), best.x[:3], atol=1e-5
    )
    assert (
        negative(
            np.append(
                (fitted.alpha / concentration).numpy(),
                math.log(float(concentration[0])),
            )
        )
        <= best.fun + 1e-7
    )


@pytest.mark.oracle
def test_a_planted_shared_dispersion_and_concentration_are_recovered() -> None:
    # The draws plant r = 3 and a + b = 12 in every state; scored at the
    # planted states, at 1,500 observations both land within 10%.
    data = _counts(2, exposure=True)
    counts, posterior = data["counts"], data["posterior"]
    assert posterior is not None
    posterior = (posterior == posterior.max(dim=1, keepdim=True).values).double()
    assert counts is not None
    assert posterior is not None
    dispersion = (
        _nb_start().reestimate(counts, posterior, data["exposure"]).emissions.dispersion
    )
    assert abs(float(dispersion[0]) - 3.0) / 3.0 < 0.1
    beta_binomial = _successes(2)
    planted = beta_binomial["posterior"]
    planted = (planted == planted.max(dim=1, keepdim=True).values).double()
    fitted = _bb_start().reestimate(beta_binomial["successes"], planted).emissions
    assert abs(float((fitted.alpha + fitted.beta)[0]) - 12.0) / 12.0 < 0.1


@pytest.mark.smoke
def test_the_tie_is_carried_by_the_pair_and_refused_when_broken() -> None:
    pair = IndependentCountPair(_nb_start(), _bb_start())
    data = _counts(3, exposure=False)
    counts, posterior = data["counts"], data["posterior"]
    assert counts is not None
    assert posterior is not None
    observations = torch.stack(
        [counts, torch.minimum(counts, torch.tensor(40.0))], dim=-1
    )
    fitted = pair.reestimate(observations, posterior).emissions
    assert fitted.total.tied
    assert fitted.successes.tied
    assert not _nb_start(tied=False).reestimate(counts, posterior).emissions.tied
    with pytest.raises(ValueError, match="tied dispersion"):
        NegativeBinomialEmission([1.0, 2.0], [1.0, 1.0], tied=True)
    with pytest.raises(ValueError, match="tied concentration"):
        BetaBinomialEmission([5.0, 5.0], [1.0, 1.0], [1.0, 2.0], tied=True)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_untied_dispersion_under_a_varying_exposure_maximizes_the_likelihood() -> (
    None
):
    # The term profiling drops is kept where the exposure varies, so each
    # state's r is the likelihood's maximum at its mean; before #933 it sat a
    # relative 5.6e-4 off on the tied draw.
    data = _counts(1, exposure=True)
    counts, posterior = data["counts"], data["posterior"]
    assert counts is not None
    assert posterior is not None
    fitted = (
        _nb_start(tied=False).reestimate(counts, posterior, data["exposure"]).emissions
    )
    for state in range(3):

        def negative(log_r: float, state: int = state) -> float:
            dispersion = fitted.dispersion.clone()
            dispersion[state] = math.exp(log_r)
            family = NegativeBinomialEmission(dispersion, fitted.mean)
            scores = family.log_density(counts, data["exposure"])[:, state]
            return -float((posterior[:, state] * scores).sum())

        best = minimize_scalar(
            negative, bounds=(-3.0, 5.0), method="bounded", options={"xatol": 1e-10}
        )
        found = float(fitted.dispersion[state])
        assert abs(found - math.exp(best.x)) / math.exp(best.x) < 1e-6
