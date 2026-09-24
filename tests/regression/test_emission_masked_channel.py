"""An unobserved channel: zero exposure or zero trials (issue #933, R2).

A zero exposure marks an observation's total **unobserved**, and a zero trial
count its successes. Such an entry scores log 1 and the M step drops it,
whatever count it carries, so the referee is the same family with those rows
removed --- checked bitwise, in the density and in the fit. The count pair
takes one covariate per channel, so each channel is masked on its own.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import BetaBinomialEmission, NegativeBinomialEmission
from snakes_and_ladders.sim.count_pairs import IndependentCountPair

from tests._rows import every_value

DISPERSION = torch.tensor([2.0, 5.0, 0.8], dtype=torch.float64)
MEAN = torch.tensor([1.5, 6.0, 20.0], dtype=torch.float64)
ALPHA = torch.tensor([2.0, 3.0, 0.7], dtype=torch.float64)
BETA = torch.tensor([4.0, 2.0, 1.1], dtype=torch.float64)
TRIALS = torch.tensor([30.0, 30.0, 30.0], dtype=torch.float64)
N_OBS = 400


def _draw(seed: int) -> dict[str, torch.Tensor]:
    """Counts, covariates with about a fifth masked per channel, and a posterior."""
    rng = np.random.default_rng([933, 2, seed])
    exposure = rng.uniform(0.2, 3.0, N_OBS)
    exposure[rng.random(N_OBS) < 0.2] = 0.0
    trials = rng.integers(1, 40, N_OBS).astype(float)
    trials[rng.random(N_OBS) < 0.2] = 0.0
    # A masked entry carries an arbitrary count: it must not be read.
    totals = rng.negative_binomial(2.0, 0.3, N_OBS).astype(float)
    successes = rng.binomial(np.maximum(trials, 1).astype(np.int64), 0.4).astype(float)
    successes[trials == 0] = rng.integers(0, 50, int((trials == 0).sum()))
    posterior = rng.random((N_OBS, 3))
    return {
        "exposure": torch.as_tensor(exposure).unsqueeze(-1),
        "trials": torch.as_tensor(trials).unsqueeze(-1),
        "totals": torch.as_tensor(totals),
        "successes": torch.as_tensor(successes),
        "posterior": torch.as_tensor(posterior / posterior.sum(axis=1, keepdims=True)),
    }


def _equal_parameters(first: object, second: object) -> None:
    left = first.named_parameters()  # type: ignore[attr-defined]
    right = second.named_parameters()  # type: ignore[attr-defined]
    assert left.keys() == right.keys()
    for name, tensor in left.items():
        assert torch.equal(tensor, right[name]), name


@pytest.mark.critical
@pytest.mark.oracle
def test_an_unobserved_total_scores_log_one_and_the_rest_as_without_it() -> None:
    def check(seed: int) -> None:
        data = _draw(seed)
        family = NegativeBinomialEmission(DISPERSION, MEAN)
        observed = data["exposure"][:, 0] > 0
        scores = family.log_density(data["totals"], data["exposure"])
        assert torch.equal(
            scores[~observed],
            torch.zeros(int((~observed).sum()), 3, dtype=torch.float64),
        )
        assert torch.equal(
            scores[observed],
            family.log_density(data["totals"][observed], data["exposure"][observed]),
        )

    every_value(range(3), check)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_negative_binomial_m_step_is_the_fit_without_unobserved_totals() -> None:
    def check(seed: int) -> None:
        data = _draw(seed)
        family = NegativeBinomialEmission(DISPERSION, MEAN)
        observed = data["exposure"][:, 0] > 0
        masked = family.reestimate(data["totals"], data["posterior"], data["exposure"])
        removed = family.reestimate(
            data["totals"][observed],
            data["posterior"][observed],
            data["exposure"][observed],
        )
        _equal_parameters(masked.emissions, removed.emissions)
        assert (masked.iterations, masked.at_boundary) == (
            removed.iterations,
            removed.at_boundary,
        )

    every_value(range(3), check)


@pytest.mark.critical
@pytest.mark.oracle
def test_unobserved_successes_score_log_one_and_fit_as_without_them() -> None:
    def check(seed: int) -> None:
        data = _draw(seed)
        family = BetaBinomialEmission(TRIALS, ALPHA, BETA)
        observed = data["trials"][:, 0] > 0
        scores = family.log_density(data["successes"], data["trials"])
        assert torch.equal(
            scores[~observed],
            torch.zeros(int((~observed).sum()), 3, dtype=torch.float64),
        )
        assert torch.equal(
            scores[observed],
            family.log_density(data["successes"][observed], data["trials"][observed]),
        )
        masked = family.reestimate(data["successes"], data["posterior"], data["trials"])
        removed = family.reestimate(
            data["successes"][observed],
            data["posterior"][observed],
            data["trials"][observed],
        )
        _equal_parameters(masked.emissions, removed.emissions)

    every_value(range(3), check)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_pair_masks_each_channel_on_its_own() -> None:
    # A row with no total still informs the successes, and one with no
    # successes the total: each channel's fit is its own family's without its
    # own masked rows, and a row masked in both scores log 1.
    data = _draw(7)
    pair = IndependentCountPair(
        NegativeBinomialEmission(DISPERSION, MEAN),
        BetaBinomialEmission(TRIALS, ALPHA, BETA),
    )
    observations = torch.stack([data["totals"], data["successes"]], dim=-1)
    covariate = torch.cat([data["exposure"], data["trials"]], dim=-1)
    scores = pair.log_density(observations, covariate)
    total = NegativeBinomialEmission(DISPERSION, MEAN).log_density(
        data["totals"], data["exposure"]
    )
    success = BetaBinomialEmission(TRIALS, ALPHA, BETA).log_density(
        data["successes"], data["trials"]
    )
    assert torch.equal(scores, total + success)
    both = (data["exposure"][:, 0] == 0) & (data["trials"][:, 0] == 0)
    assert bool(both.any())
    assert torch.equal(
        scores[both], torch.zeros(int(both.sum()), 3, dtype=torch.float64)
    )

    fitted = pair.reestimate(observations, data["posterior"], covariate).emissions
    has_total = data["exposure"][:, 0] > 0
    has_successes = data["trials"][:, 0] > 0
    alone_total = NegativeBinomialEmission(DISPERSION, MEAN).reestimate(
        data["totals"][has_total],
        data["posterior"][has_total],
        data["exposure"][has_total],
    )
    alone_successes = BetaBinomialEmission(TRIALS, ALPHA, BETA).reestimate(
        data["successes"][has_successes],
        data["posterior"][has_successes],
        data["trials"][has_successes],
    )
    _equal_parameters(fitted.total, alone_total.emissions)
    _equal_parameters(fitted.successes, alone_successes.emissions)


@pytest.mark.analytic
def test_a_masked_entry_leaves_the_gradient_finite() -> None:
    # The zero rate's `0 * log 0` is kept out of both branches of the `where`,
    # so a gradient through the density is a number, not `nan`.
    mean = MEAN.clone().requires_grad_(True)
    family = NegativeBinomialEmission(DISPERSION, mean)
    exposure = torch.tensor([[0.0], [1.5]], dtype=torch.float64)
    (gradient,) = torch.autograd.grad(
        family.log_density(torch.tensor([4.0, 3.0]), exposure).sum(), mean
    )
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.smoke
def test_a_negative_or_fractional_covariate_is_still_refused() -> None:
    family = NegativeBinomialEmission(DISPERSION, MEAN)
    with pytest.raises(ValueError, match="non-negative"):
        family.log_density(torch.tensor([1.0]), torch.tensor([[-1.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        family.log_density(torch.tensor([1.0]), torch.tensor([[float("nan")]]))
    successes = BetaBinomialEmission(TRIALS, ALPHA, BETA)
    for bad in (-1.0, 2.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            successes.log_density(torch.tensor([1.0]), torch.tensor([[bad]]))
