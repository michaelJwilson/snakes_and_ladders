"""A trial count per observation, and what it does and does not reproduce (#631).

``BetaBinomialEmission`` keeps its per-state trial count as the reference (so
:attr:`mean` and :attr:`variance` keep a value) and a covariate overrides it.
At a constant covariate, scoring is bitwise (one broadcast, no reduction); the
M step is not, since its sums reorder when ``n`` is a vector (issue #648).
"""

from __future__ import annotations

import pytest
import torch
from snakes_and_ladders.emissions import BetaBinomialEmission
from snakes_and_ladders.sandbox.count_emissions import (
    BetaBinomialEmission as Conserved,
)

TRIALS = torch.tensor([20.0, 20.0])
ALPHA = torch.tensor([2.0, 5.0])
BETA = torch.tensor([5.0, 2.0])
SEED = 20260916


@pytest.fixture
def observations() -> tuple[torch.Tensor, torch.Tensor]:
    """Counts and a normalized posterior, from one seeded generator."""
    generator = torch.Generator().manual_seed(SEED)
    counts = torch.randint(0, 21, (1, 200), generator=generator).to(torch.float64)
    posterior = torch.rand(1, 200, 2, generator=generator, dtype=torch.float64)
    return counts, posterior / posterior.sum(dim=-1, keepdim=True)


@pytest.mark.critical
@pytest.mark.oracle
def test_scoring_at_a_constant_covariate_is_the_conserved_family_bitwise() -> None:
    # The half of the ticket's bit-for-bit claim that holds. `log_density` is
    # one broadcast expression, so a constant `n` folded per state and the same
    # `n` per observation evaluate the same terms in the same order.
    counts = torch.tensor([0, 1, 4, 19, 20])
    live = BetaBinomialEmission(TRIALS, ALPHA, BETA)
    conserved = Conserved(TRIALS, ALPHA, BETA)
    constant = torch.full((5, 1), 20.0, dtype=torch.float64)

    assert torch.equal(
        live.log_density(counts, constant), conserved.log_density(counts)
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_the_m_step_at_a_constant_covariate_agrees_to_a_tolerance(
    observations: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # A tolerance, and data-dependent: exact on this draw, 1.0e-06 relative in
    # `alpha` on another 200-observation draw, four orders outside
    # `_solve_beta_binomial`'s 1e-10 (issue #648). The bound holds on both.
    counts, posterior = observations
    live = BetaBinomialEmission(TRIALS, ALPHA, BETA)
    conserved = Conserved(TRIALS, ALPHA, BETA)
    constant = torch.full_like(counts, 20.0)

    fitted = live.reestimate(counts, posterior, constant).emissions
    reference = conserved.reestimate(counts, posterior).emissions
    moved = max(
        float(((fitted.alpha - reference.alpha).abs() / reference.alpha).max()),
        float(((fitted.beta - reference.beta).abs() / reference.beta).max()),
    )

    assert moved < 2e-06, f"the M step moved {moved:.3e}, past what #648 recorded"


@pytest.mark.critical
@pytest.mark.oracle
def test_no_covariate_is_the_conserved_family(
    observations: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # The path every existing caller takes, which is what makes this an
    # extension rather than a second implementation. The beta-binomial's M step
    # is still bitwise -- #649's matmul rewrite is in the negative binomial's
    # profiled mean, which this family does not have.
    counts, posterior = observations
    live = BetaBinomialEmission(TRIALS, ALPHA, BETA)
    conserved = Conserved(TRIALS, ALPHA, BETA)

    fitted = live.reestimate(counts, posterior).emissions
    reference = conserved.reestimate(counts, posterior).emissions

    assert torch.equal(live.log_density(counts[0]), conserved.log_density(counts[0]))
    assert torch.equal(fitted.alpha, reference.alpha)
    assert torch.equal(fitted.beta, reference.beta)


@pytest.mark.critical
@pytest.mark.smoke
def test_an_observation_above_its_own_trial_count_scores_negative_infinity() -> None:
    # On the joint form's precedent: the support is a property of the pair, so
    # a sequence carrying one impossible site is scored, not refused.
    live = BetaBinomialEmission(TRIALS, ALPHA, BETA)
    counts = torch.tensor([5, 9])
    supplied = torch.tensor([[20.0], [8.0]])

    scored = live.log_density(counts, supplied)

    assert torch.isfinite(scored[0]).all()
    assert torch.isinf(scored[1]).all()


@pytest.mark.critical
@pytest.mark.infra
def test_a_trial_count_materialised_per_state_is_refused() -> None:
    # The layout is the whole cost: shaped `(..., 1)` three of the nine lgamma
    # terms are `n_obs * K`, which is what the per-state count costs today;
    # materialised `(..., K)` all nine are, and the term count triples.
    live = BetaBinomialEmission(TRIALS, ALPHA, BETA)

    with pytest.raises(ValueError, match="singleton axis"):
        live.log_density(torch.tensor([1, 2]), torch.full((2, 2), 20.0))


@pytest.mark.critical
@pytest.mark.infra
def test_the_state_count_is_read_off_alpha_not_the_trial_count() -> None:
    # A trial count supplied per observation says nothing about how many states
    # there are, so reading `n_states` off it would be wrong the moment one is.
    live = BetaBinomialEmission(TRIALS, ALPHA, BETA)

    assert live.n_states == int(ALPHA.shape[0])
