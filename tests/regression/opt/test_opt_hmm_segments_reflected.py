"""Transition kernels per sequence, and states reflected in their success rate (issue #933, R9).

`baum_welch_family` takes ``log_transition`` of shape ``(n_sequences,
length - 1, m, m)``, one kernel per step of each sequence, held fixed as the
per-step kernel is. `ReflectedEmission` doubles a beta-binomial or count-pair
family into ``2K`` states whose second half emits at rate ``1 - p``. Referees:
identical per-sequence kernels are the shared per-step kernel bitwise, and
distinct ones score each sequence as its own chain; a reflection scores as the
exchanged family and its M step maximizes the unfolded likelihood, found by
`scipy.optimize`; and a planted reflected chain is recovered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from snakes_and_ladders.emissions import BetaBinomialEmission, NegativeBinomialEmission
from snakes_and_ladders.opt.hmm import baum_welch_family
from snakes_and_ladders.sim.count_pairs import IndependentCountPair, ReflectedEmission

TRIALS = 40.0


def _kernel(stay: float, m: int) -> torch.Tensor:
    matrix = np.full((m, m), (1.0 - stay) / (m - 1))
    np.fill_diagonal(matrix, stay)
    return torch.log(torch.as_tensor(matrix))


def _chains(
    family: BetaBinomialEmission | ReflectedEmission, n: int, length: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sticky chains over the family's states, and their draws."""
    rng = np.random.default_rng([933, 9, seed])
    m = family.n_states
    states = np.empty((n, length), dtype=np.int64)
    states[:, 0] = rng.integers(0, m, n)
    for t in range(1, length):
        move = rng.random(n) > 0.9
        states[:, t] = np.where(move, rng.integers(0, m, n), states[:, t - 1])
    return np.asarray(family.sample(states, rng)), states


@pytest.mark.critical
@pytest.mark.oracle
def test_identical_per_sequence_kernels_are_the_per_step_kernel() -> None:
    family = BetaBinomialEmission([TRIALS] * 2, [2.0, 8.0], [8.0, 2.0])
    draws, _ = _chains(family, 6, 30, 0)
    per_step = _kernel(0.9, 2).expand(29, 2, 2).clone()
    initial = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    shared = baum_welch_family(draws, initial, per_step, family, max_iterations=5)
    each = baum_welch_family(
        draws, initial, per_step.expand(6, 29, 2, 2).clone(), family, max_iterations=5
    )
    assert shared.log_likelihood == each.log_likelihood
    assert torch.equal(shared.log_initial, each.log_initial)
    for name, value in shared.emissions.named_parameters().items():
        assert torch.equal(value, each.emissions.named_parameters()[name])


@pytest.mark.critical
@pytest.mark.oracle
def test_distinct_kernels_score_each_sequence_as_its_own_chain() -> None:
    # One iteration returns the likelihood at the given parameters; with a
    # kernel per sequence it is the sum of each sequence scored alone.
    family = BetaBinomialEmission([TRIALS] * 2, [2.0, 8.0], [8.0, 2.0])
    draws, _ = _chains(family, 4, 25, 1)
    stays = [0.6, 0.8, 0.95, 0.99]
    kernels = torch.stack([_kernel(stay, 2).expand(24, 2, 2) for stay in stays])
    initial = torch.log(torch.tensor([0.3, 0.7], dtype=torch.float64))
    together = baum_welch_family(draws, initial, kernels, family, max_iterations=1)
    alone = sum(
        baum_welch_family(
            draws[i : i + 1], initial, kernels[i], family, max_iterations=1
        ).log_likelihood
        for i in range(4)
    )
    assert abs(together.log_likelihood - alone) <= 1e-10 * abs(alone)
    with pytest.raises(ValueError, match="is neither"):
        baum_welch_family(draws, initial, kernels[:3], family, max_iterations=1)


@pytest.mark.critical
@pytest.mark.analytic
def test_a_reflected_state_scores_as_the_exchanged_family() -> None:
    base = BetaBinomialEmission([TRIALS] * 2, [2.0, 5.0], [9.0, 3.0])
    reflection = ReflectedEmission(base)
    counts = torch.arange(0.0, TRIALS + 1.0)
    scores = reflection.log_density(counts)
    exchanged = BetaBinomialEmission([TRIALS] * 2, [9.0, 3.0], [2.0, 5.0])
    assert torch.equal(scores[:, :2], base.log_density(counts))
    torch.testing.assert_close(
        scores[:, 2:], exchanged.log_density(counts), rtol=0.0, atol=1e-12
    )
    torch.testing.assert_close(
        scores[:, 2:], base.log_density(TRIALS - counts), rtol=0.0, atol=1e-12
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_the_reflected_m_step_maximizes_the_unfolded_likelihood() -> None:
    # The referee maximizes the 4-state expected log-likelihood over the two
    # base states' (a, b) through `log_density` of the unfolded family, with
    # the tie imposed by construction and no reflection in sight.
    rng = np.random.default_rng(933)
    counts = torch.as_tensor(
        rng.binomial(40, rng.beta(3.0, 7.0, 2_000)), dtype=torch.float64
    )
    posterior = torch.as_tensor(rng.dirichlet(np.ones(4), 2_000))
    start = ReflectedEmission(
        BetaBinomialEmission([TRIALS] * 2, [2.0, 5.0], [5.0, 2.0])
    )
    fitted = start.reestimate(counts, posterior).emissions
    assert isinstance(fitted, ReflectedEmission)

    def negative(point: np.ndarray) -> float:
        a, b = np.exp(point[:2]), np.exp(point[2:])
        family = ReflectedEmission(BetaBinomialEmission([TRIALS] * 2, a, b))
        return -float((posterior * family.log_density(counts)).sum())

    best = minimize(
        negative,
        np.log([2.0, 5.0, 5.0, 2.0]),
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-10},
    )
    base = fitted.base
    assert isinstance(base, BetaBinomialEmission)
    found = np.log(np.concatenate([base.alpha.numpy(), base.beta.numpy()]))
    np.testing.assert_allclose(found, best.x, atol=1e-4)
    assert negative(found) <= best.fun + 1e-7


@pytest.mark.end2end
def test_a_planted_reflected_count_pair_chain_is_recovered() -> None:
    # Two base states at depth 30 and 90 with allele rates 0.2 and 0.35, each
    # reflected: four states over 40 chains of 150. Fitted from a perturbed
    # start with a sticky kernel per sequence, each base state's rate lands
    # within 0.02 and its depth within 5%.
    truth = ReflectedEmission(
        IndependentCountPair(
            NegativeBinomialEmission([8.0, 8.0], [30.0, 90.0]),
            BetaBinomialEmission([TRIALS] * 2, [4.0, 7.0], [16.0, 13.0]),
        )
    )
    draws, _ = _chains(truth, 40, 150, 2)
    start = ReflectedEmission(
        IndependentCountPair(
            NegativeBinomialEmission([3.0, 3.0], [20.0, 120.0]),
            BetaBinomialEmission([TRIALS] * 2, [3.0, 4.0], [7.0, 6.0]),
        )
    )
    kernels = _kernel(0.9, 4).expand(40, 149, 4, 4).clone()
    initial = torch.full((4,), math.log(0.25), dtype=torch.float64)
    fit = baum_welch_family(draws, initial, kernels, start, max_iterations=200)
    base = fit.emissions.base  # type: ignore[attr-defined]
    order = np.argsort(base.total.mean.numpy())
    np.testing.assert_allclose(base.total.mean.numpy()[order], [30.0, 90.0], rtol=0.05)
    rate = (base.successes.alpha / base.successes.concentration).numpy()[order]
    folded = np.minimum(rate, 1.0 - rate)
    np.testing.assert_allclose(folded, [0.2, 0.35], atol=0.02)


@pytest.mark.smoke
def test_a_reflection_without_a_common_trial_count_is_refused() -> None:
    reflection = ReflectedEmission(
        BetaBinomialEmission([10.0, 20.0], [1.0, 1.0], [1.0, 1.0])
    )
    with pytest.raises(ValueError, match="reflects successes"):
        reflection.reestimate(
            torch.tensor([1.0, 2.0]), torch.full((2, 4), 0.25, dtype=torch.float64)
        )
