"""Mixture EM under a per-observation covariate, and seeding in rate space (issue #933, R1).

`expectation_maximization`, `anneal_assignments` and the mixture E step take a
covariate and hand it to the family's `log_density` and `reestimate`;
`MixtureInstance` carries one to every start. Referees: a unit covariate is
the covariate-free fit at #648's floor, the covariate-free route is untouched
bitwise, and a mixture drawn under a varying exposure and trial count is
recovered from starts placed in `rate_space`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import BetaBinomialEmission, NegativeBinomialEmission
from sal.opt.emission_mixture import (
    expectation_maximization,
    plus_plus_start,
)
from sal.opt.mixture import e_step, mixture_log_likelihood
from sal.sample.mixture_anneal import anneal_assignments
from sal.search.mixture_starts import (
    MixtureInstance,
    emission_seeding,
    polish,
)
from sal.sim.count_pairs import (
    IndependentCountPair,
    IndependentCountPairSeeding,
    rate_space,
)

#: The declared floor for a reordered sum through a bisection (#648).
FLOOR = 2e-06
TRIALS = 30.0
N_SAMPLES = 3_000
TRUTH = IndependentCountPair(
    NegativeBinomialEmission([6.0, 6.0, 6.0], [3.0, 12.0, 40.0]),
    BetaBinomialEmission([TRIALS] * 3, [2.0, 10.0, 16.0], [18.0, 10.0, 4.0]),
)
WEIGHTS = np.array([0.3, 0.3, 0.4])


def _draw(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs under a varying exposure and trial count, their covariate and labels."""
    rng = np.random.default_rng([933, 1, seed])
    labels = rng.choice(3, size=N_SAMPLES, p=WEIGHTS)
    covariate = np.stack(
        [rng.uniform(0.3, 3.0, N_SAMPLES), rng.integers(5, 60, N_SAMPLES)], axis=1
    ).astype(np.float64)
    pairs = TRUTH.sample(labels, rng, covariate=torch.as_tensor(covariate))
    return np.asarray(pairs, dtype=np.float64), covariate, labels


def _seeding() -> IndependentCountPairSeeding:
    return IndependentCountPairSeeding(
        dispersion=2.0, concentration=10.0, trials=TRIALS
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_a_unit_covariate_is_the_covariate_free_fit() -> None:
    # Unit exposure and the declared trial count: the same model. The E step
    # agrees bitwise; the dispersion is solved per state under a covariate and
    # in lockstep without, so the fit is pinned at the floor.
    pairs, _, _ = _draw(0)
    pairs[:, 1] = np.minimum(pairs[:, 1], TRIALS)
    unit = np.stack([np.ones(N_SAMPLES), np.full(N_SAMPLES, TRIALS)], axis=1)
    start = _seeding()(pairs[[0, 1, 2]])
    weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    values = torch.as_tensor(pairs)
    free = e_step(values, torch.log(weights), start)
    conditioned = e_step(
        values, torch.log(weights), start, covariate=torch.as_tensor(unit)
    )
    assert torch.equal(free[0], conditioned[0])
    assert torch.equal(free[1], conditioned[1])
    plain = expectation_maximization(pairs, weights, start, max_iterations=20)
    covaried = expectation_maximization(
        pairs, weights, start, max_iterations=20, covariate=unit
    )
    assert abs(plain.log_likelihood - covaried.log_likelihood) < FLOOR * abs(
        plain.log_likelihood
    )
    for name, value in plain.components.named_parameters().items():
        other = covaried.components.named_parameters()[name]
        assert float(((value - other).abs() / value.abs()).max()) < FLOOR, name


@pytest.mark.critical
@pytest.mark.oracle
def test_rate_space_is_the_count_over_its_covariate() -> None:
    pairs = np.array([[6.0, 3.0], [0.0, 0.0], [10.0, 10.0]])
    covariate = np.array([[2.0, 6.0], [0.5, 4.0], [4.0, 20.0]])
    rows = rate_space(pairs, covariate, TRIALS)
    np.testing.assert_array_equal(rows[:, 0], [3.0, 0.0, 2.5])
    np.testing.assert_array_equal(rows[:, 1], [15.0, 0.0, 15.0])
    seeded = _seeding()(rows)
    np.testing.assert_array_equal(seeded.total.mean.numpy(), [3.0, 1e-6, 2.5])
    rate = (rows[:, 1] + 0.5) / (TRIALS + 1.0)
    np.testing.assert_allclose(seeded.successes.mean.numpy(), rate * TRIALS)
    with pytest.raises(ValueError, match="strictly positive"):
        rate_space(pairs, np.zeros_like(covariate), TRIALS)
    with pytest.raises(ValueError, match="must both be"):
        rate_space(pairs, covariate[:2], TRIALS)


@pytest.mark.end2end
def test_a_mixture_under_a_varying_covariate_is_recovered_from_rate_space_starts() -> (
    None
):
    # Planted rates 3, 12 and 40 per unit exposure and allele rates 0.1, 0.5
    # and 0.8, drawn under exposures in [0.3, 3] and trials in [5, 60).
    # Seeded by D-squared sampling on the rate-space rows and fitted under the
    # covariate, each component lands within 10% of its planted rate.
    pairs, covariate, labels = _draw(1)
    rows = rate_space(pairs, covariate, TRIALS)
    start = plus_plus_start(rows, 3, _seeding(), np.random.default_rng(933))
    fitted = expectation_maximization(
        pairs,
        torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
        start,
        max_iterations=300,
        covariate=covariate,
    )
    assert isinstance(fitted.components, IndependentCountPair)
    order = np.argsort(fitted.components.total.mean.numpy())
    np.testing.assert_allclose(
        fitted.components.total.mean.numpy()[order], [3.0, 12.0, 40.0], rtol=0.1
    )
    np.testing.assert_allclose(
        (fitted.components.successes.mean / TRIALS).numpy()[order],
        [0.1, 0.5, 0.8],
        atol=0.03,
    )
    reference = float(
        mixture_log_likelihood(
            torch.as_tensor(pairs),
            torch.log(torch.as_tensor(WEIGHTS)),
            TRUTH,
            covariate=torch.as_tensor(covariate),
        )
    )
    assert fitted.log_likelihood >= reference
    assigned = fitted.responsibilities.argmax(dim=1).numpy()
    assert float((order[assigned] == labels).mean()) > 0.9


@pytest.mark.oracle
def test_an_instance_carries_its_covariate_to_the_starts_and_the_polish() -> None:
    # The instance's reference is the likelihood under its covariate, a start
    # seeds from the rate-space rows, and the polish climbs past the truth.
    pairs, covariate, labels = _draw(2)
    instance = MixtureInstance(
        observations=pairs,
        labels=labels,
        weights=WEIGHTS,
        truth=TRUTH,
        at=_seeding(),
        covariate=covariate,
        seeding_rows=rate_space(pairs, covariate, TRIALS),
    )
    assert instance.reference == float(
        mixture_log_likelihood(
            torch.as_tensor(pairs),
            torch.log(torch.as_tensor(WEIGHTS)),
            TRUTH,
            covariate=torch.as_tensor(covariate),
        )
    )
    seeded = emission_seeding(instance, np.random.default_rng(0)).components
    assert isinstance(seeded, IndependentCountPair)
    means = seeded.total.mean.numpy()
    assert bool(np.isin(means, np.maximum(instance.rows[:, 0], 1e-6)).all())
    polished = polish(instance, seeded, passes=200)
    assert polished.log_likelihoods[-1] >= instance.reference


@pytest.mark.smoke
def test_the_anneal_conditions_on_the_covariate() -> None:
    pairs, covariate, _ = _draw(3)
    start = _seeding()(rate_space(pairs, covariate, TRIALS)[[0, 1, 2]])
    run = anneal_assignments(
        pairs,
        torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
        start,
        [4.0, 2.0, 1.0],
        np.random.default_rng(0),
        covariate=covariate,
    )
    weights = torch.as_tensor(run.weights)
    rescored = float(
        mixture_log_likelihood(
            torch.as_tensor(pairs),
            torch.log(weights),
            run.components,
            covariate=torch.as_tensor(covariate),
        )
    )
    assert rescored == run.log_likelihood
