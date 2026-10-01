"""A mixture fit that a collapsed component, a round-off score or one failed seeding no longer ends (issue #1136).

- **A collapsed component** is planted: two generating components and a
  third seeded far from every pair, which the first E step empties. On
  `main` the M step raised ("mean and weight must be positive, got nan and
  0.0"); here the component is held, the fit reports it, and the two others
  recover their generating means.
- **D² seeding** is handed a divergence that rounds a seed's own score to
  ``-1e-15``, as a floating-point divergence does; the draws are those of the
  same divergence floored at zero, bitwise.
- **A best-of start** with one seeding that raises hands over the best of the
  others and names the one it skipped.
- **The sampling starts** read the mixture's own likelihood: the components
  handed over are the objective's at the chain's draw, not a snap to a row.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    EmissionFamily,
    NegativeBinomialEmission,
)
from sal.opt.emission_mixture import (
    CountPairSeeding,
    EmissionMixtureObjective,
    build_like,
    expectation_maximization,
)
from sal.opt.mixture import emission_mixture_plus_plus, mixture_log_likelihood
from sal.opt.objective import autograd_value_and_gradient, value_and_gradient
from sal.search import mixture_starts
from sal.search.mixture_starts import (
    BestOf,
    chain_seeding,
    emission_objective,
    instance_from,
)
from sal.search.projection import Seeding
from sal.sim.count_pairs import IndependentCountPair
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture

SEED = 1136

#: The planted means; the third component is seeded at 1e12, where its
#: posterior underflows to zero at the first E step.
MEANS = (30.0, 200.0)


def _planted() -> tuple[np.ndarray, np.ndarray, CountPairEmission]:
    """Pairs from two count-pair components, their labels, and a three-component start."""
    rng = np.random.default_rng(SEED)
    truth = CountPairEmission(
        [5.0, 20.0], list(MEANS), [2.0, 8.0], [8.0, 2.0], [40.0, 40.0], joint=False
    )
    labels = rng.integers(0, 2, 4_000)
    start = CountPairEmission(
        [5.0, 20.0, 5.0],
        [25.0, 180.0, 1e12],
        [2.0, 8.0, 5.0],
        [8.0, 2.0, 5.0],
        [40.0, 40.0, 40.0],
        joint=False,
    )
    return truth.sample(labels, rng).astype(np.float64), labels, start


@pytest.mark.end2end
def test_an_emptied_component_is_held_and_the_rest_recover() -> None:
    pairs, labels, start = _planted()

    fit = expectation_maximization(pairs, np.full(3, 1.0 / 3.0), start)

    fitted = fit.components
    assert isinstance(fitted, CountPairEmission)
    assert fit.frozen == (2,)
    assert fit.termination.converged
    assert float(fitted.total.mean[2]) == 1e12
    for k, planted in enumerate(MEANS):
        observed = pairs[labels == k, 0].mean()
        assert float(fitted.total.mean[k]) == pytest.approx(observed, rel=0.02)
        assert float(fitted.total.mean[k]) == pytest.approx(planted, rel=0.05)


@pytest.mark.oracle
def test_every_pair_family_reports_the_state_either_channel_held() -> None:
    pairs, _, start = _planted()
    posterior = torch.zeros(pairs.shape[0], 3, dtype=torch.float64)
    posterior[:, 0] = 0.5
    posterior[:, 1] = 0.5
    successes = start.successes
    assert successes is not None
    independent = IndependentCountPair(start.total, successes)
    # The two states with data are the states fitted alone, bitwise.
    alone = NegativeBinomialEmission(
        start.total.dispersion[:2], start.total.mean[:2]
    ).reestimate(pairs[:, 0], posterior[:, :2])

    for family in (start, independent):
        held = family.reestimate(pairs, posterior)
        emissions = held.emissions
        assert isinstance(emissions, CountPairEmission | IndependentCountPair)
        assert held.frozen == (2,)
        assert torch.equal(emissions.total.mean[:2], alone.emissions.mean)
        assert float(emissions.total.mean[2]) == 1e12


@pytest.mark.oracle
def test_d_squared_seeding_floors_a_round_off_negative_score() -> None:
    values = np.arange(50, dtype=np.float64)

    def rounded(seed: float, candidates: np.ndarray) -> np.ndarray:
        # A divergence that lands a hair below zero at its own seed.
        scores = np.asarray((candidates - seed) ** 2, dtype=np.float64)
        scores[candidates == seed] = -1e-15
        return scores

    def floored(seed: float, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(np.maximum(rounded(seed, candidates), 0.0))

    for generator in range(20):
        got = emission_mixture_plus_plus(
            values, 5, rounded, np.random.default_rng(generator)
        )
        want = emission_mixture_plus_plus(
            values, 5, floored, np.random.default_rng(generator)
        )
        assert np.array_equal(got, want)


def _instance() -> mixture_starts.MixtureInstance:
    params = fixture("emission_mixture", "ci").params
    return instance_from(
        simulate_emission_mixture(params),
        CountPairSeeding(
            float(params.components.total.dispersion.mean()),
            float(params.components.concentration.mean()),
            joint=True,
        ),
    )


@pytest.mark.smoke
def test_a_best_of_skips_a_seeding_that_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _instance()
    honest = mixture_starts.STARTS["data"]
    calls = iter(range(100))

    def flaky(
        instance: mixture_starts.MixtureInstance, rng: np.random.Generator
    ) -> Seeding[EmissionFamily]:
        if next(calls) == 1:
            msg = "mean and weight must be positive, got nan and 0.0"
            raise ValueError(msg)
        return honest(instance, rng)

    monkeypatch.setitem(mixture_starts.STARTS, "data", flaky)
    seeding = BestOf("data", 3)(instance, np.random.default_rng(SEED))

    assert "skipped 1 that failed (1: mean and weight" in seeding.diagnostics
    assert len(seeding.path) == 2

    def refusing(
        _instance: mixture_starts.MixtureInstance, _rng: np.random.Generator
    ) -> Seeding[EmissionFamily]:
        msg = "refused"
        raise ValueError(msg)

    monkeypatch.setitem(mixture_starts.STARTS, "data", refusing)
    with pytest.raises(ValueError, match="every one of 2 seedings"):
        BestOf("data", 2)(instance, np.random.default_rng(SEED))


@pytest.mark.oracle
def test_the_sampling_starts_read_the_mixtures_own_likelihood() -> None:
    instance = _instance()
    objective = emission_objective(instance)
    theta = objective.initial()
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    k = instance.n_components
    uniform = torch.full((k,), -math.log(k), dtype=torch.float64)

    # The objective is the count-pair mixture's negative log-likelihood.
    assert float(objective(theta)) == pytest.approx(
        -float(mixture_log_likelihood(values, uniform, objective.components(theta))),
        rel=1e-14,
    )
    # A chain's handover is its last draw's components, not a row snapped to.
    seeding = chain_seeding(instance, np.random.default_rng(SEED))
    handed, last = seeding.components, seeding.path[-1][1]
    assert isinstance(handed, CountPairEmission)
    assert isinstance(last, CountPairEmission)
    assert torch.equal(handed.total.mean, last.total.mean)
    rows = np.asarray(instance.observations, dtype=np.float64)
    assert not bool(np.isin(handed.total.mean.numpy(), rows[:, 0]).any())


@pytest.mark.smoke
def test_build_like_rebuilds_each_count_family() -> None:
    pair = CountPairEmission(
        [5.0, 9.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], joint=True
    )
    for family in (
        pair,
        pair.total,
        IndependentCountPair(
            pair.total, BetaBinomialEmission([40.0, 40.0], [2.0, 9.0], [8.0, 3.0])
        ),
    ):
        rebuilt = build_like(family)(family.named_parameters())
        assert type(rebuilt) is type(family)
        assert all(
            torch.equal(rebuilt.named_parameters()[name], value)
            for name, value in family.named_parameters().items()
        )
    with pytest.raises(TypeError, match="no build"):
        build_like(object())  # type: ignore[arg-type]
    EmissionMixtureObjective(np.zeros((3, 2)), pair, build_like(pair))


@pytest.mark.oracle
def test_the_gradient_on_distinct_counts_is_the_gradient_to_rounding() -> None:
    # The opt-in sums each parameter's gradient over the distinct counts; the
    # value is the same expression, and the gradient the same to rounding.
    instance = _instance()
    start = emission_objective(instance)
    plain = EmissionMixtureObjective(
        np.asarray(instance.observations, dtype=np.float64),
        start.components(start.initial()),
        build_like(start.components(start.initial())),
    )
    theta = start.initial() + 0.05 * torch.randn(
        start.n_parameters,
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float64,
    )

    value, gradient = value_and_gradient(start, theta)
    want_value, want_gradient = value_and_gradient(plain, theta)

    assert float(value) == pytest.approx(float(want_value), rel=1e-14)
    np.testing.assert_allclose(
        gradient.numpy(), want_gradient.numpy(), rtol=1e-10, atol=1e-10
    )


@pytest.mark.oracle
def test_a_reused_distinct_count_is_the_recomputed_one() -> None:
    from sal.emissions.rising import DistinctCache, distinct_values, reusing_distinct

    rng = np.random.default_rng(SEED)
    first = torch.as_tensor(rng.integers(0, 50, (500, 1)), dtype=torch.float64)
    # Same shape and sum, different values: the key collides, the check does not.
    second = first.flip(0).clone()
    second[0, 0], second[1, 0] = second[0, 0] + 1.0, second[1, 0] - 1.0
    cache: DistinctCache = {}

    with reusing_distinct(cache):
        for values in (first, second, first, second):
            got = distinct_values(values)
            want = torch.unique(values, return_inverse=True)
            assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    assert sum(len(entries) for entries in cache.values()) == 2


def _families() -> dict[str, tuple[EmissionFamily, Callable[[np.ndarray], np.ndarray]]]:
    """Each family the compiled gradient takes, and how its observations are read from its draws."""
    joint = CountPairEmission(
        [6.0, 12.0, 3e5],
        [20.0, 60.0, 40.0],
        [2.0, 9.0, 3e5],
        [8.0, 3.0, 2e5],
        joint=True,
    )
    independent = CountPairEmission(
        [6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], [40.0, 40.0], joint=False
    )
    successes = independent.successes
    assert successes is not None
    return {
        "joint": (joint, lambda pairs: pairs),
        "independent": (independent, lambda pairs: pairs),
        "IndependentCountPair": (
            IndependentCountPair(independent.total, successes),
            lambda pairs: pairs,
        ),
        "negative-binomial": (independent.total, lambda counts: counts),
        "beta-binomial": (successes, lambda counts: counts),
    }


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(_families()))
def test_the_compiled_gradient_is_autograds(name: str) -> None:
    # Every lgamma and digamma difference the kernel forms is a prefix sum
    # over integers; autograd through `__call__` is the reference. The joint
    # family carries a component at shape 3e5, past the series threshold.
    family, read = _families()[name]
    rng = np.random.default_rng([SEED, len(name)])
    labels = rng.integers(0, family.n_states, 3_000)
    observations = read(np.asarray(family.sample(labels, rng), dtype=np.float64))
    objective = EmissionMixtureObjective(observations, family, build_like(family))
    theta = objective.initial() + 0.1 * torch.randn(
        objective.n_parameters,
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float64,
    )

    value, gradient = objective.value_and_gradient(theta)
    want_value, want_gradient = autograd_value_and_gradient(objective, theta)

    assert objective._route is not None
    # Measured: 1.3e-15 relative in the value, 1.8e-11 absolute in a
    # gradient of magnitude up to 1.4e3.
    assert float(value) == pytest.approx(float(want_value), rel=1e-13)
    np.testing.assert_allclose(
        gradient.numpy(), want_gradient.numpy(), rtol=0, atol=1e-9
    )


@pytest.mark.oracle
def test_outside_the_kernel_the_gradient_is_autograds_bitwise() -> None:
    family, _ = _families()["independent"]
    rng = np.random.default_rng(SEED)
    pairs = np.asarray(family.sample(rng.integers(0, 2, 500), rng), dtype=np.float64)
    covariate = np.concatenate([np.ones((500, 1)), np.full((500, 1), 40.0)], axis=-1)
    conditioned = EmissionMixtureObjective(
        pairs, family, build_like(family), covariate=covariate
    )
    theta = conditioned.initial()
    # Run past overflow, where the chain rejects and autograd scores it.
    overflowed = theta.clone()
    overflowed[-1] = 800.0
    compiled = EmissionMixtureObjective(pairs, family, build_like(family))

    assert conditioned._route is None
    for objective, point in ((conditioned, theta), (compiled, overflowed)):
        got = objective.value_and_gradient(point)
        want = autograd_value_and_gradient(objective, point)
        assert all(
            torch.equal(a, b) or (a.isnan().all() and b.isnan().all())
            for a, b in zip(got, want, strict=True)
        )
