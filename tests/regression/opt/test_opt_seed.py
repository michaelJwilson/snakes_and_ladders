"""`seed` from data alone, and a `Start` every EM fit takes as it is (issue #1085).

Referees: each method is bitwise the rule it names, from the same generator,
so the seeding `search.mixture_starts` ran before this is the one it runs
now; and `expectation_maximization(observations, *start)` is the fit a
hand-built weight vector and family start, bitwise. The public score
`seed_scores` (#1236) draws the indices the private `_seed_scores` drew on
main 25dde304, on 8 of 8 generators, and scores a 1-D Gaussian row set as
the hand-computed ``(y - c) ** 2 / (2 sigma ** 2)``, bitwise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.emissions import CountPairEmission, EmissionFamily, GaussianEmission
from sal.opt.em import EmConfig
from sal.opt.emission_mixture import (
    ComponentsAt,
    CountPairSeeding,
    SeedMethod,
    Start,
    expectation_maximization,
    plus_plus_start,
    seed,
    seed_scores,
    uniform_start,
)
from sal.opt.mixture import emission_mixture_plus_plus, kmeans_plus_plus
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture


def _problem() -> tuple[np.ndarray, int, CountPairSeeding]:
    """The CI draw as a caller holds it: observations, a count, a seam; no truth."""
    params = fixture("emission_mixture", "ci").params
    components = params.components
    assert isinstance(components, CountPairEmission)
    declared = components.trials
    at = CountPairSeeding(
        dispersion=5.0,
        concentration=10.0,
        joint=components.joint,
        trials=None if declared is None else float(np.unique(declared.numpy()).item()),
    )
    observations = simulate_emission_mixture(params).observations.astype(float)
    return observations, params.n_components, at


def _kmeans(
    rows: np.ndarray, k: int, at: ComponentsAt, rng: np.random.Generator
) -> EmissionFamily:
    return at(kmeans_plus_plus(rows, k, rng))


RULES: dict[SeedMethod, Callable[..., EmissionFamily]] = {
    SeedMethod.UNIFORM: uniform_start,
    SeedMethod.PLUS_PLUS: plus_plus_start,
    SeedMethod.KMEANS: _kmeans,
}


@pytest.mark.oracle
@pytest.mark.parametrize("method", list(SeedMethod))
def test_each_method_is_the_rule_it_names_bitwise(method: SeedMethod) -> None:
    observations, n_components, at = _problem()

    start = seed(
        observations, n_components, at, method=method, rng=np.random.default_rng(7)
    )
    by_hand = RULES[method](observations, n_components, at, np.random.default_rng(7))

    values = torch.as_tensor(observations, dtype=torch.float64)
    assert torch.equal(
        start.components.log_density(values), by_hand.log_density(values)
    )
    assert np.array_equal(start.weights, np.full(n_components, 1.0 / n_components))


@pytest.mark.oracle
def test_a_start_unpacks_into_the_fit_a_hand_built_one_runs() -> None:
    observations, n_components, at = _problem()
    start = seed(
        observations,
        n_components,
        at,
        method=SeedMethod.PLUS_PLUS,
        rng=np.random.default_rng(1),
    )
    config = EmConfig(max_iterations=5, tolerance=0.0)

    unpacked = expectation_maximization(observations, *start, config=config)
    by_hand = expectation_maximization(
        observations,
        torch.full((n_components,), 1.0 / n_components, dtype=torch.float64),
        start.components,
        config=config,
    )

    assert unpacked.log_likelihood == by_hand.log_likelihood
    assert torch.equal(unpacked.weights, by_hand.weights)
    assert isinstance(start, Start)


#: The indices `_seed_scores` seeded on, on main 25dde304, for the CI draw at
#: three components, from `np.random.default_rng(seed)` for seeds 0..7 (#1236).
TODAYS_INDICES = (
    (765, 259, 40),
    (425, 841, 127),
    (753, 255, 746),
    (730, 217, 677),
    (653, 417, 879),
    (603, 716, 450),
    (400, 292, 357),
    (850, 776, 713),
)


@pytest.mark.oracle
def test_the_public_score_draws_todays_indices_and_scores_bitwise() -> None:
    observations, n_components, at = _problem()
    rows = np.asarray(observations, dtype=np.float64)
    indices = np.arange(rows.shape[0], dtype=np.float64)
    score = seed_scores(observations, at)

    # The scores: the family seeded at one row, its divergence written out.
    for index in (0, 5, 37):
        written = (
            at(rows[[index]])
            .bregman_divergence(torch.as_tensor(rows, dtype=torch.float64))[:, 0]
            .numpy()
        )
        assert np.array_equal(score(float(index), indices), written)

    # The indices: the draw main recorded, and the components `plus_plus_start`
    # seeds on from the same generator.
    values = torch.as_tensor(observations, dtype=torch.float64)
    for generator, today in enumerate(TODAYS_INDICES):
        chosen = emission_mixture_plus_plus(
            indices, n_components, score, np.random.default_rng(generator)
        ).astype(np.int64)
        assert tuple(chosen.tolist()) == today
        shipped = plus_plus_start(
            observations, n_components, at, np.random.default_rng(generator)
        )
        assert torch.equal(
            at(rows[chosen]).log_density(values), shipped.log_density(values)
        )


@pytest.mark.analytic
def test_the_public_score_is_half_the_squared_distance_on_a_gaussian() -> None:
    # A Gaussian at scale sigma has divergence (y - c) ** 2 / (2 sigma ** 2):
    # at sigma = 2 over rows 0, 1, 3, seeded on row 0, that is 0, 1/8, 9/8,
    # and seeded on row 2, 9/8, 1/2, 0; every value is exact in binary.
    rows = np.array([0.0, 1.0, 3.0])

    def at(chosen: np.ndarray) -> GaussianEmission:
        located = np.asarray(chosen, dtype=np.float64).reshape(-1)
        return GaussianEmission(located, np.full(located.shape, 2.0), 1e-6)

    score = seed_scores(rows, at)
    candidates = np.arange(3, dtype=np.float64)
    assert np.array_equal(score(0.0, candidates), [0.0, 0.125, 1.125])
    assert np.array_equal(score(2.0, candidates), [1.125, 0.5, 0.0])
