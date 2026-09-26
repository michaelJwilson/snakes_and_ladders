"""`seed` from data alone, and a `Start` every EM fit takes as it is (issue #1085).

Referees: each method is bitwise the rule it names, from the same generator,
so the seeding `search.mixture_starts` ran before this is the one it runs
now; and `expectation_maximization(observations, *start)` is the fit a
hand-built weight vector and family start, bitwise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.emissions import CountPairEmission, EmissionFamily
from sal.opt.em import EmConfig
from sal.opt.emission_mixture import (
    ComponentsAt,
    CountPairSeeding,
    SeedMethod,
    Start,
    expectation_maximization,
    plus_plus_start,
    seed,
    uniform_start,
)
from sal.opt.mixture import kmeans_plus_plus
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
