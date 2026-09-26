"""Seed, fit, refine and restart in one call (issue #1085).

Referees: ``restarts=1`` is the hand-wired seed, EM and split-and-merge on
the generator's first spawned stream, bitwise; more restarts return the best
of the same hand-wired runs; and the result always carries a termination.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.emissions import CountPairEmission
from sal.opt.em import EmConfig
from sal.opt.emission_mixture import (
    CountPairSeeding,
    SeedMethod,
    expectation_maximization,
    seed,
)
from sal.opt.split_merge import REFINE_CANDIDATES, fit, split_and_merge
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture

CONFIG = EmConfig(max_iterations=30, tolerance=1e-6)


def _problem() -> tuple[np.ndarray, int, CountPairSeeding]:
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


def _by_hand(
    observations: np.ndarray,
    n_components: int,
    at: CountPairSeeding,
    stream: np.random.Generator,
) -> float:
    start = seed(
        observations, n_components, at, method=SeedMethod.PLUS_PLUS, rng=stream
    )
    fitted = expectation_maximization(observations, *start, config=CONFIG)
    refined = split_and_merge(
        fitted, observations, at, candidates=REFINE_CANDIDATES, config=CONFIG
    )
    return refined.fit.log_likelihood


@pytest.mark.oracle
def test_one_restart_is_the_hand_wired_chain_and_more_keep_the_best() -> None:
    observations, n_components, at = _problem()
    streams = np.random.default_rng(11).spawn(3)
    hand = [_by_hand(observations, n_components, at, stream) for stream in streams]

    one = fit(
        observations,
        n_components,
        at,
        method=SeedMethod.PLUS_PLUS,
        rng=np.random.default_rng(11),
        refine=True,
        config=CONFIG,
    )
    three = fit(
        observations,
        n_components,
        at,
        method=SeedMethod.PLUS_PLUS,
        rng=np.random.default_rng(11),
        restarts=3,
        refine=True,
        config=CONFIG,
    )

    assert one.log_likelihood == hand[0]
    assert three.log_likelihood == max(hand)
    # Required from #1100 on; until then the fit fills it at every return.
    assert three.termination is not None


@pytest.mark.smoke
def test_no_restart_and_a_refined_covariate_fit_are_refused() -> None:
    observations, n_components, at = _problem()
    with pytest.raises(ValueError, match="restarts"):
        fit(
            observations,
            n_components,
            at,
            method="uniform",
            rng=np.random.default_rng(0),
            restarts=0,
        )
    with pytest.raises(ValueError, match="covariate"):
        fit(
            observations,
            n_components,
            at,
            method="uniform",
            rng=np.random.default_rng(0),
            refine=True,
            covariate=np.ones((observations.shape[0], 2)),
        )
