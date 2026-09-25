"""The spatial seams under a covariate, on a fixture that carries one (#658 item 5).

The canonical instance is categorical, so `test_covariate_reaches_the_fit.py`
could assert only that the seams run; this count-pair instance carries a
varying covariate. `external_field` scored through its own `log_density` while
`fit_spatio_sequential` handed it a posterior computed with the covariate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sal.likelihood.spatio_sequential import (
    class_log_density,
    external_field,
    marginal_log_likelihood_torch,
)
from sal.sim.count_pairs import IndependentCountPair
from sal.sim.count_pairs_rust import fine_instance
from sal.sim.fixtures import fixture
from sal.sim.spatio_sequential import (
    SpatioSequentialParams,
    gated_log_density,
)

PROBLEM = "spatio_sequential_counts"


def _instance() -> tuple[SpatioSequentialParams, np.ndarray, np.ndarray]:
    loaded = fine_instance(fixture(PROBLEM, "ci").path)
    return loaded.params, loaded.observations, loaded.labels


def _covariate(observations: np.ndarray, spread: float) -> np.ndarray:
    """A per-vertex exposure and a trial count that covers the drawn successes.

    Separate streams: a trial count from the total would score `-inf`.
    """
    n_positions, n_nodes = observations.shape[:2]
    rng = np.random.default_rng(19)
    covariate = np.empty((n_positions, n_nodes, 2))
    covariate[..., 0] = rng.uniform(1.0 / spread, spread, size=(1, n_nodes))
    covariate[..., 1] = int(observations[..., 1].max()) + 3
    return covariate


@pytest.mark.analytic
@pytest.mark.critical
def test_every_spatial_seam_changes_when_the_covariate_does() -> None:
    """The claim the replaced test named and could not make.

    Four seams, each asserted to move; a dropped covariate passes `isfinite`.
    """
    params, observations, labels = _instance()
    covaried = replace(params, covariate=_covariate(observations, spread=4.0))

    plain_gated = gated_log_density(params, observations)
    plain_class = class_log_density(params, observations, labels)
    plain_field = external_field(params, observations, labels)
    plain_torch = float(marginal_log_likelihood_torch(params, observations, labels))

    assert not np.allclose(gated_log_density(covaried, observations), plain_gated)
    assert not np.allclose(
        class_log_density(covaried, observations, labels), plain_class
    )
    assert not np.allclose(external_field(covaried, observations, labels), plain_field)
    assert (
        abs(
            float(marginal_log_likelihood_torch(covaried, observations, labels))
            - plain_torch
        )
        > 1.0
    )


@pytest.mark.analytic
def test_the_field_and_the_posterior_agree_about_the_model() -> None:
    """The defect: a field scored without the covariate the posterior was scored with.

    Given the same posterior, the field moves with the covariate (was `0.000e+00`).
    """
    params, observations, labels = _instance()
    covaried = replace(params, covariate=_covariate(observations, spread=4.0))
    from sal.likelihood.spatio_sequential import class_posteriors

    posterior = class_posteriors(covaried, observations, labels).posterior

    told = external_field(covaried, observations, labels, posterior)
    untold = external_field(params, observations, labels, posterior)

    moved = float(np.abs(told - untold).max())
    assert moved > 1.0, f"the field moved {moved:.3e}; it was 0.000e+00 before #658"


@pytest.mark.smoke
def test_a_neutral_covariate_leaves_every_seam_where_it_was() -> None:
    """The other half: what the covariate does nothing to, it does nothing to.

    Neutral: exposure one, declared trials; to tolerance, as the order differs.
    """
    params, observations, labels = _instance()
    family = params.emissions[0]
    assert isinstance(family, IndependentCountPair)
    neutral = np.ones((*observations.shape[:2], 2))
    neutral[..., 1] = family.successes.trials.numpy()[0]
    covaried = replace(params, covariate=neutral)

    np.testing.assert_allclose(
        class_log_density(covaried, observations, labels),
        class_log_density(params, observations, labels),
        rtol=1e-11,
    )
