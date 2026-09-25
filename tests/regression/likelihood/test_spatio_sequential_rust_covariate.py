"""The Rust coupled E step under a covariate (issue #658).

Under a covariate the density depends on the count and what it is scored
against; the caller tabulates the `(count, covariate)` pairs and hands each
observation its row, so `src/coupled.rs` changes only its index width. Pinned
against the NumPy oracle (`likelihood/CLAUDE.md`) and against itself uncovaried.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sal.likelihood.rust import spatio_sequential as rust
from sal.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
)
from sal.sim.count_pairs import CountPairInstance
from sal.sim.fixtures import fixture
from sal.sim.rust.count_pairs import fine_instance
from sal.sim.spatio_sequential import SpatioSequentialParams

PROBLEM = "spatio_sequential_counts"

#: Relative agreement required of an evidence and a field, both sums over
#: positions and vertices, so an absolute bound at one size does not transfer
#: (root `CLAUDE.md`). Measured at the ci instance under a covariate: 2.7e-15
#: on the evidence and 2.9e-15 on the field, one thread.
_RELATIVE = 1e-11

#: The posteriors are probabilities in [0, 1], so they are compared absolutely.
#: Measured 1.0e-09, which is the forward--backward's own accumulation and not
#: this change's: the uncovaried comparison reports the same order.
_ABSOLUTE = 1e-7


def _instance() -> CountPairInstance:
    return fine_instance(fixture(PROBLEM, "ci").path)


def _covariate(instance: CountPairInstance) -> np.ndarray:
    """A per-vertex exposure and a per-vertex trial count, both varying.

    Per vertex, the tabulated case; trials cover the independently drawn successes.
    """
    observations = instance.observations
    n_positions, n_nodes = observations.shape[:2]
    rng = np.random.default_rng(4)
    covariate = np.empty((n_positions, n_nodes, 2))
    covariate[..., 0] = rng.uniform(0.5, 2.0, size=(1, n_nodes))
    covariate[..., 1] = int(observations[..., 1].max()) + rng.integers(
        0, 6, size=(1, n_nodes)
    )
    return covariate


def _covaried(instance: CountPairInstance) -> SpatioSequentialParams:
    return replace(instance.params, covariate=_covariate(instance))


@pytest.mark.oracle
def test_the_covaried_rust_e_step_and_field_match_the_numpy_oracle() -> None:
    """The fast backend conditions, and on the same numbers; the field is the
    seam #658 found unthreaded, and both paths now score with it."""
    instance = _instance()
    params = _covaried(instance)
    observations, labels = instance.observations, instance.labels

    expected = class_posteriors(params, observations, labels)
    actual = rust.class_posteriors(params, observations, labels)

    np.testing.assert_allclose(
        actual.log_evidence, expected.log_evidence, rtol=_RELATIVE
    )
    np.testing.assert_allclose(
        actual.posterior, expected.posterior, rtol=_RELATIVE, atol=_ABSOLUTE
    )
    np.testing.assert_allclose(
        actual.pairwise, expected.pairwise, rtol=_RELATIVE, atol=_ABSOLUTE
    )
    np.testing.assert_allclose(
        rust.external_field(params, observations, labels, expected.posterior),
        external_field(params, observations, labels, expected.posterior),
        rtol=_RELATIVE,
    )


@pytest.mark.smoke
@pytest.mark.critical
def test_without_a_covariate_the_rows_are_the_counts() -> None:
    """The uncovaried path is unchanged, not merely equivalent.

    Rows are the counts, count-major, asserted on the arrays themselves.
    """
    instance = _instance()
    observations = instance.observations

    totals, successes, total_table, success_table = rust.emission_rows(
        instance.params, observations
    )

    assert np.array_equal(totals, observations[..., 0])
    assert np.array_equal(successes, observations[..., 1])
    assert total_table.shape[0] == int(observations[..., 0].max()) + 1
    assert success_table.shape[0] == int(observations[..., 1].max()) + 1


@pytest.mark.smoke
def test_the_covaried_tabulation_stays_a_table() -> None:
    """The reason this is a table and not a per-observation score array.

    The outer product, 20,736 and 41, against 64,000 observations at ci. Sorting
    the distinct pairs instead cost 135.6 ms of a 141.4 ms E step (0.6x the
    oracle); the outer product is 10x faster to build.
    """
    instance = _instance()
    params = _covaried(instance)
    observations = instance.observations
    n_observations = observations.shape[0] * observations.shape[1]

    _, _, total_table, success_table = rust.emission_rows(params, observations)

    assert total_table.shape[0] < n_observations
    assert success_table.shape[0] < n_observations


@pytest.mark.analytic
def test_the_covariate_changes_the_answer() -> None:
    """A backend that quietly dropped one would pass every test above."""
    instance = _instance()
    observations, labels = instance.observations, instance.labels

    plain = rust.class_posteriors(instance.params, observations, labels)
    covaried = rust.class_posteriors(_covaried(instance), observations, labels)

    assert np.abs(covaried.log_evidence - plain.log_evidence).max() > 1.0
