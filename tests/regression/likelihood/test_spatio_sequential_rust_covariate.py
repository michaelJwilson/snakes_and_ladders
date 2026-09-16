"""The Rust coupled E step under a covariate (issue #658).

The kernel tabulated `log p(count | class, state)` and indexed it by the count.
Under a covariate the density is a function of the count *and* what it is
scored against, so there is no table indexed by the count alone — which is why
this looked like a kernel redesign. It is not. The kernel never knew what the
index meant, only that the table was built from it, so the caller tabulates the
distinct `(count, covariate)` pairs and hands the row each observation falls
in. The only change in `src/coupled.rs` is the width of that index.

Pinned against the NumPy path, which is the oracle (`likelihood/CLAUDE.md`),
and against itself uncovaried, which must be what it was.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from snakes_and_ladders.likelihood import spatio_sequential_rust as rust
from snakes_and_ladders.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
)
from snakes_and_ladders.sim.count_pairs import CountPairInstance
from snakes_and_ladders.sim.count_pairs_rust import fine_instance
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.spatio_sequential import SpatioSequentialParams

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

    Per-vertex rather than per-observation because that is the case the
    tabulation is for --- a library size repeated down the positions --- and
    because it is what makes the row count worth reporting.

    The trial count must cover the successes actually drawn: the two channels
    are drawn independently (`sim/count_pairs.py`), so a trial count taken from
    the total scores `-inf` wherever the successes exceeded it, and the whole
    comparison comes back `nan`.
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
def test_the_covaried_rust_e_step_matches_the_numpy_oracle() -> None:
    """The claim: the fast backend conditions, and on the same numbers."""
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


@pytest.mark.oracle
def test_the_covaried_rust_field_matches_the_numpy_oracle() -> None:
    """The field is the seam #658 found unthreaded; both paths now score with it."""
    instance = _instance()
    params = _covaried(instance)
    observations, labels = instance.observations, instance.labels
    posterior = class_posteriors(params, observations, labels).posterior

    np.testing.assert_allclose(
        rust.external_field(params, observations, labels, posterior),
        external_field(params, observations, labels, posterior),
        rtol=_RELATIVE,
    )


@pytest.mark.structural
@pytest.mark.critical
def test_without_a_covariate_the_rows_are_the_counts() -> None:
    """The uncovaried path is unchanged, not merely equivalent.

    The rows *are* the counts and the table is count-major, which is what this
    module tabulated before #658. Asserted on the arrays rather than on a
    result, because a result can agree while the tabulation has silently become
    a different one --- and the row count is the whole memory argument.
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


@pytest.mark.structural
def test_the_covaried_tabulation_stays_a_table() -> None:
    """The reason this is a table and not a per-observation score array.

    The rows are every `(count, covariate)` combination --- the outer product,
    20,736 and 41 against 64,000 observations at the ci instance --- not one
    row per observation. Were it one per observation the table would be the
    score array itself and the kernel's reason for existing would be gone.

    The outer product is larger than the *distinct* pairs, which the first
    version of this tabulated. It is also 10x faster to build, because those
    were found by sorting an `(S * V, 2)` array once per call: 135.6 ms of a
    141.4 ms E step, which put this backend at 0.6x the oracle it exists to
    beat. Fewer rows was the wrong thing to optimize.
    """
    instance = _instance()
    params = _covaried(instance)
    observations = instance.observations
    n_observations = observations.shape[0] * observations.shape[1]

    _, _, total_table, success_table = rust.emission_rows(params, observations)

    assert total_table.shape[0] < n_observations
    assert success_table.shape[0] < n_observations


@pytest.mark.mathematical
def test_the_covariate_changes_the_answer() -> None:
    """A backend that quietly dropped one would pass every test above."""
    instance = _instance()
    observations, labels = instance.observations, instance.labels

    plain = rust.class_posteriors(instance.params, observations, labels)
    covaried = rust.class_posteriors(_covaried(instance), observations, labels)

    assert np.abs(covaried.log_evidence - plain.log_evidence).max() > 1.0
