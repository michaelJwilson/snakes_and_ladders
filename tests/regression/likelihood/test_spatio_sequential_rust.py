"""The Rust coupled E step against the NumPy oracle (issue #399).

`likelihood/CLAUDE.md`: the reference implementation is the oracle and it
stays, and a backend is accepted or rejected against a stated tolerance
rather than adjusted until it matches. The tolerances below are the measured
ones, and the measurement says which quantity carries which form.

The two paths do not share their arithmetic. The oracle evaluates three
`lgamma` calls per count and runs forward--backward through `logsumexp`; the
kernel reads a table the oracle's own families filled and runs the scaled
recursion, each position's row maximum divided out. So agreement is a
tolerance, and one of the two is measurably better normalized: at the ci
instance the oracle's state posterior departs from summing to one by up to
1.3e-9, which is the whole of the difference between them, and the kernel's
rows sum to one exactly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.emissions import CategoricalEmission
from snakes_and_ladders.likelihood import spatio_sequential_rust as rust
from snakes_and_ladders.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
)
from snakes_and_ladders.sim.count_pairs import (
    CountPairInstance,
    IndependentCountPair,
)
from snakes_and_ladders.sim.count_pairs_rust import binned_instance, fine_instance
from snakes_and_ladders.sim.fixtures import fixture

from tests._scale import stress_only

PROBLEM = "spatio_sequential_counts"

#: Relative agreement required of a log evidence and of a field: both are sums
#: over positions and vertices, so an absolute bound fixed at one size does not
#: transfer to another (root `CLAUDE.md`). Measured at the ci instance: 3.1e-15
#: on the evidence and 3.8e-15 on the field.
LOG_TOLERANCE = 1e-12

#: Absolute agreement required of a state posterior and a pairwise posterior.
#: Absolute rather than relative because both are probabilities on ``[0, 1]``
#: and the entries the two paths differ on are the ones near zero, where a
#: relative bound is a bound on nothing. Measured at the ci instance: 1.3e-9,
#: which is exactly the oracle's own departure from normalization.
POSTERIOR_TOLERANCE = 1e-8

#: Vertices of the 5K instance the slice test scores, spread across every
#: planted band so all ten classes have members. 64 is what the ci instance
#: carries, so the two comparisons are at one size.
SLICE_VERTICES = 64


def _ci() -> CountPairInstance:
    """The ci instance, drawn once."""
    return fine_instance(fixture(PROBLEM, "ci").path)


def _slice(instance: CountPairInstance) -> tuple[np.ndarray, np.ndarray]:
    """``SLICE_VERTICES`` vertices spread over the lattice, with their labels."""
    n_nodes = instance.observations.shape[1]
    taken = np.arange(0, n_nodes, n_nodes // SLICE_VERTICES)[:SLICE_VERTICES]
    return (
        np.ascontiguousarray(instance.observations[:, taken]),
        np.ascontiguousarray(instance.labels[taken]),
    )


def _agree(
    instance: CountPairInstance, observations: np.ndarray, labels: np.ndarray
) -> None:
    """Both kernels against both oracles on one set of observations."""
    expected = class_posteriors(instance.params, observations, labels)
    actual = rust.class_posteriors(instance.params, observations, labels)

    np.testing.assert_allclose(
        actual.log_evidence, expected.log_evidence, rtol=LOG_TOLERANCE
    )
    np.testing.assert_allclose(
        actual.posterior, expected.posterior, atol=POSTERIOR_TOLERANCE
    )
    np.testing.assert_allclose(
        actual.pairwise, expected.pairwise, atol=POSTERIOR_TOLERANCE
    )
    (
        np.testing.assert_allclose(actual.posterior.sum(axis=-1), 1.0, atol=1e-12),
        "the kernel's own posterior must be normalized, whatever the oracle's is",
    )

    np.testing.assert_allclose(
        rust.external_field(instance.params, observations, labels, expected.posterior),
        external_field(instance.params, observations, labels, expected.posterior),
        rtol=LOG_TOLERANCE,
    )


@pytest.mark.oracle
def test_the_rust_e_step_and_field_match_the_numpy_oracle() -> None:
    # At the declared ci instance, where the oracle is the transfer matrix:
    # given the labelling the classes decouple and each chain's evidence and
    # state posterior are the exact forward recursion, which the NumPy path
    # runs and this kernel does not share.
    instance = _ci()

    _agree(instance, instance.observations, instance.labels)


@pytest.mark.oracle
@stress_only("the 5K instance's 1.0e8 draws are simulated before anything is scored")
def test_the_rust_e_step_matches_the_oracle_on_a_slice_of_the_5k_instance() -> None:
    # The same comparison at the declared size's counts, class count and state
    # count, on 64 of its 5,041 vertices --- spread across the planted bands,
    # so every class has members --- because the oracle's cost at all 5,041 is
    # minutes per sweep and the kernel's advantage is exactly that.
    instance = binned_instance(fixture(PROBLEM, "stress").path, 10)
    observations, labels = _slice(instance)

    assert np.unique(labels).size == instance.params.n_classes
    _agree(instance, observations, labels)


@pytest.mark.structural
def test_the_tables_are_the_families_own_log_density() -> None:
    # The kernel does no arithmetic of its own on a count: it reads what the
    # families wrote. A table that had drifted from them would be a second
    # emission model nothing else knows about.
    instance = _ci()
    params = instance.params
    totals, successes = rust.emission_tables(params, instance.observations)

    for m, family in enumerate(params.emissions):
        assert isinstance(family, IndependentCountPair)
        counts = torch.arange(totals.shape[0], dtype=torch.float64)
        np.testing.assert_array_equal(
            totals[:, m, :], family.total.log_density(counts).numpy()
        )
        counts = torch.arange(successes.shape[0], dtype=torch.float64)
        np.testing.assert_array_equal(
            successes[:, m, :], family.successes.log_density(counts).numpy()
        )


@pytest.mark.edge_case
def test_a_count_past_the_table_is_refused_rather_than_clamped() -> None:
    # The table is built from the counts it will be indexed by, so a count the
    # caller added afterwards has no entry. Reading whatever is at that offset
    # would be a silently wrong density; clamping would be a silently wrong
    # model.
    instance = _ci()
    params = instance.params
    totals, successes = rust.emission_tables(params, instance.observations)
    counts = np.ascontiguousarray(
        instance.observations[..., 0], dtype=np.uint16
    ).reshape(-1)
    counts[0] = totals.shape[0]
    arguments = (
        counts,
        np.ascontiguousarray(instance.observations[..., 1], dtype=np.uint16).reshape(
            -1
        ),
        np.ascontiguousarray(instance.labels, dtype=np.int64),
        totals.reshape(-1),
        successes.reshape(-1),
        np.ascontiguousarray(np.log(params.initial)).reshape(-1),
        np.ascontiguousarray(
            np.broadcast_to(
                np.log(params.transition),
                (params.n_classes, params.n_states, params.n_states),
            )
        ).reshape(-1),
        params.n_positions,
        instance.observations.shape[1],
        params.n_classes,
        params.n_states,
        np.empty(params.n_classes * params.n_positions * params.n_states),
        np.empty(
            params.n_classes
            * (params.n_positions - 1)
            * params.n_states
            * params.n_states
        ),
        np.empty(params.n_classes),
    )

    with pytest.raises(ValueError, match="past the table's extent"):
        oxi_snakes_and_ladders.class_posteriors(*arguments)


@pytest.mark.edge_case
def test_a_family_that_is_not_a_count_pair_is_refused() -> None:
    # The kernel tabulates two channels by their integer counts. A categorical
    # family has no channels to tabulate, and the refusal names the class
    # rather than failing later on a shape.
    instance = _ci()
    params = replace(
        instance.params,
        emissions=tuple(
            CategoricalEmission(np.full((instance.params.n_states, 3), 1.0 / 3.0))
            for _ in range(instance.params.n_classes)
        ),
    )

    with pytest.raises(TypeError, match="two-channel count emission"):
        rust.class_posteriors(params, instance.observations, instance.labels)
