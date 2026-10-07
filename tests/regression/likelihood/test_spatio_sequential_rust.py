"""The Rust coupled E step against the NumPy oracle (issue #399).

The oracle (`likelihood/CLAUDE.md`) evaluates three `lgamma` calls per count
and runs forward--backward through `logsumexp`; the kernel reads a table the
oracle's families filled and runs the scaled recursion. At ci the oracle's
state posterior departs from summing to one by up to 1.3e-9, the whole of the
difference; the kernel's rows sum to one exactly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.backend import Backend
from sal.emissions import CategoricalEmission, GaussianEmission
from sal.likelihood import spatio_sequential
from sal.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
    rust,
)
from sal.sim.count_pairs import (
    CountPairInstance,
    IndependentCountPair,
)
from sal.sim.count_pairs.rust import binned_instance, fine_instance
from sal.sim.fixtures import fixture
from sal.sim.spatio_sequential import (
    SpatioSequentialParams,
    simulate_spatio_sequential,
)

from tests._scale import stress_only

PROBLEM = "spatio_sequential_counts"

#: Relative agreement required of a log evidence and of a field: both are sums
#: over positions and vertices, so an absolute bound fixed at one size does not
#: transfer to another (root `CLAUDE.md`). Measured at the ci instance: 3.1e-15
#: on the evidence and 3.8e-15 on the field.
LOG_TOLERANCE = 1e-12

#: Absolute on probabilities: they differ near zero, where relative bounds
#: nothing. Measured at ci: 1.3e-9, the oracle's normalization error.
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


@pytest.mark.oracle
@pytest.mark.backend
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


@pytest.mark.smoke
def test_a_count_past_the_table_is_refused_rather_than_clamped() -> None:
    # The table is built from the counts it will be indexed by, so a count the
    # caller added afterwards has no entry. Reading whatever is at that offset
    # would be a silently wrong density; clamping would be a silently wrong
    # model.
    instance = _ci()
    params = instance.params
    totals, successes = rust.emission_tables(params, instance.observations)
    counts = np.ascontiguousarray(
        instance.observations[..., 0], dtype=np.uint32
    ).reshape(-1)
    counts[0] = totals.shape[0]
    arguments = (
        counts,
        np.ascontiguousarray(instance.observations[..., 1], dtype=np.uint32).reshape(
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
        oxisal.class_posteriors(*arguments)


def _categorical(tier: str) -> tuple[SpatioSequentialParams, np.ndarray, np.ndarray]:
    """The categorical model at ``tier``, one draw: the default model of the fit."""
    params = fixture("spatio_sequential", tier).params
    data = simulate_spatio_sequential(params, np.random.default_rng(0))
    return params, data.observations, data.labels


@pytest.mark.oracle
@pytest.mark.parametrize("tier", ["ci", "stress"])
def test_the_rust_route_matches_the_oracle_on_the_categorical_model(tier: str) -> None:
    # Issue #1298: the categorical model is one channel by symbol, and the
    # three entry points the fit routes through agree with the NumPy oracle
    # at the count model's tolerances. Measured: 9.6e-15 on the posterior,
    # 1.7e-14 on the pairwise, 4.2e-15 relative on the field, the evidence
    # bitwise at stress.
    params, observations, labels = _categorical(tier)
    expected = class_posteriors(params, observations, labels)
    actual = class_posteriors(params, observations, labels, backend=Backend.RUST)

    np.testing.assert_allclose(
        actual.log_evidence, expected.log_evidence, rtol=LOG_TOLERANCE
    )
    np.testing.assert_allclose(
        actual.posterior, expected.posterior, atol=POSTERIOR_TOLERANCE
    )
    np.testing.assert_allclose(
        actual.pairwise, expected.pairwise, atol=POSTERIOR_TOLERANCE
    )
    np.testing.assert_allclose(
        external_field(params, observations, labels, backend=Backend.RUST),
        external_field(params, observations, labels),
        rtol=LOG_TOLERANCE,
    )
    np.testing.assert_allclose(
        spatio_sequential.labelled_log_likelihood(
            params, observations, labels, backend=Backend.RUST
        ),
        spatio_sequential.labelled_log_likelihood(params, observations, labels),
        rtol=LOG_TOLERANCE,
    )


@pytest.mark.oracle
@pytest.mark.backend
def test_the_categorical_table_is_the_family_s_own_log_density() -> None:
    # The table is the stored log emission matrix, transposed, so each score
    # is the family's `log_density` bit for bit: the kernel adds nothing.
    params, observations, _ = _categorical("ci")
    table = rust.symbol_table(params, observations).table

    for m, family in enumerate(params.emissions):
        symbols = torch.arange(table.shape[0])
        np.testing.assert_array_equal(
            table[:, m, :], family.log_density(symbols).numpy()
        )


@pytest.mark.smoke
def test_a_family_the_kernel_does_not_tabulate_is_refused() -> None:
    # The kernel tabulates two count channels or one symbol. Any other family,
    # a mixture of the two, two alphabets, or a symbol past the alphabet is
    # refused before the kernel is reached, and the refusal says which.
    params, observations, labels = _categorical("ci")
    gaussian = GaussianEmission(
        np.zeros(params.n_states), np.ones(params.n_states), 1e-6
    )
    pair = _ci().params.emissions[0]
    wider = CategoricalEmission(np.full((params.n_states, 4), 0.25))

    with pytest.raises(TypeError, match="or the categorical emission"):
        rust.class_posteriors(
            replace(params, emissions=(gaussian, gaussian)), observations, labels
        )
    with pytest.raises(TypeError, match="one emission type in every class"):
        rust.class_posteriors(
            replace(params, emissions=(params.emissions[0], pair)),
            observations,
            labels,
        )
    with pytest.raises(ValueError, match="one alphabet"):
        rust.class_posteriors(
            replace(params, emissions=(params.emissions[0], wider)),
            observations,
            labels,
        )
    past = observations.copy()
    past[0, 0] = 3
    with pytest.raises(ValueError, match=r"must lie in \[0, 3\)"):
        rust.class_posteriors(params, past, labels)
    instance = _ci()
    with pytest.raises(ValueError, match="carry two channels"):
        rust.class_posteriors(params, instance.observations, instance.labels)


@pytest.mark.backend
@pytest.mark.oracle
def test_the_enum_reaches_the_rust_kernel_bitwise() -> None:
    # #828: `Backend.RUST` at the NumPy module's three functions is the twin's
    # own call, bitwise, and any other member is refused by name.
    instance = _ci()
    observations, labels = instance.observations, instance.labels
    via_enum = spatio_sequential.class_posteriors(
        instance.params, observations, labels, backend=Backend.RUST
    )
    direct = rust.class_posteriors(instance.params, observations, labels)
    np.testing.assert_array_equal(via_enum.posterior, direct.posterior)
    assert spatio_sequential.labelled_log_likelihood(
        instance.params, observations, labels, backend=Backend.RUST
    ) == rust.labelled_log_likelihood(instance.params, observations, labels)
    np.testing.assert_array_equal(
        spatio_sequential.external_field(
            instance.params, observations, labels, backend=Backend.RUST
        ),
        rust.external_field(instance.params, observations, labels),
    )
    with pytest.raises(ValueError, match="not numba"):
        spatio_sequential.labelled_log_likelihood(
            instance.params, observations, labels, backend=Backend.NUMBA
        )
