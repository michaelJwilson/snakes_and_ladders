"""Recovering the planted labelling of the 5,041-vertex coupled instance (issue #399).

At every bin factor block ascent recovers each vertex's class up to a
permutation, found by linear assignment (``M!`` is 3.6 million at ``M = 10``).
Binning is exact for the negative binomial (``NB(f r, p)``), asserted on the
data; for the beta-binomial ``BetaBinomial(f n, a, b)`` (`aggregate`) keeps
the mean and overstates the variance over fourfold, a declared
misspecification of which only the labelling is claimed. Tiers are the
fixture's, measured (`DEV.md`).
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.likelihood.spatio_sequential import (
    class_posteriors,
    external_field,
)
from sal.search.spatio_sequential import (
    LabelSolver,
    fit_spatio_sequential,
    label_accuracy,
)
from sal.sim.count_pairs import (
    TOTAL,
    CountPairInstance,
    IndependentCountPair,
    aggregate,
)
from sal.sim.fixtures import KEY, fixture
from sal.sim.rust.count_pairs import binned_instance

from tests._scale import at_bin, stress_only

PROBLEM = "spatio_sequential_counts"

#: One round: exact after the first at factors 10 and 5 (`STATUS.md`);
#: alpha-expansion is 20 of 23 s a round at 5,041 vertices.
BLOCKS = 1

#: The start corrupts the planting: the claim is correction. From uniform at
#: ``M = 10`` the first field is at chance (`STATUS.md`; start choice is #306).
PERTURBED = 0.3
START_SEED = 7

#: The labelling must come back exactly at every factor. Each vertex carries
#: thousands of positions of evidence, so a vertex left in the wrong class is
#: a defect and not a hard case; the measurement is in `STATUS.md`.
REQUIRED_ACCURACY = 1.0


def _instance(factor: int) -> CountPairInstance:
    """The declared 5K instance at one bin factor: simulated once, then binned."""
    return binned_instance(fixture(PROBLEM, KEY).path, factor)


def _start(instance: CountPairInstance) -> np.ndarray:
    """The planted labelling with :data:`PERTURBED` of its vertices redrawn."""
    rng = np.random.default_rng(START_SEED)
    labels = instance.labels.copy()
    hit = rng.choice(labels.size, size=int(PERTURBED * labels.size), replace=False)
    labels[hit] = rng.integers(0, instance.params.n_classes, size=hit.size)
    return labels


def _fitted(instance: CountPairInstance) -> np.ndarray:
    """Block ascent from a corrupted planting, parameters held at the truth."""
    return fit_spatio_sequential(
        instance.params,
        instance.observations,
        np.random.default_rng(START_SEED),
        solver=LabelSolver.ALPHA_EXPANSION,
        n_blocks=BLOCKS,
        labels=_start(instance),
        fit_parameters=False,
        backend=Backend.RUST,
    ).labels


@pytest.mark.end2end
@at_bin("factor", PROBLEM)
def test_the_planted_labelling_is_recovered_at_every_bin_factor(factor: int) -> None:
    # The full test of one declared instance: simulate, bin, fit, assert. The
    # key factor is the largest of the three that fits the key budget, which
    # is what makes it the key one.
    instance = _instance(factor)
    start = label_accuracy(_start(instance), instance.labels, instance.params.n_classes)

    accuracy = label_accuracy(
        _fitted(instance), instance.labels, instance.params.n_classes
    )

    assert start < 0.8, "the start must have something to correct"
    assert accuracy >= REQUIRED_ACCURACY, (
        f"bin factor {factor}: {accuracy:.4f} of vertices recovered from {start:.4f}"
    )


@pytest.mark.end2end
@stress_only("the 5K instance's 1.0e8 draws are simulated before anything is scored")
def test_the_field_alone_names_every_vertex_class() -> None:
    # No prior, no solver: the lowest-field class is the planted one at every
    # vertex; a NaN class reads 0.113, the class-0 fraction.
    instance = _instance(fixture(PROBLEM, KEY).params.key_factor)
    posterior = class_posteriors(
        instance.params,
        instance.observations,
        instance.labels,
        backend=Backend.RUST,
    ).posterior
    field = external_field(
        instance.params,
        instance.observations,
        instance.labels,
        posterior,
        backend=Backend.RUST,
    )

    assert np.isfinite(field).all()
    np.testing.assert_array_equal(np.argmin(field, axis=1), instance.labels)


@pytest.mark.end2end
@stress_only("the 5K instance's 1.0e8 draws are simulated before anything is summed")
def test_the_negative_binomial_channel_aggregates_on_the_declared_instance() -> None:
    # `aggregate`'s exactness on the 5K counts, within one hidden state:
    # binned totals have the aggregated mean and variance.
    factor = fixture(PROBLEM, KEY).params.key_factor
    instance = _instance(factor)
    model = instance.params
    blocks = instance.states.reshape(model.n_classes, -1, factor)

    fine = _instance(1).params.emissions
    for m in range(model.n_classes):
        family = fine[m]
        assert isinstance(family, IndependentCountPair)
        aggregated = aggregate(family, factor).total
        members = np.flatnonzero(instance.labels == m)
        constant = blocks[m].min(axis=1) == blocks[m].max(axis=1)
        for k in range(model.n_states):
            bins = np.flatnonzero(constant & (blocks[m, :, 0] == k))
            values = instance.observations[np.ix_(bins, members)][..., TOTAL].astype(
                np.float64
            )
            assert values.size > 10_000
            np.testing.assert_allclose(
                values.mean(), float(aggregated.mean[k]), rtol=0.02
            )
            np.testing.assert_allclose(
                values.var(), float(aggregated.variance[k]), rtol=0.10
            )
