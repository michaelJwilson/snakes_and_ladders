"""Recovering the planted labelling of the 5,041-vertex coupled instance (issue #399).

The claim the declared instance exists to support: at every bin factor, block
ascent over the coupled model recovers the class each vertex was planted in,
up to a permutation of the class names --- which is all a class label is.
The permutation is found by linear assignment on the contingency table
(`search.spatio_sequential.label_accuracy`), since ``M!`` at ``M = 10`` is
3.6 million.

**Which channel is checked for what.** Binning sums the counts of ``f``
consecutive positions in both channels. That is exact for the negative
binomial --- a sum of ``f`` independent ``NB(r, p)`` counts is ``NB(f r, p)``,
so the coarse instance's mean and dispersion are ``f`` times the fine one's,
and the test below asserts them on the data. It is **not** exact for the
beta-binomial: a sum of ``f`` beta-binomials is not beta-binomial, and
``BetaBinomial(f n, a, b)`` --- the closest member of the family, and what
`sim.count_pairs.aggregate` returns --- keeps the mean and overstates the
variance by more than a factor of four. So at a coarse factor the second
channel is a declared misspecification, and the only thing asserted of it is
that the labelling still comes back: no parameter of it is claimed.

The tiers are the fixture file's, not this module's: each bin factor carries
the marker its declaration names, measured rather than assumed (`DEV.md`).
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.spatio_sequential_rust import RUST_BACKEND
from snakes_and_ladders.search.spatio_sequential import (
    LabelSolver,
    fit_spatio_sequential,
    label_accuracy,
)
from snakes_and_ladders.sim.count_pairs import (
    TOTAL,
    CountPairInstance,
    IndependentCountPair,
    aggregate,
)
from snakes_and_ladders.sim.count_pairs_rust import binned_instance
from snakes_and_ladders.sim.fixtures import KEY, fixture

from tests._scale import at_bin, stress_only

PROBLEM = "spatio_sequential_counts"

#: Block-ascent rounds. One is enough: measured at bin factors 10 and 5, the
#: labelling is exact after the first and does not move in the second
#: (`STATUS.md`), and the alpha-expansion is 20 of the 23 seconds a round
#: costs at 5,041 vertices, so a second round buys nothing and doubles the
#: budget.
BLOCKS = 1

#: The fraction of the planted labelling the start corrupts, and its seed.
#: The start is a corrupted planting rather than a uniform draw for the
#: reason the ci-size test in `test_spatio_sequential_fit.py` starts from the
#: planted labels: what is claimed here is that the label block *corrects* a
#: labelling, and from a uniform start at ``M = 10`` the first E step's class
#: densities are ten mixtures of the same data and the field they give is
#: measured at chance (`STATUS.md`). Which start the block ascent can be
#: driven from is issue #306's question and not this fixture's.
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
    """Block ascent from a corrupted planting with the parameters held at the truth.

    ``fit_parameters=False``: what is claimed here is label recovery, and
    re-estimating the emissions at the same time would leave a failure
    ambiguous between the two blocks.
    """
    return fit_spatio_sequential(
        instance.params,
        instance.observations,
        np.random.default_rng(START_SEED),
        solver=LabelSolver.ALPHA_EXPANSION,
        n_blocks=BLOCKS,
        labels=_start(instance),
        fit_parameters=False,
        backend=RUST_BACKEND,
    ).labels


@pytest.mark.simulated_truth
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


@pytest.mark.simulated_truth
@stress_only("the 5K instance's 1.0e8 draws are simulated before anything is scored")
def test_the_field_alone_names_every_vertex_class() -> None:
    # The same recovery without the spatial prior or the label solver: the
    # class of lowest external field at the planted labelling is the planted
    # class, at every vertex. It is what makes the fit above a correction
    # rather than a search, and it is the statement that a defect in the
    # emission table fails --- with one class's evidence NaN the argmin is
    # constant and this reads 0.113, the fraction of vertices in class 0.
    instance = _instance(fixture(PROBLEM, KEY).params.key_factor)
    posterior = RUST_BACKEND.class_posteriors(
        instance.params, instance.observations, instance.labels
    ).posterior
    field = RUST_BACKEND.external_field(
        instance.params, instance.observations, instance.labels, posterior
    )

    assert np.isfinite(field).all()
    np.testing.assert_array_equal(np.argmin(field, axis=1), instance.labels)


@pytest.mark.simulated_truth
@stress_only("the 5K instance's 1.0e8 draws are simulated before anything is summed")
def test_the_negative_binomial_channel_aggregates_on_the_declared_instance() -> None:
    # The exactness `aggregate` claims, on the 5K instance's own counts rather
    # than on the ci instance's: over the bins whose positions share a hidden
    # state --- two states are two values of p, and only within one state are
    # the summands identically distributed --- the binned totals have the
    # aggregated family's mean and variance.
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
