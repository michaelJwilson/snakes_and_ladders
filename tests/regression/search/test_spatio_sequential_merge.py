"""A move that lowers the number of classes: merge the pair that most raises the joint (issue #933, R7).

`merge_step` tries every occupied pair, refits once, and keeps the best merge
whose penalized joint is at least the input's. Stress lattice at 16 positions
on a half/half planting (at four the joint merges on 4 of 8, at 8 on none,
margin 8.0 nats; at 16, 96). Referees: a class planted as a split copy is
merged back into its partner; two true classes are not merged at zero
penalty; the kept merge is the best by `labelled_log_likelihood`; a large
penalty merges.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sal.likelihood.spatio_sequential import labelled_log_likelihood
from sal.search.spatio_sequential import merge_step
from sal.sim.fixtures import fixture
from sal.sim.spatio_sequential import simulate_spatio_sequential

from tests._rows import every_value


def _distinct(seed: int) -> tuple:  # type: ignore[type-arg]
    """The stress fixture at 16 positions, drawn on a planted half/half labelling."""
    truth = replace(fixture("spatio_sequential", "stress").params, n_positions=16)
    side = truth.graph.shape[0]
    planted = (np.arange(side * side) % side < side // 2).astype(np.int64)
    data = simulate_spatio_sequential(
        truth, np.random.default_rng([933, 7, seed]), labels=planted
    )
    return truth, data, planted


def _split_instance(seed: int) -> tuple:  # type: ignore[type-arg]
    """:func:`_distinct`, then class 0 split at random into 0 and a copy, class 2."""
    truth, data, planted = _distinct(seed)
    rng = np.random.default_rng([933, 7, seed, 1])
    labels = planted.copy()
    zeros = np.flatnonzero(labels == 0)
    labels[rng.choice(zeros, size=zeros.size // 2, replace=False)] = 2
    split = replace(
        truth,
        n_classes=3,
        initial=np.vstack([truth.initial, truth.initial[:1]]),
        emissions=(*truth.emissions, truth.emissions[0]),
    )
    return truth, split, data, labels


@pytest.mark.critical
@pytest.mark.oracle
def test_a_class_split_from_its_copy_is_merged_back() -> None:
    def check(seed: int) -> None:
        truth, split, data, labels = _split_instance(seed)
        merge = merge_step(split, data.observations, labels)
        assert merge.merged == (0, 2)
        assert set(np.unique(merge.labels)) == {0, 1}
        np.testing.assert_array_equal(merge.labels, _distinct(seed)[2])
        before = labelled_log_likelihood(split, data.observations, labels)
        assert merge.criterion >= before

    every_value(range(3), check)


@pytest.mark.critical
@pytest.mark.oracle
def test_the_kept_merge_is_the_best_pair_by_the_joint() -> None:
    def check(seed: int) -> None:
        _, split, data, labels = _split_instance(seed)
        merge = merge_step(split, data.observations, labels)
        for kept, emptied, value in merge.tried:
            merged = np.where(labels == emptied, kept, labels)
            assert np.unique(merged).size == 2
            assert value <= merge.criterion
        assert merge.criterion == max(value for _, _, value in merge.tried)
        assert merge.criterion == labelled_log_likelihood(
            merge.params, data.observations, merge.labels
        )

    every_value(range(3), check)


@pytest.mark.oracle
def test_classes_the_truth_keeps_apart_are_not_merged() -> None:
    def check(seed: int) -> None:
        truth, data, labels = _distinct(seed)
        merge = merge_step(truth, data.observations, labels)
        assert merge.merged is None
        assert merge.labels is labels or np.array_equal(merge.labels, labels)
        assert merge.criterion == labelled_log_likelihood(
            truth, data.observations, labels
        )
        assert len(merge.tried) == 1
        assert merge.tried[0][2] < merge.criterion

    every_value(range(3), check)


@pytest.mark.smoke
def test_a_penalty_charges_each_occupied_class_and_is_non_negative() -> None:
    truth, data, labels = _distinct(0)
    free = merge_step(truth, data.observations, labels)
    (_, _, merged_joint) = free.tried[0]
    gap = free.criterion - merged_joint
    forced = merge_step(truth, data.observations, labels, penalty=gap + 1.0)
    assert forced.merged == (0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        merge_step(truth, data.observations, labels, penalty=-1.0)
