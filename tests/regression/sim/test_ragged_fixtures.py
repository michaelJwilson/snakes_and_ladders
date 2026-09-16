"""The two declared ragged instances, and what their segmentation buys.

Issue #666. `ragged_hmm` and `spatio_sequential_ragged` are the same models as
`hmm` and `spatio_sequential`; only the segmentation differs, which is what
`PROBLEMS.md` means by a row with two keys.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.sim import fixtures
from snakes_and_ladders.sim.spatio_sequential import simulate_spatio_sequential


@pytest.mark.critical
@pytest.mark.structural
def test_the_ragged_hmm_fixture_declares_segments_that_differ() -> None:
    """A fixture that declared equal lengths would measure the other instance."""
    params = fixtures.fixture("ragged_hmm", "ci").params
    lengths = params.segment_lengths
    assert len(set(lengths)) > 1, "a ragged fixture whose lengths are equal is not one"
    assert min(lengths) == 2, "the shortest a segment may be is exercised"
    assert params.n_sequences == len(lengths)
    # The claim the shape is chosen for: padding to the longest would be mostly
    # padding, which is what the compiled path exists to avoid.
    waste = 1 - sum(lengths) / (len(lengths) * max(lengths))
    assert waste > 0.5, f"the fixture wastes only {waste:.1%} in a padded block"


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_batch_took_one_transition_fewer_per_boundary() -> None:
    """`T - S` transitions, not `T - 1`: each boundary is a restart."""
    lengths = fixtures.fixture("ragged_hmm", "ci").params.segment_lengths
    assert sum(lengths) - len(lengths) == sum(one - 1 for one in lengths)


@pytest.mark.critical
@pytest.mark.simulated_truth
def test_the_coupled_simulator_restarts_at_every_boundary() -> None:
    """The draw is a different draw, and the seed is not what changed.

    The same parameters, the same seed, segmented and not. If the segmentation
    were ignored the two would be identical; they are not, and the positions
    that differ begin at a boundary.
    """
    declared = fixtures.fixture("spatio_sequential_ragged", "ci").params
    assert declared.segments is not None
    whole = type(declared)(**{**declared.__dict__, "segments": None})

    segmented = simulate_spatio_sequential(declared, np.random.default_rng(0))
    unsegmented = simulate_spatio_sequential(whole, np.random.default_rng(0))

    assert not np.array_equal(segmented.observations, unsegmented.observations), (
        "the segmentation changed nothing, so the chain did not restart"
    )
    assert declared.segment_lengths == (2, 4)
    assert whole.segment_lengths == (declared.n_positions,)
