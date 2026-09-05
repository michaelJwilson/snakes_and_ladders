"""The shared categorical sampler, and the guard two copies of it had lost.

`sample_rows` replaced three near-identical private helpers. Two of them
omitted the clamp the third had, so the point of the consolidation is not
that there is now one copy but that the surviving copy is the correct one.
That is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.numerics import sample_rows


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_degenerate_row_always_yields_its_certain_category() -> None:
    # The one case with an answer that owes nothing to the sampling scheme.
    distributions = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rows = np.array([0, 1, 0, 1, 0])
    drawn = sample_rows(np.random.default_rng(0), distributions, rows)
    assert drawn.tolist() == [1, 0, 1, 0, 1]


@pytest.mark.critical
@pytest.mark.mathematical
def test_the_empirical_frequencies_match_the_distribution() -> None:
    # Against the probabilities themselves, at a sample size where the
    # Monte Carlo error is an order of magnitude below the tolerance:
    # sqrt(p(1-p)/n) <= 5e-4 at n = 1e6, checked to 5e-3.
    distributions = np.array([[0.1, 0.6, 0.3]])
    draws = 1_000_000
    drawn = sample_rows(
        np.random.default_rng(1), distributions, np.zeros(draws, dtype=int)
    )
    frequencies = np.bincount(drawn, minlength=3) / draws
    assert_allclose(frequencies, distributions[0], atol=5e-3)


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_row_summing_below_one_cannot_yield_an_index_past_the_end() -> None:
    # The guard, and the reason this module exists. A normalized row can sum
    # to 1 - 4e-16 after rounding, leaving a sliver of the unit interval above
    # its own total; `rng.random` returns values in [0, 1), so a draw lands
    # there eventually. Without the clamp the draw is past every column and
    # the obvious formulations report category `n`, one past the alphabet.
    #
    # Constructed rather than sampled: the event has probability ~4e-16 per
    # draw, so waiting for it is not a test.
    distributions = np.array([[0.3, 0.3, 0.4 - 5e-16]])
    assert distributions.sum() < 1.0

    class _AtTheTop:
        """A generator returning a draw inside that sliver."""

        def random(self, size: tuple[int, ...] | int) -> np.ndarray:
            return np.full(size, 0.9999999999999998)

    drawn = sample_rows(_AtTheTop(), distributions, np.zeros(4, dtype=int))  # type: ignore[arg-type]
    assert drawn.tolist() == [2, 2, 2, 2]
    assert int(drawn.max()) < distributions.shape[1]


@pytest.mark.critical
@pytest.mark.structural
def test_every_row_is_selectable() -> None:
    distributions = np.eye(4)
    rows = np.arange(4)
    drawn = sample_rows(np.random.default_rng(2), distributions, rows)
    assert drawn.tolist() == [0, 1, 2, 3]


@pytest.mark.critical
@pytest.mark.structural
def test_one_draw_is_consumed_per_entry() -> None:
    # The stream cost must not depend on the outcome, or two callers seeded
    # alike would diverge on data rather than on their seeds.
    distributions = np.array([[0.5, 0.5]])
    first = np.random.default_rng(3)
    sample_rows(first, distributions, np.zeros(7, dtype=int))
    remaining = first.random(3)

    second = np.random.default_rng(3)
    second.random(7)
    assert_allclose(remaining, second.random(3))


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_one_dimensional_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected distributions of shape"):
        sample_rows(np.random.default_rng(0), np.array([0.5, 0.5]), np.zeros(2, int))
