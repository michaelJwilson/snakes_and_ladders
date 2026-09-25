"""The Rust categorical sampler against the NumPy oracle it replaces.

The uniforms are drawn in Python and handed to Rust, so the implementations
differ in arithmetic alone and the assertion is exact equality: a draw moved
to the neighbouring category is a defect. Checked: the left-to-right cumulative
sum, the clamp on the last column, and the first crossing. A pairwise sum would
be more accurate than ``np.cumsum`` and still wrong: the oracle is the definition.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.backend import Backend
from sal.numerics import sample_rows

from tests._rows import every_row


def oracle(
    rng: np.random.Generator, distributions: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    return sample_rows(rng, distributions, rows, backend=Backend.PYTHON)


def accelerated(
    rng: np.random.Generator, distributions: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    return sample_rows(rng, distributions, rows, backend=Backend.RUST)


SEED = 20260904


@pytest.mark.critical
@pytest.mark.oracle
def test_the_rust_sampler_is_bit_identical_to_the_oracle() -> None:
    def check(n_rows: int, n_categories: int, n_draws: int) -> None:
        distributions = np.random.default_rng(1).dirichlet(
            np.ones(n_categories), size=n_rows
        )
        rows = np.random.default_rng(2).integers(n_rows, size=n_draws)

        assert np.array_equal(
            oracle(np.random.default_rng(SEED), distributions, rows),
            accelerated(np.random.default_rng(SEED), distributions, rows),
        )

    every_row([(4, 4, 200_000), (3, 2, 50_000), (8, 7, 5_000), (1, 3, 100)], check)


@pytest.mark.critical
@pytest.mark.oracle
def test_both_agree_on_a_row_that_does_not_quite_sum_to_one() -> None:
    # A row leaving a sliver above its own total: both implementations must
    # clamp a draw there to the last category.
    distributions = np.array([[0.5, 0.5 - 4e-16]])
    assert float(distributions.sum()) < 1.0
    rows = np.zeros(200_000, dtype=np.int64)

    expected = oracle(np.random.default_rng(7), distributions, rows)
    realized = accelerated(np.random.default_rng(7), distributions, rows)

    assert np.array_equal(expected, realized)
    assert int(realized.max()) == distributions.shape[1] - 1


@pytest.mark.critical
@pytest.mark.smoke
def test_a_degenerate_distribution_selects_its_only_supported_category() -> None:
    # Point masses make the answer known without reference to either
    # implementation, so this catches a row-indexing error the randomized
    # comparison above would show only as a disagreement.
    distributions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rows = np.array([0, 1, 0, 1, 1, 0])

    realized = accelerated(np.random.default_rng(3), distributions, rows)

    assert realized.tolist() == [0, 2, 0, 2, 2, 0]


@pytest.mark.critical
@pytest.mark.oracle
def test_the_generator_is_consumed_identically_by_both() -> None:
    # A caller may swap the implementations without its stream diverging, and
    # that is a property of the port rather than a coincidence: both draw one
    # uniform per entry, in order, before doing any lookup.
    distributions = np.random.default_rng(1).dirichlet(np.ones(3), size=2)
    rows = np.random.default_rng(2).integers(2, size=1000)

    for sampler in (oracle, accelerated):
        rng = np.random.default_rng(SEED)
        sampler(rng, distributions, rows)
        assert rng.random() == pytest.approx(
            np.random.default_rng(SEED).random(size=1001)[-1]
        )


@pytest.mark.critical
@pytest.mark.oracle
def test_a_non_contiguous_input_gives_the_same_answer() -> None:
    # `numerics_rust` borrows via `ascontiguousarray`, so a sliced view must
    # still agree with the oracle.
    distributions = np.random.default_rng(1).dirichlet(np.ones(4), size=4)
    rows = np.random.default_rng(2).integers(4, size=2000)
    sliced = rows[::2]
    assert not sliced.flags["C_CONTIGUOUS"]

    assert np.array_equal(
        oracle(np.random.default_rng(SEED), distributions, sliced),
        accelerated(np.random.default_rng(SEED), distributions, sliced),
    )


@pytest.mark.critical
@pytest.mark.smoke
def test_a_one_dimensional_distribution_is_refused() -> None:
    # Matching the oracle's own refusal: a 1-D distribution has no row to
    # select, and broadcasting past it would sample from the wrong thing.
    with pytest.raises(ValueError, match="n_rows, n_categories"):
        accelerated(np.random.default_rng(0), np.array([0.5, 0.5]), np.array([0]))


@pytest.mark.critical
@pytest.mark.smoke
def test_a_row_index_past_the_distributions_is_refused() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        accelerated(
            np.random.default_rng(0),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([2]),
        )
