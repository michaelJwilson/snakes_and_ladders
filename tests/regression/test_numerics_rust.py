"""The Rust categorical sampler against the NumPy oracle it replaces.

Root ``CLAUDE.md`` requires every accelerated kernel to keep its NumPy
implementation as the oracle, and this comparison is stronger than most:
because the uniforms are drawn in Python and handed to Rust, the two
implementations differ in arithmetic alone. So the assertion is **exact
equality**, not a tolerance — a categorical index is discrete, and a port
that moved one draw into the neighbouring category would be a defect, not a
rounding difference.

What is checked is the arithmetic that can actually diverge: the left-to-right
cumulative sum, the clamp on the last column, and the choice of the first
crossing. A pairwise summation in Rust would be *more* accurate than
``np.cumsum`` and would still be wrong here, because the oracle is the
definition.
"""

from __future__ import annotations

import numpy as np
import pytest
from phylo.numerics import sample_rows as oracle
from phylo.numerics_rust import sample_rows as accelerated

SEED = 20260904


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize(
    ("n_rows", "n_categories", "n_draws"),
    [(4, 4, 200_000), (3, 2, 50_000), (8, 7, 5_000), (1, 3, 100)],
)
def test_the_rust_sampler_is_bit_identical_to_the_oracle(
    n_rows: int, n_categories: int, n_draws: int
) -> None:
    distributions = np.random.default_rng(1).dirichlet(
        np.ones(n_categories), size=n_rows
    )
    rows = np.random.default_rng(2).integers(n_rows, size=n_draws)

    assert np.array_equal(
        oracle(np.random.default_rng(SEED), distributions, rows),
        accelerated(np.random.default_rng(SEED), distributions, rows),
    )


@pytest.mark.critical
@pytest.mark.edge_case
@pytest.mark.oracle
def test_both_agree_on_a_row_that_does_not_quite_sum_to_one() -> None:
    # The case `phylo.numerics`' docstring calls out as ordinary float64
    # behaviour: the row leaves a sliver of the unit interval above its own
    # total, and a draw landing there crosses no column. Both implementations
    # must clamp it to the last category rather than report an index past the
    # end of the alphabet.
    distributions = np.array([[0.5, 0.5 - 4e-16]])
    assert float(distributions.sum()) < 1.0
    rows = np.zeros(200_000, dtype=np.int64)

    expected = oracle(np.random.default_rng(7), distributions, rows)
    realized = accelerated(np.random.default_rng(7), distributions, rows)

    assert np.array_equal(expected, realized)
    assert int(realized.max()) == distributions.shape[1] - 1


@pytest.mark.critical
@pytest.mark.edge_case
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
@pytest.mark.edge_case
@pytest.mark.oracle
def test_a_non_contiguous_input_gives_the_same_answer() -> None:
    # Borrowing rather than copying makes stride a real concern where it was
    # not before. `phylo.numerics_rust` normalizes with `ascontiguousarray`
    # -- free when the array already is contiguous -- so a sliced view must
    # still agree with the oracle rather than reading every other element of
    # something else.
    distributions = np.random.default_rng(1).dirichlet(np.ones(4), size=4)
    rows = np.random.default_rng(2).integers(4, size=2000)
    sliced = rows[::2]
    assert not sliced.flags["C_CONTIGUOUS"]

    assert np.array_equal(
        oracle(np.random.default_rng(SEED), distributions, sliced),
        accelerated(np.random.default_rng(SEED), distributions, sliced),
    )


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_one_dimensional_distribution_is_refused() -> None:
    # Matching the oracle's own refusal: a 1-D distribution has no row to
    # select, and broadcasting past it would sample from the wrong thing.
    with pytest.raises(ValueError, match="n_rows, n_categories"):
        accelerated(np.random.default_rng(0), np.array([0.5, 0.5]), np.array([0]))


@pytest.mark.critical
@pytest.mark.edge_case
def test_a_row_index_past_the_distributions_is_refused() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        accelerated(
            np.random.default_rng(0),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([2]),
        )
