"""The bicycle construction, held to the algebra it is defined by.

A bicycle code is ``H = [A | A^T]`` for a circulant ``A``, rows deleted to
the target rate (``sec:ldpc:bicycle``). Every claim here is exact over GF(2)
and needs no tolerance: the circulant against a shift of its own first row,
the degrees against the weight asked for, the self-orthogonality
``H H^T = 0`` against a dense product, the rate against the rank, and the
minimum distance against an enumeration of the code. The deletion heuristic
is refereed by the alternative it is chosen over -- deleting at random.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import enumerate_codewords
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.ldpc import (
    ParityCheck,
    bicycle_code,
    generator_matrix,
)

CIRCULANT_WEIGHT = 3


def _code(n_bits: int, n_checks: int, seed: int) -> ParityCheck:
    return bicycle_code(n_bits, n_checks, CIRCULANT_WEIGHT, np.random.default_rng(seed))


def _first_row(code: ParityCheck) -> np.ndarray:
    """The circulant's first row, read back off the left half of ``H``."""
    return np.flatnonzero(code.dense()[0, : code.n_bits // 2])


def _circulant(first_row: np.ndarray, size: int) -> np.ndarray:
    """The circulant of ``first_row``, rolled a row at a time.

    Built here rather than imported: the construction assembles the same
    matrix as edges, and a referee sharing that code would check nothing.
    """
    top = np.zeros(size, dtype=np.uint8)
    top[first_row] = 1
    return np.stack([np.roll(top, i) for i in range(size)])


# --- the matrix the construction defines ---------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("n_bits", [12, 96, 996])
def test_the_halves_are_a_circulant_and_its_transpose(n_bits: int) -> None:
    """`H = [A | A^T]` for the circulant of the first row `H` itself carries."""
    code = _code(n_bits, n_bits // 2, seed=n_bits)
    dense = code.dense()

    left, right = dense[:, : n_bits // 2], dense[:, n_bits // 2 :]

    expected = _circulant(_first_row(code), n_bits // 2)
    np.testing.assert_array_equal(left, expected)
    np.testing.assert_array_equal(right, expected.T)


@pytest.mark.mathematical
@pytest.mark.parametrize("n_bits", [12, 96, 996])
def test_the_degrees_before_deletion_are_the_circulant_weight(n_bits: int) -> None:
    """Every column carries `w` ones and every row `2w`, at every size."""
    code = _code(n_bits, n_bits // 2, seed=n_bits + 1)

    assert code.n_checks == n_bits // 2
    assert code.n_edges == n_bits * CIRCULANT_WEIGHT
    assert np.all(code.column_weights == CIRCULANT_WEIGHT)
    assert np.all(code.row_weights == 2 * CIRCULANT_WEIGHT)


@pytest.mark.mathematical
@pytest.mark.parametrize("n_checks", [48, 40, 32])
def test_the_row_space_is_self_orthogonal(n_checks: int) -> None:
    """`H H^T = 0` over GF(2), before and after deletion: the CSS condition.

    Nothing classical needs it. It holds because a circulant and its
    transpose are both polynomials in the cyclic shift and so commute,
    making `H H^T = A A^T + A^T A = 0` over GF(2), and deleting rows keeps
    it. The quantum half of issue #362 rests on it, so it is asserted here.
    """
    code = _code(96, n_checks, seed=5)
    dense = code.dense().astype(np.int64)

    product = (dense @ dense.T) % 2

    assert code.n_checks == n_checks
    assert not np.any(product)


@pytest.mark.mathematical
@pytest.mark.parametrize("n_bits", [12, 96, 996])
def test_the_all_ones_word_is_a_codeword(n_bits: int) -> None:
    """Every row has even weight `2w`, so `d <= n` whatever the draw."""
    code = _code(n_bits, n_bits // 2, seed=n_bits + 2)

    syndrome = code.syndrome(np.ones(n_bits, dtype=np.uint8))

    assert not np.any(syndrome)


# --- the rate the deletion buys -------------------------------------------------


@pytest.mark.mathematical
def test_deletion_raises_the_rate_and_keeps_the_row_weight() -> None:
    """Deleting 12 of 48 rows at `n = 96` takes `k` from 50 to 60, `2w` fixed.

    `k = n - rank H` is at least the design rate's `n - m` at every count,
    since deleting a row cannot raise the rank, and rises with each deletion.
    A deleted row takes ones out of `2w` columns, so column weights fall and
    row weights do not move.
    """
    codes = [_code(96, n_checks, seed=6) for n_checks in (48, 44, 40, 36)]

    dimensions = [int(generator_matrix(code).shape[0]) for code in codes]

    assert dimensions == [50, 52, 56, 60]
    for code, dimension in zip(codes, dimensions, strict=True):
        assert dimension >= code.n_bits - code.n_checks
        assert np.all(code.row_weights == 2 * CIRCULANT_WEIGHT)
        assert np.all(code.column_weights <= CIRCULANT_WEIGHT)


@pytest.mark.structural
def test_deleting_the_heaviest_rows_beats_deleting_at_random() -> None:
    # The construction chooses which rows to delete rather than deleting any
    # twelve, and what it chooses them for is uniform column weights. The
    # referee is the alternative: 20 random deletions of the same count,
    # scored by the sum of squared column weights, which is minimized when
    # the weights are level.
    code = _code(96, 36, seed=7)
    full = _circulant(_first_row(code), 48)
    dense = np.concatenate([full, full.T], axis=1)

    chosen = float((code.column_weights.astype(float) ** 2).sum())

    rng = np.random.default_rng(8)
    drawn = [
        float((dense[rng.permutation(48)[:36]].sum(axis=0).astype(float) ** 2).sum())
        for _ in range(20)
    ]
    assert chosen <= min(drawn)


# --- what an enumeration says about the declared instance -----------------------


@pytest.mark.oracle
def test_the_declared_instance_has_the_distance_enumeration_gives() -> None:
    """The 12-bit fixture is a `[12, 6, 4]` code: 64 codewords, lightest of weight 4."""
    code = fixture("bicycle", "ci").params.code()

    words = enumerate_codewords(code)

    weights = words.sum(axis=1)
    assert words.shape[0] == 2**6
    assert int(weights[weights > 0].min()) == 4
    assert int(weights.max()) == code.n_bits
    assert len({tuple(column) for column in code.dense().T}) == code.n_bits


@pytest.mark.structural
def test_two_draws_from_one_generator_differ_and_seeds_agree() -> None:
    # `sim/CLAUDE.md`: a generator, never a seed. The circulant's first row is
    # drawn from the generator, so two draws differ and two alike agree.
    rng = np.random.default_rng(9)
    first, second = _draw(rng), _draw(rng)

    assert first != second
    assert _draw(np.random.default_rng(10)) == _draw(np.random.default_rng(10))


def _draw(rng: np.random.Generator) -> tuple[int, ...]:
    code = bicycle_code(96, 48, CIRCULANT_WEIGHT, rng)
    return tuple(int(v) for v in np.flatnonzero(code.dense()[0]))


# --- refusals -------------------------------------------------------------------


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("n_bits", "n_checks", "weight", "message"),
    [
        (11, 5, 3, "even"),
        (12, 6, 1, "circulant_weight"),
        (12, 6, 7, "circulant_weight"),
        (12, 0, 3, "n_checks"),
        (12, 7, 3, "n_checks"),
    ],
)
def test_a_shape_the_construction_cannot_build_is_refused(
    n_bits: int, n_checks: int, weight: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bicycle_code(n_bits, n_checks, weight, np.random.default_rng(11))


@pytest.mark.edge_case
def test_a_rate_past_what_the_length_supports_is_refused() -> None:
    """Deleting to 12 of 48 rows leaves a bit in no check, which is refused."""
    with pytest.raises(ValueError, match="left a bit in no check"):
        bicycle_code(96, 12, CIRCULANT_WEIGHT, np.random.default_rng(12))
