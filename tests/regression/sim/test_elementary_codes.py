"""Codes whose referee is a closed form, held to it exactly.

Issue #594. Everything asserted here is an **equality over integers** or an
agreement at machine precision, because that is what these codes are for: a
perfect code's sphere-packing count holds with no slack, a repetition code's
error probability has a closed form on every channel, and a single parity
check's exact posterior *is* the tanh rule. None of it is a tolerance chosen
to make a result pass.

The perfection equalities earn their place: building Golay's generator matrix
and reading it as a parity check gives a `(23, 11)` code at `d = 8` that looks
plausible, enumerates, and decodes --- and fails the equality immediately.
That is the mistake this file caught while it was being written.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import (
    decode,
    enumerate_codewords,
    exact_decoding,
)
from snakes_and_ladders.sim.elementary_codes import (
    crc_remainder,
    golay_code,
    hamming_code,
    hamming_correct,
    is_perfect,
    repetition_code,
    repetition_error_rate,
    single_parity_check,
    syndrome,
)


@pytest.mark.oracle
def test_the_single_parity_check_posterior_is_the_tanh_rule() -> None:
    # The reason this code is in the tree: its exact bitwise posterior is the
    # check-node update every LDPC and turbo decode here performs, so brute
    # force pins that update rather than a second implementation of it. One
    # sweep suffices because the code has one check and no cycles.
    code = single_parity_check(6)
    rng = np.random.default_rng(594)

    worst = 0.0
    for _ in range(200):
        ratios = rng.normal(scale=2.0, size=6)
        exact = exact_decoding(code, ratios)
        swept = decode(code, ratios, max_iterations=1)
        worst = max(
            worst, float(np.abs(exact.posterior_llr - swept.posterior_llr).max())
        )

    assert worst < 1e-13


@pytest.mark.analytic
@pytest.mark.parametrize(("m", "n_bits", "dimension"), [(3, 7, 4), (4, 15, 11)])
def test_a_hamming_code_is_perfect(m: int, n_bits: int, dimension: int) -> None:
    # `2^k (1 + n) == 2^n`, an equality over integers. A dropped row or a
    # permuted column changes the dimension and the equation fails at once.
    code = hamming_code(m)

    assert code.n_bits == n_bits
    assert len(enumerate_codewords(code)) == 2**dimension
    assert is_perfect(n_bits, 2**dimension, 1)


@pytest.mark.analytic
def test_the_golay_code_is_perfect_and_its_extension_is_not() -> None:
    # `2^12 (1 + 23 + 253 + 1771) == 2^23` exactly. The extended code is *not*
    # perfect and that is not a defect: adding a parity bit buys distance 8,
    # which corrects three errors and detects a fourth, leaving the spheres no
    # longer covering. Asserted so the extension is not read as strictly better.
    code = golay_code()
    extended = golay_code(extended=True)

    assert (code.n_bits, len(enumerate_codewords(code))) == (23, 2**12)
    assert (extended.n_bits, len(enumerate_codewords(extended))) == (24, 2**12)
    assert is_perfect(23, 2**12, 3)
    assert not is_perfect(24, 2**12, 3)


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: hamming_code(3), 3),
        (lambda: hamming_code(3, extended=True), 4),
        (lambda: golay_code(), 7),
        (lambda: golay_code(extended=True), 8),
    ],
)
def test_the_minimum_distance_is_the_one_the_literature_states(
    build: object, expected: int
) -> None:
    # The invariant that separates these codes from any linear code of the
    # same shape, read off every codeword rather than from the construction.
    words = enumerate_codewords(build())  # type: ignore[operator]
    weights = words.sum(axis=1)

    assert int(weights[weights > 0].min()) == expected


@pytest.mark.analytic
def test_the_syndrome_of_a_hamming_error_is_the_index_of_the_flipped_bit() -> None:
    # Why the columns are written in increasing order: syndrome decoding is a
    # lookup with no table. Checked at *every* position, since an off-by-one in
    # the column order would still pass at one of them.
    code = hamming_code(3)
    zero = np.zeros(code.n_bits, dtype=np.int64)

    for position in range(code.n_bits):
        received = zero.copy()
        received[position] ^= 1
        pattern = syndrome(code, received)
        index = int(sum(int(bit) << place for place, bit in enumerate(pattern)))

        assert index == position + 1
        np.testing.assert_array_equal(hamming_correct(code, 3, received), zero)


@pytest.mark.analytic
def test_the_repetition_error_rate_is_its_closed_form() -> None:
    # An analytic curve, which no fixture in this tree had. The erasure case is
    # exact by construction; the binary symmetric case is the binomial tail
    # with the tie at even length counted as half, and is checked against a
    # direct sum over every error pattern.
    for length in (3, 4, 5):
        for probability in (0.05, 0.2, 0.4):
            assert repetition_error_rate(length, erasure=probability) == pytest.approx(
                probability**length
            )
            direct = 0.0
            for pattern in range(2**length):
                flips = bin(pattern).count("1")
                weight = probability**flips * (1 - probability) ** (length - flips)
                if 2 * flips > length:
                    direct += weight
                elif 2 * flips == length:
                    direct += 0.5 * weight
            assert repetition_error_rate(
                length, crossover=probability
            ) == pytest.approx(direct)


@pytest.mark.analytic
def test_a_repetition_code_carries_exactly_two_words() -> None:
    for length in (2, 4, 7):
        words = enumerate_codewords(repetition_code(length))

        assert len(words) == 2
        assert words.sum(axis=1).tolist() == [0, length]


@pytest.mark.analytic
def test_the_crc_remainder_is_what_division_leaves() -> None:
    # Polynomial long division over GF(2), checked against the property that
    # defines it: the shifted message plus its remainder is divisible, so
    # recomputing the remainder of the transmitted word gives zero.
    generator = np.array([1, 0, 1, 1])
    rng = np.random.default_rng(593)

    for _ in range(50):
        message = rng.integers(0, 2, size=9)
        remainder = crc_remainder(message, generator)
        transmitted = np.concatenate([message, remainder])

        assert remainder.size == generator.size - 1
        assert not crc_remainder(transmitted, generator).any()


@pytest.mark.smoke
def test_the_constructions_refuse_a_shape_they_cannot_build() -> None:
    with pytest.raises(ValueError, match="at least two bits"):
        repetition_code(1)
    with pytest.raises(ValueError, match="at least two bits"):
        single_parity_check(1)
    with pytest.raises(ValueError, match="at least two check bits"):
        hamming_code(1)
    with pytest.raises(ValueError, match="leading one"):
        crc_remainder(np.array([1, 0]), np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="exactly one of erasure or crossover"):
        repetition_error_rate(3)
