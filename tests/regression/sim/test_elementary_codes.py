"""Codes whose referee is a closed form, held to it exactly.

Issue #594. Equalities over integers or agreement at machine precision: a
perfect code's sphere-packing count, a repetition code's error probability on
every channel, and a single parity check's posterior as the tanh rule. The
perfection equality caught Golay's generator read as a parity check, a
plausible `(23, 11)` code at `d = 8`.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sal.likelihood.algebraic import hamming_correct, syndrome
from sal.likelihood.ldpc import (
    decode,
    enumerate_codewords,
    exact_decoding,
)
from sal.sim.elementary_codes import (
    crc_remainder,
    golay_code,
    hamming_code,
    is_perfect,
    repetition_code,
    repetition_error_rate,
    single_parity_check,
)
from sal.sim.ldpc import (
    BinaryErasureChannel,
    BinarySymmetricChannel,
    encode,
    generator_matrix,
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


@pytest.mark.oracle
@pytest.mark.analytic
@pytest.mark.parametrize(("m", "n_bits", "dimension"), [(3, 7, 4), (4, 15, 11)])
def test_a_hamming_code_is_perfect(m: int, n_bits: int, dimension: int) -> None:
    # `2^k (1 + n) == 2^n`, an equality over integers. A dropped row or a
    # permuted column changes the dimension and the equation fails at once.
    code = hamming_code(m)

    assert code.n_bits == n_bits
    assert len(enumerate_codewords(code)) == 2**dimension
    assert is_perfect(n_bits, 2**dimension, 1)


@pytest.mark.oracle
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


@pytest.mark.oracle
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


#: The crossover the maximum-likelihood decode is read at. Any value below a
#: half orders the codewords by Hamming distance, so the referee is the
#: distance and the number only fixes where `exact_decoding` is evaluated.
BSC_FLIP = 0.05

#: Received words drawn at the `(15, 11)` Hamming code, where the 2**15 space
#: is 2.7 s to walk and 500 seeded draws are 0.05 s.
HAMMING_DRAWS = 500

#: Weight-four error patterns drawn at Golay, past its correction radius.
GOLAY_DRAWS = 100


def _nearest(words: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, int, int]:
    """The nearest codeword to ``received``, its distance, and the tie count (BSC ML)."""
    distance = (words != received[np.newaxis, :]).sum(axis=1)
    smallest = int(distance.min())
    return words[int(distance.argmin())], smallest, int((distance == smallest).sum())


@pytest.mark.critical
@pytest.mark.oracle
def test_syndrome_correction_is_the_nearest_codeword_enumeration_returns() -> None:
    # The rung below (#734): brute-force ML over `enumerate_codewords`. At
    # `(7, 4)`, all 128 received words: nearest unique at distance <= 1, and
    # `hamming_correct` and `ml_codeword` flip that bit on all 128; `(15, 11)`
    # on 500 draws (2**15 exhaustively, 2.7 s, also clean). Golay: all 2,048
    # patterns of weight <= 3 on two codewords decode to the sent word, no tie
    # in 4,096. At weight four all 100 draws sit at distance 3 from another
    # codeword: the covering radius, not a decoder defect.
    rng = np.random.default_rng(734)
    magnitude = np.log((1.0 - BSC_FLIP) / BSC_FLIP)

    code = hamming_code(3)
    words = enumerate_codewords(code).astype(np.int64)
    ties = 0
    for value in range(2**code.n_bits):
        received = np.array(
            [(value >> place) & 1 for place in range(code.n_bits)], dtype=np.int64
        )
        nearest, distance, tied = _nearest(words, received)
        ties += tied - 1
        assert distance <= 1
        np.testing.assert_array_equal(hamming_correct(code, 3, received), nearest)
        np.testing.assert_array_equal(
            exact_decoding(code, (1.0 - 2.0 * received) * magnitude).ml_codeword,
            nearest,
        )
    assert ties == 0

    code = hamming_code(4)
    words = enumerate_codewords(code).astype(np.int64)
    for _ in range(HAMMING_DRAWS):
        received = rng.integers(0, 2, size=code.n_bits)
        nearest, distance, tied = _nearest(words, received)
        assert distance <= 1
        assert tied == 1
        np.testing.assert_array_equal(hamming_correct(code, 4, received), nearest)

    code = golay_code()
    words = enumerate_codewords(code).astype(np.int64)
    patterns = np.zeros((1, code.n_bits), dtype=np.int64)
    for weight in (1, 2, 3):
        support = np.array(list(itertools.combinations(range(code.n_bits), weight)))
        block = np.zeros((support.shape[0], code.n_bits), dtype=np.int64)
        np.put_along_axis(block, support, 1, axis=1)
        patterns = np.vstack([patterns, block])
    assert patterns.shape == (1 + 23 + 253 + 1771, code.n_bits)

    for sent in (words[0], words[1]):
        for pattern in patterns:
            nearest, distance, tied = _nearest(words, sent ^ pattern)
            assert tied == 1
            np.testing.assert_array_equal(nearest, sent)

    wrong = 0
    for _ in range(GOLAY_DRAWS):
        pattern = np.zeros(code.n_bits, dtype=np.int64)
        pattern[rng.choice(code.n_bits, size=4, replace=False)] = 1
        nearest, distance, tied = _nearest(words, pattern)
        assert (distance, tied) == (3, 1)
        wrong += int(bool(nearest.any()))
    assert wrong == GOLAY_DRAWS


@pytest.mark.oracle
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


# --- end to end: a planted message through a channel and back -------------------

#: Draws per code at the three constructions a draw costs 0.10 to 0.19 ms. At
#: 2,000 the three-sigma binomial interval is 2.6e-2 on the widest rate below.
END2END_DRAWS = 2000

#: Draws at Golay, where `exact_decoding` enumerates 4,096 codewords per draw
#: and one draw is 7.5 ms: 400 of them are 3.0 s of the 3.6 s this costs, which
#: is inside the 10 s a per-pull-request test may take and is why the four
#: codes are read in one run rather than deferred to the release tier.
GOLAY_END2END_DRAWS = 400

#: One-sigma binomial intervals a realized rate may sit from its closed form
#: before it is a disagreement. Three; the largest departure measured over the
#: four codes is 1.2.
BINOMIAL_INTERVALS = 3.0

#: The crossover the repetition code is read at. Majority vote over five
#: copies fails on three, so the rate is measurable at 2,000 draws.
REPETITION_FLIP = 0.2

#: The erasure probability the single parity check is read at. It resolves one
#: erasure and no more, so its failure rate is `P(>= 2 erasures)` exactly.
PARITY_ERASURE = 0.1

#: The crossover the Hamming code is read at, and the one Golay is read at.
#: Golay corrects three errors, so a rate visible at 400 draws needs a channel
#: that delivers four: 0.02 leaves the closed form at 1.0e-3 and the draws
#: could not refute it.
HAMMING_FLIP = 0.05
GOLAY_FLIP = 0.08


def _message_positions(generator: np.ndarray) -> np.ndarray:
    """The columns `sim.ldpc.encode` leaves the message standing in.

    `generator_matrix` is systematic: the identity columns are asserted, not assumed.
    """
    found: list[int] = []
    for column in range(generator.shape[1]):
        bits = generator[:, column]
        if bits.sum() == 1 and int(np.argmax(bits)) == len(found):
            found.append(column)
        if len(found) == generator.shape[0]:
            break
    positions = np.array(found, dtype=np.int64)
    np.testing.assert_array_equal(
        generator[:, positions], np.eye(generator.shape[0], dtype=generator.dtype)
    )
    return positions


def _perfect_block_error_rate(n_bits: int, crossover: float, correctable: int) -> float:
    """`1 - P(at most t flips)`, which is a perfect code's block error rate.

    Tiling spheres make it the binomial tail exactly (:func:`is_perfect`).
    """
    inside = sum(
        math.comb(n_bits, flips)
        * crossover**flips
        * (1.0 - crossover) ** (n_bits - flips)
        for flips in range(correctable + 1)
    )
    return float(1.0 - inside)


def _interval(rate: float, draws: int) -> float:
    """`BINOMIAL_INTERVALS` standard errors of a rate over `draws` draws."""
    return BINOMIAL_INTERVALS * math.sqrt(rate * (1.0 - rate) / draws)


@pytest.mark.end2end
def test_the_four_codes_recover_the_planted_message_at_their_closed_form_rate() -> None:
    """Realized 0.0535, 0.1975, 0.0435 and 0.0925 against 0.0579, 0.1869,
    0.0444 and 0.1070; the widest departure is 1.2 binomial intervals of 3.0
    allowed, the two perfect codes lost no word the channel corrupted inside
    their radius, and the parity check decided none of its draws wrongly.

    End to end per code: seeded message, `sim.ldpc.encode`, `sim.ldpc`'s
    channel, the decoder, the message read off the systematic columns. Closed
    forms: repetition (5, 1) at `p = 0.2` (:func:`repetition_error_rate`);
    parity (8, 7) at `eps = 0.1` (`P(>= 2 erasures)`, one sweep of
    `likelihood.ldpc.decode`); Hamming (7, 4) at `p = 0.05` and Golay (23, 12)
    at `p = 0.08` (the binomial tail past the radius). Separately, over 2,000
    and 400 draws no word corrupted inside the radius failed.
    """
    rng = np.random.default_rng(729)

    code = repetition_code(5)
    positions = _message_positions(generator_matrix(code))
    channel = BinarySymmetricChannel(REPETITION_FLIP)
    lost = 0
    for _ in range(END2END_DRAWS):
        message = rng.integers(0, 2, size=positions.size).astype(np.uint8)
        ratios = channel.log_likelihood_ratios(encode(code, message), rng)
        decoded = exact_decoding(code, ratios).ml_codeword[positions]
        lost += int(not np.array_equal(decoded, message))
    expected = repetition_error_rate(5, crossover=REPETITION_FLIP)
    repetition_rate = lost / END2END_DRAWS
    assert abs(repetition_rate - expected) < _interval(expected, END2END_DRAWS)

    code = single_parity_check(8)
    positions = _message_positions(generator_matrix(code))
    erasure = BinaryErasureChannel(PARITY_ERASURE)
    undecided = decided_wrong = 0
    for _ in range(END2END_DRAWS):
        message = rng.integers(0, 2, size=positions.size).astype(np.uint8)
        ratios = erasure.log_likelihood_ratios(encode(code, message), rng)
        decoding = decode(code, ratios, max_iterations=1)
        if np.any(decoding.posterior_llr == 0.0):
            undecided += 1
        elif not np.array_equal(decoding.bits[positions], message):
            decided_wrong += 1
    survives = (1.0 - PARITY_ERASURE) ** 8 + 8 * PARITY_ERASURE * (
        1.0 - PARITY_ERASURE
    ) ** 7
    parity_rate = undecided / END2END_DRAWS
    assert decided_wrong == 0
    assert abs(parity_rate - (1.0 - survives)) < _interval(
        1.0 - survives, END2END_DRAWS
    )

    code = hamming_code(3)
    positions = _message_positions(generator_matrix(code))
    channel = BinarySymmetricChannel(HAMMING_FLIP)
    lost = inside_the_radius = 0
    for _ in range(END2END_DRAWS):
        message = rng.integers(0, 2, size=positions.size).astype(np.uint8)
        word = encode(code, message)
        ratios = channel.log_likelihood_ratios(word, rng)
        received = (ratios < 0.0).astype(np.int64)
        if not np.array_equal(hamming_correct(code, 3, received)[positions], message):
            lost += 1
            inside_the_radius += int((received != word).sum() <= 1)
    expected = _perfect_block_error_rate(7, HAMMING_FLIP, 1)
    hamming_rate = lost / END2END_DRAWS
    assert inside_the_radius == 0
    assert abs(hamming_rate - expected) < _interval(expected, END2END_DRAWS)

    code = golay_code()
    positions = _message_positions(generator_matrix(code))
    channel = BinarySymmetricChannel(GOLAY_FLIP)
    lost = inside_the_radius = 0
    for _ in range(GOLAY_END2END_DRAWS):
        message = rng.integers(0, 2, size=positions.size).astype(np.uint8)
        word = encode(code, message)
        ratios = channel.log_likelihood_ratios(word, rng)
        if not np.array_equal(
            exact_decoding(code, ratios).ml_codeword[positions], message
        ):
            lost += 1
            inside_the_radius += int(
                ((ratios < 0.0).astype(np.uint8) != word).sum() <= 3
            )
    expected = _perfect_block_error_rate(23, GOLAY_FLIP, 3)
    golay_rate = lost / GOLAY_END2END_DRAWS
    assert inside_the_radius == 0
    assert abs(golay_rate - expected) < _interval(expected, GOLAY_END2END_DRAWS)

    assert (repetition_rate, parity_rate, hamming_rate, golay_rate) == (
        0.0535,
        0.1975,
        0.0435,
        0.0925,
    )


#: Published sphere-packing counts, not summed by `math.comb` as `is_perfect`
#: does: MacKay 2003 §13.2, and Golay `1 + 23 + 253 + 1771 = 2048`.
PUBLISHED_BALLS: tuple[tuple[int, int, int, int], ...] = (
    # (n, 2^k, t, |ball of radius t|)
    (7, 16, 1, 8),
    (15, 2048, 1, 16),
    (23, 4096, 3, 2048),
    (5, 2, 2, 16),
)


@pytest.mark.oracle
def test_the_perfect_codes_are_the_ones_the_hamming_bound_holds_with_equality() -> None:
    # `2^k |ball| == 2^n` exactly for Hamming (7,4), (15,11), Golay (23,12)
    # and odd repetition codes; strictly under otherwise. Integers.
    for n_bits, n_messages, correctable, ball in PUBLISHED_BALLS:
        assert n_messages * ball == 2**n_bits
        assert is_perfect(n_bits, n_messages, correctable)

    # Strict inequality where the spheres leave the space uncovered: the
    # extended codes, which buy distance and stop tiling, and a dimension or
    # a length moved by one.
    for n_bits, n_messages, correctable in (
        (8, 16, 1),  # extended Hamming, d = 4
        (24, 4096, 3),  # extended Golay, d = 8
        (7, 8, 1),  # the right radius at the wrong dimension
        (6, 2, 2),  # the even repetition code
    ):
        volume = sum(math.comb(n_bits, radius) for radius in range(correctable + 1))
        assert n_messages * volume < 2**n_bits
        assert not is_perfect(n_bits, n_messages, correctable)

    # And the other side of the bound, which is not a looser code but no code
    # at all: 16 spheres of radius two in seven bits need 464 words of the
    # 128 there are, so the (7, 4) code corrects one error and not two.
    assert 16 * sum(math.comb(7, radius) for radius in range(3)) == 464
    assert not is_perfect(7, 16, 2)


@pytest.mark.oracle
def test_the_crc_remainder_is_the_published_long_division() -> None:
    # Two external divisions: the textbook CRC example (11010011101100 by
    # 1011, remainder 100) and 1101000 by 1011 by hand, remainder 001. Exact.
    generator = np.array([1, 0, 1, 1])
    published = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0])

    assert crc_remainder(published, generator).tolist() == [1, 0, 0]
    assert crc_remainder(np.array([1, 1, 0, 1]), generator).tolist() == [0, 0, 1]

    # A shift-and-xor over Python integers, sharing nothing: all 256 eight-bit
    # messages at CRC-8, x^8 + x^2 + x + 1.
    crc8 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1])
    divisor = 0b100000111
    for value in range(256):
        message = np.array([(value >> bit) & 1 for bit in range(7, -1, -1)])
        register = value << 8
        for shift in range(7, -1, -1):
            if register >> (shift + 8) & 1:
                register ^= divisor << shift
        expected = [(register >> bit) & 1 for bit in range(7, -1, -1)]

        assert crc_remainder(message, crc8).tolist() == expected
