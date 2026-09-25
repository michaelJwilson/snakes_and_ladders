"""Reed--Solomon, and the field under it, against what each one guarantees.

Issue #594. ``GF(16)``'s associativity and distributivity over all 4,096
triples. Distance is an equality (Singleton), read off all 512 codewords of
``RS(7, 3)``. The decoder is exact to ``t`` and past it the behaviour is
reported: a bounded-distance decoder is confidently wrong on some inputs.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sal.likelihood.algebraic import DecodingFailure, decode, syndromes
from sal.sim.galois import (
    PRIMITIVE,
    bits_from_symbols,
    field,
    symbols_from_bits,
)
from sal.sim.ldpc import BinarySymmetricChannel
from sal.sim.reed_solomon import ReedSolomon, encode, reed_solomon

from tests._rows import every_row, every_value


@pytest.mark.analytic
def test_the_field_is_a_field() -> None:
    # Exhaustive over GF(16): 4,096 triples for each law. The axioms are what
    # every later step silently assumes, so they are checked rather than
    # trusted to the construction.
    gf = field(4)
    elements = range(gf.order)

    for a in elements:
        for b in elements:
            assert gf.multiply(a, b) == gf.multiply(b, a)
            for c in elements:
                assert gf.multiply(gf.multiply(a, b), c) == gf.multiply(
                    a, gf.multiply(b, c)
                )
                assert gf.multiply(a, b ^ c) == gf.multiply(a, b) ^ gf.multiply(a, c)


@pytest.mark.analytic
def test_the_generator_reaches_every_nonzero_element() -> None:
    # What "primitive" means, and the property the logarithm table depends on:
    # alpha's powers run through every nonzero element exactly once before
    # repeating. A non-primitive polynomial still builds a ring and would make
    # the discrete logarithm partial, which is a silent wrong answer.
    def check(m: int) -> None:
        gf = field(m)

        powers = sorted(gf.alpha(exponent) for exponent in range(gf.nonzero))

        assert powers == list(range(1, gf.order))
        for value in range(1, gf.order):
            assert gf.multiply(value, gf.inverse(value)) == 1

    every_value(sorted(PRIMITIVE), check)


@pytest.mark.smoke
def test_zero_has_no_logarithm_and_says_so() -> None:
    # The usual trick is `log(0) = -1` propagating quietly through a decode.
    with pytest.raises(ZeroDivisionError, match="no multiplicative inverse"):
        field(4).inverse(0)


@pytest.mark.oracle
def test_reed_solomon_meets_the_singleton_bound_with_equality() -> None:
    # Maximum distance separable, read off every codeword: `d = n - k + 1`
    # exactly, not `>=`. The 512 words of RS(7,3) are enumerable, which is the
    # only way to assert a minimum rather than exhibit one pair.
    code = reed_solomon(3, 3)

    words = np.array(
        [
            encode(code, np.asarray(message))
            for message in itertools.product(range(8), repeat=3)
        ]
    )
    weights = (words != 0).sum(axis=1)

    assert len(words) == 512
    assert int(weights[weights > 0].min()) == code.minimum_distance == 5


@pytest.mark.analytic
def test_encoding_is_systematic_and_lands_in_the_code() -> None:
    code = reed_solomon(3, 3)
    rng = np.random.default_rng(594)

    for _ in range(50):
        message = rng.integers(0, 8, size=3)
        word = encode(code, message)

        np.testing.assert_array_equal(word[: code.n_message], message)
        assert not syndromes(code, word).any()


@pytest.mark.oracle
def test_the_decoder_is_exact_within_its_guarantee() -> None:
    # Every draw, not most: within `t` symbol errors the decoder is a solver
    # and not an estimator, so anything short of exact recovery is a defect.
    def check(m: int, k: int, errors: int) -> None:
        code = reed_solomon(m, k)
        rng = np.random.default_rng(1000 + errors)
        assert errors <= code.correctable

        for _ in range(200):
            message = rng.integers(0, code.field.order, size=k)
            word = encode(code, message)
            received = word.copy()
            for position in rng.choice(code.n_symbols, size=errors, replace=False):
                received[position] ^= int(rng.integers(1, code.field.order))

            np.testing.assert_array_equal(decode(code, received), word)

    every_row([(3, 3, 0), (3, 3, 1), (3, 3, 2), (4, 9, 3)], check)


#: Error patterns past the guarantee, at three symbols on a code that
#: corrects two.
PAST_GUARANTEE_DRAWS = 300


def _codebook(code: ReedSolomon) -> np.ndarray:
    """Every codeword of ``code``, as the systematic encoder writes it."""
    return np.array(
        [
            encode(code, np.asarray(message))
            for message in itertools.product(
                range(code.field.order), repeat=code.n_message
            )
        ]
    )


def _nearest(words: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, int, int]:
    """The nearest codeword in symbol distance, that distance, and how many tie."""
    distance = (words != received[np.newaxis, :]).sum(axis=1)
    smallest = int(distance.min())
    return words[int(distance.argmin())], smallest, int((distance == smallest).sum())


@pytest.mark.critical
@pytest.mark.oracle
def test_the_algebraic_decode_is_the_nearest_codeword_enumeration_returns() -> None:
    # The rung below (#734): ML over RS(7,3)'s 512 codewords by symbol
    # distance. All 1,079 patterns of weight <= 2 on two codewords (2,158
    # decodes): the algebraic answer is the unique nearest and the sent word.
    # At weight three, 300 draws: 35 decode (unique nearest at distance 2,
    # never the sent word) and 265 refuse (2 to 7 tie at distance 3): ML's own split.
    code = reed_solomon(3, 3)
    words = _codebook(code)
    assert words.shape == (512, 7)
    rng = np.random.default_rng(734)

    for sent in (words[0], words[137]):
        for weight in (0, 1, 2):
            assert weight <= code.correctable
            for positions in itertools.combinations(range(code.n_symbols), weight):
                for magnitudes in itertools.product(
                    range(1, code.field.order), repeat=weight
                ):
                    received = sent.copy()
                    for position, magnitude in zip(positions, magnitudes, strict=True):
                        received[position] ^= magnitude
                    nearest, distance, tied = _nearest(words, received)

                    assert (distance, tied) == (weight, 1)
                    np.testing.assert_array_equal(nearest, sent)
                    np.testing.assert_array_equal(decode(code, received), nearest)

    sent = words[137]
    decoded = refused = 0
    for _ in range(PAST_GUARANTEE_DRAWS):
        received = sent.copy()
        for position in rng.choice(code.n_symbols, size=3, replace=False):
            received[position] ^= int(rng.integers(1, code.field.order))
        nearest, distance, tied = _nearest(words, received)
        try:
            answer = decode(code, received)
        except (DecodingFailure, RuntimeError):
            refused += 1
            assert distance == 3
            assert 2 <= tied <= 7
            continue
        decoded += 1
        assert (distance, tied) == (2, 1)
        np.testing.assert_array_equal(answer, nearest)
        assert not np.array_equal(answer, sent)

    assert (decoded, refused) == (35, 265)


@pytest.mark.analytic
def test_past_the_guarantee_it_refuses_or_is_confidently_wrong() -> None:
    # Past `t` the shape is pinned (mostly refused, some miscorrected, none
    # right by luck), so decoding three errors would fail, not pass.
    code = reed_solomon(3, 3)
    rng = np.random.default_rng(7)
    refused = wrong = recovered = 0

    for _ in range(300):
        message = rng.integers(0, 8, size=3)
        word = encode(code, message)
        received = word.copy()
        for position in rng.choice(7, size=3, replace=False):
            received[position] ^= int(rng.integers(1, 8))
        try:
            recovered += int(np.array_equal(decode(code, received), word))
        except (DecodingFailure, RuntimeError):
            refused += 1
        else:
            wrong += int(not np.array_equal(decode(code, received), word))

    assert recovered == 0
    assert refused + wrong == 300
    assert refused > wrong


@pytest.mark.smoke
def test_a_wasteful_or_impossible_shape_is_refused() -> None:
    # An odd parity count is legal and wastes a symbol: `t` rounds down while
    # `d` does not, so a reader comparing them finds them disagree. Refused
    # with the two lengths that would not.
    with pytest.raises(ValueError, match="is odd, so one is wasted"):
        reed_solomon(3, 4)
    with pytest.raises(ValueError, match="does not fit"):
        reed_solomon(3, 7)
    with pytest.raises(KeyError, match="no primitive polynomial"):
        field(9)


@pytest.mark.analytic
def test_the_bit_packing_round_trips_and_localises_a_flip() -> None:
    # What carries a symbol code onto a binary channel. Two claims: the
    # mapping is a bijection over every symbol of GF(8), and one flipped bit
    # corrupts exactly one symbol -- which is why the symbol error rate is
    # `1 - (1 - p)^m` and not `p`.
    m = 3
    symbols = np.arange(8)
    bits = bits_from_symbols(symbols, m)
    assert bits.size == symbols.size * m
    assert np.array_equal(symbols_from_bits(bits, m), symbols)

    for position in range(bits.size):
        flipped = bits.copy()
        flipped[position] ^= 1
        differing = symbols_from_bits(flipped, m) != symbols
        assert int(differing.sum()) == 1
        assert int(np.flatnonzero(differing)[0]) == position // m


@pytest.mark.infra
def test_the_packing_refuses_what_it_would_have_to_truncate() -> None:
    with pytest.raises(ValueError, match="symbols must lie in"):
        bits_from_symbols(np.array([8]), 3)
    with pytest.raises(ValueError, match="do not divide into symbols"):
        symbols_from_bits(np.zeros(7, dtype=np.int64), 3)


# --- end to end: a planted message over a binary channel ------------------------

#: Draws of the end-to-end run. At 4,000 the three-sigma binomial interval on
#: the rate below is 1.2e-2, and the run is 0.4 s.
END2END_DRAWS = 4000

#: The crossover of the binary channel the symbols are carried over. A symbol
#: of `GF(8)` is three bits, so this is a symbol error rate of
#: `1 - (1 - p) ** 3 = 0.1426` and `RS(7, 3)` sees more than its two
#: correctable symbols often enough to measure at 4,000 draws.
END2END_FLIP = 0.05

#: One-sigma binomial intervals the realized block error rate may sit from the
#: closed form. Three; realized 0.6.
BINOMIAL_INTERVALS = 3.0


@pytest.mark.critical
@pytest.mark.end2end
def test_the_planted_message_survives_every_channel_word_inside_the_guarantee() -> None:
    """Realized block error rate 0.0673 against the bounded-distance closed
    form 0.0650, 0.6 binomial intervals of 3.0 allowed; 3,731 draws inside the
    guarantee all returned the planted message and none of the 269 past it was
    right by chance.

    End to end: three `GF(8)` symbols, :func:`encode`, 21 bits, `sim.ldpc`'s
    BSC at `END2END_FLIP`, back through `symbols_from_bits`, :func:`decode`
    (Berlekamp--Massey, Chien, Forney); a symbol fails if any of its bits does.
    Of the 269 past `t`, 226 refused and 43 miscorrected, so the error rate
    equals the share past `t`, asserted; `eq:bounded-distance` is an equality.
    """
    code = reed_solomon(3, 3)
    channel = BinarySymmetricChannel(END2END_FLIP)
    rng = np.random.default_rng(729)
    inside = past = lucky = lost_inside = failures = 0

    for _ in range(END2END_DRAWS):
        message = rng.integers(0, code.field.order, size=code.n_message)
        word = encode(code, message)
        ratios = channel.log_likelihood_ratios(
            bits_from_symbols(word, code.m).astype(np.uint8), rng
        )
        received = symbols_from_bits((ratios < 0.0).astype(np.uint8), code.m)
        corrupted = int((received != word).sum())
        try:
            recovered = decode(code, received)[: code.n_message]
        except (DecodingFailure, RuntimeError):
            failures += 1
            past += 1
            continue
        if corrupted <= code.correctable:
            inside += 1
            lost_inside += int(not np.array_equal(recovered, message))
        else:
            past += 1
            lucky += int(np.array_equal(recovered, message))

    assert lost_inside == 0
    assert lucky == 0
    assert (inside, past, failures) == (3731, 269, 226)

    symbol_flip = 1.0 - (1.0 - END2END_FLIP) ** code.m
    expected = 1.0 - sum(
        math.comb(code.n_symbols, errors)
        * symbol_flip**errors
        * (1.0 - symbol_flip) ** (code.n_symbols - errors)
        for errors in range(code.correctable + 1)
    )
    realized = past / END2END_DRAWS
    assert realized == 0.06725
    assert abs(realized - expected) < BINOMIAL_INTERVALS * math.sqrt(
        expected * (1.0 - expected) / END2END_DRAWS
    )


#: ``alpha^i`` in ``GF(16)`` under ``x^4 + x + 1`` from the printed table (Lin
#: and Costello 2004, Table 2.8; MacKay 2003), least significant bit first.
GF16_POWERS: tuple[int, ...] = (1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9)


@pytest.mark.oracle
def test_the_field_reproduces_the_published_gf_sixteen_table() -> None:
    # `field(4)` against the printed table and log against its inverse, exact.
    # The `@cache` is cleared so this test runs the recursion itself.
    field.cache_clear()
    gf = field(4)

    assert [gf.alpha(exponent) for exponent in range(15)] == list(GF16_POWERS)
    assert [int(gf.exponential[exponent]) for exponent in range(15)] == list(
        GF16_POWERS
    )
    for exponent, value in enumerate(GF16_POWERS):
        assert int(gf.logarithm[value]) == exponent

    # Two products read off the published table rather than off the code:
    # alpha^5 * alpha^7 = alpha^12 = 15, and alpha^11 * alpha^9 = alpha^20 =
    # alpha^5 = 6, the wrap the doubled exponential table exists for.
    assert gf.multiply(GF16_POWERS[5], GF16_POWERS[7]) == GF16_POWERS[12]
    assert gf.multiply(GF16_POWERS[11], GF16_POWERS[9]) == GF16_POWERS[5]


@pytest.mark.oracle
def test_every_nonzero_element_satisfies_fermat_and_its_own_inverse() -> None:
    # Fermat in `GF(2^m)`: `a^(2^m - 1) = 1` over all 501 nonzero elements,
    # and `a * a^-1 = 1`. Exact.
    def check(m: int) -> None:
        field.cache_clear()
        gf = field(m)

        for value in range(1, gf.order):
            assert gf.power(value, gf.nonzero) == 1
            assert gf.multiply(value, gf.inverse(value)) == 1
            assert gf.divide(value, value) == 1
            # The discrete logarithm is the exponent the group law gives it.
            assert gf.alpha(int(gf.logarithm[value])) == value

    every_value(sorted(PRIMITIVE), check)
