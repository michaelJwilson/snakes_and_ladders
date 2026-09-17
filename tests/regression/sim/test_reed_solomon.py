"""Reed--Solomon, and the field under it, against what each one guarantees.

Issue #594. Three kinds of claim, kept apart:

* **The field is checked exhaustively.** ``GF(16)`` has sixteen elements, so
  associativity and distributivity are asserted over all 4,096 triples rather
  than sampled. A field that fails one of those fails silently everywhere.
* **The code's distance is an equality.** Reed--Solomon meets the Singleton
  bound exactly, and on ``RS(7, 3)`` all 512 codewords are enumerated to read
  the minimum distance off rather than take it from the construction.
* **The decoder is exact to ``t`` and measured past it.** Within the guarantee
  every draw is asserted to recover the sent word; beyond it the behaviour is
  *reported*, because a bounded-distance decoder is confidently wrong on some
  inputs and asserting otherwise would assert something false.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.sim.galois import PRIMITIVE, field
from snakes_and_ladders.sim.reed_solomon import (
    DecodingFailure,
    decode,
    encode,
    reed_solomon,
    syndromes,
)


@pytest.mark.mathematical
@pytest.mark.ldpc
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


@pytest.mark.mathematical
@pytest.mark.ldpc
@pytest.mark.parametrize("m", sorted(PRIMITIVE))
def test_the_generator_reaches_every_nonzero_element(m: int) -> None:
    # What "primitive" means, and the property the logarithm table depends on:
    # alpha's powers run through every nonzero element exactly once before
    # repeating. A non-primitive polynomial still builds a ring and would make
    # the discrete logarithm partial, which is a silent wrong answer.
    gf = field(m)

    powers = sorted(gf.alpha(exponent) for exponent in range(gf.nonzero))

    assert powers == list(range(1, gf.order))
    for value in range(1, gf.order):
        assert gf.multiply(value, gf.inverse(value)) == 1


@pytest.mark.edge_case
@pytest.mark.ldpc
def test_zero_has_no_logarithm_and_says_so() -> None:
    # The usual trick is `log(0) = -1` propagating quietly through a decode.
    with pytest.raises(ZeroDivisionError, match="no multiplicative inverse"):
        field(4).inverse(0)


@pytest.mark.oracle
@pytest.mark.ldpc
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


@pytest.mark.mathematical
@pytest.mark.ldpc
def test_encoding_is_systematic_and_lands_in_the_code() -> None:
    code = reed_solomon(3, 3)
    rng = np.random.default_rng(594)

    for _ in range(50):
        message = rng.integers(0, 8, size=3)
        word = encode(code, message)

        np.testing.assert_array_equal(word[: code.n_message], message)
        assert not syndromes(code, word).any()


@pytest.mark.oracle
@pytest.mark.ldpc
@pytest.mark.parametrize(
    ("m", "k", "errors"), [(3, 3, 0), (3, 3, 1), (3, 3, 2), (4, 9, 3)]
)
def test_the_decoder_is_exact_within_its_guarantee(m: int, k: int, errors: int) -> None:
    # Every draw, not most: within `t` symbol errors the decoder is a solver
    # and not an estimator, so anything short of exact recovery is a defect.
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


@pytest.mark.mathematical
@pytest.mark.ldpc
def test_past_the_guarantee_it_refuses_or_is_confidently_wrong() -> None:
    # Reported, not asserted to refuse: a bounded-distance decoder returns the
    # nearest codeword, and past `t` the nearest one can be the wrong one. The
    # test pins the *shape* of that -- most refused, some miscorrected, none
    # recovered by luck -- so a change that started recovering three errors on
    # a two-error code would fail here rather than look like an improvement.
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


@pytest.mark.edge_case
@pytest.mark.ldpc
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
