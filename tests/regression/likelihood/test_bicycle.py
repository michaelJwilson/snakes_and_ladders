"""Decoding a bicycle code: the same decoder, held to the same oracles.

``sec:ldpc:bicycle``. On the 12-bit fixture, enumeration over 64 codewords
gives the bitwise MAP and the ML codeword, which bound belief propagation. At
96 and 996 bits the claim is a comparison with a Gallager draw of the same
length and degrees on shared seeds. Agreement with the general flooding is a
row of `test_ldpc.py::test_flooding_reaches_the_general_fixed_point_on_a_loopy_code`.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import decode, exact_decoding
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.ldpc import (
    BinaryErasureChannel,
    BinaryInputGaussianChannel,
    BinarySymmetricChannel,
    Channel,
    ParityCheck,
    all_zero_transmission,
)

#: Seeds shared by the two constructions, so a difference between them is the
#: code and not the noise.
SEEDS = range(100, 120)

MAX_ITERATIONS = 200


def _failures(code: ParityCheck, channel: Channel) -> int:
    """Blocks of :data:`SEEDS` the decoder did not resolve into a codeword."""
    return sum(
        not decode(
            code,
            all_zero_transmission(code, channel, np.random.default_rng(seed)),
            max_iterations=MAX_ITERATIONS,
        ).decoded
        for seed in SEEDS
    )


# --- against the enumeration, at the length it reaches --------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("channel_of", ["symmetric_channel", "gaussian_channel"])
def test_the_exact_decodings_bound_belief_propagation(channel_of: str) -> None:
    """Over 200 transmissions of the 12-bit fixture the enumerated decodings are
    better: bits 295 against 348 and blocks 30 against 171 (symmetric channel),
    129 against 285 and 29 against 110 (Gaussian).
    """
    params = fixture("bicycle", "ci").params
    code, channel = params.code(), getattr(params, channel_of)()

    decoder_bits = decoder_blocks = exact_bits = exact_blocks = 0
    for seed in range(200, 400):
        llr = all_zero_transmission(code, channel, np.random.default_rng(seed))
        decoded = decode(code, llr, max_iterations=MAX_ITERATIONS)
        exact = exact_decoding(code, llr)
        decoder_bits += int(decoded.bits.sum())
        decoder_blocks += int(np.any(decoded.bits))
        exact_bits += int((exact.posterior_llr < 0.0).sum())
        exact_blocks += int(np.any(exact.ml_codeword))

    expected = {
        "symmetric_channel": (295, 348, 30, 171),
        "gaussian_channel": (129, 285, 29, 110),
    }[channel_of]
    assert exact_bits <= decoder_bits
    assert exact_blocks <= decoder_blocks
    assert (exact_bits, decoder_bits, exact_blocks, decoder_blocks) == expected


# --- against a random construction of the same length and degrees ---------------


@pytest.mark.stress
@pytest.mark.end2end
def test_the_bicycle_code_fails_more_blocks_than_a_gallager_draw_at_96_bits() -> None:
    """At 96 bits, weights (3, 6), dimension 48 against 50, the bicycle code loses
    all six settings, blocks of 20: 2, 12 vs 1, 5 (symmetric); 20, 20 vs 3, 12
    (erasure); 9, 20 vs 1, 15 (Gaussian).
    """
    bicycle = fixture("bicycle", "stress").params.code()
    gallager = fixture("ldpc", "stress").params.code()
    channels: list[Channel] = [
        BinarySymmetricChannel(0.02),
        BinarySymmetricChannel(0.06),
        BinaryErasureChannel(0.30),
        BinaryErasureChannel(0.40),
        BinaryInputGaussianChannel(0.7),
        BinaryInputGaussianChannel(0.9),
    ]

    measured = [(_failures(bicycle, c), _failures(gallager, c)) for c in channels]

    assert measured == [(2, 1), (12, 5), (20, 3), (20, 12), (9, 1), (20, 15)]
    assert all(bicycle_failures > draw for bicycle_failures, draw in measured)


@pytest.mark.release
@pytest.mark.end2end
def test_the_bicycle_waterfall_sits_above_a_gallager_draws_at_996_bits() -> None:
    """At 996 bits and matched degrees the bicycle code fails at least as many
    blocks of 20 as the Gallager draw at every one of twelve settings, and more
    at eleven: 3, 11, 19, 20 against 0, 0, 0, 7 on the symmetric channel; 0, 8,
    19, 20 against 0, 0, 3, 18 on the erasure channel; 1, 6, 17, 20 against 0, 0,
    0, 16 on the Gaussian. The (3,6) threshold 0.4294 referees the Gallager draw
    (all blocks at 0.30, where the bicycle code fails 8); it states no oracle here.
    """
    bicycle = fixture("bicycle", "release").params.code()
    gallager = fixture("ldpc", "release").params.code()
    channels: list[Channel] = (
        [BinarySymmetricChannel(p) for p in (0.02, 0.04, 0.06, 0.08)]
        + [BinaryErasureChannel(e) for e in (0.20, 0.30, 0.40, 0.45)]
        + [BinaryInputGaussianChannel(s) for s in (0.6, 0.7, 0.8, 0.9)]
    )

    measured = [(_failures(bicycle, c), _failures(gallager, c)) for c in channels]

    assert measured == [
        (3, 0),
        (11, 0),
        (19, 0),
        (20, 7),
        (0, 0),
        (8, 0),
        (19, 3),
        (20, 18),
        (1, 0),
        (6, 0),
        (17, 0),
        (20, 16),
    ]
    assert all(bicycle_failures >= draw for bicycle_failures, draw in measured)
