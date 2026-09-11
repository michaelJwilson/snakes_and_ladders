"""Decoding a bicycle code: the same decoder, held to the same oracles.

The construction is new and nothing downstream is (``sec:ldpc:bicycle``), so
what is checked here is that the code goes through the existing machinery and
where it lands. Enumeration over the 64 codewords of the 12-bit fixture gives
both exact decodings, and the optimality of each -- the bitwise MAP minimizes
bit errors, the maximum-likelihood codeword block errors -- bounds what
belief propagation can do on a graph full of short cycles. At 96 and 996 bits
no enumeration reaches, and the claim is a comparison: the bicycle code
against a Gallager draw of the same length and degrees, on shared seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import decode, exact_decoding
from snakes_and_ladders.likelihood.message_passing import MessageSchedule, sum_product
from snakes_and_ladders.sim.factor_graph import from_parity_check
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
    the better of each pair: 295 bit errors against the decoder's 348 and 30
    block errors against its 171 on the symmetric channel, 129 against 285 and
    29 against 110 on the Gaussian.

    Neither ordering is a coincidence to be re-measured. The bitwise MAP
    minimizes the expected bit error rate and the maximum-likelihood codeword
    the block error rate, so an enumeration cannot lose to a decoder on the
    quantity it optimizes; the gap is what a Bethe approximation costs on a
    graph whose girth is four.
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


@pytest.mark.oracle
def test_the_decoder_reaches_the_general_fixed_point_on_the_bicycle_graph() -> None:
    """The specialized decoder and the general flooding sum-product agree to 2e-12
    on the 12-bit fixture, which has cycles: the same Bethe fixed point from two
    implementations, and so the adapter carries this construction as it does the
    other."""
    params = fixture("bicycle", "ci").params
    code = params.code()
    llr = all_zero_transmission(
        code, params.symmetric_channel(), np.random.default_rng(1)
    )
    graph = from_parity_check(code, llr)
    assert not graph.is_tree()

    decoded = decode(code, llr, early_stop=False, max_iterations=3000, tolerance=1e-12)
    general = sum_product(
        graph, schedule=MessageSchedule.FLOODING, tolerance=1e-12, max_iterations=3000
    )

    marginals = np.array(
        [
            np.log(general.variable[f"x{i}"][0]) - np.log(general.variable[f"x{i}"][1])
            for i in range(code.n_bits)
        ]
    )
    np.testing.assert_allclose(decoded.posterior_llr, marginals, rtol=0, atol=2e-12)


# --- against a random construction of the same length and degrees ---------------


@pytest.mark.stress
@pytest.mark.simulated_truth
def test_the_bicycle_code_fails_more_blocks_than_a_gallager_draw_at_96_bits() -> None:
    """At 96 bits, column weight 3 and row weight 6 in both, the bicycle code
    loses at all six settings: 2 and 12 blocks of 20 against 1 and 5 on the
    symmetric channel, 20 and 20 against 3 and 12 on the erasure channel, 9 and
    20 against 1 and 15 on the Gaussian.

    The circulant is what costs it. Its dimension, 48 against the Gallager
    draw's 50, is the smaller of the two, so the deficit is not bought rate.
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
@pytest.mark.simulated_truth
def test_the_bicycle_waterfall_sits_above_a_gallager_draws_at_996_bits() -> None:
    """At 996 bits and matched degrees the bicycle code fails at least as many
    blocks of 20 as the Gallager draw at every one of twelve settings, and more
    at eleven: 3, 11, 19, 20 against 0, 0, 0, 7 on the symmetric channel; 0, 8,
    19, 20 against 0, 0, 3, 18 on the erasure channel; 1, 6, 17, 20 against 0, 0,
    0, 16 on the Gaussian.

    The erasure column carries the point. The (3,6) ensemble's threshold
    0.4294 referees the Gallager draw, which still resolves every block at
    0.30; the bicycle code fails 8 of 20 there. A threshold is a property of
    the ensemble a code is drawn from, and this code is not drawn from that
    one, which is why its fixture states no oracle.
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
