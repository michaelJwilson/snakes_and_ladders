"""Successive cancellation, the list that fixes it, and what each one is held to.

Issue #593. Four referees, in order of strength, and each answers a different
question:

1. **enumeration** gives the maximum-likelihood codeword at the declared
   instance, so the gap successive cancellation leaves is *measured* rather
   than cited --- and that gap is the reason list decoding exists;
2. **a naive recursive reference** shares the arithmetic and none of the
   layout, so ``SC`` is pinned to it **bitwise**;
3. **``SCL(1) == SC``** and **``SCL(2^k) == ML``** pin the fork-and-prune
   machinery at both ends: at one path there is nothing to prune, and at
   ``2^k`` nothing is pruned, so the search is exhaustive;
4. **monotonicity in the list size** is what a list is for, checked over
   shared seeds rather than asserted from theory --- a larger list explores a
   superset only because the metric orders the same way, which is a property
   of this implementation.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import exact_decoding
from snakes_and_ladders.sandbox.polar import (
    PolarCode,
    PolarParams,
    load_polar_params,
    parity_check,
    polar_transform,
)
from snakes_and_ladders.sandbox.polar_decoding import (
    decode_sc,
    decode_scl,
    is_codeword,
    transmitted,
)
from snakes_and_ladders.sandbox.polar_reference import decode_sc_reference
from snakes_and_ladders.sim.ldpc import (
    BinaryErasureChannel,
    BinaryInputGaussianChannel,
    BinarySymmetricChannel,
    Channel,
    ParityCheck,
    all_zero_transmission,
)

#: The declared instance every test here shares: `N = 16`, `k = 8`, so
#: enumeration over 256 codewords is the maximum-likelihood oracle.
DRAWS = 200

#: The declared instance, read off the sandbox's own file as `test_polar.py` does.
FIXTURES = Path(__file__).parent / "fixtures" / "polar"


def _declared() -> PolarParams:
    """The `ci` instance the sandbox declares."""
    return load_polar_params(FIXTURES / "ci.yaml")


def _instance() -> tuple[PolarCode, ParityCheck]:
    """The `ci` code and its parity check, which is what the oracle reads."""
    code = _declared().code()
    return code, parity_check(code)


def _channels() -> tuple[Channel, ...]:
    """The three channels the fixture declares, at its declared parameters."""
    declared = _declared()
    return (
        BinaryInputGaussianChannel(declared.noise_scale),
        BinarySymmetricChannel(declared.flip_probability),
        BinaryErasureChannel(declared.erasure_probability),
    )


@pytest.mark.oracle
def test_successive_cancellation_is_the_reference_implementation_bitwise() -> None:
    # Same arithmetic, same order, different layout: `polar.py` carries a path
    # axis and prunes at the leaves, the reference recurses one candidate with
    # a Python call per node. A tolerance here would hide a gather that
    # reorders the traversal, so the two are pinned bitwise over every channel.
    code, check = _instance()
    for channel in _channels():
        for seed in range(DRAWS):
            ratios = all_zero_transmission(check, channel, np.random.default_rng(seed))
            assert np.array_equal(
                decode_sc(code, ratios).source, decode_sc_reference(code, ratios)
            )


@pytest.mark.smoke
def test_the_list_at_one_path_is_plain_successive_cancellation() -> None:
    # True by construction -- `decode_sc` calls `decode_scl` -- and asserted
    # because the construction is what a later optimization would break.
    code, check = _instance()
    for seed in range(DRAWS):
        ratios = all_zero_transmission(
            check, BinaryInputGaussianChannel(1.0), np.random.default_rng(seed)
        )
        plain = decode_sc(code, ratios)
        listed = decode_scl(code, ratios, list_size=1)
        assert np.array_equal(plain.source, listed.source)
        assert plain.metric == listed.metric


@pytest.mark.oracle
def test_an_exhaustive_list_is_maximum_likelihood() -> None:
    # At `2^k` paths nothing is pruned, so the list *is* the codebook and the
    # answer must be the enumerated maximum-likelihood one. This is what pins
    # the fork, the metric and the survivor bookkeeping against an oracle
    # rather than against a second run of themselves.
    code, check = _instance()
    exhaustive = 2**code.n_info
    for seed in range(20):
        ratios = all_zero_transmission(
            check, BinaryInputGaussianChannel(1.2), np.random.default_rng(seed)
        )
        assert np.array_equal(
            decode_scl(code, ratios, exhaustive).codeword,
            exact_decoding(check, ratios).ml_codeword,
        )


@pytest.mark.oracle
def test_the_gap_to_maximum_likelihood_is_what_the_list_closes() -> None:
    # The measurement the ticket exists for, and the reason `N = 16` is the
    # `ci` size rather than `8`: successive cancellation commits to each bit
    # in order and cannot revisit, so it is not maximum likelihood, and the
    # list closes most of what it gives up.
    #
    # At sigma = 1.0 over 200 shared draws: SC 98 block errors, SCL(4) 64,
    # enumeration ML 62 --- the list recovers 34 of the 36 blocks SC loses,
    # and the 62 are the channel's, beyond any decoder's reach.
    code, check = _instance()
    channel = BinaryInputGaussianChannel(1.0)

    failures = {"sc": 0, "scl4": 0, "ml": 0}
    for seed in range(DRAWS):
        ratios = all_zero_transmission(check, channel, np.random.default_rng(seed))
        failures["sc"] += int(decode_sc(code, ratios).codeword.any())
        failures["scl4"] += int(decode_scl(code, ratios, 4).codeword.any())
        failures["ml"] += int(exact_decoding(check, ratios).ml_codeword.any())

    assert failures["ml"] <= failures["scl4"] < failures["sc"]
    assert failures["sc"] - failures["ml"] >= 20
    assert failures["scl4"] - failures["ml"] <= 5


@pytest.mark.end2end
def test_a_longer_list_never_decodes_worse_on_shared_seeds() -> None:
    # What a list is for, measured rather than assumed: the metric must order
    # candidates so that a larger list explores a superset. Block errors are
    # counted on the same draws for every size, so the comparison is paired.
    code, check = _instance()
    channel = BinaryInputGaussianChannel(1.0)
    sizes = (1, 2, 4, 8)

    failures = []
    for size in sizes:
        failed = 0
        for seed in range(DRAWS):
            ratios = all_zero_transmission(check, channel, np.random.default_rng(seed))
            failed += int(decode_scl(code, ratios, size).codeword.any())
        failures.append(failed)

    assert all(later <= earlier for earlier, later in pairwise(failures)), failures
    assert failures[-1] < failures[0]


@pytest.mark.analytic
def test_every_decoded_word_lies_in_the_code() -> None:
    # A decoder that returned something outside the code would be wrong in a
    # way no error rate reveals: the frozen source positions must be zero,
    # which is the definition rather than a consequence.
    code, check = _instance()
    for seed in range(50):
        ratios = all_zero_transmission(
            check, BinarySymmetricChannel(0.1), np.random.default_rng(seed)
        )
        for size in (1, 4):
            decoding = decode_scl(code, ratios, size)
            assert is_codeword(code, decoding.codeword)
            assert np.array_equal(
                decoding.codeword, transmitted(code, decoding.message)
            )


@pytest.mark.analytic
def test_a_noiseless_channel_returns_the_message_that_was_sent() -> None:
    # The end-to-end identity, at every list size: encode, hand the decoder
    # certain ratios, and the message comes back. It is the one case where the
    # answer is known without an oracle.
    code, _ = _instance()
    rng = np.random.default_rng(593)
    for _ in range(20):
        message = rng.integers(0, 2, size=code.n_info)
        word = code.encode(message)
        certain = (1.0 - 2.0 * word.astype(float)) * 30.0
        for size in (1, 2, 8):
            assert np.array_equal(decode_scl(code, certain, size).message, message)


@pytest.mark.smoke
def test_the_decoder_refuses_what_it_cannot_answer_for() -> None:
    code, _ = _instance()
    with pytest.raises(ValueError, match="ratios"):
        decode_scl(code, np.zeros(code.n_bits - 1))
    with pytest.raises(ValueError, match="at least one path"):
        decode_scl(code, np.zeros(code.n_bits), list_size=0)


@pytest.mark.oracle
def test_the_frozen_positions_carry_their_evidence_into_the_metric() -> None:
    # The usual bug in a list decoder: a frozen bit forks nothing, so its
    # penalty is easy to skip -- and then an unlikely prefix costs the same as
    # a likely one and the list prunes the wrong paths. Certain ratios that
    # *disagree* with a frozen zero must raise the metric above what an
    # agreeing channel gives.
    code, _ = _instance()
    frozen_row = polar_transform(code.n_stages)[int(code.frozen[0])]
    outside = (1.0 - 2.0 * frozen_row.astype(float)) * 30.0
    inside = np.full(code.n_bits, 30.0)

    # A word that puts a one on a frozen source position is *not* in the code,
    # so a decoder held to the frozen set must pay for every certain ratio
    # that disagrees with it. The all-ones received word would not do: it is
    # the transform's last row and lies in the code.
    assert decode_scl(code, inside, 1).metric == pytest.approx(0.0, abs=1e-9)
    assert decode_scl(code, outside, 1).metric > 25.0
