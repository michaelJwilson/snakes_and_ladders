"""Successive cancellation, the list that fixes it, and what each one is held to.

Issue #593. Enumeration gives the ML codeword, so SC's gap is measured; a naive
recursive reference pins ``SC`` bitwise; ``SCL(1) == SC`` and
``SCL(2^k) == ML`` pin the list at both ends; monotonicity in the list size
is checked over shared seeds, as a property of this implementation.
"""

from __future__ import annotations

import itertools
from itertools import pairwise

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import (
    DecodingAlgorithm,
    decode,
    exact_decoding,
)
from snakes_and_ladders.likelihood.polar import (
    crc_checks,
    crc_encode,
    decode_sc,
    decode_scl,
    is_codeword,
    transmitted,
)
from snakes_and_ladders.sandbox.polar_reference import decode_sc_reference
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.ldpc import (
    BinaryErasureChannel,
    BinaryInputGaussianChannel,
    BinarySymmetricChannel,
    Channel,
    ParityCheck,
    all_zero_transmission,
)
from snakes_and_ladders.sim.polar import (
    PolarCode,
    PolarParams,
    gaussian_polar_code,
    parity_check,
    polar_transform,
)

from tests._fixtures import FIXTURES_DIR

#: The declared instance every test here shares: `N = 16`, `k = 8`, so
#: enumeration over 256 codewords is the maximum-likelihood oracle.
DRAWS = 200

#: The declared instance, read off the sandbox's own file as `test_polar.py` does.
FIXTURES = FIXTURES_DIR / "polar"


def _declared() -> PolarParams:
    """The declared `ci` instance, through the registry (#826)."""
    params = fixture("polar", "ci").params
    assert isinstance(params, PolarParams)
    return params


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
    # Why `N = 16` is `ci`: at sigma = 1.0 over 200 shared draws SC makes 98
    # block errors, SCL(4) 64, ML 62; the list recovers 34 of SC's 36.
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


#: The `N = 4` code whose information set is the transform's last three rows.
#: Its dual is the single all-ones row, so its Tanner graph carries one check
#: and no cycle, which is where sum-product is exact.
CYCLE_FREE = PolarCode(2, np.array([1, 2, 3]))


@pytest.mark.critical
@pytest.mark.oracle
def test_successive_cancellation_against_min_sum_on_the_same_parity_check() -> None:
    # The rung below (#734): `ldpc.decode` on `polar.parity_check`, paired.
    # Cycle-free (`CYCLE_FREE`, a single dual check): min-sum is blockwise ML,
    # sum-product bitwise, both right on 200 of 200; SC differs on 4 (2 both
    # wrong, 1 each way), so counts tie at 55. Declared `N = 16` (28 check
    # pairs overlap in 2+ positions): ML 62, min-sum 81, sum-product 89, SC
    # 98, so BP is nearer ML than SC, against the ticket's expectation.
    channel = BinaryInputGaussianChannel(1.0)
    check = parity_check(CYCLE_FREE)
    exact_min_sum = exact_sum_product = differing = 0
    counts = {"sc": 0, "ml": 0}
    for seed in range(DRAWS):
        ratios = all_zero_transmission(check, channel, np.random.default_rng(seed))
        exact = exact_decoding(check, ratios)
        min_sum = decode(check, ratios, algorithm=DecodingAlgorithm.MIN_SUM)
        sum_product = decode(check, ratios)
        word = decode_sc(CYCLE_FREE, ratios).codeword

        exact_min_sum += int(np.array_equal(min_sum.bits, exact.ml_codeword))
        exact_sum_product += int(
            np.array_equal(
                sum_product.bits, (exact.posterior_llr < 0.0).astype(np.uint8)
            )
        )
        differing += int(not np.array_equal(word, exact.ml_codeword))
        counts["sc"] += int(word.any())
        counts["ml"] += int(exact.ml_codeword.any())

    assert (exact_min_sum, exact_sum_product) == (DRAWS, DRAWS)
    assert differing == 4
    assert counts == {"sc": 55, "ml": 55}

    code, check = _instance()
    overlap = check.dense().astype(np.int64) @ check.dense().astype(np.int64).T
    np.fill_diagonal(overlap, 0)
    blocks = {"ml": 0, "min_sum": 0, "sum_product": 0, "sc": 0}
    for seed in range(DRAWS):
        ratios = all_zero_transmission(check, channel, np.random.default_rng(seed))
        blocks["ml"] += int(exact_decoding(check, ratios).ml_codeword.any())
        blocks["min_sum"] += int(
            decode(check, ratios, algorithm=DecodingAlgorithm.MIN_SUM).bits.any()
        )
        blocks["sum_product"] += int(decode(check, ratios).bits.any())
        blocks["sc"] += int(decode_sc(code, ratios).codeword.any())

    assert int((overlap >= 2).sum()) // 2 == 28
    assert blocks == {"ml": 62, "min_sum": 81, "sum_product": 89, "sc": 98}


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
    # A frozen bit forks nothing, so its penalty is easy to skip: ratios
    # disagreeing with a frozen zero must raise the metric.
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


#: CRC-3, ``x^3 + x + 1``, most significant bit first: three check bits on the
#: declared code's eight message bits leave a five-bit payload, 32 codewords.
CRC3 = np.array([1, 0, 1, 1])


def _crc_codebook(code: PolarCode) -> list[np.ndarray]:
    """Every codeword whose message passes CRC-3: the outer code, enumerated."""
    payload_bits = code.n_info - (CRC3.size - 1)
    return [
        code.encode(crc_encode(np.array(bits), CRC3))
        for bits in itertools.product((0, 1), repeat=payload_bits)
    ]


@pytest.mark.oracle
def test_crc_aided_exhaustive_list_is_maximum_likelihood_over_the_outer_code() -> None:
    # Every CRC-valid codeword scored by ``sum (1 - 2c) L`` (`sim.ldpc`'s sign);
    # the exhaustive list must return it and report the check passed (#826).
    code = _declared().code()
    codebook = _crc_codebook(code)
    assert len(codebook) == 32
    channel = BinaryInputGaussianChannel(sigma=1.0)
    rng = np.random.default_rng(826)
    for _ in range(12):
        payload = rng.integers(0, 2, size=code.n_info - 3)
        sent = code.encode(crc_encode(payload, CRC3))
        llr = channel.log_likelihood_ratios(sent, rng)
        decoded = decode_scl(code, llr, list_size=2**code.n_info, crc=CRC3)
        scores = [
            float(np.sum((1 - 2 * word.astype(float)) * llr)) for word in codebook
        ]
        best = codebook[int(np.argmax(scores))]
        assert decoded.crc_passed is True
        np.testing.assert_array_equal(decoded.codeword, best)
        assert crc_checks(decoded.message, CRC3)


@pytest.mark.analytic
def test_crc_aided_decoding_falls_back_to_the_best_metric_when_no_survivor_checks() -> (
    None
):
    # At list size one there is one survivor; if its check fails the decoder
    # returns it and says so, rather than inventing a second candidate.
    code = _declared().code()
    llr = np.where(code.encode(np.zeros(code.n_info, dtype=np.int64)) == 0, 4.0, -4.0)
    plain = decode_scl(code, llr, list_size=1)
    aided = decode_scl(code, llr, list_size=1, crc=CRC3)
    np.testing.assert_array_equal(plain.codeword, aided.codeword)
    assert aided.crc_passed is crc_checks(aided.message, CRC3)
    assert plain.crc_passed is None
    with pytest.raises(ValueError, match="cannot carry"):
        crc_checks(np.zeros(3, dtype=np.int64), CRC3)


@pytest.mark.end2end
def test_crc_aided_list_decoding_recovers_more_blocks_than_the_plain_list() -> None:
    # The step the ticket names, at a size the per-pull-request tier affords:
    # N = 64 at rate 1/2 with CRC-8 (x^8 + x^2 + x + 1), L = 8, 120 blocks on
    # shared seeds through a Gaussian channel. Judged against the payload that
    # was sent; the pull request carries the N = 256 sweep (#826).
    crc8 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1])
    code = gaussian_polar_code(6, 32, 0.8)
    channel = BinaryInputGaussianChannel(sigma=0.8)
    rng = np.random.default_rng(2026)
    plain_errors = aided_errors = 0
    for _ in range(120):
        payload = rng.integers(0, 2, size=code.n_info - 8)
        sent = code.encode(crc_encode(payload, crc8))
        llr = channel.log_likelihood_ratios(sent, rng)
        plain_errors += int(not np.array_equal(decode_scl(code, llr, 8).codeword, sent))
        aided_errors += int(
            not np.array_equal(decode_scl(code, llr, 8, crc=crc8).codeword, sent)
        )
    assert aided_errors < plain_errors, (aided_errors, plain_errors)
