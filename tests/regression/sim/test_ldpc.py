"""The parity-check ensemble, its channels and its encoder, held to their definitions.

A code is a null space, so the encoder is pinned by ``H c = 0`` and by
nothing that shares code with it; the offsets layout is pinned by the dense
product it stands in for; a channel is pinned by the closed form of its
log-likelihood ratio and by a Monte Carlo count against its parameter; and
the all-zero shortcut is pinned per realization, on a real codeword, as the
sign identity the textbook argues for (``sec:ldpc``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import enumerate_codewords
from snakes_and_ladders.sim.factor_graph import from_parity_check
from snakes_and_ladders.sim.ldpc import (
    LLR_CAP,
    MAX_ENCODABLE_BITS,
    BinaryErasureChannel,
    BinaryInputGaussianChannel,
    BinarySymmetricChannel,
    ParityCheck,
    all_zero_transmission,
    encode,
    gallager_code,
    generator_matrix,
)

COLUMN_WEIGHT, ROW_WEIGHT = 3, 6


def _code(n_bits: int, seed: int) -> ParityCheck:
    return gallager_code(n_bits, COLUMN_WEIGHT, ROW_WEIGHT, np.random.default_rng(seed))


# --- the ensemble --------------------------------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("n_bits", [12, 96, 996])
def test_a_gallager_draw_has_the_declared_degrees(n_bits: int) -> None:
    """Every column of a (3,6) draw has three ones and every row six, at every size."""
    code = _code(n_bits, seed=n_bits)

    assert code.n_checks == n_bits * COLUMN_WEIGHT // ROW_WEIGHT
    assert code.n_edges == n_bits * COLUMN_WEIGHT
    assert np.all(code.column_weights == COLUMN_WEIGHT)
    assert np.all(code.row_weights == ROW_WEIGHT)
    dense = code.dense()
    assert np.all(dense.sum(axis=0) == COLUMN_WEIGHT)
    assert np.all(dense.sum(axis=1) == ROW_WEIGHT)


@pytest.mark.mathematical
def test_the_bands_make_at_least_column_weight_minus_one_rows_dependent() -> None:
    """The rows of each band sum to the all-ones vector, so `k >= n - m + 2` for (3,6)."""
    code = _code(96, seed=2)

    k = int(generator_matrix(code).shape[0])

    assert k >= code.n_bits - code.n_checks + COLUMN_WEIGHT - 1
    bands = code.dense().reshape(COLUMN_WEIGHT, -1, code.n_bits)
    assert np.all(bands.sum(axis=1) == 1)


@pytest.mark.oracle
def test_the_offsets_layout_is_the_dense_matrix() -> None:
    """The syndrome through the offsets equals `H c mod 2` by the dense product."""
    code = _code(48, seed=3)
    rng = np.random.default_rng(4)
    dense = code.dense()

    assert np.array_equal(ParityCheck.from_dense(dense).dense(), dense)
    for _ in range(20):
        word = rng.integers(0, 2, code.n_bits)
        np.testing.assert_array_equal(
            code.syndrome(word), (dense.astype(np.int64) @ word) % 2
        )


@pytest.mark.structural
def test_two_draws_from_one_generator_differ_and_seeds_agree() -> None:
    # `sim/CLAUDE.md`: a generator, never a seed. The ensemble is drawn from
    # the generator, so two draws differ and two generators alike agree.
    rng = np.random.default_rng(5)
    first, second = _draw(rng), _draw(rng)

    assert first != second
    assert _draw(np.random.default_rng(6)) == _draw(np.random.default_rng(6))


def _draw(rng: np.random.Generator) -> tuple[int, ...]:
    code = gallager_code(24, COLUMN_WEIGHT, ROW_WEIGHT, rng)
    return tuple(int(v) for v in code.edge_check)


# --- the encoder ---------------------------------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("n_bits", [24, 96, MAX_ENCODABLE_BITS - 2])
def test_every_encoded_word_is_in_the_null_space(n_bits: int) -> None:
    """`H c = 0` on 20 random messages per size, and distinct messages encode distinctly."""
    code = _code(n_bits, seed=n_bits + 1)
    rng = np.random.default_rng(7)
    k = int(generator_matrix(code).shape[0])

    words = {encode(code, rng.integers(0, 2, size=k)).tobytes() for _ in range(20)}

    assert len(words) == 20
    for packed in words:
        assert not np.any(code.syndrome(np.frombuffer(packed, dtype=np.uint8)))


@pytest.mark.oracle
def test_the_enumerated_code_has_dimension_n_minus_rank() -> None:
    """A (3,6) code at `n = 12` has `2 ** k` codewords, every one with zero syndrome."""
    code = _code(12, seed=8)

    words = enumerate_codewords(code)

    k = int(generator_matrix(code).shape[0])
    assert words.shape == (2**k, 12)
    assert len({row.tobytes() for row in words}) == 2**k
    assert not np.any((code.dense().astype(np.int64) @ words.T) % 2)


# --- the channels --------------------------------------------------------------


@pytest.mark.mathematical
def test_the_binary_symmetric_ratio_is_the_closed_form() -> None:
    """Every ratio is `+-log((1 - p) / p)`, negative exactly at a flipped bit."""
    p = 0.11
    code = _code(96, seed=9)
    rng = np.random.default_rng(10)
    codeword = encode(
        code, rng.integers(0, 2, size=int(generator_matrix(code).shape[0]))
    )

    llr = BinarySymmetricChannel(p).log_likelihood_ratios(codeword, rng)

    np.testing.assert_allclose(np.abs(llr), math.log((1 - p) / p), rtol=1e-15)
    received = (llr < 0).astype(np.uint8)
    flipped = received ^ codeword
    assert 0 < flipped.sum() < codeword.size


@pytest.mark.simulated_truth
@pytest.mark.parametrize(("p", "seed"), [(0.05, 11), (0.2, 12), (0.45, 13)])
def test_the_flip_count_matches_the_flip_probability(p: float, seed: int) -> None:
    """Flips on 20,000 bits fall within four binomial standard deviations of `n p`."""
    code = _code(19_998, seed=1)

    llr = all_zero_transmission(
        code, BinarySymmetricChannel(p), np.random.default_rng(seed)
    )

    flips = int((llr < 0).sum())
    assert abs(flips - p * code.n_bits) < 4 * math.sqrt(code.n_bits * p * (1 - p))


@pytest.mark.mathematical
@pytest.mark.simulated_truth
def test_the_erasure_channel_is_zero_or_certain() -> None:
    """An erased bit is exactly zero; a delivered one is `+-LLR_CAP` with the bit's sign,
    and the erasure count sits inside four standard deviations of `n epsilon`."""
    epsilon = 0.3
    code = _code(19_998, seed=14)
    rng = np.random.default_rng(15)
    codeword = np.zeros(code.n_bits, dtype=np.uint8)
    codeword[::7] = 1
    # A word, not a codeword: the channel does not care, and the sign check needs ones.

    llr = BinaryErasureChannel(epsilon).log_likelihood_ratios(codeword, rng)

    erased = llr == 0
    assert np.all(llr[~erased] == (1 - 2 * codeword[~erased].astype(float)) * LLR_CAP)
    n = code.n_bits
    assert abs(erased.sum() - epsilon * n) < 4 * math.sqrt(n * epsilon * (1 - epsilon))


@pytest.mark.mathematical
@pytest.mark.simulated_truth
def test_the_gaussian_ratio_has_the_closed_form_moments() -> None:
    """`L = 2 y / sigma^2` on the zero word has mean `2 / sigma^2` and variance `4 / sigma^2`."""
    sigma = 0.8
    code = _code(19_998, seed=16)

    llr = all_zero_transmission(
        code, BinaryInputGaussianChannel(sigma), np.random.default_rng(17)
    )

    n = code.n_bits
    mean, variance = 2 / sigma**2, 4 / sigma**2
    assert abs(llr.mean() - mean) < 4 * math.sqrt(variance / n)
    assert abs(llr.var() - variance) < 4 * variance * math.sqrt(2 / n)


@pytest.mark.mathematical
@pytest.mark.parametrize(
    "channel", [BinarySymmetricChannel(0.08), BinaryErasureChannel(0.3)]
)
def test_a_codeword_through_the_channel_is_the_zero_word_up_to_sign(
    channel: BinarySymmetricChannel | BinaryErasureChannel,
) -> None:
    """On one noise realization the ratios of `c` are those of `0` negated where `c_i = 1`.

    The output symmetry `all_zero_transmission` rests on, per realization: a
    flip or an erasure is applied to the bit whatever its value, so the two
    transmissions differ only by the sign the codeword imposes.
    """
    code = _code(96, seed=18)
    codeword = encode(
        code,
        np.random.default_rng(19).integers(
            0, 2, size=int(generator_matrix(code).shape[0])
        ),
    )

    on_zero = all_zero_transmission(code, channel, np.random.default_rng(20))
    on_codeword = channel.log_likelihood_ratios(codeword, np.random.default_rng(20))

    np.testing.assert_array_equal(
        on_codeword, (1 - 2 * codeword.astype(float)) * on_zero
    )


# --- the factor graph ----------------------------------------------------------


@pytest.mark.oracle
def test_the_parity_check_factor_graph_scores_the_definition() -> None:
    """`log_density` is `-c . L` on every enumerated codeword and `-inf` off the code."""
    code = _code(12, seed=21)
    rng = np.random.default_rng(22)
    llr = all_zero_transmission(code, BinaryInputGaussianChannel(1.0), rng)
    graph = from_parity_check(code, llr)

    for word in enumerate_codewords(code):
        assignment = {f"x{i}": int(b) for i, b in enumerate(word)}
        assert math.isclose(
            graph.log_density(assignment), -float(word @ llr), abs_tol=1e-12
        )
    outside = enumerate_codewords(code)[1].copy()
    outside[0] ^= 1
    assert np.any(code.syndrome(outside))
    assert (
        graph.log_density({f"x{i}": int(b) for i, b in enumerate(outside)}) == -np.inf
    )


# --- refusals ------------------------------------------------------------------


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("n_bits", "column_weight", "row_weight"),
    [(12, 6, 3), (12, 1, 6), (12, 6, 6), (13, 3, 6), (0, 3, 6)],
)
def test_degrees_that_do_not_make_a_regular_code_are_refused(
    n_bits: int, column_weight: int, row_weight: int
) -> None:
    with pytest.raises(ValueError, match="row_weight"):
        gallager_code(n_bits, column_weight, row_weight, np.random.default_rng(0))


@pytest.mark.edge_case
def test_a_repeated_edge_an_empty_check_and_an_unchecked_bit_are_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        ParityCheck.from_edges(3, 2, np.array([0, 0, 1, 2]), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="every check"):
        ParityCheck.from_edges(3, 2, np.array([0, 1, 2]), np.array([0, 0, 0]))
    with pytest.raises(ValueError, match="every bit"):
        ParityCheck.from_edges(3, 2, np.array([0, 1]), np.array([0, 1]))
    with pytest.raises(ValueError, match="outside"):
        ParityCheck.from_edges(3, 2, np.array([0, 3]), np.array([0, 1]))


@pytest.mark.edge_case
def test_channel_parameters_outside_their_ranges_are_refused() -> None:
    with pytest.raises(ValueError, match="flip_probability"):
        BinarySymmetricChannel(0.5)
    with pytest.raises(ValueError, match="erasure_probability"):
        BinaryErasureChannel(1.0)
    with pytest.raises(ValueError, match="sigma"):
        BinaryInputGaussianChannel(0.0)
    with pytest.raises(ValueError, match="0/1"):
        BinarySymmetricChannel(0.1).log_likelihood_ratios(
            np.array([0, 2]), np.random.default_rng(0)
        )


@pytest.mark.edge_case
def test_the_encoder_refuses_a_wrong_length_and_a_code_past_its_size() -> None:
    code = _code(24, seed=23)
    with pytest.raises(ValueError, match="message bits"):
        encode(code, np.zeros(3, dtype=np.uint8))
    with pytest.raises(ValueError, match=str(MAX_ENCODABLE_BITS)):
        generator_matrix(_code(MAX_ENCODABLE_BITS + 4, seed=24))
    with pytest.raises(ValueError, match="shape"):
        from_parity_check(code, np.zeros(23))
