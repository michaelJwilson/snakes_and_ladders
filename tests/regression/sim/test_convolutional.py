"""The shift register, its trellis and the turbo encoder, held to their definitions.

A convolutional code is a linear map, so the encoder is pinned by linearity
against its own generator matrix and by ``H c = 0`` through a parity-check
matrix built by a nullspace that shares no code with it; the trellis is
pinned by running the register a second way -- the polynomial division the
generators state -- rather than by re-reading the arrays; the tail is pinned
by the state it leaves; and the interleaver is pinned as a permutation, its
inverse composing to the identity.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.ldpc import enumerate_codewords
from snakes_and_ladders.sim.convolutional import (
    MAX_PARITY_CHECK_BITS,
    encode_stream,
    inverse_permutation,
    octal_taps,
    parity_check,
    random_interleaver,
    recursive_systematic_trellis,
    terminate,
    turbo_code,
    turbo_encode,
    turbo_generator_matrix,
)
from snakes_and_ladders.sim.fixtures import fixture

#: The default register: ``1 + D + D**2`` over ``1 + D**2``, memory 2.
FEEDBACK, FEEDFORWARD, MEMORY = 0o7, 0o5, 2

#: LTE's constituent encoder, the second register the textbook names.
LTE = (0o13, 0o15, 3)


# --- the generators and the trellis ---------------------------------------------


@pytest.mark.oracle
def test_an_octal_generator_reads_as_the_polynomial_the_textbook_states() -> None:
    """`7` is `1 + D + D**2` and `5` is `1 + D**2`, the published reading."""
    assert octal_taps(0o7, 2).tolist() == [1, 1, 1]
    assert octal_taps(0o5, 2).tolist() == [1, 0, 1]
    assert octal_taps(0o13, 3).tolist() == [1, 0, 1, 1]
    assert octal_taps(0o15, 3).tolist() == [1, 1, 0, 1]


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("polynomial", "memory"),
    [(0o17, 2), (0o2, 2), (0o6, 3)],
)
def test_a_generator_that_does_not_fit_the_register_is_refused(
    polynomial: int, memory: int
) -> None:
    # Too many bits, or a zero constant term: both build a different register
    # silently, which is the failure a reader of the fixture cannot see.
    with pytest.raises(ValueError, match="octal"):
        octal_taps(polynomial, memory)


@pytest.mark.edge_case
def test_a_feedback_generator_without_its_top_term_is_refused() -> None:
    # `6` at memory 2 is `1 + D`: a memory-1 register spelled as a memory-2
    # one. Its two edges into a state collide, so the trellis it would build
    # is not the regular one every recursion here assumes.
    with pytest.raises(ValueError, match="D\\*\\*2"):
        recursive_systematic_trellis(0o6, 0o5, 2)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("feedback", "feedforward", "memory"), [(FEEDBACK, FEEDFORWARD, MEMORY), LTE]
)
def test_the_trellis_runs_the_polynomial_division_the_generators_state(
    feedback: int, feedforward: int, memory: int
) -> None:
    """The register's output equals ``b(D) / a(D)`` applied to the input, term by term.

    The independent computation: the recursive systematic encoder's parity
    stream is the input filtered by the rational transfer function the two
    generators name, which is here run as an explicit long division over
    GF(2) on a list of bits, sharing nothing with the ``(state, input)``
    arrays under test.
    """
    trellis = recursive_systematic_trellis(feedback, feedforward, memory)
    a, b = octal_taps(feedback, memory), octal_taps(feedforward, memory)
    rng = np.random.default_rng(1)
    inputs = rng.integers(0, 2, 40).astype(np.uint8)

    parity, _ = encode_stream(trellis, inputs)

    # d[t] = u[t] + sum_{i >= 1} a_i d[t - i]; p[t] = sum_i b_i d[t - i].
    d: list[int] = []
    expected = []
    for u in inputs:
        value = int(u)
        for i in range(1, memory + 1):
            if a[i] and len(d) >= i:
                value ^= d[-i]
        d.append(value)
        emitted = int(b[0]) * value
        for i in range(1, memory + 1):
            if b[i] and len(d) > i:
                emitted ^= d[-1 - i]
        expected.append(emitted & 1)
    np.testing.assert_array_equal(parity, np.array(expected, dtype=np.uint8))


@pytest.mark.mathematical
@pytest.mark.parametrize(
    ("feedback", "feedforward", "memory"), [(FEEDBACK, FEEDFORWARD, MEMORY), LTE]
)
def test_the_two_edges_leaving_a_state_enter_different_states(
    feedback: int, feedforward: int, memory: int
) -> None:
    """Every state has two outgoing and two incoming edges, so the trellis is regular.

    The property the decoders rest on: an input bit is recoverable from a
    state pair, which is what lets `sim.factor_graph.from_trellis` carry no
    variable for it.
    """
    trellis = recursive_systematic_trellis(feedback, feedforward, memory)

    assert np.all(trellis.next_state[:, 0] != trellis.next_state[:, 1])
    for u in (0, 1):
        assert sorted(trellis.next_state[:, u].tolist()) == list(
            range(trellis.n_states)
        )


@pytest.mark.mathematical
@pytest.mark.parametrize(
    ("feedback", "feedforward", "memory"), [(FEEDBACK, FEEDFORWARD, MEMORY), LTE]
)
def test_the_tail_empties_the_register_from_every_state(
    feedback: int, feedforward: int, memory: int
) -> None:
    """`m` tail steps reach the zero state from each of the `2 ** m`, and no fewer."""
    trellis = recursive_systematic_trellis(feedback, feedforward, memory)
    rng = np.random.default_rng(2)

    for _ in range(10):
        message = rng.integers(0, 2, 20).astype(np.uint8)
        inputs = terminate(trellis, message)

        assert inputs.size == message.size + memory
        np.testing.assert_array_equal(inputs[: message.size], message)
        _, states = encode_stream(trellis, inputs)
        assert states[-1] == 0
        # The register is recursive, so the tail is not zeros: a decoder that
        # assumed it were would terminate a different trellis.
        assert states[message.size] == 0 or np.any(inputs[message.size :] == 1)


# --- the interleaver ------------------------------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("length", [12, 256, 1024])
def test_the_interleaver_is_a_permutation_and_its_inverse_undoes_it(
    length: int,
) -> None:
    """`inverse[order[i]] = i`, and both compositions are the identity."""
    order = random_interleaver(length, np.random.default_rng(233))
    inverse = inverse_permutation(order)
    identity = np.arange(length)

    np.testing.assert_array_equal(np.sort(order), identity)
    np.testing.assert_array_equal(order[inverse], identity)
    np.testing.assert_array_equal(inverse[order], identity)


@pytest.mark.structural
def test_two_interleavers_from_one_generator_differ_and_seeds_agree() -> None:
    # `sim/CLAUDE.md`: a generator, never a seed. The permutation is the
    # whole of the randomness in the construction, so drawing it inside the
    # call would make every member of an ensemble the same code.
    rng = np.random.default_rng(5)
    first, second = random_interleaver(64, rng), random_interleaver(64, rng)

    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(
        random_interleaver(64, np.random.default_rng(6)),
        random_interleaver(64, np.random.default_rng(6)),
    )


@pytest.mark.edge_case
def test_an_array_that_repeats_an_index_is_refused_as_a_permutation() -> None:
    with pytest.raises(ValueError, match="not a permutation"):
        inverse_permutation(np.array([0, 1, 1, 3]))


@pytest.mark.structural
def test_the_declared_instances_draw_the_interleaver_their_seed_names() -> None:
    # The fixture declares a seed and not a permutation, so two loads of one
    # file must give one code; a fixture that drew a fresh interleaver would
    # make every claim about "the instance" a claim about one run of it.
    params = fixture("turbo", "ci").params

    np.testing.assert_array_equal(params.code().interleaver, params.code().interleaver)
    assert params.code().message_length == params.message_length


# --- the encoder ----------------------------------------------------------------


@pytest.mark.mathematical
@pytest.mark.parametrize("message_length", [12, 40])
def test_encoding_is_gf2_linear_and_systematic(message_length: int) -> None:
    """`c(u + v) = c(u) + c(v)`, and the first `K` bits of a word are the message."""
    code = turbo_code(
        FEEDBACK, FEEDFORWARD, MEMORY, message_length, np.random.default_rng(3)
    )
    rng = np.random.default_rng(4)
    generator = turbo_generator_matrix(code)

    for _ in range(10):
        u = rng.integers(0, 2, message_length).astype(np.uint8)
        v = rng.integers(0, 2, message_length).astype(np.uint8)

        word = turbo_encode(code, u)
        np.testing.assert_array_equal(word[:message_length], u)
        np.testing.assert_array_equal(
            word,
            ((u.astype(np.int64) @ generator.astype(np.int64)) & 1).astype(np.uint8),
        )
        np.testing.assert_array_equal(
            turbo_encode(code, u ^ v), turbo_encode(code, u) ^ turbo_encode(code, v)
        )


@pytest.mark.mathematical
def test_the_block_length_and_rate_are_three_k_plus_four_m() -> None:
    """Both encoders terminated: `K + m` systematic, two parity streams, one tail."""
    code = turbo_code(FEEDBACK, FEEDFORWARD, MEMORY, 12, np.random.default_rng(3))
    where = code.slices()

    assert code.block_length == 3 * 12 + 4 * MEMORY
    assert turbo_encode(code, np.zeros(12, dtype=np.uint8)).size == code.block_length
    assert code.rate == pytest.approx(12 / (3 * 12 + 4 * MEMORY))
    covered = sorted(
        index for span in where.values() for index in range(span.start, span.stop)
    )
    assert covered == list(range(code.block_length))


@pytest.mark.oracle
def test_the_code_is_the_null_space_a_parity_check_matrix_defines() -> None:
    """Every encoded word satisfies `H c = 0`, and `H`'s null space is the code.

    The independent construction: `H` comes from a GF(2) nullspace of the
    generator, and the codewords are re-derived from `H` alone by issue
    #340's `enumerate_codewords`, which never sees the shift register. Two
    sets of `2 ** K` words, built from opposite ends, are compared.
    """
    code = turbo_code(FEEDBACK, FEEDFORWARD, MEMORY, 10, np.random.default_rng(7))
    check = parity_check(code)
    rng = np.random.default_rng(8)

    assert check.n_bits == code.block_length
    for _ in range(20):
        message = rng.integers(0, 2, code.message_length).astype(np.uint8)
        assert not np.any(check.syndrome(turbo_encode(code, message)))
    from_generator = {
        turbo_encode(code, message).tobytes()
        for message in (
            (
                np.arange(2**code.message_length)[:, None]
                >> np.arange(code.message_length)
            )
            & 1
        ).astype(np.uint8)
    }

    assert {word.tobytes() for word in enumerate_codewords(check)} == from_generator


@pytest.mark.edge_case
def test_the_dense_parity_check_is_refused_past_its_stated_length() -> None:
    # Cubic, and the trellis decoders are linear: past the ceiling the
    # caller wanted those, and a refusal says so rather than running for
    # minutes.
    code = turbo_code(
        FEEDBACK, FEEDFORWARD, MEMORY, MAX_PARITY_CHECK_BITS, np.random.default_rng(9)
    )

    with pytest.raises(ValueError, match=str(MAX_PARITY_CHECK_BITS)):
        parity_check(code)


@pytest.mark.edge_case
def test_a_message_of_the_wrong_length_or_outside_gf2_is_refused() -> None:
    code = turbo_code(FEEDBACK, FEEDFORWARD, MEMORY, 8, np.random.default_rng(10))

    with pytest.raises(ValueError, match="shape"):
        turbo_encode(code, np.zeros(7, dtype=np.uint8))
    with pytest.raises(ValueError, match="0/1"):
        turbo_encode(code, np.full(8, 2, dtype=np.uint8))
