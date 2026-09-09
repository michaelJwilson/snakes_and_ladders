"""BCJR and Viterbi on one trellis, against enumeration and against sum-product.

Two referees, and they answer different questions. Enumerating all ``2 ** K``
messages says the posterior is *right*; the general message passing on
:func:`snakes_and_ladders.sim.factor_graph.from_trellis` says the
specialization here is the *same computation* as the one every other problem
class in this repository runs. Neither replaces the other, which is the
pairing issue #340 established for the parity-check decoder and this extends
rather than duplicates.

The tolerances are the measured agreements, stated as the module they pin:
BCJR against enumeration at ``1e-9`` on a posterior ratio measured at
``2.8e-14``, and against the tree schedule at ``1e-9`` on one measured at
``2.7e-15``. Both are absolute on a log-odds ratio, which is a per-bit
quantity and does not grow with the block length, unlike the summed
log-likelihood ``likelihood/CLAUDE.md`` states a relative bound for.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.likelihood.convolutional import (
    bcjr,
    exact_bitwise_posterior,
    viterbi,
)
from snakes_and_ladders.likelihood.message_passing import (
    MessageSchedule,
    max_product,
    sum_product,
)
from snakes_and_ladders.sim.convolutional import (
    IMPOSSIBLE_EDGE,
    Trellis,
    encode_stream,
    recursive_systematic_trellis,
    terminate,
)
from snakes_and_ladders.sim.factor_graph import Factor, FactorGraph, from_trellis
from snakes_and_ladders.sim.ldpc import BinaryInputGaussianChannel

#: Agreement with an exact answer on a log-odds ratio. Measured at 2.8e-14
#: against enumeration and 2.7e-15 against the tree schedule at the sizes
#: below; five orders of margin, so a real disagreement fails and rounding
#: does not.
EXACT_TOLERANCE = 1e-9

FEEDBACK, FEEDFORWARD, MEMORY = 0o7, 0o5, 2


def _received(
    trellis: Trellis, message: np.ndarray, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Terminate, encode and transmit; return the two streams' channel ratios."""
    inputs = terminate(trellis, message)
    parity, _ = encode_stream(trellis, inputs)
    ratios = BinaryInputGaussianChannel(sigma).log_likelihood_ratios(
        np.concatenate([inputs, parity]), rng
    )
    return ratios[: inputs.size], ratios[inputs.size :]


# --- against enumeration --------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("message_length", [6, 10])
def test_bcjr_posteriors_are_the_exact_bitwise_map(message_length: int) -> None:
    """The forward-backward ratio equals the sum over all `2 ** K` messages.

    The whole claim of the decoder: no approximation is involved on a
    terminated trellis, since it is a chain and message passing is exact on
    one, so equality may be asserted rather than a departure reported.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(233)

    for _ in range(5):
        message = rng.integers(0, 2, message_length).astype(np.uint8)
        systematic, parity = _received(trellis, message, 0.9, rng)

        decoding = bcjr(trellis, systematic, parity)
        exact = exact_bitwise_posterior(trellis, systematic, parity, message_length)

        np.testing.assert_allclose(
            decoding.posterior_llr[:message_length],
            exact.posterior_llr,
            atol=EXACT_TOLERANCE,
        )
        assert decoding.log_evidence == pytest.approx(
            exact.log_evidence, abs=EXACT_TOLERANCE
        )


@pytest.mark.oracle
def test_viterbi_returns_the_maximum_likelihood_message_with_a_pinned_margin() -> None:
    """The best path is the enumerated maximum, and the runner-up is not tied to it.

    ``sim/CLAUDE.md``: a degenerate fixture proves nothing. The margin is
    asserted positive per draw, so what is compared is a maximum and not a
    tie-break; the smallest margin the draws produced is reported by the
    assertion that fails if any is zero.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(11)

    for _ in range(10):
        message = rng.integers(0, 2, 10).astype(np.uint8)
        systematic, parity = _received(trellis, message, 1.0, rng)
        exact = exact_bitwise_posterior(trellis, systematic, parity, 10)

        assert exact.margin > 1e-6
        np.testing.assert_array_equal(
            viterbi(trellis, systematic, parity)[:10], exact.ml_message
        )


@pytest.mark.oracle
def test_the_two_decodings_of_one_trellis_are_different_answers() -> None:
    """A received word on which the bitwise and blockwise MAP disagree.

    ``likelihood/CLAUDE.md``: two decodings of one model are different
    answers, and a decoder computing one and reporting the other survives a
    suite with no case where they diverge. This is that case, found by
    searching seeds and pinned here as one.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(2)
    message = rng.integers(0, 2, 8).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.6, rng)
    exact = exact_bitwise_posterior(trellis, systematic, parity, 8)

    bitwise = (exact.posterior_llr < 0.0).astype(np.uint8)

    assert not np.array_equal(bitwise, exact.ml_message)
    np.testing.assert_array_equal(
        viterbi(trellis, systematic, parity)[:8], exact.ml_message
    )
    np.testing.assert_array_equal(
        (bcjr(trellis, systematic, parity).posterior_llr[:8] < 0.0).astype(np.uint8),
        bitwise,
    )


# --- against the general message passing ----------------------------------------


@pytest.mark.oracle
def test_the_trellis_graph_is_a_chain_and_sum_product_gives_bcjr() -> None:
    """The tree schedule's beliefs and `log Z` are BCJR's, edge for edge.

    `from_trellis` puts the observation on the transition, so the input
    bit's posterior is read off the pairwise belief rather than off a
    variable's; that read is the only arithmetic this test does that the
    decoder does not.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(19)
    message = rng.integers(0, 2, 8).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    graph = from_trellis(trellis, systematic, parity)

    assert graph.is_tree()
    marginals = sum_product(graph, schedule=MessageSchedule.TREE)
    decoding = bcjr(trellis, systematic, parity)

    assert marginals.exact
    assert marginals.log_partition == pytest.approx(
        decoding.log_evidence, abs=EXACT_TOLERANCE
    )
    for step in range(systematic.size):
        belief = marginals.factor[f"T{step}"]
        states = np.arange(trellis.n_states)
        zero = belief[states, trellis.next_state[:, 0]].sum()
        one = belief[states, trellis.next_state[:, 1]].sum()
        assert np.log(zero / one) == pytest.approx(
            decoding.posterior_llr[step], abs=EXACT_TOLERANCE
        )


@pytest.mark.oracle
def test_max_product_on_the_trellis_graph_gives_the_viterbi_path() -> None:
    """The general max-product's assignment is the specialized Viterbi's, per draw."""
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(21)

    for _ in range(5):
        message = rng.integers(0, 2, 8).astype(np.uint8)
        systematic, parity = _received(trellis, message, 1.0, rng)
        graph = from_trellis(trellis, systematic, parity)

        assignment, _ = max_product(graph, schedule=MessageSchedule.TREE)
        path = [assignment[f"s{t}"] for t in range(systematic.size + 1)]
        from_graph = np.array(
            [
                0 if trellis.next_state[path[t], 0] == path[t + 1] else 1
                for t in range(systematic.size)
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(viterbi(trellis, systematic, parity), from_graph)


# --- the a priori input and the boundary conditions ------------------------------


@pytest.mark.mathematical
def test_an_a_priori_ratio_enters_the_posterior_exactly_once() -> None:
    """`posterior = channel + a priori + extrinsic`, the decomposition eq:extrinsic.

    The identity a turbo iteration rests on: the extrinsic ratio is what is
    left after removing what the decoder was told, so a decoder that leaked
    the a priori term back into its extrinsic output would fail here rather
    than in a waterfall nobody can read.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(31)
    message = rng.integers(0, 2, 10).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    apriori = rng.normal(size=systematic.size)

    decoding = bcjr(trellis, systematic, parity, apriori)

    np.testing.assert_allclose(
        decoding.posterior_llr,
        systematic + apriori + decoding.extrinsic_llr,
        atol=1e-12,
    )


@pytest.mark.oracle
def test_an_a_priori_ratio_is_the_same_evidence_as_a_second_channel() -> None:
    """A priori `L_a` equals enumerating with `L_a` added to the systematic stream.

    An a priori log-odds on the input bit is, by definition, evidence about
    that bit from elsewhere; adding it to the systematic channel ratio is
    the same posterior. Enumeration referees the equality, so the a priori
    path is pinned by the same oracle as the rest and not by construction.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(37)
    message = rng.integers(0, 2, 10).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    apriori = rng.normal(size=systematic.size)

    decoding = bcjr(trellis, systematic, parity, apriori)
    exact = exact_bitwise_posterior(trellis, systematic + apriori, parity, 10)

    np.testing.assert_allclose(
        decoding.posterior_llr[:10], exact.posterior_llr, atol=EXACT_TOLERANCE
    )


@pytest.mark.mathematical
def test_an_unterminated_decoder_computes_a_different_posterior() -> None:
    """Telling the decoder the register was not emptied changes the answer.

    The boundary condition is not decoration: an unterminated backward
    recursion starts uniform over states, so it admits paths the terminated
    code does not contain. A decoder that ignored the flag would pass every
    other test here, since both branches are otherwise identical.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(41)
    message = rng.integers(0, 2, 10).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)

    terminated = bcjr(trellis, systematic, parity, terminated=True)
    free = bcjr(trellis, systematic, parity, terminated=False)

    assert free.log_evidence > terminated.log_evidence
    assert np.abs(free.posterior_llr - terminated.posterior_llr).max() > 1e-3


@pytest.mark.edge_case
def test_streams_of_disagreeing_length_are_refused() -> None:
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)

    with pytest.raises(ValueError, match="one-dimensional"):
        bcjr(trellis, np.zeros(6), np.zeros(5))
    with pytest.raises(ValueError, match="one-dimensional"):
        from_trellis(trellis, np.zeros(6), np.zeros(5))


@pytest.mark.edge_case
def test_an_impossible_edge_is_a_finite_floor_and_not_minus_infinity() -> None:
    """A hard zero would make the general sum-product return `nan`, not a number.

    The failure the floor prevents, asserted rather than described. At the
    first steps of a terminated trellis most states are unreachable, so their
    incoming edges are all impossible; the general implementation shifts a row
    by its maximum before exponentiating, and a row that is entirely `-inf`
    becomes `-inf - (-inf)`. Rebuilding the same graph with `-inf` in place of
    `IMPOSSIBLE_EDGE` is what this asserts against, so the constant is pinned
    by the thing it exists for.
    """
    trellis = recursive_systematic_trellis(FEEDBACK, FEEDFORWARD, MEMORY)
    rng = np.random.default_rng(43)
    message = rng.integers(0, 2, 8).astype(np.uint8)
    systematic, parity = _received(trellis, message, 1.0, rng)
    graph = from_trellis(trellis, systematic, parity)

    assert np.isfinite(sum_product(graph, schedule=MessageSchedule.TREE).log_partition)

    hard = FactorGraph(
        graph.variables,
        [
            Factor(
                factor.name,
                factor.variables,
                np.where(factor.log_table <= IMPOSSIBLE_EDGE, -np.inf, factor.log_table),
            )
            for factor in graph.factors
        ],
    )

    assert not np.isfinite(
        sum_product(hard, schedule=MessageSchedule.TREE).log_partition
    )
