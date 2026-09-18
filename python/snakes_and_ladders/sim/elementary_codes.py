"""The codes whose maximum-likelihood decoder is closed form, and CRC.

Issue #594. Every code in this repository is refereed by enumerating its
``2^k`` codewords, which is exact and stops where the exponent does --- at the
`ci` fixture. The codes here are the ones whose ML decoder has a **closed
form at every size**, so each is a referee that does not run out, and each
pins something the tree already runs:

* **single parity check** --- its exact bitwise posterior *is* the tanh
  product rule, so it pins `likelihood.ldpc`'s check-node update, the update
  every LDPC and turbo decode in this repository performs, against brute
  force rather than against another implementation of itself;
* **repetition** --- the floor every other code is measured against, with an
  analytic error probability on all three channels, which no fixture in the
  tree has;
* **Hamming** and **Golay** --- *perfect* codes, where the sphere-packing
  count holds with **equality** and not as a bound, which is the sharpest
  structural invariant available and fails loudly if a construction drifts.

**Perfection is an equality, and that is why these codes are here.** A code
of minimum distance ``2t + 1`` can correct ``t`` errors, and the spheres of
radius ``t`` around its codewords are disjoint; perfect means they also
*cover* the space, leaving nothing over:
``2^k * sum_{i<=t} C(n, i) == 2^n`` exactly. A test asserts that equation
rather than a tolerance, so a wrong column order or a dropped row cannot pass.

**CRC** is here rather than in a polar module because #593's CRC-aided list
decoding needs it and it is polynomial division over GF(2), which belongs with
the elementary codes.

Reed--Solomon and the ``GF(2^m)`` arithmetic it needs are the same ticket's
second part and live in :mod:`snakes_and_ladders.sim.reed_solomon` and
:mod:`snakes_and_ladders.sim.galois`. What is **not** here is BCH, which that
machinery makes cheap afterwards and which nothing yet asks for. Nor is a codeword
enumerator: :func:`snakes_and_ladders.likelihood.ldpc.enumerate_codewords`
spans the generator matrix and checks each word against ``H c = 0``, which is
what these constructions are refereed with rather than a second copy of it.

See ``sec:ldpc:algebraic`` of ``docs/tex/textbook.tex`` (Blahut 2003;
MacKay 2003), and ``eq:sphere-packing`` for the perfection equality.
"""

from __future__ import annotations

import math

import numpy as np

from snakes_and_ladders.sim.ldpc import ParityCheck, null_space


def repetition_code(n_bits: int) -> ParityCheck:
    """The ``(n, 1)`` code: every bit equals the first.

    Parameters
    ----------
    n_bits : int
        Block length, at least 2.

    Returns
    -------
    ParityCheck
        ``n - 1`` checks, each tying bit ``0`` to one other bit.

    Raises
    ------
    ValueError
        If ``n_bits`` is below 2, where there is no check to write.
    """
    if n_bits < 2:
        msg = f"a repetition code needs at least two bits, got {n_bits}"
        raise ValueError(msg)
    dense = np.zeros((n_bits - 1, n_bits), dtype=np.int64)
    for row in range(n_bits - 1):
        dense[row, 0] = 1
        dense[row, row + 1] = 1
    return ParityCheck.from_dense(dense)


def single_parity_check(n_bits: int) -> ParityCheck:
    """The ``(n, n-1)`` code: one all-ones row, so every word has even weight."""
    if n_bits < 2:
        msg = f"a single parity check needs at least two bits, got {n_bits}"
        raise ValueError(msg)
    return ParityCheck.from_dense(np.ones((1, n_bits), dtype=np.int64))


def hamming_code(m: int, *, extended: bool = False) -> ParityCheck:
    """The perfect ``(2^m - 1, 2^m - 1 - m)`` code, or its extended form.

    ``H``'s columns are the ``2^m - 1`` nonzero ``m``-bit vectors, in
    increasing order, so a nonzero syndrome **is** the binary index of the
    flipped bit and correction is a lookup rather than a search. Extending
    appends an overall parity bit and a row of ones, which lifts the minimum
    distance from 3 to 4 --- the extended ``(8, 4)`` is also the first-order
    Reed--Muller code ``RM(1, 3)`` and the ``N = 8`` polar code, one object
    under three constructions (#593).

    Parameters
    ----------
    m : int
        Number of check bits, at least 2.
    extended : bool
        Append the overall parity bit.

    Returns
    -------
    ParityCheck
        The code's parity-check matrix.
    """
    if m < 2:
        msg = f"a Hamming code needs at least two check bits, got {m}"
        raise ValueError(msg)
    n_bits = 2**m - 1
    columns = np.array(
        [[(value >> bit) & 1 for bit in range(m)] for value in range(1, n_bits + 1)],
        dtype=np.int64,
    )
    dense = columns.T
    if extended:
        dense = np.hstack([dense, np.zeros((m, 1), dtype=np.int64)])
        dense = np.vstack([dense, np.ones((1, n_bits + 1), dtype=np.int64)])
    return ParityCheck.from_dense(dense)


#: The 11 generators of the binary Golay code's ``(23, 12)`` parity check, as
#: the cyclic shifts of ``x^11 + x^9 + x^7 + x^6 + x^5 + x + 1``. Written as
#: the polynomial rather than as a 23x12 table because the table is what a
#: reader cannot check and the polynomial is what the literature states.
_GOLAY_POLYNOMIAL = (1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1)


def golay_code(*, extended: bool = False) -> ParityCheck:
    """The perfect ``(23, 12)`` binary Golay code, or its extended ``(24, 12)``.

    The largest code in this repository whose exact maximum-likelihood decoder
    still enumerates at its **real** length rather than at a shrunken fixture:
    ``2^12 = 4096`` codewords. It is perfect at ``d = 7``, correcting any three
    errors with ``2^12 * (1 + 23 + 253 + 1771) == 2^23`` exactly; extending it
    gives the ``(24, 12)`` code at ``d = 8``.

    **The shifts of the generator polynomial are the generator matrix, not the
    parity check**, so the check is taken as its null space --- through
    :func:`snakes_and_ladders.sim.ldpc.null_space`, the one home of the GF(2)
    elimination here. Reading the shifts as ``H`` instead builds a ``(23, 11)``
    code at ``d = 8`` that is not Golay and is not perfect, which is what the
    perfection equality below catches.

    Returns
    -------
    ParityCheck
        The parity check of the cyclic code generated by
        :data:`_GOLAY_POLYNOMIAL`.
    """
    n_bits = 23
    degree = len(_GOLAY_POLYNOMIAL) - 1
    rows = n_bits - degree
    generator = np.zeros((rows, n_bits), dtype=np.int64)
    for row in range(rows):
        for position, coefficient in enumerate(_GOLAY_POLYNOMIAL):
            generator[row, row + position] = coefficient
    if extended:
        parity = generator.sum(axis=1) % 2
        generator = np.hstack([generator, parity[:, np.newaxis]])
    return ParityCheck.from_dense(null_space(generator))


def is_perfect(n_bits: int, n_messages: int, correctable: int) -> bool:
    """Whether the Hamming spheres of radius ``t`` tile the space exactly.

    ``n_messages * sum_{i <= t} C(n, i) == 2^n``, as an equality over integers
    --- no tolerance, because the statement is combinatorial and either holds
    or does not.
    """
    volume = sum(math.comb(n_bits, radius) for radius in range(correctable + 1))
    return bool(n_messages * volume == 2**n_bits)


def repetition_error_rate(
    n_bits: int, *, erasure: float | None = None, crossover: float | None = None
) -> float:
    """The closed-form block error probability of the ``(n, 1)`` code.

    Exactly one of ``erasure`` or ``crossover`` is given.

    * **Erasure channel.** The decoder fails only when *every* copy is erased,
      so the probability is ``eps ** n`` --- there is no wrong answer, only a
      missing one.
    * **Binary symmetric channel.** Majority vote fails when more than half
      the copies flip, the binomial tail above ``n / 2``, with a tie broken
      arbitrarily and so counted as half at even ``n``.

    Returns
    -------
    float
        The block error probability.

    Raises
    ------
    ValueError
        If neither or both channels are named, or a probability is outside
        ``[0, 1]``.
    """
    if (erasure is None) == (crossover is None):
        msg = "name exactly one of erasure or crossover"
        raise ValueError(msg)
    if erasure is not None:
        if not 0.0 <= erasure <= 1.0:
            msg = f"an erasure probability lies in [0, 1], got {erasure}"
            raise ValueError(msg)
        return float(erasure**n_bits)
    assert crossover is not None
    if not 0.0 <= crossover <= 1.0:
        msg = f"a crossover probability lies in [0, 1], got {crossover}"
        raise ValueError(msg)
    total = 0.0
    for flipped in range(n_bits + 1):
        weight = math.comb(n_bits, flipped) * crossover**flipped
        weight *= (1.0 - crossover) ** (n_bits - flipped)
        if 2 * flipped > n_bits:
            total += weight
        elif 2 * flipped == n_bits:
            total += 0.5 * weight
    return float(total)


def crc_remainder(message: np.ndarray, polynomial: np.ndarray) -> np.ndarray:
    """``message`` shifted by the polynomial's degree, modulo it, over GF(2).

    Polynomial long division with exclusive-or for subtraction: the remainder
    is the check the receiver recomputes. #593's CRC-aided list decoding picks
    the survivor whose remainder is zero, which is why this is here rather
    than inside that module.

    Parameters
    ----------
    message : np.ndarray
        Bits, most significant first.
    polynomial : np.ndarray
        Generator coefficients, most significant first, leading coefficient 1.

    Returns
    -------
    np.ndarray
        The remainder, one bit shorter than ``polynomial``.

    Raises
    ------
    ValueError
        If the generator's leading coefficient is not 1, where the division is
        not defined.
    """
    generator = np.asarray(polynomial, dtype=np.int64) % 2
    if generator.size < 2 or generator[0] != 1:
        msg = "a CRC generator is at least two bits with a leading one"
        raise ValueError(msg)
    degree = generator.size - 1
    register = np.concatenate(
        [np.asarray(message, dtype=np.int64) % 2, np.zeros(degree, dtype=np.int64)]
    )
    for position in range(register.size - degree):
        if register[position]:
            register[position : position + generator.size] ^= generator
    return np.asarray(register[-degree:])


def syndrome(code: ParityCheck, word: np.ndarray) -> np.ndarray:
    """``H x`` over GF(2), for a word of the code's length."""
    bits = np.asarray(word, dtype=np.int64) % 2
    if bits.shape != (code.n_bits,):
        msg = f"word is {bits.shape}, expected {(code.n_bits,)}"
        raise ValueError(msg)
    return np.asarray((code.dense() @ bits) % 2)


def hamming_correct(code: ParityCheck, m: int, word: np.ndarray) -> np.ndarray:
    """Correct one error by reading the syndrome as a bit index.

    The whole reason the columns are written in increasing order: a nonzero
    syndrome is the binary expansion of the flipped position, so decoding is a
    table lookup with no table.
    """
    bits = np.asarray(word, dtype=np.int64) % 2
    pattern = syndrome(code, bits)[:m]
    position = int(sum(int(bit) << index for index, bit in enumerate(pattern)))
    if position == 0:
        return bits
    corrected = bits.copy()
    corrected[position - 1] ^= 1
    return corrected
