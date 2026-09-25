"""Decoders for the algebraic codes: Reed--Solomon by syndromes, Hamming by its syndrome table.

The codes are constructed in ``sim/`` --- :mod:`sal.sim.reed_solomon`
and :mod:`sal.sim.elementary_codes` --- and decoding them is
inference, which ``sim/CLAUDE.md`` says that directory does not perform (issue
#830). The decoders moved here unchanged: Berlekamp--Massey and Chien search
over the field arithmetic the code carries, and the Hamming correction that
reads a syndrome as a bit index. Every test that pinned them pins them from
here, bitwise.
"""

from __future__ import annotations

import numpy as np

from sal.sim.galois import (
    polynomial_evaluate,
    polynomial_multiply,
)
from sal.sim.ldpc import ParityCheck
from sal.sim.reed_solomon import ReedSolomon


class DecodingFailure(RuntimeError):
    """More errors than the code can correct, so no decode is returned.

    Carries what was established before the contradiction --- how many errors
    the locator claimed and how many roots the search found --- because those
    two disagreeing *is* the detection, and a caller tuning a channel wants
    the numbers rather than the word "failed".
    """

    def __init__(self, degree: int, roots: int) -> None:
        super().__init__(
            f"the error locator has degree {degree} and {roots} roots in the "
            "field; they disagree, so more symbols are wrong than this code "
            "can correct and no codeword is returned"
        )
        self.degree = degree
        self.roots = roots


def syndromes(code: ReedSolomon, received: np.ndarray) -> np.ndarray:
    """``S_j = r(alpha^j)`` for ``j`` in ``1..n-k``; all zero iff a codeword."""
    gf = code.field
    coefficients = np.asarray(received, dtype=np.int64)[::-1]
    return np.asarray(
        [
            polynomial_evaluate(coefficients, gf.alpha(power), gf)
            for power in range(1, code.n_parity + 1)
        ],
        dtype=np.int64,
    )


def berlekamp_massey(code: ReedSolomon, values: np.ndarray) -> np.ndarray:
    """The shortest linear recursion the syndromes satisfy, as its locator.

    The connection polynomial of the Berlekamp--Massey recursion: its roots are
    the inverses of the error positions, which is what the Chien search then
    finds. Returned low degree first with a constant term of one.
    """
    gf = code.field
    locator = np.array([1], dtype=np.int64)
    previous = np.array([1], dtype=np.int64)
    length = 0
    since = 1
    discrepancy_previous = 1
    for position in range(len(values)):
        discrepancy = int(values[position])
        for index in range(1, length + 1):
            if index < len(locator):
                discrepancy ^= gf.multiply(
                    int(locator[index]), int(values[position - index])
                )
        if discrepancy == 0:
            since += 1
            continue
        scale = gf.divide(discrepancy, discrepancy_previous)
        shifted = np.zeros(since + len(previous), dtype=np.int64)
        for index, coefficient in enumerate(previous):
            shifted[since + index] = gf.multiply(scale, int(coefficient))
        updated = np.zeros(max(len(locator), len(shifted)), dtype=np.int64)
        updated[: len(locator)] = locator
        updated[: len(shifted)] ^= shifted
        if 2 * length <= position:
            previous = locator
            length = position + 1 - length
            discrepancy_previous = discrepancy
            since = 1
        else:
            since += 1
        locator = updated
    return locator


def chien_search(code: ReedSolomon, locator: np.ndarray) -> list[int]:
    """The error positions, by trying every field element as a root.

    A root ``alpha^-i`` of the locator says position ``i`` is wrong. Trying all
    ``2^m - 1`` elements is the whole search: at these lengths it is cheaper
    than being clever, and it is the step whose count the decoder compares
    against the locator's degree to detect an uncorrectable word.
    """
    gf = code.field
    found: list[int] = []
    for position in range(code.n_symbols):
        point = gf.alpha(-position)
        if polynomial_evaluate(locator, point, gf) == 0:
            found.append(position)
    return found


def _formal_derivative(coefficients: np.ndarray) -> np.ndarray:
    """``d/dx`` in characteristic two: the odd-degree terms, shifted down.

    Every even-degree term differentiates to zero because its coefficient
    doubles, and doubling is zero here. That is not a simplification of the
    real derivative; it *is* the derivative in this field, and Forney's
    formula is stated with it.
    """
    derivative = np.zeros(max(len(coefficients) - 1, 1), dtype=np.int64)
    for power in range(1, len(coefficients)):
        if power % 2:
            derivative[power - 1] = coefficients[power]
    return derivative


def decode(code: ReedSolomon, received: np.ndarray) -> np.ndarray:
    """Correct up to ``t`` symbol errors, or refuse.

    Syndromes, then Berlekamp--Massey, then a Chien search, then Forney's
    formula for the magnitudes. The word is returned corrected; a word already
    in the code is returned unchanged without running any of it.

    Raises
    ------
    DecodingFailure
        When the locator's degree and its number of roots in the field
        disagree, which is what an uncorrectable word looks like from inside.
    """
    gf = code.field
    word = np.asarray(received, dtype=np.int64).copy()
    if word.shape != (code.n_symbols,):
        msg = f"received word is {word.shape}, expected {(code.n_symbols,)}"
        raise ValueError(msg)

    values = syndromes(code, word)
    if not values.any():
        return word

    locator = berlekamp_massey(code, values)
    positions = chien_search(code, locator)
    degree = len(locator) - 1
    if degree != len(positions) or degree > code.correctable:
        raise DecodingFailure(degree, len(positions))

    # Forney: the magnitude at position i is `omega(x) / lambda'(x)` evaluated
    # at the root, with `omega = S * lambda` truncated to the parity length.
    product = polynomial_multiply(values, locator, gf)
    evaluator = product[: code.n_parity]
    derivative = _formal_derivative(locator)
    for position in positions:
        point = gf.alpha(-position)
        numerator = polynomial_evaluate(evaluator, point, gf)
        denominator = polynomial_evaluate(derivative, point, gf)
        if denominator == 0:
            raise DecodingFailure(degree, len(positions))
        # Forney with the generator's first root at `alpha^1`: the magnitude
        # is `omega / lambda'` at the root, with **no** leading power of the
        # locator. The `X^(1-b)` factor of the general formula is one here,
        # and carrying it anyway decodes every single-error word to the wrong
        # magnitude while locating it correctly --- which reads as a decoder
        # that finds the error and cannot fix it.
        magnitude = gf.divide(numerator, denominator)
        word[code.n_symbols - 1 - position] ^= magnitude

    if syndromes(code, word).any():
        msg = (
            f"the locator found {len(positions)} error positions and the "
            "corrected word is still not a codeword, so the magnitudes solve "
            "no consistent system: more symbols are wrong than this code can "
            "correct"
        )
        raise RuntimeError(msg)
    return word


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
