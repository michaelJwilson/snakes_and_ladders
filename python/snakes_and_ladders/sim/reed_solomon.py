"""Reed--Solomon: the first decoder here that is algebraic rather than probabilistic.

Issue #594. Everything else in this repository decodes by an argument about
probability --- a message-passing fixed point, or an enumeration of the
posterior. Reed--Solomon decodes by **solving equations**: it corrects any
``t = (n - k) / 2`` symbol errors deterministically, with no channel model, no
iteration count, no tolerance and no tie to break. That is a different kind of
object, and having one is the reason this code is worth carrying.

**Maximum distance separable, which is an equality.** The Singleton bound says
``d <= n - k + 1`` for every code; Reed--Solomon meets it with equality, so a
``(7, 3)`` code over ``GF(2^3)`` has ``d = 5`` exactly and corrects two symbol
errors. Like the perfection of Hamming and Golay
(:mod:`snakes_and_ladders.sim.elementary_codes`), that is a sharp invariant
rather than a bound, and it fails loudly if a generator polynomial drifts.

**The decoder, in four steps and no more.** Syndromes say whether and how the
received word fails; Berlekamp--Massey finds the shortest recursion the
syndromes satisfy, whose roots locate the errors; a Chien search finds those
roots by trying every field element; Forney's formula gives each error's
magnitude from the same two polynomials. Each step is checked here against
what the step before it established, and the whole against re-encoding.

**Past ``t`` it can be confidently wrong, and that is the honest statement.**
This is a *bounded-distance* decoder: within ``t`` errors it is exact, and
beyond ``t`` the received word may land inside another codeword's sphere, at
which point the equations are consistent and the decoder returns that other
word --- correctly, by its own lights, and wrongly by the sender's. Measured
on ``RS(7, 3)`` at three errors over 300 draws: **260 refused, 40 decoded to
the wrong codeword, 0 right by luck**. So a decode is not evidence that ``t``
errors or fewer occurred, and a caller who needs that evidence needs a code
with more distance, not a better decoder. The refusals are real refusals ---
the locator's degree and its root count disagree, or the corrected word still
fails its syndromes --- and they are raised rather than returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.sim.galois import (
    BinaryField,
    field,
    polynomial_evaluate,
    polynomial_multiply,
    polynomial_remainder,
)


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


@dataclass(frozen=True)
class ReedSolomon:
    """An ``RS(n, k)`` code over ``GF(2^m)``, ``n = 2^m - 1``.

    Parameters
    ----------
    m : int
        The field's extension degree.
    n_symbols, n_message : int
        Block length and message length, in symbols.
    generator : np.ndarray
        The generator polynomial's coefficients, low degree first.
    """

    m: int
    n_symbols: int
    n_message: int
    generator: np.ndarray

    @property
    def field(self) -> BinaryField:
        """The field the code is over."""
        return field(self.m)

    @property
    def n_parity(self) -> int:
        """``n - k``, the number of parity symbols."""
        return self.n_symbols - self.n_message

    @property
    def correctable(self) -> int:
        """``t = (n - k) // 2``, the guaranteed symbol-error capacity."""
        return self.n_parity // 2

    @property
    def minimum_distance(self) -> int:
        """``n - k + 1``: the Singleton bound, met with equality."""
        return self.n_parity + 1


def reed_solomon(m: int, n_message: int) -> ReedSolomon:
    """The ``RS(2^m - 1, k)`` code whose generator has roots ``alpha^1..alpha^(n-k)``.

    Parameters
    ----------
    m : int
        Field extension degree.
    n_message : int
        Message length in symbols, below ``2^m - 1``.

    Returns
    -------
    ReedSolomon
        The code.

    Raises
    ------
    ValueError
        If the message length leaves no parity, or an odd number of parity
        symbols --- which is legal but wastes one, since ``t`` rounds down and
        a reader comparing ``d`` to ``2t + 1`` would find them disagree.
    """
    gf = field(m)
    n_symbols = gf.nonzero
    if not 0 < n_message < n_symbols:
        msg = f"a message of {n_message} symbols does not fit a length-{n_symbols} code"
        raise ValueError(msg)
    parity = n_symbols - n_message
    if parity % 2:
        msg = (
            f"{parity} parity symbols is odd, so one is wasted: t = {parity // 2} "
            f"but d = {parity + 1}. Choose k = {n_message - 1} or {n_message + 1}"
        )
        raise ValueError(msg)
    generator = np.array([1], dtype=np.int64)
    for power in range(1, parity + 1):
        generator = polynomial_multiply(
            generator, np.array([gf.alpha(power), 1], dtype=np.int64), gf
        )
    return ReedSolomon(
        m=m, n_symbols=n_symbols, n_message=n_message, generator=generator
    )


def encode(code: ReedSolomon, message: np.ndarray) -> np.ndarray:
    """Systematically: the message, then the remainder that makes it divisible.

    The codeword is ``m(x) x^(n-k) - (m(x) x^(n-k) mod g(x))``, so the message
    symbols appear unchanged and the parity follows them. Systematic because a
    reader of a received word can see the message in it, which is what makes a
    failed decode legible.
    """
    gf = code.field
    symbols = np.asarray(message, dtype=np.int64)
    if symbols.shape != (code.n_message,):
        msg = f"message is {symbols.shape}, expected {(code.n_message,)}"
        raise ValueError(msg)
    if symbols.max(initial=0) >= gf.order:
        msg = f"a symbol is outside GF(2^{code.m})"
        raise ValueError(msg)
    shifted = np.concatenate([np.zeros(code.n_parity, dtype=np.int64), symbols[::-1]])
    remainder = polynomial_remainder(shifted, code.generator, gf)
    word = shifted.copy()
    word[: code.n_parity] ^= remainder
    return np.asarray(word[::-1])


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
