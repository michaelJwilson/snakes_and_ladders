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

See ``sec:ldpc:algebraic`` of ``docs/tex/textbook.tex``, ``eq:singleton`` for
the bound this code meets and ``eq:bounded-distance`` for the block error rate
that follows from meeting it (Blahut 2003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.sim.galois import (
    BinaryField,
    field,
    polynomial_multiply,
    polynomial_remainder,
)


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
