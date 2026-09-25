"""Successive cancellation, written the way the textbook states it.

Issue #593. The oracle for :mod:`sal.likelihood.polar`, kept
per root `CLAUDE.md`'s rule that an accelerated loop keeps the implementation
it stands in for. The difference is only the layout: this recurses over
subvectors with one Python call per node and no path axis, decoding a single
candidate; `polar.py` carries ``L`` candidates as one array with a path
dimension and prunes at the leaves, which is what makes a list affordable and
what a reader cannot check by eye.

The arithmetic is identical --- the same ``f`` and ``g``, the same order, the
same hard decision at a leaf --- so the suite pins the two **bitwise** rather
than to a tolerance. A tolerance there would hide exactly the bug this exists
to catch: a fork or a gather that reorders the traversal and decodes a
different code on some inputs and the same one on most.

It decodes one path and therefore referees ``L = 1`` alone. The other end of
`polar.py`'s machinery has its own oracle and needs no second implementation:
at ``L = 2^k`` the list prunes nothing, so it is an exhaustive search and must
equal :func:`sal.likelihood.ldpc.exact_decoding`.
"""

from __future__ import annotations

import numpy as np

from sal.sim.polar import PolarCode


def _decode(llr: np.ndarray, frozen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One subtree: its source bits and the codeword bits they produce.

    Parameters
    ----------
    llr : np.ndarray
        The observations this subtree sees, length ``m``.
    frozen : np.ndarray
        Boolean, length ``m``: which of its positions are held to zero.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(source, codeword)``, both length ``m``.
    """
    if llr.size == 1:
        if frozen[0]:
            bit = np.zeros(1, dtype=np.int64)
        else:
            bit = np.array([int(llr[0] < 0.0)], dtype=np.int64)
        return bit, bit.copy()

    half = llr.size // 2
    left_llr = np.array(
        [
            2.0
            * np.arctanh(
                np.clip(
                    np.tanh(np.clip(llr[i] / 2.0, -30.0, 30.0))
                    * np.tanh(np.clip(llr[half + i] / 2.0, -30.0, 30.0)),
                    -1.0 + 1e-15,
                    1.0 - 1e-15,
                )
            )
            for i in range(half)
        ]
    )
    left_source, left_word = _decode(left_llr, frozen[:half])

    right_llr = np.array(
        [
            llr[half + i] + (1.0 - 2.0 * float(left_word[i])) * llr[i]
            for i in range(half)
        ]
    )
    right_source, right_word = _decode(right_llr, frozen[half:])

    return (
        np.concatenate([left_source, right_source]),
        np.concatenate([(left_word + right_word) % 2, right_word]),
    )


def decode_sc_reference(code: PolarCode, llr: np.ndarray) -> np.ndarray:
    """The source vector plain successive cancellation decides, all ``N`` positions.

    Parameters
    ----------
    code : PolarCode
    llr : np.ndarray
        ``N`` channel log-likelihood ratios, positive favouring zero.

    Returns
    -------
    np.ndarray
        The ``N`` pre-transform bits, frozen positions included, so the
        comparison against
        :func:`sal.likelihood.polar.decode_sc` covers what the
        decoder believed everywhere and not only where the message sat.

    Raises
    ------
    ValueError
        If ``llr`` is not ``N`` long.
    """
    values = np.asarray(llr, dtype=float).reshape(-1)
    if values.size != code.n_bits:
        msg = f"the code takes {code.n_bits} ratios, got {values.size}"
        raise ValueError(msg)
    frozen = np.ones(code.n_bits, dtype=bool)
    frozen[code.information] = False
    source, _ = _decode(values, frozen)
    return np.asarray(source, dtype=np.uint8)


__all__ = ["decode_sc_reference"]
