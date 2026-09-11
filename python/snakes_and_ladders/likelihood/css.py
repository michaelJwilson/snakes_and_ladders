"""Decoding a CSS code: the criterion degeneracy forces, and its exact oracle.

Classical decoding asks whether the decoder recovered the word that was sent.
A CSS code cannot ask that. The syndrome ``s = H e`` leaves the error known
only up to ``ker H``, and a residual ``r = e + ê`` inside ``rowspace(H)`` is a
*stabilizer*: it acts as the identity on the codespace, so the decode succeeds
with ``ê != e``. A residual in ``ker H`` outside the row space is a *logical
operator* and the decode fails. The criterion is therefore membership of the
quotient ``ker(H) / rowspace(H)``, which is :func:`decode_succeeds`, and it is
written and pinned here before any decoder is called: every number in
``sec:ldpc:css`` is meaningless if it is wrong.

**Maximum likelihood is a sum, not a maximum** (``eq:coset-ml``). The optimal
decoder returns the *coset* of highest total probability, summed over its
members, which is a different decoder from the one returning the single
likeliest error -- the second maximizes a term, the first the sum the term
belongs to. :func:`error_cosets` computes both by enumerating every error at a
length where ``2 ** n`` reaches, so the gap between them is measured rather
than argued, and the degenerate rate is the floor every other decoder here is
reported against.

**Belief propagation is run unchanged.** ``sec:ldpc``'s decoder solves
``H ĉ = 0`` from channel ratios, and that is already a syndrome decoder:
sending the zero word over a binary symmetric channel makes the received word
the error itself, so the decoder's codeword estimate ``ĉ`` *is* the residual
``e + ê``, and the quantum criterion reads "is ``ĉ`` a stabilizer" where the
classical one read "is ``ĉ`` zero". The estimate depends on the error only
through its syndrome, by the coset symmetry ``sec:ldpc`` already proves for
the all-zero word, and the suite checks that rather than assuming it. So no
second decoder appears, and the difference between the two sections is the
question asked of one output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.enumeration import refuse_oversized
from snakes_and_ladders.likelihood.ldpc import (
    DEFAULT_MAX_ITERATIONS,
    DecodingAlgorithm,
    decode,
)
from snakes_and_ladders.sim.css import CssCode, sample_x_error
from snakes_and_ladders.sim.ldpc import BinarySymmetricChannel


def decode_succeeds(code: CssCode, error: np.ndarray, estimate: np.ndarray) -> bool:
    """Whether correcting ``error`` with ``estimate`` leaves the codespace fixed.

    ``(error + estimate) in rowspace(H)``, by the rank test of
    :meth:`~snakes_and_ladders.sim.css.CssCode.is_stabilizer`, and not
    ``error == estimate``. Two things the one test decides. A residual that is
    a nonzero stabilizer reads as a success, which is the degeneracy the
    classical criterion has no room for. And a correction with the wrong
    syndrome reads as a failure without a second branch, since
    ``rowspace(H)`` sits inside ``ker H``: a residual outside ``ker H`` is in
    neither.

    Parameters
    ----------
    code : CssCode
    error : np.ndarray
        The ``X`` error that occurred, ``n`` bits.
    estimate : np.ndarray
        The correction the decoder returns, ``n`` bits.
    """
    residual = np.asarray(error, dtype=np.uint8) ^ np.asarray(estimate, dtype=np.uint8)
    return code.is_stabilizer(residual)


# --- the oracle: exact degenerate maximum likelihood by enumeration -------------


@dataclass(frozen=True)
class ErrorCosets:
    """Every error of a code, partitioned by syndrome and then by coset.

    The two decoders below share this partition and differ only in what they
    read off it, which is the point: the optimum under degeneracy is a
    property of the coset probabilities, and the likeliest single error is a
    property of one member.

    Parameters
    ----------
    coset_probability : np.ndarray
        ``(2 ** n_checks, 2 ** k)``. Row ``s`` is the syndrome packed
        little-endian over the checks, column ``l`` the coset label packed the
        same way over :attr:`~snakes_and_ladders.sim.css.CssCode.logical_z`.
        The entries sum to one over the whole array: every error is in exactly
        one cell.
    reachable : np.ndarray
        ``(2 ** n_checks,)`` of bool: whether any error has that syndrome.
        Rows of ``H`` that are dependent make some patterns unreachable, and
        those rows carry no probability.
    cosets_per_syndrome : np.ndarray
        ``(2 ** n_checks,)``: cells carrying probability in each row, which is
        ``2 ** k`` at every reachable syndrome and zero elsewhere.
    degenerate_label, likeliest_error_label : np.ndarray
        ``(2 ** n_checks,)``: the coset each decoder returns per syndrome --
        the one of greatest summed probability, and the one holding the single
        likeliest error. Ties in either go to the lower label, and
        ``ties`` counts the syndromes where that choice was made.
    degenerate_logical_error_rate, single_error_logical_error_rate : float
        ``1 - sum_s P(the returned coset)``: the exact probability that the
        decoder's correction differs from the error by a logical operator. No
        sampling enters either, so neither carries an interval.
    disagreements : int
        Syndromes where the two decoders return different cosets. Zero means
        degeneracy changed no decision at this instance and noise level, which
        is a measurement and not a defect.
    ties : int
        Syndromes where the degenerate decoder's maximum was not unique.
    """

    coset_probability: np.ndarray
    reachable: np.ndarray
    cosets_per_syndrome: np.ndarray
    degenerate_label: np.ndarray
    likeliest_error_label: np.ndarray
    degenerate_logical_error_rate: float
    single_error_logical_error_rate: float
    disagreements: int
    ties: int


def error_cosets(code: CssCode, flip_probability: float) -> ErrorCosets:
    """Enumerate every ``X`` error and group it by syndrome, then by coset.

    ``P(e) = p^|e| (1 - p)^(n - |e|)`` over all ``2 ** n`` errors; the
    syndrome is ``H e`` and the coset is
    :meth:`~snakes_and_ladders.sim.css.CssCode.logical_label`, which separates
    the cosets of one syndrome because two errors of that syndrome differ by
    an element of ``ker H``. Probabilities are summed rather than kept in the
    log domain: the enumeration ceiling puts ``n`` at 17, where the smallest
    term is ``p ** 17`` and no channel this is run on comes near a float64
    denormal.

    Parameters
    ----------
    code : CssCode
    flip_probability : float
        The per-qubit ``X`` rate, in ``(0, 0.5)``.

    Raises
    ------
    ValueError
        If the rate is out of range, or past
        :data:`~snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS`
        errors -- which is the size this oracle reaches and the reason
        ``sec:ldpc:css`` reports a gap at one length and a comparison at the
        other.
    """
    if not 0.0 < flip_probability < 0.5:
        msg = f"flip_probability must be in (0, 0.5), got {flip_probability}"
        raise ValueError(msg)
    n_qubits, n_checks = code.n_qubits, code.checks.n_checks
    refuse_oversized(
        2**n_qubits, what=f"2 ** {n_qubits} error patterns on {n_qubits} qubits"
    )
    errors = ((np.arange(2**n_qubits)[:, None] >> np.arange(n_qubits)) & 1).astype(
        np.uint8
    )
    weight = errors.sum(axis=1)
    probability = flip_probability**weight * (1.0 - flip_probability) ** (
        n_qubits - weight
    )
    dense = code.checks.dense().astype(np.int64)
    syndrome = _pack((errors.astype(np.int64) @ dense.T) & 1)
    label = _pack(code.logical_label(errors))
    n_labels = 2**code.n_logical
    cell = syndrome * n_labels + label
    totals = np.bincount(
        cell, weights=probability, minlength=2**n_checks * n_labels
    ).reshape(2**n_checks, n_labels)
    reachable = np.bincount(syndrome, minlength=2**n_checks) > 0
    # A coset of a reachable syndrome always carries probability, every error
    # being flippable by any logical operator, so counting the occupied cells
    # states `2 ** k` cosets per syndrome rather than assuming it.
    occupied = (totals > 0.0).sum(axis=1)
    degenerate = np.argmax(totals, axis=1)
    ties = int(
        ((totals == totals.max(axis=1, keepdims=True)).sum(axis=1) > 1)[reachable].sum()
    )
    likeliest = _likeliest_error_label(syndrome, label, weight, n_checks)
    return ErrorCosets(
        coset_probability=totals,
        reachable=reachable,
        cosets_per_syndrome=occupied,
        degenerate_label=degenerate,
        likeliest_error_label=likeliest,
        degenerate_logical_error_rate=_logical_error_rate(
            totals, reachable, degenerate
        ),
        single_error_logical_error_rate=_logical_error_rate(
            totals, reachable, likeliest
        ),
        disagreements=int((degenerate[reachable] != likeliest[reachable]).sum()),
        ties=ties,
    )


def _pack(bits: np.ndarray) -> np.ndarray:
    """Rows of 0/1 as integers, little-endian; an empty row packs to zero."""
    columns = bits.shape[-1]
    return np.asarray(bits.astype(np.int64) @ (1 << np.arange(columns)))


def _likeliest_error_label(
    syndrome: np.ndarray, label: np.ndarray, weight: np.ndarray, n_checks: int
) -> np.ndarray:
    """The coset of the lightest error of each syndrome, ties to the lower label.

    Sorting by weight and then by label makes the first error of a syndrome
    the one a decoder maximizing a single error's probability returns, the
    weight ordering being the probability ordering at any ``p < 1/2``.
    """
    order = np.lexsort((label, weight))
    _, first = np.unique(syndrome[order], return_index=True)
    labels = np.zeros(2**n_checks, dtype=np.int64)
    labels[syndrome[order][first]] = label[order][first]
    return labels


def _logical_error_rate(
    totals: np.ndarray, reachable: np.ndarray, chosen: np.ndarray
) -> float:
    """``1 - sum_s P(chosen coset)``, over the syndromes that occur."""
    rows = np.flatnonzero(reachable)
    return float(1.0 - totals[rows, chosen[rows]].sum())


# --- belief propagation, scored by the criterion above ---------------------------


@dataclass(frozen=True)
class SyndromeDecoding:
    """One decode of a CSS code, and which of the two failures it is.

    Parameters
    ----------
    correction : np.ndarray
        ``ê``, the ``X`` correction to apply, ``n`` bits.
    residual : np.ndarray
        ``e + ê``, which is the decoder's codeword estimate. A stabilizer, a
        logical operator, or -- when the decoder stopped at its cap -- not in
        ``ker H`` at all.
    succeeded : bool
        :func:`decode_succeeds`: the residual is a stabilizer.
    converged : bool
        Whether the decoder reached a word satisfying every check. A decode
        that did not is a different defect from one that converged to a
        logical coset: the first returns no usable correction, the second
        returns one that damages the encoded state, and a single block-error
        count hides which.
    iterations : int
        Iterations run.
    """

    correction: np.ndarray
    residual: np.ndarray
    succeeded: bool
    converged: bool
    iterations: int


def decode_syndrome(
    code: CssCode,
    error: np.ndarray,
    llr: np.ndarray,
    *,
    algorithm: DecodingAlgorithm = DecodingAlgorithm.SUM_PRODUCT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> SyndromeDecoding:
    """``sec:ldpc``'s decoder on a CSS code, scored by :func:`decode_succeeds`.

    The received word of a binary symmetric channel carrying the zero codeword
    *is* the error, so the decoder's codeword estimate ``ĉ`` is the residual
    ``e + ê`` and the correction is ``ê = e + ĉ``. That correction depends on
    the error only through its syndrome -- replacing ``e`` by ``e + c`` for a
    codeword ``c`` negates the ratios on ``c``'s support, and the check update
    being odd in each argument and the bit update linear carries that through
    to ``ĉ + c``, leaving ``ê`` fixed. It is the coset symmetry ``sec:ldpc``
    states for the all-zero word, and it is what makes this a decoder of the
    syndrome rather than of an error no one observes; the suite asserts it
    rather than taking it on the argument.

    Parameters
    ----------
    code : CssCode
    error, llr : np.ndarray
        An error and its ratios, as
        :func:`~snakes_and_ladders.sim.css.sample_x_error` returns them.
    algorithm : DecodingAlgorithm
        The check update, unchanged from the classical section.
    max_iterations : int
        The cap; reaching it is a failure to decode and is counted as one.
    """
    decoding = decode(
        code.checks, llr, algorithm=algorithm, max_iterations=max_iterations
    )
    residual = np.asarray(decoding.bits, dtype=np.uint8)
    correction = np.asarray(error, dtype=np.uint8) ^ residual
    return SyndromeDecoding(
        correction=correction,
        residual=residual,
        succeeded=decode_succeeds(code, error, correction),
        converged=decoding.decoded,
        iterations=decoding.iterations,
    )


@dataclass(frozen=True)
class LogicalErrorRate:
    """What a decoder achieved over one seeded run, split by how it failed.

    Parameters
    ----------
    trials : int
        Errors drawn.
    logical_failures : int
        Decodes that converged on a word satisfying every check whose residual
        is a logical operator: a correction that is applied and damages the
        encoded state.
    undecoded : int
        Decodes that reached the iteration cap. Their residual satisfies no
        syndrome, so they are counted as failures and not as a separate
        outcome, but they are a different defect and are reported apart.
    """

    trials: int
    logical_failures: int
    undecoded: int

    @property
    def failures(self) -> int:
        """Decodes that did not leave the codespace fixed."""
        return self.logical_failures + self.undecoded

    @property
    def rate(self) -> float:
        """The share of trials that failed, either way."""
        return self.failures / self.trials


def measure_logical_error_rate(
    code: CssCode,
    channel: BinarySymmetricChannel,
    rng: np.random.Generator,
    trials: int,
    *,
    algorithm: DecodingAlgorithm = DecodingAlgorithm.SUM_PRODUCT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> LogicalErrorRate:
    """Draw ``trials`` errors and decode each, counting the two failures apart.

    Parameters
    ----------
    code : CssCode
    channel : BinarySymmetricChannel
        The per-qubit ``X`` rate.
    rng : np.random.Generator
        The errors are drawn from it, one generator for the run rather than a
        seed per trial.
    trials : int
        At least one.
    algorithm : DecodingAlgorithm
    max_iterations : int

    Raises
    ------
    ValueError
        If ``trials`` is below one.
    """
    if trials < 1:
        msg = f"trials must be at least 1, got {trials}"
        raise ValueError(msg)
    logical_failures = 0
    undecoded = 0
    for _ in range(trials):
        error, llr = sample_x_error(code, channel, rng)
        result = decode_syndrome(
            code, error, llr, algorithm=algorithm, max_iterations=max_iterations
        )
        if not result.converged:
            undecoded += 1
        elif not result.succeeded:
            logical_failures += 1
    return LogicalErrorRate(trials, logical_failures, undecoded)
