"""Convolutional and turbo codes: the shift register, its trellis, and the encoder.

A rate-1/2 recursive systematic convolutional encoder is a linear feedback
shift register of memory ``m`` over GF(2), stated by two octal generator
polynomials: a feedback polynomial and a feedforward one (``sec:turbo`` in
the textbook, ``eq:rsc``). Its state is the register contents, so its
input-output relation is a hidden Markov chain on ``2 ** m`` states with two
edges leaving each -- the trellis, held here as arrays in the offsets layout
root ``CLAUDE.md`` asks for rather than as a list of node objects.

A turbo code is two such registers fed the same message in two orders through
an interleaver \\citep{berrou1993}: the first sees the message, the second a
permutation of it, and the transmitted word carries the message once and both
parity streams. Rate 1/3 unpunctured, both registers driven back to the zero
state by ``m`` tail steps each, so the transmitted length is ``3 K + 4 m`` for
a ``K``-bit message and the trellis referee of
:mod:`snakes_and_ladders.likelihood.convolutional` may terminate both ends.

**Codewords are the same objects LDPC's are.** A convolutional code is linear,
so its words are the null space of a parity-check matrix and
:func:`parity_check` produces one; that lets a turbo instance be decoded by
issue #340's parity-check machinery and referees this encoder against a
construction sharing no code with it. The channels are #340's, imported rather
than restated, and the log-likelihood ratio convention is ``eq:ldpc-llr``
unchanged: ``L_i = log p(y_i | c_i = 0) - log p(y_i | c_i = 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snakes_and_ladders.fixtures import load_declared
from snakes_and_ladders.sim.ldpc import ParityCheck, null_space

#: The log weight of an edge the trellis does not have. Finite rather than
#: ``-inf``: the general message passing shifts by a row maximum before
#: exponentiating, and a row entirely ``-inf`` -- an unreachable state's
#: incoming edges at the first steps of a terminated trellis -- becomes
#: ``-inf - (-inf)``, a ``nan`` that spreads. At ``-1e30`` the shifted term
#: underflows to exactly zero in ``float64``, so the sum is the one ``-inf``
#: would have given. The same constant floors the recursions in
#: :mod:`snakes_and_ladders.likelihood.convolutional`, so it is stated once.
IMPOSSIBLE_EDGE = -1e30

#: Past this the dense GF(2) construction of :func:`parity_check` is no
#: longer the cheap part: it is cubic in the block length, and the trellis
#: decoders of :mod:`snakes_and_ladders.likelihood.convolutional` are linear
#: in it, so a caller reaching this ceiling wanted those instead.
MAX_PARITY_CHECK_BITS = 512


def octal_taps(polynomial: int, memory: int) -> np.ndarray:
    """The GF(2) coefficients of an octal generator, constant term first.

    An octal generator is read most-significant bit first as the textbook's
    convention states: ``7`` (binary ``111``) at memory 2 is ``1 + D +
    D**2``, and ``5`` (``101``) is ``1 + D**2``.

    Parameters
    ----------
    polynomial : int
        The generator, written in octal at the call site (``0o7``).
    memory : int
        Register length ``m``; the result is ``m + 1`` long.

    Returns
    -------
    np.ndarray
        ``uint8`` of shape ``(memory + 1,)``: index ``i`` is the coefficient
        of ``D ** i``.

    Raises
    ------
    ValueError
        If the polynomial needs more than ``memory + 1`` bits, or its constant
        term is zero -- a generator without one is a delayed copy of a shorter
        generator, and a feedback polynomial without one defines no recursion.

    Examples
    --------
    >>> octal_taps(0o7, 2).tolist()
    [1, 1, 1]
    >>> octal_taps(0o5, 2).tolist()
    [1, 0, 1]
    """
    if polynomial < 0 or polynomial >> (memory + 1):
        msg = f"generator {polynomial:o} (octal) needs more than {memory + 1} bits"
        raise ValueError(msg)
    taps = np.array(
        [(polynomial >> (memory - i)) & 1 for i in range(memory + 1)], dtype=np.uint8
    )
    if not taps[0]:
        msg = f"generator {polynomial:o} (octal) has a zero constant term"
        raise ValueError(msg)
    return taps


@dataclass(frozen=True)
class Trellis:
    """A shift register's state machine, as arrays indexed by (state, input).

    The register holds ``m`` bits and the state is that word read as an integer
    with the most recent bit *least* significant, so one step is a left shift
    with the new bit entering at the bottom. One array per quantity rather than
    a node object per state: an edge is a pair of indices and a sweep over a
    trellis step is one vectorized gather, which is what the BCJR recursion
    needs.

    Parameters
    ----------
    memory : int
        Register length ``m``; there are ``2 ** m`` states.
    next_state : np.ndarray
        ``(2 ** m, 2)`` ``int64``: the state entered from ``state`` on input
        bit ``u``.
    parity : np.ndarray
        ``(2 ** m, 2)`` ``uint8``: the parity bit that edge emits. The
        systematic bit an edge emits is its input ``u``, so it is not stored.
    tail_input : np.ndarray
        ``(2 ** m,)`` ``uint8``: the input that drives the register one step
        towards zero. For a recursive encoder this is the feedback value and
        not ``0``, which is the whole of what termination costs.
    source : np.ndarray
        ``(2, 2 ** m)`` ``int64``: ``source[u, s']`` is the state that enters
        ``s'`` on input ``u``. It exists because ``next_state[:, u]`` is a
        permutation --- two states differing only in their oldest cell get
        different feedback values, the feedback polynomial having its ``D ** m``
        term --- so the forward recursion is a *gather* through this array
        rather than a scatter. `cProfile` over a ``K = 1024`` turbo decoding put
        21.9% of self time in ``numpy.ufunc.at``, which the scatter was.
    """

    memory: int
    next_state: np.ndarray
    parity: np.ndarray
    tail_input: np.ndarray
    source: np.ndarray

    @property
    def n_states(self) -> int:
        """``2 ** memory``."""
        return int(self.next_state.shape[0])


def recursive_systematic_trellis(
    feedback: int, feedforward: int, memory: int
) -> Trellis:
    """The trellis of the rate-1/2 RSC encoder ``[1, feedforward / feedback]``.

    With register contents ``s_1 .. s_m`` (most recent first), feedback
    taps ``a`` and feedforward taps ``b``, one step on input ``u`` forms the
    feedback value ``d = u + sum_{i >= 1} a_i s_i``, emits the systematic bit
    ``u`` and the parity bit ``p = b_0 d + sum_{i >= 1} b_i s_i``, and shifts
    ``d`` in (``eq:rsc``). Setting ``u = sum_{i >= 1} a_i s_i`` makes ``d =
    0``, which is :attr:`Trellis.tail_input`: ``m`` such steps empty the
    register whatever it held.

    Parameters
    ----------
    feedback, feedforward : int
        Octal generators, the feedback one first: ``(0o7, 0o5)`` is the
        memory-2 register this repository's default fixture uses.
    memory : int
        Register length ``m``, at least one.

    Returns
    -------
    Trellis

    Raises
    ------
    ValueError
        If ``memory`` is below one, or a generator does not fit it.
    """
    if memory < 1:
        msg = f"memory must be at least 1, got {memory}"
        raise ValueError(msg)
    a = octal_taps(feedback, memory)
    b = octal_taps(feedforward, memory)
    if not a[memory]:
        msg = (
            f"the feedback generator {feedback:o} (octal) has no D**{memory} term, so "
            f"the register has less than {memory} states' worth of memory"
        )
        raise ValueError(msg)
    n_states = 1 << memory
    states = np.arange(n_states, dtype=np.int64)
    # Bit i of the state word is s_{i+1}, the cell i steps back, so the most
    # recent bit is the least significant and one step is a left shift.
    cells = (states[:, None] >> np.arange(memory)[None, :]) & 1
    feedback_sum = (cells @ a[1:].astype(np.int64)) & 1
    forward_sum = (cells @ b[1:].astype(np.int64)) & 1
    next_state = np.empty((n_states, 2), dtype=np.int64)
    parity = np.empty((n_states, 2), dtype=np.uint8)
    for u in (0, 1):
        d = (u ^ feedback_sum).astype(np.int64)
        next_state[:, u] = ((states << 1) & (n_states - 1)) | d
        parity[:, u] = ((int(b[0]) * d) ^ forward_sum).astype(np.uint8)
    source = np.empty((2, n_states), dtype=np.int64)
    for u in (0, 1):
        source[u, next_state[:, u]] = states
    return Trellis(
        memory=memory,
        next_state=next_state,
        parity=parity,
        tail_input=feedback_sum.astype(np.uint8),
        source=source,
    )


def encode_stream(
    trellis: Trellis, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run the register over ``inputs`` from the zero state.

    Parameters
    ----------
    trellis : Trellis
    inputs : np.ndarray
        The input bits, one per step.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The parity bits, one per step, and the state visited after each
        step -- ``(len(inputs),)`` and ``(len(inputs) + 1,)``, the latter
        starting at the zero state.
    """
    bits = np.asarray(inputs, dtype=np.uint8)
    parity = np.empty(bits.size, dtype=np.uint8)
    states = np.empty(bits.size + 1, dtype=np.int64)
    states[0] = 0
    for t, u in enumerate(bits):
        parity[t] = trellis.parity[states[t], u]
        states[t + 1] = trellis.next_state[states[t], u]
    return parity, states


def terminate(trellis: Trellis, message: np.ndarray) -> np.ndarray:
    """``message`` followed by the ``m`` tail inputs that empty the register.

    The tail is a function of the state the message leaves, so it is
    computed by running the register rather than declared; the encoder
    asserts the final state is zero, which is what the terminated BCJR's
    backward boundary condition requires.

    Parameters
    ----------
    trellis : Trellis
    message : np.ndarray
        ``K`` input bits.

    Returns
    -------
    np.ndarray
        ``K + m`` input bits.
    """
    bits = np.asarray(message, dtype=np.uint8)
    _, states = encode_stream(trellis, bits)
    state = int(states[-1])
    tail = np.empty(trellis.memory, dtype=np.uint8)
    for step in range(trellis.memory):
        tail[step] = trellis.tail_input[state]
        state = int(trellis.next_state[state, tail[step]])
    if state != 0:
        msg = "the tail did not return the register to the zero state"
        raise RuntimeError(msg)
    return np.concatenate([bits, tail])


# --- interleavers ---------------------------------------------------------------


def random_interleaver(length: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random permutation of ``0 .. length - 1``, drawn from ``rng``.

    ``sim/CLAUDE.md``: a generator, never a seed. The permutation is the
    whole of the randomness in a turbo code's construction, and the fixture
    records the seed the generator was made from.

    Parameters
    ----------
    length : int
        ``K``, the message length the permutation acts on.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        ``int64`` of shape ``(length,)``: position ``i`` of the interleaved
        sequence carries message bit ``permutation[i]``.
    """
    return np.asarray(rng.permutation(length), dtype=np.int64)


def inverse_permutation(permutation: np.ndarray) -> np.ndarray:
    """The permutation that undoes ``permutation``.

    Returns
    -------
    np.ndarray
        ``inverse[permutation[i]] = i``.

    Raises
    ------
    ValueError
        If the argument is not a permutation of ``0 .. n - 1``.
    """
    order = np.asarray(permutation, dtype=np.int64)
    inverse = np.full(order.size, -1, dtype=np.int64)
    if order.ndim != 1 or np.any(order < 0) or np.any(order >= order.size):
        msg = "a permutation is a one-dimensional array of 0 .. n - 1"
        raise ValueError(msg)
    inverse[order] = np.arange(order.size, dtype=np.int64)
    if np.any(inverse < 0):
        msg = "the array repeats an index, so it is not a permutation"
        raise ValueError(msg)
    return inverse


# --- the turbo code -------------------------------------------------------------


@dataclass(frozen=True)
class TurboCode:
    """Two registers, one interleaver, and the layout of the transmitted word.

    Both constituent encoders are the same :class:`Trellis`; the second is
    driven by the interleaved message. Both are terminated, so the
    transmitted word is, in this order: the ``K + m`` systematic bits the
    first encoder consumed, its ``K + m`` parity bits, the ``m`` tail inputs
    the second encoder consumed, and its ``K + m`` parity bits -- ``3 K + 4
    m`` in total. The second encoder's ``K`` message bits are already carried
    by the first encoder's systematic stream, in a different order, which is
    the whole of what the interleaver buys; its ``m`` tail inputs are not,
    so they are transmitted.

    Parameters
    ----------
    trellis : Trellis
        The constituent encoder.
    interleaver : np.ndarray
        A permutation of ``0 .. K - 1``.
    """

    trellis: Trellis
    interleaver: np.ndarray

    @property
    def message_length(self) -> int:
        """``K``."""
        return int(self.interleaver.size)

    @property
    def steps(self) -> int:
        """``K + m``: trellis steps each constituent encoder runs."""
        return self.message_length + self.trellis.memory

    @property
    def block_length(self) -> int:
        """``3 K + 4 m``: bits transmitted per message."""
        return 3 * self.message_length + 4 * self.trellis.memory

    @property
    def rate(self) -> float:
        """``K / (3 K + 4 m)``, the rate the noise variance is set from."""
        return self.message_length / self.block_length

    def slices(self) -> dict[str, slice]:
        """Where each stream sits in the transmitted word.

        Returns
        -------
        dict[str, slice]
            ``systematic`` (``K + m``), ``parity_first`` (``K + m``),
            ``tail_second`` (``m``) and ``parity_second`` (``K + m``), in
            transmission order.
        """
        steps, m = self.steps, self.trellis.memory
        return {
            "systematic": slice(0, steps),
            "parity_first": slice(steps, 2 * steps),
            "tail_second": slice(2 * steps, 2 * steps + m),
            "parity_second": slice(2 * steps + m, 3 * steps + m),
        }


def turbo_code(
    feedback: int,
    feedforward: int,
    memory: int,
    message_length: int,
    rng: np.random.Generator,
) -> TurboCode:
    """Build the code: one register, one random interleaver drawn from ``rng``.

    Parameters
    ----------
    feedback, feedforward : int
        Octal generators of the constituent RSC encoder.
    memory : int
        Register length.
    message_length : int
        ``K``.
    rng : np.random.Generator

    Returns
    -------
    TurboCode
    """
    if message_length < 1:
        msg = f"message_length must be at least 1, got {message_length}"
        raise ValueError(msg)
    return TurboCode(
        recursive_systematic_trellis(feedback, feedforward, memory),
        random_interleaver(message_length, rng),
    )


def turbo_encode(code: TurboCode, message: np.ndarray) -> np.ndarray:
    """The ``3 K + 4 m`` transmitted bits of ``message``.

    Parameters
    ----------
    code : TurboCode
    message : np.ndarray
        ``K`` bits.

    Returns
    -------
    np.ndarray
        ``uint8`` of shape ``(code.block_length,)``, laid out as
        :meth:`TurboCode.slices` says.

    Raises
    ------
    ValueError
        If ``message`` is not ``K`` bits.
    """
    bits = np.asarray(message, dtype=np.uint8)
    if bits.shape != (code.message_length,):
        msg = f"message has shape {bits.shape}, the code carries {code.message_length}"
        raise ValueError(msg)
    if np.any(bits > 1):
        msg = "a message is a 0/1 array"
        raise ValueError(msg)
    first_inputs = terminate(code.trellis, bits)
    second_inputs = terminate(code.trellis, bits[code.interleaver])
    first_parity, _ = encode_stream(code.trellis, first_inputs)
    second_parity, _ = encode_stream(code.trellis, second_inputs)
    return np.concatenate(
        [
            first_inputs,
            first_parity,
            second_inputs[code.message_length :],
            second_parity,
        ]
    )


# --- the same code as a parity-check matrix -------------------------------------


def parity_check(code: TurboCode) -> ParityCheck:
    """A parity-check matrix of the turbo code, by nullspace over GF(2).

    Every linear code is the null space of some ``H``, so a turbo instance
    is a :class:`~snakes_and_ladders.sim.ldpc.ParityCheck` and issue #340's
    decoders and enumeration oracle apply to it unchanged. The matrix is
    built by encoding the ``K`` unit messages into the generator matrix and
    taking :func:`snakes_and_ladders.sim.ldpc.null_space` of that -- the same
    GF(2) elimination the parity-check encoder runs, not a second copy of
    it. The result is dense and has nothing to do with the low-density
    ensemble, which is the point: it referees this encoder against a
    construction sharing no code with it.

    Parameters
    ----------
    code : TurboCode
        With ``3 K + 4 m`` at most :data:`MAX_PARITY_CHECK_BITS`.

    Returns
    -------
    ParityCheck

    Raises
    ------
    ValueError
        Past :data:`MAX_PARITY_CHECK_BITS`, where the dense elimination is
        refused.
    """
    if code.block_length > MAX_PARITY_CHECK_BITS:
        msg = (
            f"the dense GF(2) nullspace is refused at n = {code.block_length} > "
            f"{MAX_PARITY_CHECK_BITS}; the trellis decoders are the path at length"
        )
        raise ValueError(msg)
    return ParityCheck.from_dense(null_space(turbo_generator_matrix(code)))


def turbo_generator_matrix(code: TurboCode) -> np.ndarray:
    """The ``(K, 3 K + 4 m)`` generator: row ``i`` encodes the ``i``-th unit message.

    Encoding is GF(2)-linear -- the register's state and both outputs are
    sums of past inputs -- so the codeword of any message is the sum of the
    rows it selects, which a test asserts against :func:`turbo_encode`
    rather than assuming.

    Returns
    -------
    np.ndarray
        ``uint8``.
    """
    generator = np.zeros((code.message_length, code.block_length), dtype=np.uint8)
    unit = np.zeros(code.message_length, dtype=np.uint8)
    for i in range(code.message_length):
        unit[i] = 1
        generator[i] = turbo_encode(code, unit)
        unit[i] = 0
    return generator


# --- the declared instance ------------------------------------------------------

_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "message_length",
        "memory",
        "feedback",
        "feedforward",
        "iterations",
        "eb_n0_db",
        "frames",
    }
)


@dataclass(frozen=True)
class TurboParams:
    """Fully-specified truth for a turbo-code fixture.

    The interleaver is the seed's, not the file's: at ``K = 1024`` a written
    permutation is a file no reader could check, and it is drawn again
    instead.

    Parameters
    ----------
    message_length : int
        ``K``.
    memory : int
        Register length ``m`` of both constituent encoders.
    feedback, feedforward : int
        Their octal generators, as integers in the file's octal notation.
    seed : int
        Seed the interleaver is drawn under.
    iterations : int
        Turbo iterations the declared decoding runs.
    eb_n0_db : tuple[float, ...]
        The ``E_b / N_0`` points, in dB, this instance's error rates are
        measured at.
    frames : int
        Frames per point. Stated here rather than in a caller because a bit
        error rate at a different frame count is a different number under
        the same name, and the error bar is read off it.
    """

    message_length: int
    memory: int
    feedback: int
    feedforward: int
    seed: int
    iterations: int
    eb_n0_db: tuple[float, ...]
    frames: int

    def code(self) -> TurboCode:
        """Draw the interleaver this fixture declares and build the code.

        Returns
        -------
        TurboCode
        """
        return turbo_code(
            self.feedback,
            self.feedforward,
            self.memory,
            self.message_length,
            np.random.default_rng(self.seed),
        )

    def trellis(self) -> Trellis:
        """The constituent encoder's trellis, without drawing an interleaver."""
        return recursive_systematic_trellis(
            self.feedback, self.feedforward, self.memory
        )


def load_turbo_params(path: Path) -> TurboParams:
    """Load and validate a turbo fixture yaml.

    The generators are written in the file as octal strings (``"7"``,
    ``"15"``) and read here with base 8, which is the notation every
    reference states them in; reading them as decimal would silently build a
    different register.

    Parameters
    ----------
    path : Path
        The yaml file.

    Returns
    -------
    TurboParams
    """
    raw = load_declared(path, _REQUIRED_FIELDS)
    return TurboParams(
        message_length=int(raw["message_length"]),
        memory=int(raw["memory"]),
        feedback=int(str(raw["feedback"]), 8),
        feedforward=int(str(raw["feedforward"]), 8),
        seed=int(raw["seed"]),
        iterations=int(raw["iterations"]),
        eb_n0_db=tuple(float(value) for value in raw["eb_n0_db"]),
        frames=int(raw["frames"]),
    )
