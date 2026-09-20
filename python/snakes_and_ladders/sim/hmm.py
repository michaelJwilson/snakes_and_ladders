"""Ground truth and data generation for the discrete-HMM reference instance.

A hidden state path and an observation sequence are drawn jointly by
ancestral sampling from a declared ``(pi, A, B)``, on the same footing as
:mod:`snakes_and_ladders.sim.simulate`: the truth ships with the data, and the hidden
path is retained rather than discarded, so nothing downstream that needs a
labelled sequence -- Viterbi, iterated conditional modes, an RL environment
-- has to regenerate it.

Fitting lives in :mod:`snakes_and_ladders.opt.hmm`, which imports the truth type from
here but draws no data itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from snakes_and_ladders.emissions import CategoricalEmission, EmissionFamily
from snakes_and_ladders.numerics import sample_rows
from snakes_and_ladders.ragged import Ragged

_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "lengths",
        "n_states",
        "n_symbols",
        "initial",
        "transition",
        "emission",
        "tolerance",
    }
)


@dataclass(frozen=True)
class HmmParams:
    """Fully-specified truth for an HMM fixture.

    Parameters
    ----------
    n_states : int
        Hidden states, >= 2.
    sequence_length : int
        Observations per sequence, >= 2. A length-1 sequence carries no
        transition and would leave the transition matrix unidentifiable.
    n_sequences : int
        Independent sequences simulated from the truth.
    initial : np.ndarray
        True initial distribution, shape ``(n_states,)``.
    transition : np.ndarray
        True transition matrix, shape ``(n_states, n_states)``, rows summing
        to 1.
    emissions : EmissionFamily
        What each hidden state emits. A matrix of symbol probabilities is one
        family among several (:mod:`snakes_and_ladders.emissions`); the field
        is the family rather than the matrix so a fixture whose observations
        are continuous is the same type as one whose observations are
        symbols.
    seed : int
        Seed for ``np.random.default_rng``.
    tolerance : float
        Absolute tolerance a validation test checks simulated frequencies
        against their exact or analytic counterpart within.
    """

    n_states: int
    #: One length per chain, and the only declaration of the batch's shape.
    #: `n_sequences` chains of a shared `sequence_length` is the case where
    #: these are equal, and is read off them rather than stored beside them
    #: (issue #666): a second field for a derived fact is a field that can
    #: disagree, and on a ragged batch there is no shared length for it to
    #: hold.
    lengths: tuple[int, ...]
    initial: np.ndarray
    transition: np.ndarray
    emissions: EmissionFamily
    seed: int
    tolerance: float

    @property
    def segment_lengths(self) -> tuple[int, ...]:
        """One length per chain."""
        return self.lengths

    @property
    def n_sequences(self) -> int:
        """How many chains the batch holds."""
        return len(self.lengths)

    @property
    def rectangular(self) -> bool:
        """Whether every chain is the same length."""
        return len(set(self.lengths)) == 1

    @property
    def sequence_length(self) -> int:
        """The length every chain shares.

        Raises
        ------
        ValueError
            Where the chains differ. A caller asking for *the* length of a
            ragged batch is asking a question with no answer, and returning
            the longest --- which an earlier draft of this did --- is the kind
            of quiet wrong number the segmentation exists to make impossible.
        """
        if not self.rectangular:
            msg = (
                f"the chains have lengths {self.lengths} and do not share one; "
                "read `segment_lengths`, or `n_sequences` for how many there are"
            )
            raise ValueError(msg)
        return self.lengths[0]

    @property
    def emission(self) -> np.ndarray:
        """The emission matrix, for a fixture whose emissions are categorical.

        Kept because the categorical matrix is what a reader of a discrete
        fixture means by "the emission", and every test written against one
        says so. A family that is not a matrix has no such reading and this
        raises rather than inventing one.

        Raises
        ------
        TypeError
            If the emission family is not categorical.
        """
        return _categorical(self.emissions).matrix.numpy()

    @property
    def n_symbols(self) -> int:
        """Emission alphabet size, for a categorical fixture.

        Raises
        ------
        TypeError
            If the emission family is not categorical.
        """
        return _categorical(self.emissions).n_symbols

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from an HMM fixture's declared mapping.

        ``declared`` is the mapping
        :func:`snakes_and_ladders.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        Raises
        ------
        ValueError
            If a required field is missing, a size is too small to identify the
            parameters, or a distribution has the wrong shape or does not sum
            to 1.
        """
        n_states = int(declared["n_states"])
        n_symbols = int(declared["n_symbols"])
        # Both guards predate the `lengths` spelling and were lost with the field
        # that was beside them (#667). They are not shape checks: a one-state chain
        # has no transition to identify and a one-symbol alphabet carries no
        # information, and both declare arrays that are internally consistent, so
        # `_stochastic` below passes them and the fixture is accepted.
        for name, size in (("n_states", n_states), ("n_symbols", n_symbols)):
            if size < 2:
                msg = f"{path}: {name} must be >= 2, got {size}"
                raise ValueError(msg)
        # One spelling. `n_sequences` chains of a shared `sequence_length` is the
        # equal-length case of `lengths`, so a fixture writes the lengths and the
        # loader reads them; there is nothing to keep consistent (issue #666).
        lengths = tuple(int(one) for one in declared["lengths"])
        if not lengths:
            msg = f"{path}: a batch needs at least one chain, got none"
            raise ValueError(msg)
        sequence_length = min(lengths)
        if sequence_length < 2:
            msg = (
                f"{path}: every chain carries at least 2 positions, got {lengths}; "
                "one position is an initial distribution and no transition"
            )
            raise ValueError(msg)

        initial = _stochastic(declared["initial"], (n_states,), path, "initial")
        transition = _stochastic(
            declared["transition"], (n_states, n_states), path, "transition"
        )
        emission = _stochastic(
            declared["emission"], (n_states, n_symbols), path, "emission"
        )

        return cls(
            n_states=n_states,
            lengths=lengths,
            initial=initial,
            transition=transition,
            emissions=CategoricalEmission(emission),
            seed=int(declared["seed"]),
            tolerance=float(declared["tolerance"]),
        )


def _categorical(family: EmissionFamily) -> CategoricalEmission:
    """The family as a categorical one, or a refusal naming what it is."""
    if not isinstance(family, CategoricalEmission):
        msg = (
            f"emission family {type(family).__name__} has no emission matrix "
            f"and no alphabet; read its named_parameters() instead"
        )
        raise TypeError(msg)
    return family


def _stochastic(
    raw: object, shape: tuple[int, ...], path: Path, name: str
) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != shape:
        msg = f"{path}: {name} has shape {values.shape}, expected {shape}"
        raise ValueError(msg)
    sums = values.sum(axis=-1)
    if not np.allclose(sums, 1.0):
        msg = f"{path}: {name} rows sum to {sums.tolist()}, expected 1.0"
        raise ValueError(msg)
    return values


@dataclass(frozen=True)
class SimulatedHmmDataset:
    """A simulated HMM dataset together with the hidden path and its truth.

    Ground truth ships with the data: a dataset without the hidden path and
    the generating ``(initial, transition, emission, seed)`` is not
    validation-usable.

    Parameters
    ----------
    states : np.ndarray
        Hidden states, shape ``(n_sequences, sequence_length)``, entries in
        ``[0, n_states)``.
    observations : np.ndarray
        Emitted observations, shape ``(n_sequences, sequence_length)``. Symbol
        indices for a categorical family, real values for a continuous one.
    initial, transition : np.ndarray
        The transition truth that generated ``states``.
    emissions : EmissionFamily
        The emission truth that generated ``observations``.
    seed : int
        Seed used.
    lengths : tuple[int, ...] | None
        The segment lengths where they differ, and `None` where they do not.
        Where it is given, `states` and `observations` are **flat** --- the
        segments end to end --- because there is no rectangle to put them in;
        `batch` is the carrier that reads them (issue #666).
    """

    states: np.ndarray
    observations: np.ndarray
    initial: np.ndarray
    transition: np.ndarray
    emissions: EmissionFamily
    seed: int
    lengths: tuple[int, ...]

    @property
    def rectangular(self) -> bool:
        """Whether every chain is the same length, and so whether the arrays are 2-D."""
        return len(set(self.lengths)) == 1

    @property
    def batch(self) -> Ragged:
        """The observations as segments, whatever shape they are stored in."""
        flat = self.observations
        if self.rectangular:
            flat = flat.reshape((-1, *flat.shape[2:]))
        return Ragged(flat, self.lengths)

    @property
    def emission(self) -> np.ndarray:
        """The emission matrix, for a categorical dataset.

        Raises
        ------
        TypeError
            If the emission family is not categorical.
        """
        return _categorical(self.emissions).matrix.numpy()


def simulate_sequences(
    params: HmmParams, rng: np.random.Generator | None = None
) -> SimulatedHmmDataset:
    """Draw hidden state paths and observation sequences by ancestral sampling.

    Parameters
    ----------
    params : HmmParams
        The generating truth.
    rng : np.random.Generator | None
        Generator to draw from. ``None`` builds one from ``params.seed``,
        which is the stream every fixture was drawn on; a caller drawing an
        ensemble passes its own (issue #829).

    Returns
    -------
    SimulatedHmmDataset
        The hidden paths, the emitted observations, and the generating
        truth.
    """
    rng = np.random.default_rng(params.seed) if rng is None else rng
    # Segments of one length are drawn together, which is what keeps the draw
    # vectorized. Where every chain is the same length --- every fixture that
    # predates #666 --- there is one group, the calls below are the calls this
    # simulator has always made, and the stream is therefore the same one: a
    # seeded fixture's data does not move because the simulator learned to
    # segment. A ragged batch is several groups, in declared length order.
    lengths = params.segment_lengths
    order: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        order.setdefault(length, []).append(index)

    drawn_states: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * len(lengths)
    drawn_values: list[np.ndarray] = [np.empty(0)] * len(lengths)
    for length, members in order.items():
        block = np.empty((len(members), length), dtype=np.int64)
        columns: list[np.ndarray] = []
        # Every chain in the group restarts here, at the initial distribution.
        block[:, 0] = rng.choice(params.n_states, size=len(members), p=params.initial)
        columns.append(params.emissions.sample(block[:, 0], rng))
        for t in range(1, length):
            block[:, t] = sample_rows(rng, params.transition, block[:, t - 1])
            columns.append(params.emissions.sample(block[:, t], rng))
        values = np.stack(columns, axis=1)
        for row, index in enumerate(members):
            drawn_states[index] = block[row]
            drawn_values[index] = values[row]

    if params.rectangular:
        states = np.stack(drawn_states, axis=0)
        observations = np.stack(drawn_values, axis=0)
    else:
        states = np.concatenate(drawn_states)
        observations = np.concatenate(drawn_values)
    return SimulatedHmmDataset(
        states=states,
        observations=observations,
        initial=params.initial,
        transition=params.transition,
        emissions=params.emissions,
        seed=params.seed,
        lengths=params.lengths,
    )
