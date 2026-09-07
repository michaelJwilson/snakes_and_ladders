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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snakes_and_ladders.emissions import CategoricalEmission, EmissionFamily
from snakes_and_ladders.fixtures import load_declared
from snakes_and_ladders.numerics_rust import sample_rows

_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "n_sequences",
        "sequence_length",
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
    sequence_length: int
    n_sequences: int
    initial: np.ndarray
    transition: np.ndarray
    emissions: EmissionFamily
    seed: int
    tolerance: float

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


def _categorical(family: EmissionFamily) -> CategoricalEmission:
    """The family as a categorical one, or a refusal naming what it is."""
    if not isinstance(family, CategoricalEmission):
        msg = (
            f"emission family {type(family).__name__} has no emission matrix "
            f"and no alphabet; read its named_parameters() instead"
        )
        raise TypeError(msg)
    return family


def load_hmm_params(path: Path) -> HmmParams:
    """Load and validate an HMM fixture yaml.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    HmmParams
        The parsed, validated truth.

    Raises
    ------
    ValueError
        If a required field is missing, a size is too small to identify the
        parameters, or a distribution has the wrong shape or does not sum
        to 1.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)

    n_states = int(raw["n_states"])
    n_symbols = int(raw["n_symbols"])
    sequence_length = int(raw["sequence_length"])
    if n_states < 2:
        msg = f"{path}: n_states must be >= 2, got {n_states}"
        raise ValueError(msg)
    if n_symbols < 2:
        msg = f"{path}: n_symbols must be >= 2, got {n_symbols}"
        raise ValueError(msg)
    if sequence_length < 2:
        msg = f"{path}: sequence_length must be >= 2, got {sequence_length}"
        raise ValueError(msg)

    initial = _stochastic(raw["initial"], (n_states,), path, "initial")
    transition = _stochastic(
        raw["transition"], (n_states, n_states), path, "transition"
    )
    emission = _stochastic(raw["emission"], (n_states, n_symbols), path, "emission")

    return HmmParams(
        n_states=n_states,
        sequence_length=sequence_length,
        n_sequences=int(raw["n_sequences"]),
        initial=initial,
        transition=transition,
        emissions=CategoricalEmission(emission),
        seed=int(raw["seed"]),
        tolerance=float(raw["tolerance"]),
    )


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
    """

    states: np.ndarray
    observations: np.ndarray
    initial: np.ndarray
    transition: np.ndarray
    emissions: EmissionFamily
    seed: int

    @property
    def emission(self) -> np.ndarray:
        """The emission matrix, for a categorical dataset.

        Raises
        ------
        TypeError
            If the emission family is not categorical.
        """
        return _categorical(self.emissions).matrix.numpy()


def simulate_sequences(params: HmmParams) -> SimulatedHmmDataset:
    """Draw hidden state paths and observation sequences by ancestral sampling.

    Parameters
    ----------
    params : HmmParams
        The generating truth.

    Returns
    -------
    SimulatedHmmDataset
        The hidden paths, the emitted observations, and the generating
        truth.
    """
    rng = np.random.default_rng(params.seed)
    states = np.empty((params.n_sequences, params.sequence_length), dtype=np.int64)
    columns: list[np.ndarray] = []
    states[:, 0] = rng.choice(
        params.n_states, size=params.n_sequences, p=params.initial
    )
    columns.append(params.emissions.sample(states[:, 0], rng))
    for t in range(1, params.sequence_length):
        states[:, t] = sample_rows(rng, params.transition, states[:, t - 1])
        columns.append(params.emissions.sample(states[:, t], rng))
    return SimulatedHmmDataset(
        states=states,
        observations=np.stack(columns, axis=1),
        initial=params.initial,
        transition=params.transition,
        emissions=params.emissions,
        seed=params.seed,
    )
