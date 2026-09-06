"""Ground truth and data generation for the discrete-HMM reference instance.

A hidden state path and an observation sequence are drawn jointly by
ancestral sampling from a declared ``(pi, A, B)``, on the same footing as
:mod:`phylo.sim.simulate`: the truth ships with the data, and the hidden
path is retained rather than discarded, so nothing downstream that needs a
labelled sequence -- Viterbi, iterated conditional modes, an RL environment
-- has to regenerate it.

Fitting lives in :mod:`phylo.opt.hmm`, which imports the truth type from
here but draws no data itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phylo.fixtures import load_declared
from phylo.numerics_rust import sample_rows

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
    n_symbols : int
        Emission alphabet size, >= 2.
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
    emission : np.ndarray
        True emission matrix, shape ``(n_states, n_symbols)``, rows summing
        to 1.
    seed : int
        Seed for ``np.random.default_rng``.
    tolerance : float
        Absolute tolerance a validation test checks simulated frequencies
        against their exact or analytic counterpart within.
    """

    n_states: int
    n_symbols: int
    sequence_length: int
    n_sequences: int
    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    seed: int
    tolerance: float


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
        n_symbols=n_symbols,
        sequence_length=sequence_length,
        n_sequences=int(raw["n_sequences"]),
        initial=initial,
        transition=transition,
        emission=emission,
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
        Emitted symbols, shape ``(n_sequences, sequence_length)``, entries
        in ``[0, n_symbols)``.
    initial, transition, emission : np.ndarray
        The truth that generated ``states`` and ``observations``.
    seed : int
        Seed used.
    """

    states: np.ndarray
    observations: np.ndarray
    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    seed: int


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
    observations = np.empty(
        (params.n_sequences, params.sequence_length), dtype=np.int64
    )
    states[:, 0] = rng.choice(
        params.n_states, size=params.n_sequences, p=params.initial
    )
    observations[:, 0] = sample_rows(rng, params.emission, states[:, 0])
    for t in range(1, params.sequence_length):
        states[:, t] = sample_rows(rng, params.transition, states[:, t - 1])
        observations[:, t] = sample_rows(rng, params.emission, states[:, t])
    return SimulatedHmmDataset(
        states=states,
        observations=observations,
        initial=params.initial,
        transition=params.transition,
        emission=params.emission,
        seed=params.seed,
    )
