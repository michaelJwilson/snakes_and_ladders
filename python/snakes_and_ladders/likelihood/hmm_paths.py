"""Every hidden path, enumerated: the oracle two decoders are separated by.

Exponential in the sequence length and deliberately so, on the same footing
as :mod:`snakes_and_ladders.likelihood.brute_force`: it shares no recursion with the
forward algorithm in :mod:`snakes_and_ladders.opt.hmm`, so agreement between them is
evidence rather than a tautology.

**What it exists to separate.** An HMM admits two different answers to "which
hidden path produced this?", and they are not approximations of each other:

* **Viterbi decoding** returns the single path of highest joint probability
  ``P(path, observations)`` --- one object, maximized as a whole.
* **Posterior decoding** returns, at each site independently, the state of
  highest marginal probability ``P(state_t | observations)``.

Where the two disagree, the posterior-decoded sequence can be a path the
model assigns *zero* probability to, because nothing constrains consecutive
choices to be compatible. Where they agree --- which is most fixtures --- a
decoder that computes one and reports the other passes every test. That is
the reason :func:`snakes_and_ladders.sim.canonical.ambiguous_hmm` exists and the reason
this enumeration does.

Both answers are read off the same enumeration here, so neither is trusted to
validate the other.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.enumeration import (
    MAX_ENUMERABLE_CONFIGURATIONS,
    refuse_oversized,
)
from snakes_and_ladders.sim.hmm import HmmParams

#: A path is one configuration, so this is
#: :data:`snakes_and_ladders.enumeration.MAX_ENUMERABLE_CONFIGURATIONS` under the name the
#: callers here use. Kept as an alias rather than a second number, so the two
#: cannot drift (issue #230).
MAX_ENUMERABLE_PATHS = MAX_ENUMERABLE_CONFIGURATIONS


@dataclass(frozen=True)
class PathEnumeration:
    """Both decodings of one observation sequence, and the evidence.

    Parameters
    ----------
    viterbi : np.ndarray
        The single highest-joint-probability path, shape ``(T,)``.
    viterbi_log_probability : float
        ``log P(viterbi, observations)``. A joint, not a conditional.
    posterior : np.ndarray
        Single-site marginals ``P(state_t | observations)``, shape
        ``(T, n_states)``, each row summing to 1.
    posterior_path : np.ndarray
        Per-site ``argmax`` of ``posterior``, shape ``(T,)``. Not in general
        a path the model favours, and not in general the Viterbi path.
    log_likelihood : float
        ``log P(observations)``, summed over every path.
    """

    viterbi: np.ndarray
    viterbi_log_probability: float
    posterior: np.ndarray
    posterior_path: np.ndarray
    log_likelihood: float

    def decoders_agree(self) -> bool:
        """Whether the two decodings coincide site for site."""
        return bool(np.array_equal(self.viterbi, self.posterior_path))


def emission_log_density(params: HmmParams, observations: np.ndarray) -> np.ndarray:
    """Score every observation under every hidden state.

    Parameters
    ----------
    params : HmmParams
        Supplies the emission family.
    observations : np.ndarray
        One sequence, shape ``(T,)``.

    Returns
    -------
    np.ndarray
        Shape ``(T, n_states)``. A log-probability for a categorical family
        and a log *density* for a continuous one, so entries may be positive
        and so may the evidence assembled from them.
    """
    family = params.emissions
    return np.asarray(
        family.log_density(
            torch.as_tensor(observations, dtype=family.observation_dtype)
        ).numpy()
    )


def path_log_probability(
    params: HmmParams, path: np.ndarray, observations: np.ndarray
) -> float:
    """``log P(path, observations)`` from the definition, term by term.

    Written out rather than recursed, so it can referee a recursion.
    """
    return _path_log_probability(
        np.log(params.initial),
        np.log(params.transition),
        emission_log_density(params, observations),
        path,
    )


def _path_log_probability(
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_density: np.ndarray,
    path: np.ndarray,
) -> float:
    """One path's joint log-probability from pre-computed per-site scores."""
    total = float(log_initial[path[0]] + log_density[0, path[0]])
    for step in range(1, len(path)):
        total += float(
            log_transition[path[step - 1], path[step]] + log_density[step, path[step]]
        )
    return total


def enumerate_hidden_paths(
    params: HmmParams,
    observations: np.ndarray,
    *,
    max_paths: int = MAX_ENUMERABLE_PATHS,
) -> PathEnumeration:
    """Both decodings, by summing and maximizing over all ``k ** T`` paths.

    Parameters
    ----------
    params : HmmParams
        Supplies ``initial``, ``transition`` and ``emissions``. The fixture's
        ``sequence_length`` is not consulted; the length of ``observations``
        is, so a shorter slice can be enumerated than the fixture declares.
    observations : np.ndarray
        One observation sequence, shape ``(T,)``. What an entry means is the
        emission family's to say: a symbol index for a categorical family, a
        real value for a continuous one.
    max_paths : int
        Refuse above this many paths.

    Raises
    ------
    ValueError
        If ``observations`` is empty, lies outside the emission family's
        support, or would need more than ``max_paths`` paths.

    Notes
    -----
    ``log_likelihood`` is bounded above by zero only where the emission family
    is discrete. For a continuous family the terms summed here are densities,
    so the evidence is a density too and may exceed 1.
    """
    length = int(observations.shape[0])
    if length == 0:
        msg = "observations must be non-empty"
        raise ValueError(msg)
    params.emissions.validate(observations)
    refuse_oversized(
        params.n_states**length,
        what=f"{params.n_states}**{length} hidden paths",
        limit=max_paths,
    )

    log_initial = np.log(params.initial)
    log_transition = np.log(params.transition)
    log_density = emission_log_density(params, observations)

    log_joint: list[float] = []
    paths: list[np.ndarray] = []
    for candidate in itertools.product(range(params.n_states), repeat=length):
        path = np.array(candidate, dtype=np.int64)
        paths.append(path)
        log_joint.append(
            _path_log_probability(log_initial, log_transition, log_density, path)
        )

    joint = np.array(log_joint)
    shift = float(joint.max())
    weights = np.exp(joint - shift)
    log_likelihood = shift + float(np.log(weights.sum()))

    posterior = np.zeros((length, params.n_states))
    for path, weight in zip(paths, weights, strict=True):
        posterior[np.arange(length), path] += weight
    posterior /= posterior.sum(axis=1, keepdims=True)

    best = int(joint.argmax())
    return PathEnumeration(
        viterbi=paths[best],
        viterbi_log_probability=float(joint[best]),
        posterior=posterior,
        posterior_path=posterior.argmax(axis=1).astype(np.int64),
        log_likelihood=log_likelihood,
    )
