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


def path_log_probability(
    params: HmmParams, path: np.ndarray, observations: np.ndarray
) -> float:
    """``log P(path, observations)`` from the definition, term by term.

    Written out rather than recursed, so it can referee a recursion.
    """
    log_initial = np.log(params.initial)
    log_transition = np.log(params.transition)
    log_emission = np.log(params.emission)

    total = float(log_initial[path[0]] + log_emission[path[0], observations[0]])
    for step in range(1, len(path)):
        total += float(
            log_transition[path[step - 1], path[step]]
            + log_emission[path[step], observations[step]]
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
        Supplies ``initial``, ``transition`` and ``emission``. The fixture's
        ``sequence_length`` is not consulted; the length of ``observations``
        is, so a shorter slice can be enumerated than the fixture declares.
    observations : np.ndarray
        Symbol indices, shape ``(T,)``.
    max_paths : int
        Refuse above this many paths.

    Raises
    ------
    ValueError
        If ``observations`` is empty, names a symbol outside the alphabet, or
        would need more than ``max_paths`` paths.
    """
    length = int(observations.shape[0])
    if length == 0:
        msg = "observations must be non-empty"
        raise ValueError(msg)
    if int(observations.min()) < 0 or int(observations.max()) >= params.n_symbols:
        msg = (
            f"observations must lie in [0, {params.n_symbols}), got "
            f"[{int(observations.min())}, {int(observations.max())}]"
        )
        raise ValueError(msg)
    refuse_oversized(
        params.n_states**length,
        what=f"{params.n_states}**{length} hidden paths",
        limit=max_paths,
    )

    log_joint: list[float] = []
    paths: list[np.ndarray] = []
    for candidate in itertools.product(range(params.n_states), repeat=length):
        path = np.array(candidate, dtype=np.int64)
        paths.append(path)
        log_joint.append(path_log_probability(params, path, observations))

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
