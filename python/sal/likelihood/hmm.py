"""An HMM's evidence and its most probable path at given parameters (issues #997, #1059).

Evaluators, not fits: :func:`hmm_log_likelihood` is the summed evidence of
every sequence and :func:`viterbi` the max-product path, both at parameters
the caller holds, with no gradient taken. Each has a compiled route
(``oxisal.hmm_score``, ``oxisal.hmm_viterbi``) over the families
:func:`~sal.opt.hmm.estimation.compiled_family` names and a NumPy or torch
oracle for the rest. They sat in :mod:`sal.opt.hmm.estimation` beside the EM
drivers until #1059 moved them to the directory of evaluators.
"""

from __future__ import annotations

import numpy as np
import torch

from sal import oxisal
from sal.backend import Backend, refuse_backend
from sal.emissions import EmissionFamily
from sal.opt.hmm.estimation import compiled_family
from sal.opt.hmm.forward import forward_log_likelihood_from_density


def viterbi(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    backend: Backend = Backend.RUST,
) -> tuple[np.ndarray, float]:
    """The most probable hidden path of every sequence, and their total log-probability.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n_sequences, length)``: symbols, real values or counts, as
        ``emissions`` scores them.
    log_initial, log_transition : torch.Tensor
        Log-probabilities, ``(m,)`` and ``(m, m)``.
    emissions : EmissionFamily
        The emission family.
    backend : Backend
        :data:`~sal.backend.Backend.RUST`, the default,
        decodes the sequences in parallel in ``oxisal.
        hmm_viterbi`` where ``emissions`` is exactly a categorical, a
        one-channel Gaussian or a count family; any other family, and
        :data:`~sal.backend.Backend.PYTHON`, take the NumPy
        recursion here, which is the oracle (issue #997).

    Returns
    -------
    tuple[np.ndarray, float]
        The paths, ``(n_sequences, length)`` ``int64``, and the sum over
        sequences of each path's joint log-probability. A tie goes to the
        lower state.
    """
    refuse_backend("viterbi", backend, (Backend.PYTHON, Backend.RUST))
    values = np.asarray(observations)
    compiled = compiled_family(emissions)
    if backend is Backend.RUST and compiled is not None and values.ndim == 2:
        name, parameters = compiled
        # Symbols and counts are read as the int64 NumPy holds them.
        dtype = np.int64 if np.issubdtype(values.dtype, np.integer) else np.float64
        states, log_probability = oxisal.hmm_viterbi(
            np.ascontiguousarray(values, dtype=dtype),
            np.ascontiguousarray(log_initial.detach().numpy(), dtype=np.float64),
            np.ascontiguousarray(
                log_transition.detach().numpy(), dtype=np.float64
            ).reshape(-1),
            name,
            parameters,
        )
        return states.reshape(values.shape), float(log_probability)
    emit = emissions.log_density(
        torch.as_tensor(values, dtype=emissions.observation_dtype)
    ).numpy()
    kernel = log_transition.detach().numpy()
    n_sequences, length = values.shape
    delta = log_initial.detach().numpy() + emit[:, 0]
    back = np.empty((n_sequences, length, delta.shape[1]), dtype=np.int64)
    for t in range(1, length):
        scores = delta[:, :, None] + kernel[None]
        back[:, t] = np.argmax(scores, axis=1)
        delta = np.take_along_axis(scores, back[:, t][:, None, :], axis=1)[:, 0]
        delta = delta + emit[:, t]
    states = np.empty((n_sequences, length), dtype=np.int64)
    states[:, -1] = np.argmax(delta, axis=1)
    for t in range(length - 1, 0, -1):
        states[:, t - 1] = np.take_along_axis(back[:, t], states[:, t, None], axis=1)[
            :, 0
        ]
    return states, float(delta.max(axis=1).sum())


def hmm_log_likelihood(
    observations: np.ndarray,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    emissions: EmissionFamily,
    backend: Backend = Backend.RUST,
) -> float:
    """The summed log-likelihood of every sequence at given parameters, with no gradient.

    The number hmmlearn's ``score`` and :func:`forward_log_likelihood_from_density`
    report. Returned as a ``float``, since no gradient is taken: a caller that
    needs one differentiates :func:`forward_log_likelihood_from_density`.

    Parameters
    ----------
    observations : np.ndarray
        Shape ``(n_sequences, length)``, as ``emissions`` scores them.
    log_initial, log_transition : torch.Tensor
        Log-probabilities, ``(m,)`` and ``(m, m)``.
    emissions : EmissionFamily
        The emission family.
    backend : Backend
        :data:`~sal.backend.Backend.RUST`, the default, runs
        the scaled forward pass over the sequences in parallel in
        ``oxisal.hmm_score`` for the families :func:`viterbi`
        compiles, and forms no ``(n_sequences, length, m)`` array;
        :data:`~sal.backend.Backend.PYTHON`, and any other
        family, take :func:`forward_log_likelihood_from_density`, the oracle
        (issue #997).

    Returns
    -------
    float
    """
    refuse_backend("hmm_log_likelihood", backend, (Backend.PYTHON, Backend.RUST))
    values = np.asarray(observations)
    compiled = compiled_family(emissions)
    if backend is Backend.RUST and compiled is not None and values.ndim == 2:
        name, parameters = compiled
        dtype = np.int64 if np.issubdtype(values.dtype, np.integer) else np.float64
        return float(
            oxisal.hmm_score(
                np.ascontiguousarray(values, dtype=dtype),
                np.ascontiguousarray(log_initial.detach().numpy(), dtype=np.float64),
                np.ascontiguousarray(
                    log_transition.detach().numpy(), dtype=np.float64
                ).reshape(-1),
                name,
                parameters,
            )
        )
    with torch.no_grad():
        return float(
            forward_log_likelihood_from_density(
                emissions.log_density(
                    torch.as_tensor(values, dtype=emissions.observation_dtype)
                ),
                log_initial,
                log_transition,
            )
        )
