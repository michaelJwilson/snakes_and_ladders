"""hmmlearn as an oracle and a timing reference for Baum--Welch and Viterbi (issues #975, #997).

hmmlearn's ``CategoricalHMM`` is an independent implementation of the
recursion :func:`sal.opt.hmm.baum_welch` runs, and runs only
in ``scripts/hmmlearn.py``, in a subprocess. :func:`baum_welch` fits for a
fixed number of iterations from a given start, so the two fits are compared
parameter for parameter rather than at a convergence each defines its own
way. Every call's inputs are :func:`sal.external.hmm_inputs.hmm_inputs`'s,
the bytes :mod:`sal.external.hmm` sends, so the two paths agree bitwise
(issue #1282).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sal.external.hmm_inputs import hmm_inputs
from sal.external.runner import run

#: The script this adapter runs.
SCRIPT = "hmmlearn"


@dataclass(frozen=True)
class Fit:
    """hmmlearn's parameters after ``iterations``, as probabilities."""

    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    iterations: int
    #: Wall seconds of ``fit`` alone.
    seconds: float
    #: Peak resident bytes ``fit`` added.
    peak_bytes: int


def baum_welch(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
    n_iter: int,
) -> Fit:
    """``n_iter`` hmmlearn iterations from the given start, in log space."""
    result = run(
        SCRIPT,
        hmm_inputs(
            observations,
            initial,
            transition,
            {"emission": emission},
            call="fit",
            n_iter=n_iter,
        ),
    )
    out = result.outputs
    return Fit(
        out["initial"],
        out["transition"],
        out["emission"],
        int(out["iterations"]),
        result.seconds,
        result.peak_bytes,
    )


@dataclass(frozen=True)
class FamilyFit:
    """hmmlearn's parameters after ``iterations`` for a Gaussian or Poisson HMM."""

    initial: np.ndarray
    transition: np.ndarray
    #: The family's parameters: ``mean`` and ``variance``, or ``rate``.
    emission: dict[str, np.ndarray]
    iterations: int
    #: Wall seconds of ``fit`` alone.
    seconds: float
    #: Peak resident bytes ``fit`` added.
    peak_bytes: int


def family_baum_welch(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission: dict[str, np.ndarray],
    n_iter: int,
) -> FamilyFit:
    """``n_iter`` hmmlearn iterations of a Gaussian or Poisson HMM (issue #997).

    ``emission`` is ``{"mean", "variance"}`` for a one-channel Gaussian and
    ``{"rate"}`` for a Poisson; the family is read from which it is. Every
    prior and floor hmmlearn applies is zero, so the update is the
    maximum-likelihood one.
    """
    result = run(
        SCRIPT,
        hmm_inputs(
            observations, initial, transition, emission, call="fit", n_iter=n_iter
        ),
    )
    out = result.outputs
    return FamilyFit(
        out["initial"],
        out["transition"],
        {name: out[name] for name in emission},
        int(out["iterations"]),
        result.seconds,
        result.peak_bytes,
    )


@dataclass(frozen=True)
class ViterbiPaths:
    """hmmlearn's Viterbi path and its joint log-probability."""

    states: np.ndarray
    log_probability: float
    #: Wall seconds of ``decode`` alone.
    seconds: float
    #: Peak resident bytes ``decode`` added.
    peak_bytes: int


def viterbi(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission: dict[str, np.ndarray],
) -> ViterbiPaths:
    """hmmlearn's Viterbi at the given parameters (issue #997).

    ``emission`` is as :func:`family_baum_welch` takes it, or
    ``{"emission": probabilities}`` for a categorical HMM.
    """
    result = run(
        SCRIPT,
        hmm_inputs(observations, initial, transition, emission, call="decode"),
    )
    out = result.outputs
    return ViterbiPaths(
        out["states"],
        float(out["log_probability"]),
        result.seconds,
        result.peak_bytes,
    )


@dataclass(frozen=True)
class Score:
    """hmmlearn's summed log-likelihood at given parameters."""

    log_likelihood: float
    #: Wall seconds of ``score`` alone.
    seconds: float
    #: Peak resident bytes ``score`` added.
    peak_bytes: int


def score(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission: dict[str, np.ndarray],
) -> Score:
    """hmmlearn's ``score`` at the given parameters (issue #997); ``emission`` as :func:`viterbi` takes it."""
    result = run(
        SCRIPT,
        hmm_inputs(observations, initial, transition, emission, call="score"),
    )
    return Score(
        float(result.outputs["log_likelihood"]),
        result.seconds,
        result.peak_bytes,
    )
