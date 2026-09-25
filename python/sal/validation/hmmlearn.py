"""hmmlearn as an oracle and a timing reference for Baum--Welch and Viterbi (issues #975, #997).

hmmlearn's ``CategoricalHMM`` is an independent implementation of the
recursion :func:`sal.opt.hmm.baum_welch` runs, and runs only
in ``scripts/hmmlearn.py``, in a subprocess. :func:`baum_welch` fits for a
fixed number of iterations from a given start, so the two fits are compared
parameter for parameter rather than at a convergence each defines its own
way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sal.validation.runner import run

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
        {
            "observations": np.ascontiguousarray(observations, dtype=np.int64),
            "initial": np.ascontiguousarray(initial, dtype=np.float64),
            "transition": np.ascontiguousarray(transition, dtype=np.float64),
            "emission": np.ascontiguousarray(emission, dtype=np.float64),
            "n_iter": np.asarray(n_iter, dtype=np.int64),
        },
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
    family = "poisson" if "rate" in emission else "gaussian"
    result = run(
        SCRIPT,
        {
            **_family_inputs(observations, initial, transition, emission, family),
            "n_iter": np.asarray(n_iter, dtype=np.int64),
        },
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
    family = (
        "categorical"
        if "emission" in emission
        else "poisson"
        if "rate" in emission
        else "gaussian"
    )
    result = run(
        SCRIPT,
        {
            **_family_inputs(observations, initial, transition, emission, family),
            "call": np.asarray("decode"),
        },
    )
    out = result.outputs
    return ViterbiPaths(
        out["states"],
        float(out["log_probability"]),
        result.seconds,
        result.peak_bytes,
    )


def _family_inputs(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission: dict[str, np.ndarray],
    family: str,
) -> dict[str, np.ndarray]:
    """The script's inputs for ``family``, every array contiguous."""
    dtype = np.float64 if family == "gaussian" else np.int64
    return {
        "observations": np.ascontiguousarray(observations, dtype=dtype),
        "initial": np.ascontiguousarray(initial, dtype=np.float64),
        "transition": np.ascontiguousarray(transition, dtype=np.float64),
        **{
            name: np.ascontiguousarray(value, dtype=np.float64)
            for name, value in emission.items()
        },
        "family": np.asarray(family),
    }


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
    family = (
        "categorical"
        if "emission" in emission
        else "poisson"
        if "rate" in emission
        else "gaussian"
    )
    result = run(
        SCRIPT,
        {
            **_family_inputs(observations, initial, transition, emission, family),
            "call": np.asarray("score"),
        },
    )
    return Score(
        float(result.outputs["log_likelihood"]),
        result.seconds,
        result.peak_bytes,
    )
