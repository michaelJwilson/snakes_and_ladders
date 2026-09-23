"""hmmlearn as an oracle and a timing reference for Baum--Welch (issue #975).

hmmlearn's ``CategoricalHMM`` is an independent implementation of the
recursion :func:`snakes_and_ladders.opt.hmm.baum_welch` runs, and runs only
in ``scripts/hmmlearn.py``, in a subprocess. :func:`baum_welch` fits for a
fixed number of iterations from a given start, so the two fits are compared
parameter for parameter rather than at a convergence each defines its own
way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.validation.runner import run

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
        int(result.peak_bytes or 0),
    )
