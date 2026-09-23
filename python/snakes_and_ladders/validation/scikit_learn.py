"""scikit-learn as an oracle and a timing reference for Gaussian mixture EM (issue #975).

``GaussianMixture`` with a diagonal covariance is an independent
implementation of the EM :func:`snakes_and_ladders.opt.mixture.expectation_maximization`
runs on a one-dimensional mixture, and runs only in
``scripts/scikit_learn.py``, in a subprocess. :func:`expectation_maximization`
fits for a fixed number of iterations from a given start with no covariance
regularization, so the two fits are compared parameter for parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.validation.runner import run

#: The script this adapter runs.
SCRIPT = "scikit_learn"


@dataclass(frozen=True)
class Fit:
    """scikit-learn's parameters after ``iterations``: weights, means, standard deviations."""

    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    iterations: int
    #: Wall seconds of ``fit`` alone.
    seconds: float


def expectation_maximization(
    observations: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    n_iter: int,
) -> Fit:
    """``n_iter`` scikit-learn EM iterations from the given start."""
    result = run(
        SCRIPT,
        {
            "observations": np.ascontiguousarray(observations, dtype=np.float64),
            "weights": np.ascontiguousarray(weights, dtype=np.float64),
            "mean": np.ascontiguousarray(mean, dtype=np.float64),
            "scale": np.ascontiguousarray(scale, dtype=np.float64),
            "n_iter": np.asarray(n_iter, dtype=np.int64),
        },
    )
    out = result.outputs
    return Fit(
        out["weights"],
        out["mean"],
        out["scale"],
        int(out["iterations"]),
        result.seconds,
    )
