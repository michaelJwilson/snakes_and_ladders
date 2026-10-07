"""hmmlearn as a timing reference for Baum--Welch (issue #975).

hmmlearn's ``CategoricalHMM`` is an independent implementation of the
recursion :func:`sal.opt.hmm.baum_welch` runs, and runs only
in ``scripts/hmmlearn.py``, in a subprocess. :func:`baum_welch` fits for a
fixed number of iterations from a given start, and returns the seconds and
peak bytes of hmmlearn's ``fit``, which
``tests/benchmarks/test_opt_external_em_bench.py`` reads. Its inputs are
:func:`sal.external.hmm_inputs.hmm_inputs`'s, the bytes
:data:`sal.external.hmm.fit` sends, so the two agree bitwise (issue #1282).
The Gaussian and Poisson fits, Viterbi and ``score`` are
:mod:`sal.external.hmm`'s calls alone since #1282, step 7.
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
