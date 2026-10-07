"""The bytes hmmlearn receives for an HMM (issue #1282, step 5).

One construction, read by :data:`sal.external.hmm.fit`,
:func:`~sal.external.hmm.viterbi` and
:func:`~sal.external.hmm.forward_log_likelihood`, and by the adapter's
:func:`sal.validation.hmmlearn.baum_welch`, so a categorical fit from either
path is the other's bitwise. NumPy alone, so the adapter and the script import no torch through
it.

The family and the call travel as ``int64`` codes, an index into
:data:`FAMILIES` and :data:`CALLS`: a session sends ``float64``, ``int64`` or
``bool`` arrays only (:func:`sal.external.transport.admit`), and a string is
none of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

#: The emission families the script builds, by code.
FAMILIES = ("categorical", "gaussian", "poisson")

#: The script's calls, by code: Baum-Welch, Viterbi, and the forward log-likelihood.
CALLS = ("fit", "decode", "score")


def family_of(emission: Mapping[str, np.ndarray]) -> str:
    """The family ``emission`` parameterizes: ``emission``, ``rate``, or ``mean`` and ``variance``."""
    if "emission" in emission:
        return "categorical"
    return "poisson" if "rate" in emission else "gaussian"


def hmm_inputs(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    emission: Mapping[str, np.ndarray],
    *,
    call: str,
    n_iter: int | None = None,
    tolerance: float | None = None,
    lengths: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """The script's inputs, every array contiguous.

    ``observations`` is ``(n_sequences, length)``, or the segments end to end,
    ``(total,)``, with ``lengths``. ``initial`` and ``transition`` are
    probabilities; ``emission`` is as :func:`family_of` reads it. ``n_iter``
    and ``tolerance`` are read by ``fit`` alone: ``tolerance`` is relative, as
    :func:`sal.opt.em.em_loop` tests it, and without it hmmlearn runs
    ``n_iter`` iterations whatever the change.
    """
    family = family_of(emission)
    dtype = np.float64 if family == "gaussian" else np.int64
    inputs = {
        "observations": np.ascontiguousarray(observations, dtype=dtype),
        "initial": np.ascontiguousarray(initial, dtype=np.float64),
        "transition": np.ascontiguousarray(transition, dtype=np.float64),
        **{
            name: np.ascontiguousarray(value, dtype=np.float64)
            for name, value in emission.items()
        },
        "family": np.asarray(FAMILIES.index(family), dtype=np.int64),
        "call": np.asarray(CALLS.index(call), dtype=np.int64),
    }
    if n_iter is not None:
        inputs["n_iter"] = np.asarray(n_iter, dtype=np.int64)
    if tolerance is not None:
        inputs["tolerance"] = np.asarray(tolerance, dtype=np.float64)
    if lengths is not None:
        inputs["lengths"] = np.ascontiguousarray(lengths, dtype=np.int64)
    return inputs
