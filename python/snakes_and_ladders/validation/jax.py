"""JAX as an oracle and a timing reference for the gradients HMC spends (issue #991).

JAX's reverse mode shares no code with PyTorch's autograd, and JAX is a second
autodiff stack the package does not take on: autodiff here is PyTorch by
decision, so JAX runs only in ``scripts/jax.py``, in a subprocess, for
validation and benchmarking.

:func:`gradients` returns JAX's value and gradient of a target at a batch of
points, with three timings: per point eagerly, per point under ``jit``, and
the whole batch under ``jit(vmap)``, the bound a sampler that evaluated many
points at once could reach. The targets are the diagonal and dense Gaussians
of :mod:`snakes_and_ladders.validation.gaussian` and the one-dimensional
Gaussian mixture of :class:`snakes_and_ladders.opt.mixture.GaussianMixtureObjective`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.validation.runner import run

#: The script this adapter runs.
SCRIPT = "jax"


@dataclass(frozen=True)
class Gradients:
    """JAX's values and gradients at a batch of points, and what they cost."""

    values: np.ndarray
    gradients: np.ndarray
    #: Median seconds per point under ``jit``, compilation excluded.
    seconds: float
    #: Median seconds per point without compilation.
    eager_seconds: float
    #: Seconds for the whole batch under ``jit(vmap)``.
    batch_seconds: float
    #: Peak resident bytes the ``jit`` loop added.
    peak_bytes: int


def gradients(
    points: np.ndarray,
    *,
    precision: np.ndarray | None = None,
    observations: np.ndarray | None = None,
    n_components: int | None = None,
    n_states: int | None = None,
) -> Gradients:
    """JAX's value and gradient at each row of ``points``.

    Give ``precision`` for the Gaussian target, ``observations`` and
    ``n_components`` for the mixture, or ``observations``, ``(n_sequences,
    length)``, and ``n_states`` for the Gaussian HMM (issue #997).
    """
    inputs: dict[str, np.ndarray] = {
        "points": np.ascontiguousarray(points, dtype=np.float64)
    }
    if precision is not None:
        inputs["target"] = np.asarray("gaussian")
        inputs["precision"] = np.ascontiguousarray(precision, dtype=np.float64)
    elif observations is not None and n_states is not None:
        inputs["target"] = np.asarray("hmm")
        inputs["observations"] = np.ascontiguousarray(observations, dtype=np.float64)
        inputs["n_states"] = np.asarray(n_states, dtype=np.int64)
    elif observations is not None and n_components is not None:
        inputs["target"] = np.asarray("mixture")
        inputs["observations"] = np.ascontiguousarray(observations, dtype=np.float64)
        inputs["n_components"] = np.asarray(n_components, dtype=np.int64)
    else:
        msg = "give a precision, or observations and n_components or n_states"
        raise ValueError(msg)
    result = run(SCRIPT, inputs)
    out = result.outputs
    return Gradients(
        values=out["values"],
        gradients=out["gradients"],
        seconds=result.seconds,
        eager_seconds=float(out["eager_seconds"]),
        batch_seconds=float(out["batch_seconds"]),
        peak_bytes=result.peak_bytes,
    )
